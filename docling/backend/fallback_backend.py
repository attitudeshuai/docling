# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Per-document fallback backends.

A fallback backend wraps the ordered backend chain declared on a format option
and presents the *same* backend interfaces the pipelines already use
(:class:`DeclarativeDocumentBackend` / :class:`PdfDocumentBackend` /
:class:`PdfPageBackend`), so pipelines run unchanged:

* document-level unavailability (missing dependency, load failure,
  ``is_valid() == False``) advances to the next candidate before any output is
  produced;
* a page that a candidate cannot parse is transparently taken over by the next
  candidate, while pages already delivered to the pipeline are never requested
  again from another backend;
* every attempt (backend, outcome, reason, served pages) is recorded on the
  wrapper instance and later copied into ``ConversionResult.backend_attempts``.

All mutable state lives on one wrapper instance, i.e. one per input document,
so concurrent documents never interfere.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

from docling_core.types.doc import BoundingBox, DoclingDocument, Size
from docling_core.types.doc.page import SegmentedPdfPage, TextCell
from PIL import Image

from docling.backend.abstract_backend import (
    AbstractDocumentBackend,
    DeclarativeDocumentBackend,
    PaginatedDocumentBackend,
)
from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.backend_options import BaseBackendOptions
from docling.datamodel.base_models import (
    BackendAttempt,
    BackendAttemptStatus,
    BackendChainEntry,
    BackendExclusionReason,
    InputFormat,
)
from docling.exceptions import BackendChainExhaustedError, DocumentLoadError

if TYPE_CHECKING:
    from docling.datamodel.document import InputDocument
    from docling.pipeline.base_pipeline import BasePipeline

_log = logging.getLogger(__name__)


@dataclass
class _AttemptRecord:
    """Mutable per-candidate state for one document (not serialized)."""

    index: int
    backend_name: str
    status: BackendAttemptStatus = BackendAttemptStatus.SKIPPED
    reason: Optional[BackendExclusionReason] = None
    detail: Optional[str] = None
    served_pages: list[int] = field(default_factory=list)
    failed_pages: list[int] = field(default_factory=list)
    touched: bool = False
    # Terminal marker: the candidate was abandoned and must never be reported
    # as selected again, even if pages already flowing through the pipeline
    # finish on it afterwards.
    retired: bool = False


class _BaseFallbackBackend(AbstractDocumentBackend):
    """Common machinery for declarative and paginated fallback wrappers."""

    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
        specs: Sequence[BackendChainEntry],
        options: Optional[BaseBackendOptions] = None,
    ) -> None:
        # Bypass the cooperative Pdf/Declarative base constructors on purpose:
        # their format/options checks are the candidates' own responsibility,
        # and the wrapper must not fail construction before it can try a chain.
        AbstractDocumentBackend.__init__(self, in_doc, path_or_stream, options)
        self._in_doc: InputDocument = in_doc
        self._specs: tuple[BackendChainEntry, ...] = tuple(specs)
        self._instances: dict[int, AbstractDocumentBackend] = {}
        self._records: list[_AttemptRecord] = [
            _AttemptRecord(index=i, backend_name=spec.backend.__name__)
            for i, spec in enumerate(self._specs)
        ]
        self._active_index: Optional[int] = None
        self._last_error: Optional[BaseException] = None
        self._exhausted_error: Optional[BackendChainExhaustedError] = None
        # Candidates that already failed while recovering another candidate's
        # page, so they are not retried for further recoveries.
        self._recovery_dead: set[int] = set()
        self._closed = False
        # The threaded pipelines call into the same wrapper from producer and
        # stage worker threads. The lock guards candidate lifecycle and attempt
        # bookkeeping; slow backend calls themselves are made without it.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _detail(exc: BaseException) -> str:
        text = str(exc).strip()
        return text or exc.__class__.__name__

    @staticmethod
    def _classify(exc: BaseException, *, page: bool = False) -> BackendExclusionReason:
        if isinstance(exc, ImportError):
            return BackendExclusionReason.DEPENDENCY_UNAVAILABLE
        if isinstance(exc, DocumentLoadError):
            return BackendExclusionReason.LOAD_FAILED
        if page:
            return BackendExclusionReason.PAGE_PROCESSING_FAILED
        return BackendExclusionReason.CONVERSION_FAILED

    def _rewind_stream(self) -> None:
        stream = self.path_or_stream
        if isinstance(stream, BytesIO) and not stream.closed:
            stream.seek(0)

    def _safe_unload_candidate(self, inst: AbstractDocumentBackend) -> None:
        """Release a candidate without closing the shared input stream.

        The default ``AbstractDocumentBackend.unload`` closes the ``BytesIO``
        passed at construction. Several candidates share that stream, so it is
        detached before unloading; the wrapper closes it exactly once itself.
        """
        stream = self.path_or_stream
        try:
            if isinstance(stream, BytesIO) and not stream.closed:
                inst.path_or_stream = None
            inst.unload()
        except Exception:
            _log.warning(
                "Error while unloading fallback candidate %s",
                type(inst).__name__,
                exc_info=True,
            )

    def _instantiate_locked(self, index: int) -> Optional[AbstractDocumentBackend]:
        """Instantiate and validate candidate ``index``. Caller holds the lock."""
        spec = self._specs[index]
        record = self._records[index]
        existing = self._instances.get(index)
        if existing is not None:
            return existing
        record.touched = True
        self._rewind_stream()
        source = self.path_or_stream
        if source is None:
            raise RuntimeError(
                f"Cannot instantiate fallback backend {spec.backend.__name__}: "
                "the input source was already unloaded."
            )
        try:
            if spec.backend_options is not None:
                inst = spec.backend(
                    self._in_doc,
                    path_or_stream=source,
                    options=spec.backend_options,
                )
            else:
                inst = spec.backend(self._in_doc, path_or_stream=source)
        except Exception as exc:
            self._last_error = exc
            record.status = BackendAttemptStatus.SKIPPED
            record.reason = self._classify(exc)
            record.detail = self._detail(exc)
            _log.warning(
                "Fallback backend %s unavailable for %s: %s",
                spec.backend.__name__,
                self.file.name,
                record.detail,
            )
            return None

        try:
            valid = inst.is_valid()
        except Exception as exc:
            self._last_error = exc
            record.status = BackendAttemptStatus.SKIPPED
            record.reason = self._classify(exc)
            record.detail = self._detail(exc)
            self._safe_unload_candidate(inst)
            return None

        if not valid:
            record.status = BackendAttemptStatus.SKIPPED
            record.reason = BackendExclusionReason.INVALID_FOR_DOCUMENT
            record.detail = "The document backend could not parse the input."
            _log.warning(
                "Fallback backend %s rejected %s", spec.backend.__name__, self.file.name
            )
            self._safe_unload_candidate(inst)
            return None

        self._instances[index] = inst
        return inst

    def _activate(self, from_index: int = 0) -> tuple[int, AbstractDocumentBackend]:
        """Return an active candidate at or after ``from_index``.

        Raises :class:`BackendChainExhaustedError` when no remaining candidate
        can be opened for this document.
        """
        with self._lock:
            if (
                self._active_index is not None
                and self._active_index >= from_index
                and self._active_index in self._instances
            ):
                return self._active_index, self._instances[self._active_index]
            index = from_index
            while index < len(self._specs):
                inst = self._instances.get(index)
                if inst is None:
                    inst = self._instantiate_locked(index)
                if inst is not None:
                    self._active_index = index
                    return index, inst
                index += 1
            error = self._exhausted()
            self._exhausted_error = error
        raise error from self._last_error

    def _abandon(
        self,
        index: int,
        reason: BackendExclusionReason,
        exc: Optional[BaseException],
    ) -> None:
        """Mark an in-use candidate as failed and release it. Idempotent."""
        if not 0 <= index < len(self._records):
            return
        with self._lock:
            self._abandon_locked(index, reason, exc)

    def _abandon_locked(
        self,
        index: int,
        reason: BackendExclusionReason,
        exc: Optional[BaseException],
    ) -> None:
        record = self._records[index]
        inst = self._instances.pop(index, None)
        record.retired = True
        record.status = BackendAttemptStatus.FAILED
        record.reason = reason
        if exc is not None:
            record.detail = self._detail(exc)
            self._last_error = exc
        if inst is not None:
            self._safe_unload_candidate(inst)
        if self._active_index == index:
            self._active_index = None

    def _invoke_active(self, op: Any) -> Any:
        """Invoke ``op(instance)`` on the active candidate, advancing on error."""
        next_index = 0
        while True:
            index, inst = self._activate(next_index)
            try:
                return op(inst)
            except Exception as exc:
                self._abandon(index, self._classify(exc), exc)
                next_index = index + 1

    def note_page_served(self, index: int, page_no: int) -> None:
        """Attribute ``page_no`` to candidate ``index`` (ownership migration).

        Each delivered page has exactly one owner: when a later candidate
        finishes a page an earlier candidate only partially handled, the page
        moves to the later candidate's ``served_pages``.
        """
        with self._lock:
            for other_index, record in enumerate(self._records):
                if other_index != index and page_no in record.served_pages:
                    record.served_pages.remove(page_no)
            record = self._records[index]
            # A retired candidate stays failed even if a page already in the
            # pipeline finishes through it after the candidate was abandoned.
            if not record.retired and record.status in (
                BackendAttemptStatus.SKIPPED,
                BackendAttemptStatus.FAILED,
            ):
                record.status = BackendAttemptStatus.SELECTED
            if page_no not in record.served_pages:
                record.served_pages.append(page_no)

    def forget_page(self, page_no: int) -> None:
        """Remove ``page_no`` from every candidate when the chain can't serve it."""
        with self._lock:
            for record in self._records:
                if page_no in record.served_pages:
                    record.served_pages.remove(page_no)

    def _record_page_failure_locked(
        self, index: int, exc: BaseException, page_no: Optional[int] = None
    ) -> None:
        record = self._records[index]
        if record.reason is None:
            record.reason = BackendExclusionReason.PAGE_PROCESSING_FAILED
            record.detail = self._detail(exc)
        if page_no is not None and page_no not in record.failed_pages:
            record.failed_pages.append(page_no)
        if record.status == BackendAttemptStatus.SKIPPED:
            record.status = BackendAttemptStatus.FAILED

    @staticmethod
    def _finalize_record_locked(record: _AttemptRecord) -> None:
        """Distinguish "still effective" from "failed mid-document".

        A candidate with page failures stays selected only if it successfully
        served a page after (in page order) its last failed page; otherwise the
        failure was the end of its contribution and it is reported failed.
        """
        if record.retired:
            return
        if record.status != BackendAttemptStatus.SELECTED or not record.failed_pages:
            return
        last_served = max(record.served_pages, default=-1)
        last_failed = max(record.failed_pages)
        if last_failed > last_served:
            record.status = BackendAttemptStatus.FAILED

    def _finalize_records_locked(self) -> None:
        for record in self._records:
            self._finalize_record_locked(record)

    # ------------------------------------------------------------- outcome

    def _exclusion_lines(self) -> list[str]:
        lines = []
        with self._lock:
            for record in self._records:
                if record.status == BackendAttemptStatus.SELECTED:
                    line = f"{record.backend_name} [selected]"
                    if record.reason is not None:
                        line += (
                            f" (page failures {record.reason.value}: {record.detail})"
                        )
                    lines.append(line)
                elif record.reason is not None:
                    lines.append(
                        f"{record.backend_name} "
                        f"[{record.status.value}: {record.reason.value}]: "
                        f"{record.detail}"
                    )
                elif record.touched:
                    lines.append(f"{record.backend_name} [failed]")
                else:
                    lines.append(
                        f"{record.backend_name} [not tried: an earlier backend "
                        "handled the document]"
                    )
        return lines

    def _exhausted(self) -> BackendChainExhaustedError:
        return BackendChainExhaustedError(
            f"No backend in the fallback chain could handle {self.file.name}: "
            + "; ".join(self._exclusion_lines())
        )

    def chain_exclusion_summary(self) -> str:
        return "; ".join(self._exclusion_lines())

    @property
    def backend_attempts(self) -> list[BackendAttempt]:
        attempts = []
        with self._lock:
            self._finalize_records_locked()
            for record in self._records:
                detail = record.detail
                if (
                    record.status == BackendAttemptStatus.SKIPPED
                    and record.reason is None
                    and not record.touched
                ):
                    detail = (
                        "Not tried: an earlier backend in the chain handled the "
                        "document."
                    )
                attempts.append(
                    BackendAttempt(
                        backend=record.backend_name,
                        status=record.status,
                        reason=record.reason,
                        detail=detail,
                        served_pages=sorted(record.served_pages),
                    )
                )
        return attempts

    # --------------------------------------------------------- shared ABC API

    def is_valid(self) -> bool:
        try:
            return bool(self._invoke_active(lambda inst: inst.is_valid()))
        except BackendChainExhaustedError:
            return False

    def unload(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._closed = True
            self._finalize_records_locked()
            instances = [
                self._instances.pop(index, None) for index in list(self._instances)
            ]
            stream = self.path_or_stream
        for inst in instances:
            if inst is not None:
                self._safe_unload_candidate(inst)
        try:
            if isinstance(stream, BytesIO) and not stream.closed:
                stream.close()
        finally:
            with self._lock:
                self.path_or_stream = None


class _FallbackDeclarativeBackend(
    _BaseFallbackBackend, DeclarativeDocumentBackend, PaginatedDocumentBackend
):
    """Fallback wrapper for chains of whole-document declarative backends."""

    def convert(self) -> DoclingDocument:
        next_index = 0
        while True:
            index, inst = self._activate(next_index)
            if not isinstance(inst, DeclarativeDocumentBackend):
                # Configuration validation prevents this; stay defensive.
                self._abandon(
                    index,
                    BackendExclusionReason.CONVERSION_FAILED,
                    RuntimeError(
                        f"{type(inst).__name__} is not a DeclarativeDocumentBackend."
                    ),
                )
                next_index = index + 1
                continue
            try:
                document = inst.convert()
            except Exception as exc:
                self._abandon(index, self._classify(exc), exc)
                next_index = index + 1
                continue
            with self._lock:
                record = self._records[index]
                if record.status == BackendAttemptStatus.SKIPPED:
                    record.status = BackendAttemptStatus.SELECTED
            return document

    def supports_pagination(self) -> bool:  # type: ignore[override]
        _, inst = self._activate(0)
        return inst.supports_pagination()

    def page_count(self) -> int:
        _, inst = self._activate(0)
        if isinstance(inst, PaginatedDocumentBackend):
            return inst.page_count()
        return 0

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        # Wrapper only: every candidate owns and validates its own formats.
        return set()


class _FallbackPdfPageBackend(PdfPageBackend):
    """Page backend proxy that moves one page across fallback candidates."""

    def __init__(
        self,
        parent: _FallbackPdfDocumentBackend,
        page_no: int,
        initial_index: Optional[int] = None,
        initial_backend: Optional[PdfPageBackend] = None,
    ) -> None:
        self._parent = parent
        self._page_no = page_no
        self._index = initial_index
        self._delegate = initial_backend
        self._exhausted = False
        self._error_detail: Optional[str] = None
        self._exhausted_error: Optional[BackendChainExhaustedError] = None

    @classmethod
    def exhausted(
        cls,
        parent: _FallbackPdfDocumentBackend,
        page_no: int,
        error: BackendChainExhaustedError,
    ) -> _FallbackPdfPageBackend:
        """A page backend for a page no candidate could serve."""
        proxy = cls(parent=parent, page_no=page_no)
        proxy._exhausted = True
        proxy._error_detail = str(error)
        proxy._exhausted_error = error
        return proxy

    @property
    def page_no(self) -> int:
        return self._page_no

    def _ensure_delegate(self) -> PdfPageBackend:
        if self._delegate is None:
            try:
                self._index, self._delegate = self._parent.acquire_page(
                    page_no=self._page_no, start_index=0
                )
            except BackendChainExhaustedError as exhausted:
                self._exhausted = True
                self._error_detail = str(exhausted)
                self._exhausted_error = exhausted
                raise
        return self._delegate

    @staticmethod
    def _unload_delegate(delegate: Optional[PdfPageBackend]) -> None:
        if delegate is None:
            return
        try:
            delegate.unload()
        except Exception:
            _log.warning(
                "Error while unloading replaced fallback page backend",
                exc_info=True,
            )

    def _switch(self, exc: BaseException) -> bool:
        """Move this page to a later candidate. False when the chain ends."""
        if self._exhausted:
            return False
        previous = self._delegate
        replacement = self._parent.recover_page(
            page_no=self._page_no,
            failed_index=self._index,
            exc=exc,
        )
        if replacement is None:
            self._exhausted = True
            self._error_detail = self._parent.chain_exclusion_summary()
            # The page never completed on any backend: drop its attribution
            # and release the last failed native page handle.
            self._parent.forget_page(self._page_no)
            self._unload_delegate(previous)
            self._delegate = None
            return False
        self._index, self._delegate = replacement
        if replacement[1] is not previous:
            # Release the candidate page that lost the page; the replacement
            # page backend is released later, through the normal pipeline path.
            self._unload_delegate(previous)
        return True

    def _invoke(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if self._exhausted and self._exhausted_error is not None:
            raise self._exhausted_error
        while True:
            delegate = self._ensure_delegate()
            try:
                result = getattr(delegate, name)(*args, **kwargs)
            except Exception as exc:
                if self._switch(exc):
                    continue
                raise
            candidate_index = self._index
            if candidate_index is None:
                raise RuntimeError("Fallback page backend has no active candidate.")
            self._parent.note_page_served(candidate_index, self._page_no)
            return result

    def is_valid(self) -> bool:
        while not self._exhausted:
            try:
                delegate = self._ensure_delegate()
            except BackendChainExhaustedError:
                return False
            try:
                valid = bool(delegate.is_valid())
            except Exception as exc:
                if self._switch(exc):
                    continue
                return False
            if valid:
                # ``is_valid`` is only a gate, not a fetched page result; the
                # page is attributed to a backend once a content method
                # succeeds.
                return True
            detail = delegate.get_error_message() or "Page failed to parse."
            if not self._switch(RuntimeError(detail)):
                return False
        return False

    def get_error_message(self) -> str:
        if self._exhausted:
            return self._error_detail or (
                "All backends in the fallback chain failed for this page."
            )
        return self._ensure_delegate().get_error_message()

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        return self._invoke("get_text_in_rect", bbox)

    def get_segmented_page(self) -> Optional[SegmentedPdfPage]:
        return self._invoke("get_segmented_page")

    def get_text_cells(self) -> Iterable[TextCell]:
        return self._invoke("get_text_cells")

    def get_visible_text_cells(self) -> Optional[list[TextCell]]:
        return self._invoke("get_visible_text_cells")

    def get_bitmap_rects(self, scale: float = 1) -> Iterable[BoundingBox]:
        return self._invoke("get_bitmap_rects", scale)

    def has_content_in(
        self,
        *,
        bbox: BoundingBox,
        chars: bool = False,
        shapes: bool = True,
        bitmaps: bool = True,
    ) -> Optional[bool]:
        return self._invoke(
            "has_content_in",
            bbox=bbox,
            chars=chars,
            shapes=shapes,
            bitmaps=bitmaps,
        )

    def get_shape_lines(
        self,
        *,
        horizontal: bool = True,
        vertical: bool = True,
        tolerance: float = 1e-3,
    ) -> Optional[list[BoundingBox]]:
        return self._invoke(
            "get_shape_lines",
            horizontal=horizontal,
            vertical=vertical,
            tolerance=tolerance,
        )

    def get_connected_shape_bounding_boxes(
        self, *, tolerance: float = 0.0
    ) -> Optional[list[BoundingBox]]:
        return self._invoke("get_connected_shape_bounding_boxes", tolerance=tolerance)

    def get_page_image(
        self, scale: float = 1, cropbox: Optional[BoundingBox] = None
    ) -> Image.Image:
        return self._invoke("get_page_image", scale, cropbox)

    def get_size(self) -> Size:
        return self._invoke("get_size")

    def unload(self) -> None:
        delegate = self._delegate
        self._delegate = None
        if delegate is not None:
            try:
                delegate.unload()
            except Exception:
                _log.warning(
                    "Error while unloading fallback page backend", exc_info=True
                )


class _FallbackPdfDocumentBackend(_BaseFallbackBackend, PdfDocumentBackend):
    """Fallback wrapper for chains of PDF (paginated, page-level) backends."""

    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
        specs: Sequence[BackendChainEntry],
        options: Optional[BaseBackendOptions] = None,
    ) -> None:
        _BaseFallbackBackend.__init__(self, in_doc, path_or_stream, specs, options)

    def _activate_pdf(self, from_index: int = 0) -> tuple[int, PdfDocumentBackend]:
        index, inst = self._activate(from_index)
        assert isinstance(inst, PdfDocumentBackend)
        return index, inst

    def page_count(self) -> int:
        return int(self._invoke_active(lambda inst: inst.page_count()))

    def get_document_outline(self) -> list[Any]:
        return self._invoke_active(lambda inst: inst.get_document_outline())

    def _expected_page_nos(self) -> list[int]:
        start_page, end_page = self._in_doc.limits.page_range
        page_count = self.page_count()
        return list(range(max(1, start_page), min(page_count, end_page) + 1))

    @staticmethod
    def _load_page_from(inst: PdfDocumentBackend, page_no: int) -> PdfPageBackend:
        """Load one 1-based page from a candidate, random or sequential."""
        if getattr(inst, "supports_random_page_access", True):
            return inst.load_page(page_no - 1)

        wanted = {page_no}
        for page_backend in inst.iter_pages():
            if page_backend.page_no in wanted:
                return page_backend
            page_backend.unload()
        raise RuntimeError(
            f"Backend {type(inst).__name__} did not yield page {page_no}."
        )

    def _get_candidate_locked(self, index: int) -> Optional[PdfDocumentBackend]:
        inst = self._instances.get(index)
        if inst is None:
            inst = self._instantiate_locked(index)
        if inst is not None:
            assert isinstance(inst, PdfDocumentBackend)
            return inst
        return None

    def acquire_page(
        self, page_no: int, start_index: int = 0
    ) -> tuple[int, PdfPageBackend]:
        """Get a page backend for ``page_no`` from the first candidate serving it.

        A random-access candidate failing the single ``load_page`` call stays
        installed for its other pages; a sequential candidate whose iterator
        dies is abandoned (it cannot resume mid-document).
        Raises :class:`BackendChainExhaustedError` when no candidate serves it.
        """
        last_exc: Optional[BaseException] = None
        index = start_index
        while index < len(self._specs):
            with self._lock:
                inst = self._get_candidate_locked(index)
            if inst is None:
                index += 1
                continue
            try:
                page_backend = self._load_page_from(inst, page_no)
            except Exception as exc:
                last_exc = exc
                with self._lock:
                    if getattr(inst, "supports_random_page_access", True):
                        self._record_page_failure_locked(index, exc, page_no)
                    else:
                        self._abandon_locked(
                            index,
                            BackendExclusionReason.PAGE_PROCESSING_FAILED,
                            exc,
                        )
                index += 1
                continue
            return index, page_backend
        with self._lock:
            error = self._exhausted()
            self._exhausted_error = error
        raise error from last_exc

    def _recovery_candidate_locked(
        self, *, after_index: int
    ) -> Optional[tuple[int, PdfDocumentBackend]]:
        index = after_index + 1
        while index < len(self._specs):
            if index in self._recovery_dead:
                index += 1
                continue
            inst = self._get_candidate_locked(index)
            if inst is not None:
                return index, inst
            self._recovery_dead.add(index)
            index += 1
        return None

    def recover_page(
        self,
        page_no: int,
        failed_index: Optional[int],
        exc: BaseException,
    ) -> Optional[tuple[int, PdfPageBackend]]:
        """Fetch one failed page from a later candidate.

        The candidate that lost the page is NOT closed: its other pages (and
        pages already flowing through the pipeline) must keep working. Only the
        failed page moves to a later candidate.
        """
        if failed_index is not None:
            with self._lock:
                self._record_page_failure_locked(failed_index, exc, page_no)
        next_index = (failed_index + 1) if failed_index is not None else 0
        while True:
            with self._lock:
                replacement = self._recovery_candidate_locked(
                    after_index=next_index - 1
                )
            if replacement is None:
                return None
            index, inst = replacement
            try:
                page_backend = self._load_page_from(inst, page_no)
            except Exception as next_exc:
                with self._lock:
                    # A sequential candidate's iterator can only be consumed
                    # once, so it cannot get a second chance for another page.
                    # A random-access candidate can: its failure here is scoped
                    # to this page only.
                    if not getattr(inst, "supports_random_page_access", True):
                        self._recovery_dead.add(index)
                    self._record_page_failure_locked(index, next_exc, page_no)
                next_index = index + 1
                continue
            return index, page_backend

    def load_page(self, page_no: int) -> PdfPageBackend:
        # ``page_no`` is 0-based at the PdfDocumentBackend boundary.
        return _FallbackPdfPageBackend(parent=self, page_no=page_no + 1)

    def iter_pages(self) -> Iterator[PdfPageBackend]:
        remaining = set(self._expected_page_nos())
        next_index = 0
        while remaining:
            with self._lock:
                active = self._active_index
                if (
                    active is not None
                    and active >= next_index
                    and active in self._instances
                ):
                    index, inst = active, self._instances[active]
                    assert isinstance(inst, PdfDocumentBackend)
                else:
                    index, inst = self._activate_pdf(next_index)
            if getattr(inst, "supports_random_page_access", True):
                # Random-access recovery: one page being unavailable on every
                # candidate must not kill the document backend or stop the
                # remaining pages. Yield an exhausted proxy for that page and
                # keep going.
                for page_no in sorted(remaining):
                    try:
                        delegate_index, page_backend = self.acquire_page(
                            page_no=page_no, start_index=index
                        )
                    except BackendChainExhaustedError as exhausted:
                        remaining.discard(page_no)
                        yield _FallbackPdfPageBackend.exhausted(
                            self, page_no, exhausted
                        )
                        continue
                    remaining.discard(page_no)
                    yield _FallbackPdfPageBackend(
                        parent=self,
                        page_no=page_no,
                        initial_index=delegate_index,
                        initial_backend=page_backend,
                    )
                return

            try:
                for page_backend in inst.iter_pages():
                    page_no = page_backend.page_no
                    if page_no not in remaining:
                        page_backend.unload()
                        continue
                    remaining.discard(page_no)
                    yield _FallbackPdfPageBackend(
                        parent=self,
                        page_no=page_no,
                        initial_index=index,
                        initial_backend=page_backend,
                    )
                return
            except BackendChainExhaustedError:
                # No candidate left at all after the sequential iterator died.
                return
            except Exception as exc:
                # The sequential candidate's iterator died mid-document (it
                # cannot resume): abandon it and take the remaining page set
                # to the next candidate.
                with self._lock:
                    self._abandon_locked(
                        index,
                        BackendExclusionReason.PAGE_PROCESSING_FAILED,
                        exc,
                    )
                next_index = index + 1

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        # Wrapper only: every candidate owns and validates the PDF format.
        return set()


class _RandomAccessFallbackPdfBackend(_FallbackPdfDocumentBackend):
    """Fallback PDF wrapper whose candidates all support random page access."""

    supports_random_page_access: ClassVar[bool] = True


class _SequentialFallbackPdfBackend(_FallbackPdfDocumentBackend):
    """Fallback PDF wrapper containing a sequential-only candidate."""

    supports_random_page_access: ClassVar[bool] = False


def get_chain_failure(
    backend: AbstractDocumentBackend,
) -> Optional[BackendChainExhaustedError]:
    """Return the chain-exhausted error recorded by a fallback backend, if any."""
    if isinstance(backend, _BaseFallbackBackend):
        return backend._exhausted_error
    return None


def get_backend_attempts(
    backend: AbstractDocumentBackend,
) -> Optional[list[BackendAttempt]]:
    """Return the fallback attempt trace of a backend, or None without a chain."""
    if isinstance(backend, _BaseFallbackBackend):
        return backend.backend_attempts
    return None


def validate_backend_chain(
    doc_format: InputFormat,
    entries: Sequence[BackendChainEntry],
    pipeline_cls: type[BasePipeline],
) -> None:
    """Reject a fallback chain incompatible with its format/pipeline.

    Entries are checked in order: each backend must be unique, declare support
    for ``doc_format``, be drivable by ``pipeline_cls``, and expose the same
    pagination capability as the primary link. Anything wrong raises
    ``ValueError`` while the converter is constructed, before documents exist.
    """
    if not entries:
        return
    primary_pagination = entries[0].resolved_capabilities.pagination
    seen: set[type[AbstractDocumentBackend]] = set()
    seen_random_access = False
    for entry in entries:
        backend_cls = entry.backend
        label = f"format {doc_format.value!r} backend {backend_cls.__name__}"

        random_access = getattr(backend_cls, "supports_random_page_access", True)
        if seen_random_access and not random_access:
            raise ValueError(
                f"Invalid fallback chain for {label}: a sequential-only "
                "(non-random-access) backend cannot follow a random-access "
                "backend; declare the sequential backend before the "
                "random-access one."
            )
        if random_access:
            seen_random_access = True

        if backend_cls in seen:
            raise ValueError(
                f"Invalid fallback chain for {label}: the backend class appears "
                "more than once."
            )
        seen.add(backend_cls)

        if doc_format not in backend_cls.supported_formats():
            supported = sorted(fmt.value for fmt in backend_cls.supported_formats())
            raise ValueError(
                f"Invalid fallback chain for {label}: the backend does not "
                f"support this format (declared formats: {supported})."
            )

        if not pipeline_cls.supports_backend_class(backend_cls):
            raise ValueError(
                f"Invalid fallback chain for {label}: the backend is not "
                f"compatible with pipeline {pipeline_cls.__name__}."
            )

        if entry.resolved_capabilities.pagination != primary_pagination:
            raise ValueError(
                f"Invalid fallback chain for {label}: pagination capability "
                f"({entry.resolved_capabilities.pagination}) differs from the "
                f"primary backend ({primary_pagination}); all chain links must "
                "be interchangeable for this format."
            )


def make_fallback_backend(
    in_doc: InputDocument,
    path_or_stream: Union[BytesIO, Path],
    entries: Sequence[BackendChainEntry],
) -> AbstractDocumentBackend:
    """Build the fallback wrapper matching the protocols of ``entries``."""
    specs = tuple(entries)
    if not specs:
        raise ValueError("Cannot build a fallback backend from an empty chain.")
    if all(issubclass(spec.backend, PdfDocumentBackend) for spec in specs):
        all_random_access = all(
            getattr(spec.backend, "supports_random_page_access", True) for spec in specs
        )
        wrapper_cls = (
            _RandomAccessFallbackPdfBackend
            if all_random_access
            else _SequentialFallbackPdfBackend
        )
        return wrapper_cls(in_doc, path_or_stream, specs)
    if all(issubclass(spec.backend, DeclarativeDocumentBackend) for spec in specs):
        return _FallbackDeclarativeBackend(in_doc, path_or_stream, specs)
    names = [spec.backend.__name__ for spec in specs]
    raise ValueError(
        "Invalid fallback chain: backends must all be PdfDocumentBackend or "
        f"all DeclarativeDocumentBackend subclasses, got {names}."
    )
