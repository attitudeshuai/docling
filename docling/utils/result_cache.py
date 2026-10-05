# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Persistent, cross-process cache of conversion results.

Unlike :mod:`docling.utils.pipeline_cache`, which only reuses initialized
pipeline instances inside one process, this module provides a *result*
ledger: finished :class:`~docling.datamodel.document.ConversionResult`
objects are persisted to disk and keyed by both the input content and the
effective conversion settings. A document that was already converted with
the same content, pipeline/backend options, limits and page range is served
from disk without entering the processing pipeline, including from a
different process.

Every lookup reports why it hit or missed (see :class:`CacheOutcome`), entries
are written atomically (temporary file + ``os.replace``), guarded by
``O_EXCL`` lock files across processes, and corrupted or half-written entries
are skipped and recomputed instead of failing the batch. If the cache
directory cannot be written, the cache degrades to plain recomputation after a
single warning.

The cache is opt-in via
:class:`docling.document_converter.ConversionResultCacheOptions` and stays
fully inert unless explicitly enabled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

from docling_core.types.doc import DoclingDocument
from pydantic import BaseModel, ConfigDict
from pydantic_core import PydanticSerializationError

from docling.datamodel.base_models import (
    ConfidenceReport,
    ConversionStatus,
    ErrorItem,
    Page,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.settings import settings
from docling.utils.pipeline_cache import create_pipeline_options_hash
from docling.utils.profiling import ProfilingItem
from docling.utils.utils import safe_version

if TYPE_CHECKING:
    from docling.datamodel.document import InputDocument
    from docling.datamodel.pipeline_options import PipelineOptions
    from docling.pipeline.base_pipeline import BasePipeline

_log = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_RESULT_KEYS = {
    "status",
    "errors",
    "pages",
    "timings",
    "confidence",
    "document",
}


class ConversionResultCacheOptions(BaseModel):
    """Configuration of the persistent conversion result cache.

    The cache is disabled by default. Enable it by passing an enabled instance
    as ``result_cache_options`` to
    :class:`~docling.document_converter.DocumentConverter`.

    Attributes:
        enabled: Whether finished conversion results are cached and reused.
        cache_dir: Directory holding the ledger. Defaults to
            ``settings.cache_dir / "conversion_results"``.
        lock_wait_seconds: How long a process waits for another process
            converting the same document to finish before giving up the
            cross-process guard and computing on its own.
        lock_poll_seconds: Polling interval while waiting on a cache lock.
        stale_lock_seconds: Age after which a lock file is considered left
            behind by a crashed process and may be taken over.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    cache_dir: Optional[Path] = None
    lock_wait_seconds: float = 3600.0
    lock_poll_seconds: float = 0.2
    stale_lock_seconds: float = 7200.0


class CacheOutcome(str, Enum):
    """Result of a cache lookup, explaining the decision."""

    HIT = "hit"
    """A valid entry matching content and settings was found."""

    MISS_NO_ENTRY = "miss_no_entry"
    """Nothing has ever been stored for this input content."""

    CORRUPT_ENTRY = "miss_corrupt_entry"
    """The entry is missing data, malformed, truncated or half-written."""

    STALE_SCHEMA = "invalidate_unsupported_schema"
    """The entry was written by an incompatible cache schema version."""

    STALE_CONTENT = "invalidate_content_changed"
    """The stored content fingerprint does not match the current input."""

    STALE_SETTINGS = "invalidate_settings_changed"
    """Options, limits or page range differ from those stored."""

    STALE_VERSION = "invalidate_docling_version"
    """The entry predates a Docling (or dependency) upgrade."""


class CacheFingerprint(NamedTuple):
    """Identity of one cacheable conversion.

    ``settings_detail`` is embedded in the entry in clear text so that an
    invalidation can be explained (which setting actually differs).
    """

    content_hash: str
    settings_hash: str
    input_format: str
    settings_detail: dict[str, Any]


class CacheDecision(NamedTuple):
    outcome: CacheOutcome
    envelope: Optional[dict[str, Any]]
    detail: str
    path: Optional[Path]


class SingleFlight:
    """In-process rendez-vous for threads converting the same input."""

    __slots__ = ("event",)

    def __init__(self) -> None:
        self.event = threading.Event()


def _md5_hex(payload: str) -> str:
    return hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8"), usedforsecurity=False).hexdigest()


def _hash_backend_options(backend_options: Optional[BaseModel]) -> Optional[str]:
    if backend_options is None:
        return None
    try:
        payload = type(backend_options).__qualname__ + backend_options.model_dump_json(
            serialize_as_any=True
        )
    except (PydanticSerializationError, ValueError, TypeError):
        payload = type(backend_options).__qualname__ + json.dumps(
            backend_options.model_dump(mode="json"),
            sort_keys=True,
            default=str,
            ensure_ascii=False,
        )
    return _md5_hex(payload)


def create_cache_fingerprint(
    in_doc: InputDocument,
    pipeline_class: type[BasePipeline],
    pipeline_options: PipelineOptions,
) -> CacheFingerprint:
    """Build the :class:`CacheFingerprint` of a pending conversion."""
    limits_dump = in_doc.limits.model_dump(mode="json")
    detail: dict[str, Any] = {
        "pipeline_class": (
            f"{pipeline_class.__module__}.{pipeline_class.__qualname__}"
        ),
        "pipeline_options_hash": create_pipeline_options_hash(pipeline_options),
        "backend_options_hash": _hash_backend_options(in_doc.backend_options),
        "limits": limits_dump,
    }
    settings_hash = _sha256_hex(
        json.dumps(
            detail,
            sort_keys=True,
            default=str,
            ensure_ascii=False,
        )
    )
    return CacheFingerprint(
        content_hash=in_doc.document_hash,
        settings_hash=settings_hash,
        input_format=in_doc.format.value,
        settings_detail=detail,
    )


class ConversionResultStore:
    """Filesystem-backed ledger of conversion results.

    One JSON entry per input content hash (sharded by hash prefix)::

        <cache_dir>/entries/<ab>/<content_hash>.json

    The entry embeds the settings fingerprint and component versions, which
    are re-checked on every lookup.
    """

    def __init__(self, options: ConversionResultCacheOptions) -> None:
        self._options = options
        base_dir = options.cache_dir or (settings.cache_dir / "conversion_results")
        self.root = Path(base_dir)
        self.entries_dir = self.root / "entries"
        self.versions = {
            "docling": safe_version("docling"),
            "docling-core": safe_version("docling-core"),
            "docling-parse": safe_version("docling-parse"),
        }
        self.available = self._probe_writable()

    # ------------------------------------------------------------------ setup

    def _probe_writable(self) -> bool:
        """Create the cache directories and verify write access once.

        Returns ``False`` (after emitting a single warning) when the directory
        cannot be created or written, so callers transparently fall back to
        recomputing every document.
        """
        try:
            self.entries_dir.mkdir(parents=True, exist_ok=True)
            probe_file = self.entries_dir / f".write-probe-{os.getpid()}"
            probe_file.write_bytes(b"ok")
            probe_file.unlink()
        except OSError as exc:
            _log.warning(
                "Conversion result cache is enabled but its directory %s is not "
                "writable (%s); falling back to converting every document and "
                "not caching anything.",
                self.root,
                exc,
            )
            return False
        return True

    def _shard_dir(self, content_hash: str) -> Path:
        return self.entries_dir / content_hash[:2]

    def entry_path(self, fingerprint: CacheFingerprint) -> Path:
        return self._shard_dir(fingerprint.content_hash) / (
            f"{fingerprint.content_hash}.json"
        )

    def _lock_path(self, content_hash: str) -> Path:
        return self._shard_dir(content_hash) / f".{content_hash}.lock"

    # ----------------------------------------------------------------- lookup

    def load(self, fingerprint: CacheFingerprint) -> CacheDecision:
        """Validate the stored entry without touching any pipeline code.

        Any parse or structure problem is reported as
        :attr:`CacheOutcome.CORRUPT_ENTRY`; the caller recomputes and
        overwrites instead of failing.
        """
        path = self.entry_path(fingerprint)
        if not path.exists():
            return CacheDecision(
                CacheOutcome.MISS_NO_ENTRY,
                None,
                "no entry exists for this content hash",
                path,
            )

        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return CacheDecision(
                CacheOutcome.CORRUPT_ENTRY,
                None,
                f"entry cannot be read as JSON: {exc}",
                path,
            )

        if not isinstance(envelope, dict):
            return CacheDecision(
                CacheOutcome.CORRUPT_ENTRY,
                None,
                "entry top-level structure is not an object",
                path,
            )
        if envelope.get("schema_version") != _SCHEMA_VERSION:
            return CacheDecision(
                CacheOutcome.STALE_SCHEMA,
                None,
                f"schema version is {envelope.get('schema_version')!r}, "
                f"expected {_SCHEMA_VERSION}",
                path,
            )

        stored_fp = envelope.get("fingerprint")
        result = envelope.get("result")
        stored_versions = envelope.get("versions")
        if not isinstance(stored_fp, dict) or not isinstance(result, dict):
            return CacheDecision(
                CacheOutcome.CORRUPT_ENTRY,
                None,
                "entry is missing a valid 'fingerprint' or 'result' section",
                path,
            )

        if stored_fp.get("content_hash") != fingerprint.content_hash:
            return CacheDecision(
                CacheOutcome.STALE_CONTENT,
                None,
                f"stored content hash {stored_fp.get('content_hash')!r} does not "
                f"match current {fingerprint.content_hash!r}",
                path,
            )
        if stored_fp.get("input_format") != fingerprint.input_format:
            return CacheDecision(
                CacheOutcome.STALE_SETTINGS,
                None,
                f"input format changed from {stored_fp.get('input_format')!r} to "
                f"{fingerprint.input_format!r}",
                path,
            )
        if stored_fp.get("settings_hash") != fingerprint.settings_hash:
            return CacheDecision(
                CacheOutcome.STALE_SETTINGS,
                None,
                "pipeline/backend options, limits or page range changed: "
                f"stored settings {stored_fp.get('settings_hash')!r} vs current "
                f"{fingerprint.settings_hash!r}",
                path,
            )

        if not isinstance(stored_versions, dict):
            return CacheDecision(
                CacheOutcome.CORRUPT_ENTRY,
                None,
                "entry is missing a valid 'versions' section",
                path,
            )
        changed_versions = [
            f"{name} {stored_versions.get(name)!r}->{self.versions.get(name)!r}"
            for name in self.versions
            if stored_versions.get(name) != self.versions[name]
        ]
        if changed_versions:
            return CacheDecision(
                CacheOutcome.STALE_VERSION,
                None,
                "component versions changed: " + ", ".join(changed_versions),
                path,
            )

        if not _RESULT_KEYS.issubset(result.keys()):
            missing = sorted(_RESULT_KEYS - result.keys())
            return CacheDecision(
                CacheOutcome.CORRUPT_ENTRY,
                None,
                f"result section is incomplete, missing: {', '.join(missing)}",
                path,
            )

        return CacheDecision(CacheOutcome.HIT, envelope, "entry matches", path)

    # ------------------------------------------------------------------ store

    def store(
        self,
        fingerprint: CacheFingerprint,
        conv_res: ConversionResult,
    ) -> bool:
        """Persist one result atomically.

        The payload is fully serialized in memory first, then written to a
        temporary file and moved into place with ``os.replace``: readers only
        ever see a complete file, and concurrent writers can only leave one
        well-formed content behind. Returns ``False`` (with a warning) when the
        write fails; cache problems never fail the conversion itself.
        """
        if not self.available:
            return False

        envelope = {
            "schema_version": _SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "versions": self.versions,
            "fingerprint": {
                "content_hash": fingerprint.content_hash,
                "settings_hash": fingerprint.settings_hash,
                "input_format": fingerprint.input_format,
                "settings": fingerprint.settings_detail,
            },
            "result": {
                "status": conv_res.status.value,
                "errors": [item.model_dump(mode="json") for item in conv_res.errors],
                "pages": [page.model_dump(mode="json") for page in conv_res.pages],
                "timings": {
                    name: item.model_dump(mode="json")
                    for name, item in conv_res.timings.items()
                },
                "confidence": conv_res.confidence.model_dump(mode="json"),
                # export_to_dict is the canonical stable DoclingDocument
                # representation, the same one used by ConversionAssets.save.
                "document": conv_res.document.export_to_dict(),
            },
        }

        try:
            payload = json.dumps(
                envelope, ensure_ascii=False, indent=2, default=str
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            _log.warning(
                "Could not serialize conversion result for caching: %s; "
                "continuing without caching this document.",
                exc,
            )
            return False

        shard_dir = self._shard_dir(fingerprint.content_hash)
        final_path = self.entry_path(fingerprint)
        tmp_path: Optional[Path] = None
        try:
            shard_dir.mkdir(parents=True, exist_ok=True)
            tmp_name = (
                f".{fingerprint.content_hash}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            tmp_path = shard_dir / tmp_name
            with tmp_path.open("wb") as tmp_file:
                tmp_file.write(payload)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_path, final_path)
            tmp_path = None
        except OSError as exc:
            _log.warning(
                "Failed to write conversion result cache entry %s: %s; continuing "
                "without caching this document.",
                final_path,
                exc,
            )
            return False
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        return True

    # ------------------------------------------------------------ cross-proc

    @contextmanager
    def lock_content(self, content_hash: str) -> Iterator[bool]:
        """Serialize conversions of identical content across processes.

        Uses an exclusively created lock file (``O_CREAT | O_EXCL``), which is
        atomic on both POSIX and Windows. Yields ``True`` when the lock was
        acquired and ``False`` after the wait timeout expires; in the latter
        case the caller computes anyway and still commits atomically, so a lock
        problem can never corrupt the ledger.
        """
        lock_path = self._lock_path(content_hash)
        acquired = False
        try:
            acquired = self._acquire_lock(lock_path)
            yield acquired
        finally:
            if acquired:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    _log.debug("Could not remove cache lock %s: %s", lock_path, exc)

    def _acquire_lock(self, lock_path: Path) -> bool:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _log.warning(
                "Cannot prepare cache lock directory %s: %s; proceeding without "
                "the cross-process guard.",
                lock_path.parent,
                exc,
            )
            return False

        deadline = time.monotonic() + self._options.lock_wait_seconds
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                # Owner still active (or crashed): steal only locks older than
                # the staleness threshold, otherwise wait for it to finish.
                if self._lock_is_stale(lock_path):
                    _log.warning(
                        "Removing stale conversion result cache lock %s; its "
                        "owner process likely crashed.",
                        lock_path,
                    )
                    try:
                        lock_path.unlink()
                    except (FileNotFoundError, OSError) as exc:
                        _log.debug("Could not remove stale lock %s: %s", lock_path, exc)
                elif time.monotonic() >= deadline:
                    self._warn_lock_timeout(lock_path)
                    return False
                else:
                    time.sleep(self._options.lock_poll_seconds)
                continue
            except OSError as exc:
                _log.warning(
                    "Failed to create cache lock %s: %s; proceeding without the "
                    "cross-process guard.",
                    lock_path,
                    exc,
                )
                return False

            try:
                os.write(
                    fd,
                    (
                        f"pid={os.getpid()} host={socket.gethostname()} "
                        f"since={datetime.now().isoformat(timespec='seconds')}"
                    ).encode(),
                )
            except OSError as exc:
                _log.debug("Could not write cache lock metadata: %s", exc)
            finally:
                os.close(fd)
            return True

    def _lock_is_stale(self, lock_path: Path) -> bool:
        try:
            age_seconds = time.time() - lock_path.stat().st_mtime
        except FileNotFoundError:
            return False
        except OSError:
            return False
        return age_seconds > self._options.stale_lock_seconds

    def _warn_lock_timeout(self, lock_path: Path) -> None:
        _log.warning(
            "Timed out after %.0fs waiting for cache lock %s held by another "
            "process; computing this document without the cross-process guard. "
            "The result is still committed atomically.",
            self._options.lock_wait_seconds,
            lock_path,
        )


def restore_result(
    in_doc: InputDocument,
    envelope: dict[str, Any],
    release_input: Callable[[InputDocument], None],
) -> ConversionResult:
    """Rebuild a ConversionResult from a cached envelope around ``in_doc``.

    The current InputDocument is kept (so ``input`` reflects this very
    request), while document structure, page information and error records are
    reconstructed from the cached payload. ``release_input`` releases the input
    backend opened during InputDocument construction, mirroring the normal
    pipeline's post-conversion cleanup.
    """
    result = envelope["result"]
    pages = [Page.model_validate(item) for item in result["pages"]]
    errors = [ErrorItem.model_validate(item) for item in result["errors"]]
    timings = {
        name: ProfilingItem.model_validate(item)
        for name, item in result["timings"].items()
    }
    conv_res = ConversionResult(
        input=in_doc,
        status=ConversionStatus(result["status"]),
        errors=errors,
        pages=pages,
        timings=timings,
        confidence=ConfidenceReport.model_validate(result["confidence"]),
        document=DoclingDocument.model_validate(result["document"]),
    )
    release_input(in_doc)
    return conv_res


class ResultCacheGate:
    """Orchestrates cache lookup, single-flight and (re)computation.

    Kept separate from ``DocumentConverter`` so the converter itself only
    decides *whether* the cache applies; this class owns the in-process
    dedup state and all hit/miss bookkeeping.
    """

    def __init__(self, store: ConversionResultStore) -> None:
        self._store = store
        # Per-key in-process single-flight guards: when the same content with
        # the same settings appears twice in one (or an overlapping) batch,
        # only one thread runs the conversion and the rest reuse its output.
        self._in_flight: dict[str, SingleFlight] = {}
        self._in_flight_guard = threading.Lock()

    def execute(
        self,
        *,
        in_doc: InputDocument,
        fingerprint: CacheFingerprint,
        raises_on_error: bool,
        run_pipeline: Callable[[], ConversionResult],
        release_input: Callable[[InputDocument], None],
    ) -> ConversionResult:
        cached_result = self._load_cached_result(
            fingerprint=fingerprint,
            in_doc=in_doc,
            raises_on_error=raises_on_error,
            release_input=release_input,
        )
        if cached_result is not None:
            return cached_result

        flight_key = f"{fingerprint.content_hash}:{fingerprint.settings_hash}"
        while True:
            with self._in_flight_guard:
                flight = self._in_flight.get(flight_key)
                is_owner = flight is None
                if is_owner:
                    flight = SingleFlight()
                    self._in_flight[flight_key] = flight

            if is_owner:
                try:
                    return self._compute_and_store_cached(
                        fingerprint=fingerprint,
                        in_doc=in_doc,
                        raises_on_error=raises_on_error,
                        run_pipeline=run_pipeline,
                        release_input=release_input,
                    )
                finally:
                    with self._in_flight_guard:
                        self._in_flight.pop(flight_key, None)
                    flight.event.set()

            flight.event.wait()
            cached_result = self._load_cached_result(
                fingerprint=fingerprint,
                in_doc=in_doc,
                raises_on_error=raises_on_error,
                release_input=release_input,
            )
            if cached_result is not None:
                return cached_result
            _log.debug(
                "The in-flight conversion of %s left no reusable cache entry; "
                "processing this document now.",
                in_doc.file,
            )

    def _compute_and_store_cached(
        self,
        *,
        fingerprint: CacheFingerprint,
        in_doc: InputDocument,
        raises_on_error: bool,
        run_pipeline: Callable[[], ConversionResult],
        release_input: Callable[[InputDocument], None],
    ) -> ConversionResult:
        with self._store.lock_content(fingerprint.content_hash) as lock_held:
            if not lock_held:
                _log.debug(
                    "Proceeding without the cross-process cache guard for %s.",
                    in_doc.file,
                )
            # Another process may have finished the same conversion while we
            # were waiting for the lock or on an in-flight thread. The miss
            # reason was already announced on the fast path, so only a hit is
            # logged again here.
            cached_result = self._load_cached_result(
                fingerprint=fingerprint,
                in_doc=in_doc,
                raises_on_error=raises_on_error,
                release_input=release_input,
                announce_miss=False,
            )
            if cached_result is not None:
                return cached_result

            conv_res = run_pipeline()
            # A failure produced with raises_on_error=False is a regular,
            # deterministic conversion outcome and is cached as such. When
            # raises_on_error=True the pipeline raises instead and no result
            # reaches here.
            if conv_res.status in {
                ConversionStatus.SUCCESS,
                ConversionStatus.PARTIAL_SUCCESS,
                ConversionStatus.FAILURE,
            }:
                self._store.store(fingerprint, conv_res)
            return conv_res

    def _load_cached_result(
        self,
        *,
        fingerprint: CacheFingerprint,
        in_doc: InputDocument,
        raises_on_error: bool,
        release_input: Callable[[InputDocument], None],
        announce_miss: bool = True,
    ) -> Optional[ConversionResult]:
        """Return a reusable cached result, or ``None`` when recomputation is due."""
        decision = self._store.load(fingerprint)

        if decision.outcome == CacheOutcome.HIT:
            # A cached FAILURE was produced by a raises_on_error=False run.
            # With raises_on_error=True a fresh conversion would raise from
            # inside the pipeline, so recompute instead to preserve semantics.
            # Check this before restoring, since restoring also releases the
            # input backend that the pipeline still needs in that case.
            if (
                raises_on_error
                and decision.envelope is not None
                and decision.envelope["result"]["status"]
                == ConversionStatus.FAILURE.value
            ):
                _log.debug(
                    "Ignoring cached FAILURE result for %s because "
                    "raises_on_error=True; recomputing.",
                    in_doc.file,
                )
                return None

            envelope = decision.envelope
            if envelope is None:
                # A HIT always carries an envelope; a missing one is treated
                # like a corrupted entry and triggers recomputation.
                _log.warning(
                    "The cache reported a hit for %s without an envelope at %s; "
                    "recomputing.",
                    in_doc.file,
                    decision.path,
                )
                return None

            try:
                conv_res = restore_result(in_doc, envelope, release_input)
            except Exception as exc:
                _log.warning(
                    "The cached conversion result at %s cannot be used: %s; "
                    "recomputing %s.",
                    decision.path,
                    exc,
                    in_doc.file,
                )
                return None

            _log.info(
                "Reusing cached conversion result for %s (content hash %s, "
                "settings hash %s).",
                in_doc.file,
                fingerprint.content_hash[:12],
                fingerprint.settings_hash[:12],
            )
            return conv_res

        if announce_miss:
            if decision.outcome == CacheOutcome.CORRUPT_ENTRY:
                _log.warning(
                    "Ignoring corrupted conversion result cache entry %s: %s; "
                    "recomputing %s.",
                    decision.path,
                    decision.detail,
                    in_doc.file,
                )
            elif decision.outcome == CacheOutcome.MISS_NO_ENTRY:
                _log.debug(
                    "Conversion result cache miss for %s: %s.",
                    in_doc.file,
                    decision.detail,
                )
            else:
                _log.info(
                    "Cached conversion result for %s is no longer valid (%s): %s; "
                    "recomputing.",
                    in_doc.file,
                    decision.outcome.value,
                    decision.detail,
                )
        return None
