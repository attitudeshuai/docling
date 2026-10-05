# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import threading
import time
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import Size
from PIL import Image
from pydantic import ValidationError

from docling.backend.abstract_backend import (
    AbstractDocumentBackend,
    PaginatedDocumentBackend,
)
from docling.backend.fallback_backend import (
    _FallbackPdfDocumentBackend,
    get_backend_attempts,
)
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.backend.noop_backend import NoOpBackend
from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import (
    BackendAttempt,
    BackendAttemptStatus,
    BackendCapabilities,
    BackendChainEntry,
    BackendExclusionReason,
    ConversionStatus,
    DocumentStream,
    FailureCategory,
    InputFormat,
)
from docling.datamodel.document import InputDocument
from docling.datamodel.settings import settings
from docling.document_converter import (
    DocumentConverter,
    FormatOption,
    NativePdfFormatOption,
    PdfFormatOption,
)
from docling.exceptions import BackendChainExhaustedError, DocumentLoadError
from docling.pipeline.asr_pipeline import AsrPipeline
from docling.pipeline.native_pdf_pipeline import NativePdfPipeline
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.video_pipeline import VideoPipeline

pytestmark = pytest.mark.cross_platform

PDF_PATH = Path("./tests/data/pdf/sources/2206.01062.pdf")


# --------------------------------------------------------------------- fakes


class _FakePage(PdfPageBackend):
    def __init__(self, backend, page_no, fail=False):
        self._doc_backend = backend
        self._no = page_no
        self._fail = fail
        type(backend).page_backend_instances.append((type(backend).__name__, page_no))

    @property
    def page_no(self):
        return self._no

    def is_valid(self):
        return not self._fail

    def get_error_message(self):
        return "synthetic bad page" if self._fail else ""

    def get_size(self):
        if self._fail:
            raise RuntimeError("synthetic page parse failure")
        return Size(width=100.0, height=200.0)

    def get_text_in_rect(self, bbox):
        return ""

    def get_segmented_page(self):
        if self._no in type(self._doc_backend).segmented_fail_pages:
            raise RuntimeError(f"synthetic segmented-page failure on {self._no}")
        return None

    def get_text_cells(self):
        return iter(())

    def get_bitmap_rects(self, scale=1):
        return iter(())

    def get_page_image(self, scale=1, cropbox=None):
        return Image.new("RGB", (2, 2))

    def unload(self):
        type(self._doc_backend).page_unloads.append(
            (type(self._doc_backend).__name__, self._no)
        )


class _FakePdfBackend(PdfDocumentBackend):
    supports_random_page_access = True
    page_count_value = 5
    fail_from_page: int | None = None
    fail_pages: set[int] = set()
    segmented_fail_pages: set[int] = set()
    init_delay: float = 0.0
    instances: list[str] = []
    instance_ids: list[tuple[str, int]] = []
    doc_unloads: list[str] = []
    page_backend_instances: list[tuple[str, int]] = []
    page_unloads: list[tuple[str, int]] = []

    def __init__(self, in_doc, path_or_stream, options=None):
        super().__init__(in_doc, path_or_stream, options)
        if self.init_delay:
            time.sleep(self.init_delay)
        type(self).instances.append(in_doc.file.name)
        type(self).instance_ids.append((type(self).__name__, id(self)))

    def is_valid(self):
        return True

    def page_count(self):
        return self.page_count_value

    def load_page(self, page_no):
        one_based = page_no + 1
        if self.fail_from_page is not None and one_based >= self.fail_from_page:
            raise RuntimeError(f"synthetic backend death at page {one_based}")
        return _FakePage(self, one_based, fail=one_based in self.fail_pages)

    def unload(self):
        type(self).doc_unloads.append(type(self).__name__)
        super().unload()


class _SequentialPdfBackend(_FakePdfBackend):
    supports_random_page_access = False
    fail_from_page = 3

    def load_page(self, page_no):
        raise NotImplementedError("sequential backend")

    def iter_pages(self):
        for one_based in range(1, self.page_count_value + 1):
            if self.fail_from_page is not None and one_based >= self.fail_from_page:
                raise RuntimeError(f"synthetic sequential death at page {one_based}")
            yield _FakePage(self, one_based)


def _reset_pdf_fakes(*classes):
    # All fakes share the class-level registries; every record carries the
    # concrete backend name so per-backend assertions filter on it.
    _FakePdfBackend.instances = []
    _FakePdfBackend.instance_ids = []
    _FakePdfBackend.doc_unloads = []
    _FakePdfBackend.page_backend_instances = []
    _FakePdfBackend.page_unloads = []


def _chain_input_doc(entries, stream_name="x.pdf"):
    stream = BytesIO(b"%PDF-fake-bytes")
    return (
        InputDocument(
            path_or_stream=stream,
            format=InputFormat.PDF,
            backend=entries[0].backend,
            filename=stream_name,
            backend_chain=entries,
        ),
        stream,
    )


class _LoadFailingMarkdown(MarkdownDocumentBackend):
    instances: list[str] = []

    def __init__(self, in_doc, path_or_stream, options=None):
        type(self).instances.append(in_doc.file.name)
        raise DocumentLoadError("synthetic unparseable markdown")


class _ImportFailingMarkdown(MarkdownDocumentBackend):
    instances: list[str] = []

    def __init__(self, in_doc, path_or_stream, options=None):
        type(self).instances.append(in_doc.file.name)
        raise ImportError("synthetic missing markdown dependency")


class _ConvertFailingMarkdown(MarkdownDocumentBackend):
    instances: list[str] = []

    def __init__(self, in_doc, path_or_stream, options=None):
        type(self).instances.append(in_doc.file.name)
        super().__init__(in_doc, path_or_stream, options)

    def convert(self):
        raise RuntimeError("synthetic convert explosion")


class _GoodMarkdown(MarkdownDocumentBackend):
    instances: list[str] = []

    def __init__(self, in_doc, path_or_stream, options=None):
        type(self).instances.append(in_doc.file.name)
        super().__init__(in_doc, path_or_stream, options)


_MD_BACKENDS = (
    _LoadFailingMarkdown,
    _ImportFailingMarkdown,
    _ConvertFailingMarkdown,
    _GoodMarkdown,
)


def _reset_md_fakes():
    for cls in _MD_BACKENDS:
        cls.instances = []


def _md_option(primary, *fallbacks):
    return FormatOption(
        pipeline_cls=SimplePipeline,
        backend=primary,
        fallback_backends=[BackendChainEntry(backend=cls) for cls in fallbacks],
    )


def _md_stream(name="doc.md"):
    return DocumentStream(name=name, stream=BytesIO(b"# Hello\n"))


# ----------------------------------------------------------- configuration


def test_backend_chain_order_options_and_capabilities():
    option = PdfFormatOption(
        fallback_backends=[
            BackendChainEntry(
                backend=PyPdfiumDocumentBackend,
                capabilities=BackendCapabilities(pagination=True, text_cells=True),
            )
        ]
    )
    chain = option.backend_chain
    assert chain[0].backend is option.backend
    assert chain[1].backend is PyPdfiumDocumentBackend
    assert chain[0].resolved_capabilities.pagination is True
    assert chain[0].resolved_capabilities.text_cells is True
    assert chain[1].resolved_capabilities.pagination is True


def _pypdfium_entry(**kwargs):
    return BackendChainEntry(backend=PyPdfiumDocumentBackend, **kwargs)


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: BackendChainEntry(backend=str), id="unknown"),
        pytest.param(
            lambda: BackendChainEntry(backend=AbstractDocumentBackend),
            id="abstract",
        ),
        pytest.param(
            lambda: PdfFormatOption(
                fallback_backends=[_pypdfium_entry(), _pypdfium_entry()]
            ),
            id="duplicate-fallback-entry",
        ),
        pytest.param(
            lambda: PdfFormatOption(
                backend=PyPdfiumDocumentBackend,
                fallback_backends=[_pypdfium_entry()],
            ),
            id="duplicate-with-primary",
        ),
        pytest.param(
            lambda: _pypdfium_entry(
                capabilities=BackendCapabilities(pagination=False, text_cells=True)
            ),
            id="capability-mismatch",
        ),
    ],
)
def test_invalid_chain_entries_rejected(factory):
    with pytest.raises(ValidationError):
        factory()


def test_invalid_chain_format_rejected_at_converter_construction():
    option = PdfFormatOption(
        fallback_backends=[BackendChainEntry(backend=MarkdownDocumentBackend)]
    )
    with pytest.raises(ValueError, match="does not support this format"):
        DocumentConverter(format_options={InputFormat.PDF: option})


class _PaginatedOnlyBackend(PaginatedDocumentBackend):
    """Supports MD by declaration but is not a DeclarativeDocumentBackend."""

    def __init__(self, in_doc, path_or_stream, options=None):
        super().__init__(in_doc, path_or_stream, options)

    def is_valid(self):
        return True

    def page_count(self):
        return 1

    def unload(self):
        super().unload()

    @classmethod
    def supports_pagination(cls):
        return True

    @classmethod
    def supported_formats(cls):
        return {InputFormat.MD}


def test_invalid_chain_pipeline_incompatibility_rejected():
    option = _md_option(MarkdownDocumentBackend, _PaginatedOnlyBackend)
    with pytest.raises(ValueError, match="not compatible with pipeline"):
        DocumentConverter(
            allowed_formats=[InputFormat.MD],
            format_options={InputFormat.MD: option},
        )


def test_supports_backend_class_matches_pipeline_protocol():
    assert SimplePipeline.supports_backend_class(MarkdownDocumentBackend)
    assert not SimplePipeline.supports_backend_class(PyPdfiumDocumentBackend)
    assert StandardPdfPipeline.supports_backend_class(PyPdfiumDocumentBackend)
    assert not StandardPdfPipeline.supports_backend_class(MarkdownDocumentBackend)
    assert NativePdfPipeline.supports_backend_class(PyPdfiumDocumentBackend)
    assert AsrPipeline.supports_backend_class(NoOpBackend)
    assert not AsrPipeline.supports_backend_class(MarkdownDocumentBackend)
    assert VideoPipeline.supports_backend_class(NoOpBackend)


# ------------------------------------------------ declarative fallback e2e


def test_declarative_fallback_on_load_failure():
    _reset_md_fakes()
    converter = DocumentConverter(
        allowed_formats=[InputFormat.MD],
        format_options={
            InputFormat.MD: _md_option(_LoadFailingMarkdown, _GoodMarkdown)
        },
    )
    result = converter.convert(_md_stream())

    assert result.status == ConversionStatus.SUCCESS
    assert result.errors == []
    attempts = result.backend_attempts
    assert [a.backend for a in attempts] == [
        "_LoadFailingMarkdown",
        "_GoodMarkdown",
    ]
    assert attempts[0].status == BackendAttemptStatus.SKIPPED
    assert attempts[0].reason == BackendExclusionReason.LOAD_FAILED
    assert "unparseable" in attempts[0].detail
    assert attempts[1].status == BackendAttemptStatus.SELECTED
    assert "Hello" in result.document.export_to_markdown()


def test_declarative_fallback_on_mid_conversion_failure():
    _reset_md_fakes()
    converter = DocumentConverter(
        allowed_formats=[InputFormat.MD],
        format_options={
            InputFormat.MD: _md_option(_ConvertFailingMarkdown, _GoodMarkdown)
        },
    )
    result = converter.convert(_md_stream())

    assert result.status == ConversionStatus.SUCCESS
    first, second = result.backend_attempts
    assert first.status == BackendAttemptStatus.FAILED
    assert first.reason == BackendExclusionReason.CONVERSION_FAILED
    assert "explosion" in first.detail
    assert second.status == BackendAttemptStatus.SELECTED


def test_declarative_chain_exhausted_without_raise_records_every_link():
    _reset_md_fakes()
    converter = DocumentConverter(
        allowed_formats=[InputFormat.MD],
        format_options={
            InputFormat.MD: _md_option(_LoadFailingMarkdown, _ImportFailingMarkdown)
        },
    )
    result = converter.convert(_md_stream(), raises_on_error=False)

    assert result.status == ConversionStatus.FAILURE
    assert result.input.valid is False
    first, second = result.backend_attempts
    assert first.status == BackendAttemptStatus.SKIPPED
    assert first.reason == BackendExclusionReason.LOAD_FAILED
    assert second.status == BackendAttemptStatus.SKIPPED
    assert second.reason == BackendExclusionReason.DEPENDENCY_UNAVAILABLE

    modules = {item.module_name for item in result.errors}
    assert "_LoadFailingMarkdown" in modules
    assert "_ImportFailingMarkdown" in modules
    assert all(
        item.category.value == "backend_failure"
        for item in result.errors
        if item.module_name
        in {
            "_LoadFailingMarkdown",
            "_ImportFailingMarkdown",
        }
    )
    # The original rejection entry is still present first.
    assert result.errors[0].component_type.value == "user_input"


def test_declarative_chain_exhausted_with_raise_keeps_exception_chain():
    _reset_md_fakes()
    converter = DocumentConverter(
        allowed_formats=[InputFormat.MD],
        format_options={
            InputFormat.MD: _md_option(_LoadFailingMarkdown, _ImportFailingMarkdown)
        },
    )
    with pytest.raises(Exception) as exc_info:
        converter.convert(_md_stream(), raises_on_error=True)

    cause = exc_info.value.__cause__
    assert isinstance(cause, DocumentLoadError)
    assert isinstance(cause, BackendChainExhaustedError)
    assert isinstance(cause.__cause__, ImportError)
    message = str(cause)
    assert "_LoadFailingMarkdown" in message
    assert "_ImportFailingMarkdown" in message


# ------------------------------------------------- paginated wrapper units


class _PdfA(_FakePdfBackend):
    fail_from_page = 3  # document-level death while fetching pages 3..5


class _PdfB(_FakePdfBackend):
    fail_from_page = None


class _PdfSingleBadPage(_FakePdfBackend):
    fail_from_page = None
    fail_pages = {3}


def test_pdf_fallback_document_death_keeps_served_pages():
    _reset_pdf_fakes(_PdfA, _PdfB, _FakePdfBackend)
    doc, stream = _chain_input_doc(
        [BackendChainEntry(backend=_PdfA), BackendChainEntry(backend=_PdfB)]
    )
    assert isinstance(doc._backend, _FallbackPdfDocumentBackend)
    assert doc.page_count == 5

    sizes = {}
    for page_backend in doc._backend.iter_pages():
        assert page_backend.is_valid()
        sizes[page_backend.page_no] = page_backend.get_size()
        page_backend.unload()

    assert sorted(sizes) == [1, 2, 3, 4, 5]
    a_pages = [p for _, p in _FakePdfBackend.page_backend_instances if _ == "_PdfA"]
    b_pages = [p for _, p in _FakePdfBackend.page_backend_instances if _ == "_PdfB"]
    # A never produced a page backend for page 3 (loading it raised); crucially
    # B is never asked for the pages A already served.
    assert a_pages == [1, 2]
    assert b_pages == [3, 4, 5]

    attempts = get_backend_attempts(doc._backend)
    assert attempts[0].served_pages == [1, 2]
    assert attempts[0].status == BackendAttemptStatus.FAILED
    assert attempts[0].reason == BackendExclusionReason.PAGE_PROCESSING_FAILED
    assert attempts[1].status == BackendAttemptStatus.SELECTED
    assert attempts[1].served_pages == [3, 4, 5]

    doc._backend.unload()
    assert stream.closed
    # Every instantiated candidate is released exactly once by the wrapper.
    assert _FakePdfBackend.doc_unloads.count("_PdfA") == 1
    assert _FakePdfBackend.doc_unloads.count("_PdfB") == 1


def test_pdf_fallback_single_bad_page_does_not_reprocess_other_pages():
    _reset_pdf_fakes(_PdfSingleBadPage, _PdfB, _FakePdfBackend)
    doc, stream = _chain_input_doc(
        [
            BackendChainEntry(backend=_PdfSingleBadPage),
            BackendChainEntry(backend=_PdfB),
        ]
    )
    sizes = {}
    for page_backend in doc._backend.iter_pages():
        assert page_backend.is_valid()
        sizes[page_backend.page_no] = page_backend.get_size()
        page_backend.unload()

    assert sorted(sizes) == [1, 2, 3, 4, 5]
    a_pages = [
        p
        for name, p in _FakePdfBackend.page_backend_instances
        if name == "_PdfSingleBadPage"
    ]
    b_pages = [
        p for name, p in _FakePdfBackend.page_backend_instances if name == "_PdfB"
    ]
    assert a_pages == [1, 2, 3, 4, 5]
    assert b_pages == [3]
    attempts = get_backend_attempts(doc._backend)
    assert attempts[0].served_pages == [1, 2, 4, 5]
    assert attempts[1].served_pages == [3]
    doc._backend.unload()
    assert stream.closed


def test_pdf_fallback_from_sequential_to_random_access():
    _reset_pdf_fakes(_SequentialPdfBackend, _PdfB, _FakePdfBackend)
    doc, stream = _chain_input_doc(
        [
            BackendChainEntry(backend=_SequentialPdfBackend),
            BackendChainEntry(backend=_PdfB),
        ]
    )
    # A sequential-only candidate makes the whole wrapper sequential.
    assert doc._backend.supports_random_page_access is False

    sizes = {}
    for page_backend in doc._backend.iter_pages():
        assert page_backend.is_valid()
        sizes[page_backend.page_no] = page_backend.get_size()
        page_backend.unload()
    assert sorted(sizes) == [1, 2, 3, 4, 5]

    b_pages = [
        p for name, p in _FakePdfBackend.page_backend_instances if name == "_PdfB"
    ]
    assert b_pages == [3, 4, 5]
    seq_pages = [
        p
        for name, p in _FakePdfBackend.page_backend_instances
        if name == "_SequentialPdfBackend"
    ]
    assert seq_pages == [1, 2]
    doc._backend.unload()
    assert stream.closed


def test_pdf_chain_exhausted_at_open_marks_invalid_with_per_link_reasons():
    class _PdfLoadBad(_FakePdfBackend):
        def __init__(self, in_doc, path_or_stream, options=None):
            super().__init__(in_doc, path_or_stream, options)
            raise DocumentLoadError("synthetic pdf load failure")

    class _PdfDepMissing(_FakePdfBackend):
        def __init__(self, in_doc, path_or_stream, options=None):
            super().__init__(in_doc, path_or_stream, options)
            raise ImportError("synthetic missing pdf dependency")

    _reset_pdf_fakes()
    doc, stream = _chain_input_doc(
        [
            BackendChainEntry(backend=_PdfLoadBad),
            BackendChainEntry(backend=_PdfDepMissing),
        ]
    )
    try:
        assert doc.valid is False
        failure = doc._backend._exhausted_error
        assert isinstance(failure, BackendChainExhaustedError)
        assert isinstance(failure.__cause__, ImportError)
        first, second = doc._backend.backend_attempts
        assert first.reason == BackendExclusionReason.LOAD_FAILED
        assert second.reason == BackendExclusionReason.DEPENDENCY_UNAVAILABLE
        assert doc._rejection.original_error is failure
    finally:
        doc._backend.unload()
        assert stream.closed


def test_stream_rewound_between_candidates():
    seen_positions: list[int] = []

    class _RecordingBad(_PdfA):
        def __init__(self, in_doc, path_or_stream, options=None):
            seen_positions.append(path_or_stream.tell())
            super().__init__(in_doc, path_or_stream, options)
            raise DocumentLoadError("bad for recording")

    class _RecordingGood(_PdfB):
        def __init__(self, in_doc, path_or_stream, options=None):
            seen_positions.append(path_or_stream.tell())
            super().__init__(in_doc, path_or_stream, options)

    stream = BytesIO(b"%PDF-data")
    doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.PDF,
        backend=_RecordingBad,
        filename="record.pdf",
        backend_chain=[
            BackendChainEntry(backend=_RecordingBad),
            BackendChainEntry(backend=_RecordingGood),
        ],
    )
    try:
        assert doc.valid is True
        assert seen_positions == [0, 0]
        assert stream.closed is False
    finally:
        doc._backend.unload()
    assert stream.closed


class _LoadSingleBadPdf(_FakePdfBackend):
    """Fails to LOAD page 3 only; every other page is healthy."""

    bad_load_pages: set[int] = {3}

    def load_page(self, page_no):
        one_based = page_no + 1
        if one_based in self.bad_load_pages:
            raise RuntimeError(f"synthetic load failure at page {one_based}")
        return _FakePage(self, one_based)


class _SegmentedFailingPdf(_FakePdfBackend):
    """get_size succeeds everywhere; get_segmented_page fails only on page 3."""

    segmented_fail_pages = {3}


def test_random_backend_kept_after_one_load_failure_and_serves_rest():
    # I-3: one load_page failure retires only that page; A keeps the rest.
    _reset_pdf_fakes()
    doc, stream = _chain_input_doc(
        [
            BackendChainEntry(backend=_LoadSingleBadPdf),
            BackendChainEntry(backend=_PdfB),
        ]
    )
    sizes = {}
    for page_backend in doc._backend.iter_pages():
        assert page_backend.is_valid()
        sizes[page_backend.page_no] = page_backend.get_size()
        page_backend.unload()
    assert sorted(sizes) == [1, 2, 3, 4, 5]

    a_loads = [
        p
        for name, p in _FakePdfBackend.page_backend_instances
        if name == "_LoadSingleBadPdf"
    ]
    b_loads = [
        p for name, p in _FakePdfBackend.page_backend_instances if name == "_PdfB"
    ]
    assert a_loads == [1, 2, 4, 5]
    assert b_loads == [3]

    attempts = get_backend_attempts(doc._backend)
    assert attempts[0].status == BackendAttemptStatus.SELECTED
    assert attempts[0].served_pages == [1, 2, 4, 5]
    assert attempts[1].status == BackendAttemptStatus.SELECTED
    assert attempts[1].served_pages == [3]
    doc._backend.unload()
    assert stream.closed


def test_page_attribution_migrates_when_later_method_fails():
    # I-4: the page completed by B is attributed to B alone.
    _reset_pdf_fakes()
    doc, stream = _chain_input_doc(
        [
            BackendChainEntry(backend=_SegmentedFailingPdf),
            BackendChainEntry(backend=_PdfB),
        ]
    )
    for page_backend in doc._backend.iter_pages():
        assert page_backend.is_valid()
        _ = page_backend.get_size()
        _ = page_backend.get_segmented_page()
        page_backend.unload()

    attempts = get_backend_attempts(doc._backend)
    served = {a.backend: set(a.served_pages) for a in attempts}
    assert served["_SegmentedFailingPdf"] == {1, 2, 4, 5}
    assert served["_PdfB"] == {3}
    doc._backend.unload()
    assert stream.closed


def test_failed_page_delegate_unloaded_on_switch():
    # I-1: replaced page delegates and the final delegate unload exactly once.
    _reset_pdf_fakes()
    doc, stream = _chain_input_doc(
        [
            BackendChainEntry(backend=_PdfSingleBadPage),
            BackendChainEntry(backend=_PdfB),
        ]
    )
    for page_backend in doc._backend.iter_pages():
        assert page_backend.is_valid()
        _ = page_backend.get_size()
        page_backend.unload()

    unloads = _FakePdfBackend.page_unloads
    assert unloads.count(("_PdfSingleBadPage", 3)) == 1
    assert unloads.count(("_PdfB", 3)) == 1
    doc._backend.unload()
    assert stream.closed


def test_concurrent_multi_page_failover_instantiates_candidate_once():
    # I-2: concurrent page failovers instantiate/release the candidate once.
    class _ConcurrentPrimary(_FakePdfBackend):
        fail_pages = {2, 3, 4, 5}

    class _ConcurrentRecovery(_FakePdfBackend):
        init_delay = 0.05

    _reset_pdf_fakes()
    doc, stream = _chain_input_doc(
        [
            BackendChainEntry(backend=_ConcurrentPrimary),
            BackendChainEntry(backend=_ConcurrentRecovery),
        ]
    )

    errors: list[BaseException] = []

    def consume(page_no: int) -> None:
        proxy = doc._backend.load_page(page_no - 1)
        try:
            assert proxy.is_valid()
            assert proxy.get_size().width == 100.0
            proxy.unload()
        except BaseException as exc:  # propagate test failures in threads
            errors.append(exc)

    threads = [
        threading.Thread(target=consume, args=(page_no,)) for page_no in range(2, 6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    recovery_ids = [
        instance_id
        for name, instance_id in _FakePdfBackend.instance_ids
        if name == "_ConcurrentRecovery"
    ]
    assert len(recovery_ids) == 1
    primary_ids = [
        instance_id
        for name, instance_id in _FakePdfBackend.instance_ids
        if name == "_ConcurrentPrimary"
    ]
    assert len(primary_ids) == 1

    attempts = get_backend_attempts(doc._backend)
    served = {a.backend: set(a.served_pages) for a in attempts}
    assert served["_ConcurrentPrimary"] == set()
    assert attempts[0].status == BackendAttemptStatus.FAILED
    assert served["_ConcurrentRecovery"] == {2, 3, 4, 5}
    doc._backend.unload()
    assert stream.closed
    assert _FakePdfBackend.doc_unloads.count("_ConcurrentRecovery") == 1
    assert _FakePdfBackend.doc_unloads.count("_ConcurrentPrimary") == 1


def test_empty_fallback_list_rejected():
    with pytest.raises(ValidationError, match="at least one entry"):
        PdfFormatOption(fallback_backends=[])


class _PaginatedMarkdown(MarkdownDocumentBackend):
    @classmethod
    def supports_pagination(cls):
        return True


def test_pagination_capability_mismatch_rejected():
    with pytest.raises(ValueError, match="pagination capability"):
        DocumentConverter(
            allowed_formats=[InputFormat.MD],
            format_options={
                InputFormat.MD: FormatOption(
                    pipeline_cls=SimplePipeline,
                    backend=MarkdownDocumentBackend,
                    fallback_backends=[BackendChainEntry(backend=_PaginatedMarkdown)],
                )
            },
        )


def test_directly_built_mixed_chain_fails_as_backend_failure():
    # Defensive: a mixed chain built directly still routes to BACKEND_FAILURE.
    stream = BytesIO(b"mixed")
    doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.MD,
        backend=MarkdownDocumentBackend,
        filename="mixed.md",
        backend_chain=[
            BackendChainEntry(backend=MarkdownDocumentBackend),
            BackendChainEntry(backend=PyPdfiumDocumentBackend),
        ],
    )
    assert doc.valid is False
    assert doc._rejection is not None
    assert doc._rejection.category == FailureCategory.BACKEND_FAILURE
    assert isinstance(doc._rejection.original_error, DocumentLoadError)


class _SequentialDieAtThree(_SequentialPdfBackend):
    page_count_value = 5


class _RandomBadLoadFour(_FakePdfBackend):
    bad_load_pages: set[int] = {4}

    def load_page(self, page_no):
        one_based = page_no + 1
        if one_based in self.bad_load_pages:
            raise RuntimeError(f"synthetic load failure at page {one_based}")
        return _FakePage(self, one_based)


def test_exhausted_single_page_does_not_kill_random_recovery_candidate():
    # I-6 (unit, wrapper.iter_pages): sequential A dies at page 3, random B
    # cannot serve page 4 but serves 3 and 5: page 4 is reported invalid and
    # B keeps serving the rest instead of being abandoned.
    _reset_pdf_fakes()
    doc, stream = _chain_input_doc(
        [
            BackendChainEntry(backend=_SequentialDieAtThree),
            BackendChainEntry(backend=_RandomBadLoadFour),
        ]
    )
    assert doc._backend.supports_random_page_access is False

    valid: dict[int, bool] = {}
    sizes: dict[int, tuple[int, float]] = {}
    for page_backend in doc._backend.iter_pages():
        page_no = page_backend.page_no
        valid[page_no] = page_backend.is_valid()
        if valid[page_no]:
            sizes[page_no] = page_backend.get_size()
        page_backend.unload()

    assert valid == {1: True, 2: True, 3: True, 4: False, 5: True}
    assert sorted(sizes) == [1, 2, 3, 5]

    attempts = get_backend_attempts(doc._backend)
    assert attempts[0].status == BackendAttemptStatus.FAILED
    assert attempts[0].served_pages == [1, 2]
    assert attempts[1].status == BackendAttemptStatus.SELECTED
    assert attempts[1].served_pages == [3, 5]
    assert attempts[1].reason == BackendExclusionReason.PAGE_PROCESSING_FAILED
    doc._backend.unload()
    assert stream.closed
    # B survived the single-page failure and is released once at the end.
    assert _FakePdfBackend.doc_unloads.count("_RandomBadLoadFour") == 1


class _SeqDieAtThreeReal(_SequentialPdfBackend):
    page_count_value = 9


class _RandomBadLoadFourReal(_FakePdfBackend):
    page_count_value = 9
    bad_load_pages: set[int] = {4}

    def load_page(self, page_no):
        one_based = page_no + 1
        if one_based in self.bad_load_pages:
            raise RuntimeError(f"synthetic load failure at page {one_based}")
        return _FakePage(self, one_based)


@pytest.mark.skipif(
    not PDF_PATH.exists(), reason="sample PDF is not available in this checkout"
)
def test_native_pipeline_single_exhausted_page_keeps_remaining_pages():
    # I-6 (end to end): sequential primary dies at page 3, random recovery
    # cannot load page 4; the document is partial with page 4 the only failed
    # page, and pages 5..9 are still produced.
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: NativePdfFormatOption(
                backend=_SeqDieAtThreeReal,
                fallback_backends=[BackendChainEntry(backend=_RandomBadLoadFourReal)],
            )
        },
    )
    result = converter.convert(PDF_PATH, raises_on_error=False)

    assert result.status == ConversionStatus.PARTIAL_SUCCESS
    assert sorted(result.document.pages) == [1, 2, 3, 5, 6, 7, 8, 9]
    first, second = result.backend_attempts
    assert first.status == BackendAttemptStatus.FAILED
    assert first.served_pages == [1, 2]
    assert second.status == BackendAttemptStatus.SELECTED
    assert second.served_pages == [3, 5, 6, 7, 8, 9]
    failed_items = [e for e in result.errors if e.page_no is not None]
    assert [e.page_no for e in failed_items] == [4]
    assert failed_items[0].category == FailureCategory.BACKEND_FAILURE


def test_abandoned_candidate_not_resurrected_by_late_inflight_call():
    # I-7: after the sequential candidate is abandoned at page 3, a late
    # successful method call on its in-flight page 2 must keep it FAILED.
    _reset_pdf_fakes()
    doc, stream = _chain_input_doc(
        [
            BackendChainEntry(backend=_SequentialDieAtThree),
            BackendChainEntry(backend=_PdfB),
        ]
    )
    pages = iter(doc._backend.iter_pages())
    page_one = next(pages)
    page_two = next(pages)
    assert page_one.is_valid() and page_two.is_valid()
    _ = page_one.get_size()
    _ = page_two.get_size()

    page_three = next(pages)  # triggers A's iterator death and recovery to B
    assert page_three.is_valid()

    # A page already flowing through the pipeline finishes after the abandon.
    _ = page_two.get_segmented_page()

    attempts = get_backend_attempts(doc._backend)
    assert attempts[0].backend == "_SequentialDieAtThree"
    assert attempts[0].status == BackendAttemptStatus.FAILED
    page_one.unload()
    page_two.unload()
    page_three.unload()
    for page_backend in pages:
        page_backend.unload()
    doc._backend.unload()
    assert stream.closed


class _MethodFailingOnThreeAndSeven(_FakePdfBackend):
    fail_pages = {3, 7}


class _LoadFailsThreeOnly(_FakePdfBackend):
    bad_load_pages: set[int] = {3}

    def load_page(self, page_no):
        one_based = page_no + 1
        if one_based in self.bad_load_pages:
            raise RuntimeError(f"synthetic load failure at page {one_based}")
        return _FakePage(self, one_based)


def test_recovery_candidate_kept_after_single_page_recovery_failure():
    # I-8: B fails to recover page 3 but can still serve page 7; it must not be
    # blacklisted for the whole document.
    _reset_pdf_fakes()
    doc, stream = _chain_input_doc(
        [
            BackendChainEntry(backend=_MethodFailingOnThreeAndSeven),
            BackendChainEntry(backend=_LoadFailsThreeOnly),
        ]
    )
    results = {}
    for page_no in (3, 7):
        proxy = doc._backend.load_page(page_no - 1)
        results[page_no] = proxy.is_valid() and proxy.get_size() is not None
        proxy.unload()

    assert results == {3: False, 7: True}
    b_pages = [
        p
        for name, p in _FakePdfBackend.page_backend_instances
        if name == "_LoadFailsThreeOnly"
    ]
    # Page 3 failed while LOADING (no page backend produced); page 7 is served.
    assert b_pages == [7]
    doc._backend.unload()
    assert stream.closed


def test_capabilities_model_is_frozen():
    capabilities = BackendCapabilities(pagination=True, text_cells=True)
    with pytest.raises(ValidationError):
        capabilities.pagination = False


def test_random_to_sequential_chain_order_rejected():
    from docling.backend.docling_parse_backend import (
        ThreadedDoclingParseDocumentBackend,
    )

    with pytest.raises(ValueError, match="sequential-only"):
        DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    backend=PyPdfiumDocumentBackend,
                    fallback_backends=[
                        BackendChainEntry(backend=ThreadedDoclingParseDocumentBackend)
                    ],
                )
            },
        )


# ------------------------------------------------------------- concurrency


def test_backend_fallback_isolated_across_concurrent_documents(monkeypatch):
    class _ConditionalFailMarkdown(MarkdownDocumentBackend):
        instances: list[str] = []

        def __init__(self, in_doc, path_or_stream, options=None):
            type(self).instances.append(in_doc.file.name)
            super().__init__(in_doc, path_or_stream, options)

        def convert(self):
            if self.file.name == "fallback.md":
                raise RuntimeError("synthetic failure for one document")
            return super().convert()

    _reset_md_fakes()
    converter = DocumentConverter(
        allowed_formats=[InputFormat.MD],
        format_options={
            InputFormat.MD: FormatOption(
                pipeline_cls=SimplePipeline,
                backend=_ConditionalFailMarkdown,
                fallback_backends=[BackendChainEntry(backend=_GoodMarkdown)],
            )
        },
    )

    sources = [
        _md_stream("primary.md"),  # succeeds on primary
        _md_stream("fallback.md"),  # needs fallback
    ]

    monkeypatch.setattr(settings.perf, "doc_batch_size", 2)
    monkeypatch.setattr(settings.perf, "doc_batch_concurrency", 2)

    results = list(converter.convert_all(sources, raises_on_error=False))
    by_name = {res.input.file.name: res for res in results}

    assert by_name["primary.md"].status == ConversionStatus.SUCCESS
    assert by_name["fallback.md"].status == ConversionStatus.SUCCESS

    primary_attempts = by_name["primary.md"].backend_attempts
    fallback_attempts = by_name["fallback.md"].backend_attempts
    assert [a.status for a in primary_attempts] == [
        BackendAttemptStatus.SELECTED,
        BackendAttemptStatus.SKIPPED,
    ]
    assert primary_attempts[1].reason is None
    assert [a.status for a in fallback_attempts] == [
        BackendAttemptStatus.FAILED,
        BackendAttemptStatus.SELECTED,
    ]

    # Per-document isolation: the shared chain spec made the primary backend
    # succeed for one document and fail for the other, without cross-talk.
    assert sorted(_ConditionalFailMarkdown.instances) == [
        "fallback.md",
        "primary.md",
    ]
    assert _GoodMarkdown.instances == ["fallback.md"]


# ------------------------------------------------------------------- e2e


class _FailPageTwoPdfium(PyPdfiumDocumentBackend):
    """Healthy document backend whose page 2 cannot report its size."""

    def load_page(self, page_no):
        page = super().load_page(page_no)
        if page_no + 1 == 2:

            def _failing_size():
                raise RuntimeError("synthetic page-2 size failure")

            page.get_size = _failing_size
        return page


@pytest.mark.skipif(
    not PDF_PATH.exists(), reason="sample PDF is not available in this checkout"
)
def test_native_pdf_pipeline_fails_over_per_page():
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: NativePdfFormatOption(
                backend=_FailPageTwoPdfium,
                fallback_backends=[BackendChainEntry(backend=PyPdfiumDocumentBackend)],
            )
        },
    )
    result = converter.convert(PDF_PATH, raises_on_error=False)

    assert result.status == ConversionStatus.SUCCESS, result.errors
    assert sorted(result.document.pages) == list(range(1, 10))

    first, second = result.backend_attempts
    assert first.backend == "_FailPageTwoPdfium"
    # A page-level failure does not retire the healthy primary backend: it
    # keeps serving every page it can; B is brought in only for page 2.
    assert first.status == BackendAttemptStatus.SELECTED
    assert first.reason == BackendExclusionReason.PAGE_PROCESSING_FAILED
    assert first.served_pages == [1, 3, 4, 5, 6, 7, 8, 9]
    assert second.backend == "PyPdfiumDocumentBackend"
    assert second.status == BackendAttemptStatus.SELECTED
    assert second.served_pages == [2]
    assert result.errors == []


@pytest.mark.skipif(
    not PDF_PATH.exists(), reason="sample PDF is not available in this checkout"
)
def test_standard_pdf_pipeline_fails_over_per_page():
    # The threaded standard pipeline consumes page backends via its producer
    # and preprocess stages; the fallback page proxy must transparently move
    # page 2 to the second backend there as well.
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=_FailPageTwoPdfium,
                fallback_backends=[BackendChainEntry(backend=PyPdfiumDocumentBackend)],
            )
        },
    )
    result = converter.convert(PDF_PATH, raises_on_error=False)

    assert result.status == ConversionStatus.SUCCESS, result.errors
    assert sorted(result.document.pages) == list(range(1, 10))
    first, second = result.backend_attempts
    assert first.status == BackendAttemptStatus.SELECTED
    assert first.served_pages == [1, 3, 4, 5, 6, 7, 8, 9]
    assert second.status == BackendAttemptStatus.SELECTED
    assert second.served_pages == [2]


def test_no_chain_declared_has_no_attempts_and_no_wrapper():
    converter = DocumentConverter(
        allowed_formats=[InputFormat.MD],
        format_options={
            InputFormat.MD: FormatOption(
                pipeline_cls=SimplePipeline,
                backend=MarkdownDocumentBackend,
            )
        },
    )
    result = converter.convert(_md_stream("plain.md"))
    assert result.status == ConversionStatus.SUCCESS
    assert result.backend_attempts == []
    assert type(result.input._backend) is MarkdownDocumentBackend


def test_backend_attempt_is_serializable():
    attempt = BackendAttempt(
        backend="X",
        status=BackendAttemptStatus.FAILED,
        reason=BackendExclusionReason.PAGE_PROCESSING_FAILED,
        detail="boom",
        served_pages=[3],
    )
    dumped = attempt.model_dump(mode="json")
    assert dumped == {
        "backend": "X",
        "status": "failed",
        "reason": "page_processing_failed",
        "detail": "boom",
        "served_pages": [3],
    }
