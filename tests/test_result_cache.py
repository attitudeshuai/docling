# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import json
import logging
import os
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from docling_core.types.doc.page import TextCellUnit

from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.base_models import (
    ConversionStatus,
    DocumentStream,
    InputFormat,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import (
    ConvertPipelineOptions,
    NativePdfPipelineOptions,
)
from docling.datamodel.settings import (
    BatchConcurrencySettings,
    DocumentLimits,
    scoped,
    settings,
)
from docling.document_converter import (
    ConversionResultCacheOptions,
    DocumentConverter,
    FormatOption,
    NativePdfFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.utils.result_cache import (
    ConversionResultStore,
    create_cache_fingerprint,
)

PDF_MULTIPAGE = Path("tests/data/pdf/sources/normal_4pages.pdf")
PDF_TEXT = Path("tests/data/pdf/sources/2305.03393v1-pg9.pdf")


def _md_stream(content: bytes = b"# Title\n\nHello **world**.\n") -> DocumentStream:
    return DocumentStream(name="hello.md", stream=BytesIO(content))


def _cache_options(tmp_path: Path, **kwargs: Any) -> ConversionResultCacheOptions:
    options: dict[str, Any] = {
        "enabled": True,
        "cache_dir": tmp_path / "result-cache",
    }
    options.update(kwargs)
    return ConversionResultCacheOptions(**options)


def _make_converter(
    *, delay: float = 0.0, **converter_kwargs: Any
) -> tuple[DocumentConverter, list[InputFormat]]:
    """Build a converter that records every real pipeline initialization."""
    calls: list[InputFormat] = []

    class _CountingConverter(DocumentConverter):
        def _get_pipeline(self, doc_format: InputFormat):
            calls.append(doc_format)
            if delay:
                # Widen the window so concurrent duplicates race into the
                # in-process single-flight guard.
                time.sleep(delay)
            return super()._get_pipeline(doc_format)

    return _CountingConverter(**converter_kwargs), calls


def _page_signature(result: ConversionResult) -> list:
    return [
        (
            page.page_no,
            page.size.width if page.size is not None else None,
            page.size.height if page.size is not None else None,
            len(page.cells),
            page.model_dump(mode="json", exclude={"predictions"}),
        )
        for page in result.pages
    ]


# --------------------------------------------------------------------- options


def test_fingerprint_is_stable_for_same_input_and_settings():
    def make_indoc() -> InputDocument:
        return InputDocument(
            path_or_stream=BytesIO(b"# Hello"),
            format=InputFormat.MD,
            backend=MarkdownDocumentBackend,
            filename="a.md",
            limits=DocumentLimits(),
        )

    fp_a = create_cache_fingerprint(
        make_indoc(), SimplePipeline, ConvertPipelineOptions()
    )
    fp_b = create_cache_fingerprint(
        make_indoc(), SimplePipeline, ConvertPipelineOptions()
    )
    assert fp_a == fp_b


def test_fingerprint_changes_with_content_and_page_range():
    base = lambda limits=None: InputDocument(  # noqa: E731
        path_or_stream=BytesIO(b"# Hello"),
        format=InputFormat.MD,
        backend=MarkdownDocumentBackend,
        filename="a.md",
        limits=limits or DocumentLimits(),
    )

    same = create_cache_fingerprint(base(), SimplePipeline, ConvertPipelineOptions())
    other_content = create_cache_fingerprint(
        InputDocument(
            path_or_stream=BytesIO(b"# Different"),
            format=InputFormat.MD,
            backend=MarkdownDocumentBackend,
            filename="a.md",
        ),
        SimplePipeline,
        ConvertPipelineOptions(),
    )
    other_range = create_cache_fingerprint(
        base(DocumentLimits(page_range=(2, 4))),
        SimplePipeline,
        ConvertPipelineOptions(),
    )

    assert other_content.content_hash != same.content_hash
    assert other_range.settings_hash != same.settings_hash


# ------------------------------------------------------------------ disabled


def test_cache_disabled_by_default_recomputes_every_time(tmp_path):
    converter, calls = _make_converter()
    converter.convert(_md_stream())
    converter.convert(_md_stream())
    assert len(calls) == 2
    # The global default cache directory location must never be created.
    assert not (tmp_path / "result-cache").exists()
    assert converter._result_store is None


# ------------------------------------------------------------------- hit/miss


def test_hit_skips_pipeline_and_matches_fresh_conversion(tmp_path):
    options = _cache_options(tmp_path)
    converter_a, calls_a = _make_converter(result_cache_options=options)
    fresh = converter_a.convert(_md_stream(), raises_on_error=False)

    # A brand-new converter simulates another process reusing the same ledger.
    converter_b, calls_b = _make_converter(result_cache_options=options)
    cached = converter_b.convert(_md_stream(), raises_on_error=False)

    assert len(calls_a) == 1
    assert calls_b == []
    assert converter_b.initialized_pipelines == {}
    assert cached.status == fresh.status == ConversionStatus.SUCCESS
    assert cached.document.export_to_dict() == fresh.document.export_to_dict()
    assert cached.document.export_to_markdown() == fresh.document.export_to_markdown()
    assert cached.errors == fresh.errors
    assert cached.input.document_hash == fresh.input.document_hash
    assert cached.input.file == fresh.input.file
    assert cached.input.format == fresh.input.format
    assert cached.input.limits == fresh.input.limits
    assert cached.pages == fresh.pages


def test_hit_restores_page_information_for_pdf(tmp_path, caplog):
    options = _cache_options(tmp_path)
    format_options = {
        InputFormat.PDF: NativePdfFormatOption(
            pipeline_options=NativePdfPipelineOptions()
        )
    }

    fresh = DocumentConverter(
        format_options=format_options, result_cache_options=options
    ).convert(PDF_TEXT)

    converter, calls = _make_converter(
        format_options=format_options, result_cache_options=options
    )
    caplog.set_level(logging.INFO, logger="docling.utils.result_cache")
    cached = converter.convert(PDF_TEXT)

    assert calls == []
    assert [p.model_dump(mode="json") for p in cached.pages] == [
        p.model_dump(mode="json") for p in fresh.pages
    ]
    assert _page_signature(cached) == _page_signature(fresh)
    assert cached.document.export_to_dict() == fresh.document.export_to_dict()
    assert cached.errors == fresh.errors
    assert "Reusing cached conversion result" in caplog.text


def test_page_range_change_forces_recomputation(tmp_path, caplog):
    options = _cache_options(tmp_path)
    format_options = {
        InputFormat.PDF: NativePdfFormatOption(
            pipeline_options=NativePdfPipelineOptions()
        )
    }
    caplog.set_level(logging.INFO)

    converter, calls = _make_converter(
        format_options=format_options, result_cache_options=options
    )

    first_two = converter.convert(PDF_MULTIPAGE, page_range=(1, 2))
    full = converter.convert(PDF_MULTIPAGE)
    first_two_again = converter.convert(PDF_MULTIPAGE, page_range=(1, 2))

    assert len(calls) == 3
    assert {p.page_no for p in first_two.pages} == {1, 2}
    assert {p.page_no for p in full.pages} == {1, 2, 3, 4}
    assert {p.page_no for p in first_two_again.pages} == {1, 2}
    assert "invalidate_settings_changed" in caplog.text


def test_pipeline_options_change_forces_recomputation(tmp_path):
    default_opts = NativePdfFormatOption(pipeline_options=NativePdfPipelineOptions())
    char_opts = NativePdfFormatOption(
        pipeline_options=NativePdfPipelineOptions(text_cell_unit=TextCellUnit.CHAR)
    )

    converter_a, calls_a = _make_converter(
        format_options={InputFormat.PDF: default_opts},
        result_cache_options=_cache_options(tmp_path),
    )
    converter_a.convert(PDF_TEXT)

    converter_b, calls_b = _make_converter(
        format_options={InputFormat.PDF: char_opts},
        result_cache_options=_cache_options(tmp_path),
    )
    changed = converter_b.convert(PDF_TEXT)

    converter_c, calls_c = _make_converter(
        format_options={InputFormat.PDF: char_opts},
        result_cache_options=_cache_options(tmp_path),
    )
    changed_again = converter_c.convert(PDF_TEXT)

    assert len(calls_a) == 1
    assert len(calls_b) == 1
    assert calls_c == []
    assert changed.status == changed_again.status == ConversionStatus.SUCCESS
    assert changed.document.export_to_dict() == changed_again.document.export_to_dict()


def test_different_contents_have_separate_entries(tmp_path):
    converter, calls = _make_converter(result_cache_options=_cache_options(tmp_path))

    converter.convert(_md_stream(b"# One\n"), raises_on_error=False)
    converter.convert(_md_stream(b"# Two\n"), raises_on_error=False)
    converter.convert(_md_stream(b"# One\n"), raises_on_error=False)

    assert len(calls) == 2
    entries = list((tmp_path / "result-cache").rglob("*.json"))
    assert len(entries) == 2


# ------------------------------------------------------------- concurrency


def test_same_input_twice_in_concurrent_batch_processed_once(tmp_path):
    converter, calls = _make_converter(
        result_cache_options=_cache_options(tmp_path), delay=0.15
    )

    # Two separate stream objects with identical bytes count as the same
    # input, plus one genuinely different document.
    sources = [
        _md_stream(b"# Same\n"),
        _md_stream(b"# Same\n"),
        _md_stream(b"# Other\n"),
    ]

    with scoped(
        perf=BatchConcurrencySettings(doc_batch_size=4, doc_batch_concurrency=4)
    ):
        results = list(converter.convert_all(sources, raises_on_error=False))

    assert len(calls) == 2
    assert [r.status for r in results] == [ConversionStatus.SUCCESS] * 3
    docs = [r.document.export_to_markdown() for r in results]
    assert docs[0] == docs[1]
    assert docs[0] != docs[2]


def test_file_lock_serializes_same_content(tmp_path):
    store = ConversionResultStore(
        _cache_options(tmp_path, lock_poll_seconds=0.02, lock_wait_seconds=10)
    )
    content_hash = "a" * 64
    started: list[float] = []
    release_first = threading.Event()
    first_inside = threading.Event()

    def first_worker() -> None:
        with store.lock_content(content_hash) as held:
            assert held
            started.append(time.monotonic())
            first_inside.set()
            release_first.wait(timeout=10)

    def second_worker() -> None:
        first_inside.wait(timeout=10)
        with store.lock_content(content_hash) as held:
            assert held
            started.append(time.monotonic())

    t1 = threading.Thread(target=first_worker)
    t2 = threading.Thread(target=second_worker)
    t1.start()
    first_inside.wait(timeout=10)
    t2.start()
    time.sleep(0.3)
    first_end = time.monotonic()
    release_first.set()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert len(started) == 2
    # The second worker only entered after the first one released the lock.
    assert started[1] >= first_end - 0.05
    assert not store._lock_path(content_hash).exists()


def test_stale_lock_is_taken_over(tmp_path):
    store = ConversionResultStore(
        _cache_options(
            tmp_path,
            lock_poll_seconds=0.02,
            lock_wait_seconds=10,
            stale_lock_seconds=1,
        )
    )
    content_hash = "b" * 64
    lock_path = store._lock_path(content_hash)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("pid=dead", encoding="utf-8")
    old_time = time.time() - 3600
    os.utime(lock_path, (old_time, old_time))

    with store.lock_content(content_hash) as held:
        assert held


# ------------------------------------------------------------ fault handling


def test_corrupted_entry_is_skipped_and_recomputed(tmp_path, caplog):
    options = _cache_options(tmp_path)
    converter = DocumentConverter(result_cache_options=options)
    converter.convert(_md_stream(), raises_on_error=False)

    entries = list((tmp_path / "result-cache").rglob("*.json"))
    assert len(entries) == 1
    # Simulate a process dying halfway through the write.
    entries[0].write_text('{"schema_version": 1, "result": {', encoding="utf-8")

    converter_b, calls = _make_converter(result_cache_options=options)
    caplog.set_level(logging.WARNING, logger="docling.utils.result_cache")
    recovered = converter_b.convert(_md_stream(), raises_on_error=False)

    assert len(calls) == 1
    assert recovered.status == ConversionStatus.SUCCESS
    assert "Ignoring corrupted conversion result cache entry" in caplog.text

    # The repaired entry is immediately reusable.
    converter_c, calls_c = _make_converter(result_cache_options=options)
    again = converter_c.convert(_md_stream(), raises_on_error=False)
    assert calls_c == []
    assert again.status == ConversionStatus.SUCCESS


def test_unwritable_cache_dir_degrades_with_one_warning(tmp_path, caplog):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")

    caplog.set_level(logging.WARNING, logger="docling.utils.result_cache")
    converter = DocumentConverter(
        result_cache_options=ConversionResultCacheOptions(
            enabled=True, cache_dir=blocker
        )
    )

    assert converter._result_store is not None
    assert converter._result_store.available is False
    # The gate must be inert, otherwise every conversion would retry the
    # unwritable directory and emit its own warning.
    assert converter._result_cache_gate is None

    result = converter.convert(_md_stream(), raises_on_error=False)
    result_again = converter.convert(_md_stream(b"# Other\n"), raises_on_error=False)
    assert result.status == ConversionStatus.SUCCESS
    assert result_again.status == ConversionStatus.SUCCESS

    warnings_from_cache = [
        record
        for record in caplog.records
        if record.name == "docling.utils.result_cache"
    ]
    assert len(warnings_from_cache) == 1
    assert "not writable" in warnings_from_cache[0].getMessage()
    assert list(blocker.rglob("*.json")) == []


def test_corrupted_batch_does_not_abort_other_documents(tmp_path, caplog):
    options = _cache_options(tmp_path)
    converter = DocumentConverter(result_cache_options=options)
    converter.convert(_md_stream(b"# A\n"), raises_on_error=False)
    converter.convert(_md_stream(b"# B\n"), raises_on_error=False)

    for entry in (tmp_path / "result-cache").rglob("*.json"):
        entry.write_bytes(b"")

    caplog.set_level(logging.WARNING)
    results = list(
        DocumentConverter(result_cache_options=options).convert_all(
            [_md_stream(b"# A\n"), _md_stream(b"# B\n")],
            raises_on_error=False,
        )
    )
    assert {r.status for r in results} == {ConversionStatus.SUCCESS}
    assert caplog.text.count("Ignoring corrupted conversion result cache entry") == 2


# -------------------------------------------------------------- invalidation


def test_version_change_invalidates_entry(tmp_path, caplog):
    options = _cache_options(tmp_path)
    converter_a = DocumentConverter(result_cache_options=options)
    converter_a.convert(_md_stream(), raises_on_error=False)

    entry = next((tmp_path / "result-cache").rglob("*.json"))
    envelope = json.loads(entry.read_text(encoding="utf-8"))
    envelope["versions"]["docling"] = "0.0.0-fake"
    entry.write_text(json.dumps(envelope), encoding="utf-8")

    converter_b, calls = _make_converter(result_cache_options=options)
    caplog.set_level(logging.INFO)
    result = converter_b.convert(_md_stream(), raises_on_error=False)

    assert len(calls) == 1
    assert result.status == ConversionStatus.SUCCESS
    assert "invalidate_docling_version" in caplog.text


# ------------------------------------------------------------------ failures


class _ExplodingPipeline(SimplePipeline):
    def _build_document(self, conv_res):  # type: ignore[override]
        raise RuntimeError("simulated pipeline boom")


class _ExplodingMdOption(FormatOption):
    pipeline_cls: type = _ExplodingPipeline
    backend: type = MarkdownDocumentBackend


def test_failure_is_cached_when_raises_disabled_but_recomputed_when_enabled(tmp_path):
    options = _cache_options(tmp_path)
    format_options = {InputFormat.MD: _ExplodingMdOption()}

    converter_a, calls_a = _make_converter(
        format_options=format_options, result_cache_options=options
    )
    failed = converter_a.convert(_md_stream(), raises_on_error=False)
    assert failed.status == ConversionStatus.FAILURE
    assert len(calls_a) == 1
    assert failed.errors[0].error_message == "simulated pipeline boom"
    assert failed.errors[0].module_name == "_ExplodingPipeline"

    converter_b, calls_b = _make_converter(
        format_options=format_options, result_cache_options=options
    )
    cached_failure = converter_b.convert(_md_stream(), raises_on_error=False)
    assert calls_b == []
    assert cached_failure.status == ConversionStatus.FAILURE
    assert cached_failure.errors == failed.errors

    converter_c, calls_c = _make_converter(
        format_options=format_options, result_cache_options=options
    )
    with pytest.raises(RuntimeError, match="failed"):
        converter_c.convert(_md_stream(), raises_on_error=True)
    assert len(calls_c) == 1
