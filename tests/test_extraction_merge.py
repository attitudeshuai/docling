# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for cross-page merging of structured extraction results."""

import copy
import json
from types import SimpleNamespace
from typing import Optional

from docling_core.types.doc import Size
from pydantic import BaseModel

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.base_models import (
    ConversionStatus,
    InputFormat,
    VlmStopReason,
)
from docling.datamodel.document import InputDocument
from docling.datamodel.extraction import (
    ExtractedPageData,
    ExtractionResult,
    FieldPresence,
    MergedExtractionData,
    PageMergeStatus,
)
from docling.datamodel.extraction_options import (
    ExtractionMergeOptions,
    FieldMergeStrategy,
)
from docling.datamodel.settings import DocumentLimits
from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline
from docling.utils.extraction_merge import merge_page_results


class _Address(BaseModel):
    city: str
    zip_code: str


class _Invoice(BaseModel):
    bill_no: str
    total: float
    note: Optional[str] = None
    address: Optional[_Address] = None
    items: list = []


def _page(page_no: int, data, errors: Optional[list] = None) -> ExtractedPageData:
    return ExtractedPageData(page_no=page_no, extracted_data=data, errors=errors or [])


# ------------------------------- merge rules -------------------------------


def test_merge_scalar_priority_and_conflict_provenance() -> None:
    # Pages are intentionally provided out of order.
    pages = [
        _page(3, {"bill_no": "C"}),
        _page(1, {"bill_no": "A"}),
        _page(2, {"bill_no": "B"}),
    ]

    last = merge_page_results(pages, _Invoice, ExtractionMergeOptions())
    assert last.status == ConversionStatus.SUCCESS
    assert last.data["bill_no"] == "C"
    info = last.fields["bill_no"]
    assert info.presence == FieldPresence.FOUND
    assert info.chosen_page == 3
    assert info.source_pages == [1, 2, 3]
    assert info.has_conflict is True
    assert [(c.page_no, c.value) for c in info.candidates] == [
        (1, "A"),
        (2, "B"),
        (3, "C"),
    ]

    first = merge_page_results(
        pages, _Invoice, ExtractionMergeOptions(strategy=FieldMergeStrategy.FIRST_PAGE)
    )
    assert first.data["bill_no"] == "A"
    assert first.fields["bill_no"].chosen_page == 1
    # Same value across pages is a selection, not a conflict.
    consistent = merge_page_results(
        [_page(1, {"bill_no": "X"}), _page(2, {"bill_no": "X"})],
        {"bill_no": "string"},
        ExtractionMergeOptions(),
    )
    assert consistent.fields["bill_no"].has_conflict is False


def test_merge_distinguishes_missing_from_empty() -> None:
    pages = [
        _page(1, {"bill_no": "A", "note": None}),
        _page(2, {"note": None}),
    ]

    merged = merge_page_results(pages, _Invoice, ExtractionMergeOptions())

    # Explicit JSON null survives as a key with null value...
    assert merged.data["note"] is None
    assert merged.fields["note"].presence == FieldPresence.EMPTY
    assert merged.fields["note"].source_pages == [1, 2]
    # ...whereas a field absent from every page is omitted from data.
    assert "total" not in merged.data
    assert merged.fields["total"].presence == FieldPresence.MISSING
    assert merged.fields["total"].candidates == []
    assert merged.fields["address"].presence == FieldPresence.MISSING
    assert merged.fields["address.city"].presence == FieldPresence.MISSING
    assert merged.fields["address.zip_code"].presence == FieldPresence.MISSING

    # String templates also drive missing-field detection.
    merged_str = merge_page_results(
        [_page(1, {"bill_no": "A"})],
        '{"bill_no": "string", "total": "number"}',
        ExtractionMergeOptions(),
    )
    assert merged_str.fields["total"].presence == FieldPresence.MISSING


def test_merge_nested_dict_lists_and_extra_fields() -> None:
    pages = [
        _page(
            1,
            {
                "bill_no": "A",
                "address": {"city": "X"},
                "items": [{"id": 1}],
                "unexpected_field": "kept",
            },
        ),
        _page(
            2,
            {
                "bill_no": "A",
                "address": {"zip_code": "Z"},
                "items": [{"id": 2}, {"id": 3}],
            },
        ),
    ]

    merged = merge_page_results(pages, _Invoice, ExtractionMergeOptions())

    assert merged.data["address"] == {"city": "X", "zip_code": "Z"}
    assert merged.fields["address.city"].source_pages == [1]
    assert merged.fields["address.zip_code"].source_pages == [2]
    # List fields concatenate in page order instead of overwriting.
    assert merged.data["items"] == [{"id": 1}, {"id": 2}, {"id": 3}]
    # Fields readable on any page but absent from the template are not lost.
    assert merged.data["unexpected_field"] == "kept"
    assert merged.fields["unexpected_field"].presence == FieldPresence.FOUND


def test_merge_excludes_failed_and_unparseable_pages_but_continues() -> None:
    pages = [
        _page(1, {"bill_no": "A"}),
        _page(2, None, errors=["model crashed"]),
        # Extraction ran but the model output was not parseable JSON: no error,
        # no dict payload.
        ExtractedPageData(page_no=3, extracted_data=None, raw_text="not json"),
        _page(4, {"bill_no": "D"}),
    ]

    merged = merge_page_results(pages, _Invoice, ExtractionMergeOptions())

    assert merged.status == ConversionStatus.PARTIAL_SUCCESS
    assert merged.pages_total == 4
    assert merged.pages_merged == 2
    assert merged.pages_failed == 1
    assert [p.status for p in merged.pages] == [
        PageMergeStatus.MERGED,
        PageMergeStatus.FAILED,
        PageMergeStatus.UNPARSEABLE,
        PageMergeStatus.MERGED,
    ]
    # The failed page keeps exactly one error message.
    failed = merged.pages[1]
    assert failed.error == "model crashed"
    assert merged.data["bill_no"] == "D"

    all_failed = merge_page_results(
        [_page(1, None, errors=["boom"]), _page(2, None, errors=["boom"])],
        _Invoice,
        ExtractionMergeOptions(),
    )
    assert all_failed.status == ConversionStatus.FAILURE
    assert all_failed.pages_merged == 0
    assert all_failed.data == {}
    assert all_failed.fields["bill_no"].presence == FieldPresence.MISSING


def test_merge_does_not_mutate_or_alias_per_page_results() -> None:
    pages = [
        _page(1, {"items": [{"id": 1}], "bill_no": "A"}),
        _page(2, {"items": [{"id": 2}], "bill_no": "B"}),
    ]
    snapshot = copy.deepcopy(pages)

    merged = merge_page_results(pages, _Invoice, ExtractionMergeOptions())
    assert [p.model_dump() for p in pages] == [p.model_dump() for p in snapshot]

    # Mutating merged output (including candidate values) cannot reach pages.
    merged.data["items"].append({"id": 99})
    merged.data["bill_no"] = "Z"
    merged.fields["bill_no"].candidates[0].value = "tampered"
    page1_data = pages[0].extracted_data
    page2_data = pages[1].extracted_data
    assert page1_data is not None and page2_data is not None
    assert page1_data["items"] == [{"id": 1}]
    assert page1_data["bill_no"] == "A"
    assert page2_data["bill_no"] == "B"


# ----------------------------- serialization -------------------------------


def test_merged_result_roundtrips_and_accepts_legacy_payloads() -> None:
    merged = merge_page_results(
        [
            _page(1, {"bill_no": "A", "note": None}),
            _page(2, None, errors=["boom"]),
        ],
        _Invoice,
        ExtractionMergeOptions(),
    )

    restored = MergedExtractionData.model_validate_json(merged.model_dump_json())
    assert restored == merged
    assert restored.data["bill_no"] == "A"
    assert restored.pages[1].error == "boom"

    # Older payloads without newly added content must load without errors.
    assert MergedExtractionData.model_validate({}).pages_merged == 0
    legacy_page = ExtractedPageData.model_validate(
        {
            "page_no": 1,
            "extracted_data": {"bill_no": "A"},
            "raw_text": None,
            "errors": [],
        }
    )
    assert legacy_page.attempts == 1

    # Results produced before merging existed carry no "merged" section; it
    # must read back as None instead of raising.
    in_doc = InputDocument.model_construct(
        file="doc.pdf", document_hash="h", format=InputFormat.PDF
    )
    legacy_result = ExtractionResult(input=in_doc, pages=[legacy_page])
    legacy_payload = legacy_result.model_dump()
    assert legacy_payload["merged"] is None

    # Callers reading a possibly-old payload get None back instead of errors.
    def read_merged(payload: dict) -> Optional[MergedExtractionData]:
        section = payload.get("merged")
        return (
            MergedExtractionData.model_validate(section)
            if section is not None
            else None
        )

    assert read_merged(legacy_payload) is None

    # The merged document-level view round-trips as a self-contained object,
    # including when read back from a slice of a serialized ExtractionResult.
    serialized = merged.model_dump_json()
    restored_from_slice = MergedExtractionData.model_validate_json(serialized)
    assert restored_from_slice == merged


# --------------------------- pipeline integration --------------------------


class _Image:
    def __init__(self, page_no: int) -> None:
        self.page_no = page_no

    def close(self) -> None:
        pass


class _PageBackend(PdfPageBackend):
    def __init__(self, page_no: int) -> None:
        self._page_no = page_no

    @property
    def page_no(self) -> int:
        return self._page_no

    def get_text_in_rect(self, bbox):
        return ""

    def get_segmented_page(self):
        return None

    def get_text_cells(self):
        return []

    def get_bitmap_rects(self, scale: float = 1):
        return []

    def get_page_image(self, scale: float = 1, cropbox=None):
        return _Image(self.page_no)

    def get_size(self) -> Size:
        return Size(width=1, height=1)

    def is_valid(self) -> bool:
        return True

    def unload(self) -> None:
        pass


class _Backend(PdfDocumentBackend):
    supports_random_page_access = False

    def __init__(self, page_count: int) -> None:
        self._page_count = page_count

    def is_valid(self) -> bool:
        return True

    def load_page(self, page_no: int) -> PdfPageBackend:
        raise AssertionError("streaming extraction must not call load_page()")

    def page_count(self) -> int:
        return self._page_count

    def iter_pages(self):
        for page_no in range(1, self._page_count + 1):
            yield _PageBackend(page_no)

    def unload(self) -> None:
        return None


class _FlakyModel:
    """Fails ``fail_attempts[page]`` times per page before succeeding."""

    def __init__(self, fail_attempts: dict[int, int]) -> None:
        self._fail_attempts = fail_attempts
        self.calls: dict[int, int] = {}

    def process_images(self, images, prompt):
        page_no = images[0].page_no
        self.calls[page_no] = self.calls.get(page_no, 0) + 1
        if self.calls[page_no] <= self._fail_attempts.get(page_no, 0):
            raise RuntimeError(f"page {page_no} failure {self.calls[page_no]}")
        yield SimpleNamespace(
            text=json.dumps({"bill_no": f"v{page_no}"}),
            stop_reason=VlmStopReason.END_OF_SEQUENCE,
        )


def _run_pipeline(
    page_count: int,
    fail_attempts: dict[int, int],
    merge_options: Optional[ExtractionMergeOptions],
) -> ExtractionResult:
    pipeline = ExtractionVlmPipeline.__new__(ExtractionVlmPipeline)
    pipeline.pipeline_options = SimpleNamespace(
        document_timeout=None,
        vlm_options=SimpleNamespace(scale=1.0),
    )
    pipeline.vlm_model = _FlakyModel(fail_attempts)

    in_doc = InputDocument.model_construct(
        file="doc.pdf",
        document_hash="hash",
        format=InputFormat.PDF,
        valid=True,
        limits=DocumentLimits(page_range=(1, page_count)),
    )
    in_doc._backend = _Backend(page_count)
    return pipeline.execute(
        in_doc,
        raises_on_error=False,
        template={"bill_no": "string", "total": "number"},
        merge_options=merge_options,
    )


def test_pipeline_retries_failed_page_and_then_merges() -> None:
    result = _run_pipeline(
        3,
        fail_attempts={2: 2},
        merge_options=ExtractionMergeOptions(enabled=True, max_page_retries=2),
    )

    assert result.status == ConversionStatus.SUCCESS
    page2 = result.pages[1]
    assert page2.attempts == 3
    assert page2.errors == []
    assert page2.extracted_data == {"bill_no": "v2"}
    assert result.merged is not None
    assert result.merged.status == ConversionStatus.SUCCESS
    assert result.merged.data["bill_no"] == "v3"
    assert result.merged.fields["total"].presence == FieldPresence.MISSING
    assert [p.attempts for p in result.merged.pages] == [1, 3, 1]


def test_pipeline_exhausted_retries_only_excludes_that_page() -> None:
    result = _run_pipeline(
        3,
        fail_attempts={2: 99},
        merge_options=ExtractionMergeOptions(enabled=True, max_page_retries=1),
    )

    assert result.status == ConversionStatus.PARTIAL_SUCCESS
    page2 = result.pages[1]
    assert page2.attempts == 2
    assert page2.errors == ["page 2 failure 2"]
    assert page2.extracted_data is None
    # Other pages merged normally and remain visible in their raw form.
    assert result.pages[0].extracted_data == {"bill_no": "v1"}
    assert result.merged is not None
    assert result.merged.status == ConversionStatus.PARTIAL_SUCCESS
    assert result.merged.pages_failed == 1
    assert result.merged.pages[1].status == PageMergeStatus.FAILED
    assert result.merged.pages_merged == 2
    assert result.merged.data["bill_no"] == "v3"


def test_pipeline_without_merge_keeps_legacy_behavior() -> None:
    # max_page_retries is ignored when merging is disabled: one attempt, no
    # document-level result, and any failed page fails the whole document.
    result = _run_pipeline(
        3,
        fail_attempts={2: 99},
        merge_options=ExtractionMergeOptions(enabled=False, max_page_retries=3),
    )

    assert result.merged is None
    assert result.status == ConversionStatus.FAILURE
    assert result.pages[1].attempts == 1
    assert result.pages[1].errors == ["page 2 failure 1"]
    assert len(result.pages) == 3
