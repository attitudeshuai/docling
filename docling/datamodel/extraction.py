# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Data models for document extraction functionality."""

from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

from docling.datamodel.base_models import ConversionStatus, ErrorItem, VlmStopReason
from docling.datamodel.document import InputDocument
from docling.datamodel.extraction_options import FieldMergeStrategy


class ExtractedPageData(BaseModel):
    """Data model for extracted content from a single page."""

    page_no: int = Field(..., description="1-indexed page number")
    extracted_data: Optional[Dict[str, Any]] = Field(
        None, description="Extracted structured data from the page"
    )
    raw_text: Optional[str] = Field(None, description="Raw extracted text")
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered during extraction for this page",
    )
    attempts: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of extraction attempts made for this page, including the "
            "initial attempt and any retries."
        ),
    )


class FieldPresence(str, Enum):
    """Presence state of a field after merging all participating pages.

    Attributes:
        FOUND: The field key is present on at least one page with a non-empty
            value.
        EMPTY: The field key is present on one or more pages, but every value
            is null/empty (e.g. JSON null, "", [], {}). Distinct from MISSING.
        MISSING: The template expects the field, but no page contains the key.
    """

    FOUND = "found"
    EMPTY = "empty"
    MISSING = "missing"


class PageMergeStatus(str, Enum):
    """Outcome of a single page with respect to the document-level merge."""

    MERGED = "merged"
    FAILED = "failed"
    UNPARSEABLE = "unparseable"


class FieldCandidate(BaseModel):
    """One observed value of a field on a specific page."""

    page_no: int = Field(..., description="1-indexed source page number")
    value: Optional[Any] = Field(
        default=None,
        description=(
            "Raw value observed on the source page. A present key with a JSON "
            "null value serializes as value=null, distinct from an absent key."
        ),
    )


class MergedFieldInfo(BaseModel):
    """Provenance of one field in the document-level merged result."""

    name: str = Field(..., description="Dotted field path, e.g. 'address.city'.")
    presence: FieldPresence = Field(
        ..., description="Whether the field was found, empty, or missing."
    )
    value: Optional[Any] = Field(
        default=None,
        description=(
            "Merged value selected by the configured strategy (scalar, list, "
            "or nested dict)."
        ),
    )
    chosen_page: Optional[int] = Field(
        default=None,
        description="Source page of the selected scalar value, if applicable.",
    )
    source_pages: List[int] = Field(
        default_factory=list,
        description="Pages on which the field key was present, in page order.",
    )
    candidates: List[FieldCandidate] = Field(
        default_factory=list,
        description=(
            "All observed values with their source pages, including losing "
            "candidates when values conflict across pages."
        ),
    )
    has_conflict: bool = Field(
        default=False,
        description="True when different scalar values were observed on different pages.",
    )


class MergedPageInfo(BaseModel):
    """Per-page participation information in the document-level merge."""

    page_no: int = Field(..., description="1-indexed page number")
    status: PageMergeStatus = Field(
        ..., description="Whether the page merged, failed, or had no mergeable data."
    )
    attempts: int = Field(
        default=1, description="Extraction attempts made for this page."
    )
    error: Optional[str] = Field(
        default=None,
        description="The single error recorded for a failed page, if any.",
    )


class MergedExtractionData(BaseModel):
    """Document-level result obtained by merging all per-page extractions.

    The merged ``data`` object follows the template structure. Every value that
    was readable on any page remains reachable either directly in ``data`` or,
    when it lost the priority resolution, in ``fields[...].candidates``.

    The model only contains plain serializable data and therefore supports
    ``model_dump_json`` / ``model_validate_json`` round-tripping. All fields
    have defaults so serialized results produced by older versions (which lack
    this object or some of its fields) can be read back without errors.
    """

    status: ConversionStatus = Field(
        default=ConversionStatus.PENDING,
        description=(
            "SUCCESS when every page merged, PARTIAL_SUCCESS when some pages "
            "were excluded, FAILURE when no page could be merged."
        ),
    )
    strategy: FieldMergeStrategy = Field(
        default=FieldMergeStrategy.LAST_PAGE,
        description="Priority rule used to select between conflicting values.",
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Merged field values; keys missing on every page are omitted.",
    )
    fields: Dict[str, MergedFieldInfo] = Field(
        default_factory=dict,
        description=(
            "Per-field provenance keyed by dotted path, including missing "
            "template fields and conflicting candidate values."
        ),
    )
    pages: List[MergedPageInfo] = Field(
        default_factory=list, description="Merge outcome of every extracted page."
    )
    pages_total: int = Field(default=0, description="Number of extracted pages.")
    pages_merged: int = Field(
        default=0, description="Number of pages that contributed to the merge."
    )
    pages_failed: int = Field(
        default=0,
        description="Number of pages that failed extraction after all retries.",
    )


class ExtractionResult(BaseModel):
    """Result of document extraction."""

    input: InputDocument
    status: ConversionStatus = ConversionStatus.PENDING
    errors: List[ErrorItem] = []

    # Pages field - always a list for consistency
    pages: List[ExtractedPageData] = Field(
        default_factory=list, description="Extracted data from each page"
    )

    merged: Optional[MergedExtractionData] = Field(
        default=None,
        description=(
            "Document-level merge of all pages. Populated only when merging is "
            "enabled; None otherwise. Never replaces or mutates 'pages'."
        ),
    )


# Type alias for template parameters that can be string, dict, or BaseModel
ExtractionTemplateType = Union[str, Dict[str, Any], BaseModel, Type[BaseModel]]
