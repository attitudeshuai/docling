# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from enum import Enum

from pydantic import BaseModel, Field


class ExtractionPromptStyle(str, Enum):
    NUEXTRACT = "nuextract"
    GRANITE_VISION = "granite_vision"


class FieldMergeStrategy(str, Enum):
    """Priority rule used when the same scalar field is found on several pages.

    Attributes:
        FIRST_PAGE: Take the value from the lowest-numbered page on which the
            field occurs.
        LAST_PAGE: Take the value from the highest-numbered page on which the
            field occurs.
    """

    FIRST_PAGE = "first_page"
    LAST_PAGE = "last_page"


class ExtractionMergeOptions(BaseModel):
    """Options for merging per-page extraction results into one document result.

    Merging is opt-in: when ``enabled`` is False (the default) extraction keeps
    returning exactly one independent result per page, with no retry or
    document-level aggregation.

    Attributes:
        enabled: Whether per-page results are merged into a document-level
            result attached to ``ExtractionResult.merged``.
        strategy: Priority rule for choosing between different values of the
            same scalar field across pages. All candidates and their source
            pages are retained regardless of the chosen value.
        max_page_retries: Number of additional attempts granted to a page whose
            extraction raises an error before the page is recorded as failed.
            Failed pages are excluded from the merge but do not fail the whole
            document as long as at least one other page merges.
    """

    enabled: bool = False
    strategy: FieldMergeStrategy = FieldMergeStrategy.LAST_PAGE
    max_page_retries: int = Field(
        default=0,
        ge=0,
        description="Extra attempts per page after the first failed extraction.",
    )
