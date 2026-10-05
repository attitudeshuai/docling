# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Merge independent per-page extraction results into one document result.

The merger is a pure, side-effect-free transformation over
``ExtractedPageData`` objects: it never mutates the per-page results and never
removes information that was readable on any page. Scalar fields found on
several pages are resolved by an explicit page-order priority; every observed
value with its source page is retained on the corresponding field info so that
conflicts remain auditable.
"""

import copy
import json
import logging
import types
import typing
from inspect import isclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.extraction import (
    ExtractedPageData,
    ExtractionTemplateType,
    FieldCandidate,
    FieldPresence,
    MergedExtractionData,
    MergedFieldInfo,
    MergedPageInfo,
    PageMergeStatus,
)
from docling.datamodel.extraction_options import (
    ExtractionMergeOptions,
    FieldMergeStrategy,
)

_log = logging.getLogger(__name__)

# A candidate is a (page_no, raw value) pair.
_Candidates = List[Tuple[int, Any]]


def _is_empty(value: Any) -> bool:
    """Return True for null/empty values, distinct from a missing key."""
    return value is None or value == "" or value == [] or value == {}


def _value_key(value: Any) -> str:
    """Stable comparison key for conflict detection across JSON values."""
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _model_skeleton(model_cls: type[BaseModel]) -> Dict[str, Any]:
    """Extract a {field: nested-skeleton-or-None} structure from a model class."""
    skeleton: Dict[str, Any] = {}
    for name, field in model_cls.model_fields.items():
        annotation = field.annotation
        origin = typing.get_origin(annotation)
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        target = annotation
        if origin in (typing.Union, types.UnionType) and len(args) == 1:
            target = args[0]
        try:
            is_model = isclass(target) and issubclass(target, BaseModel)
        except TypeError:
            is_model = False
        skeleton[name] = _model_skeleton(target) if is_model else None
    return skeleton


def _template_skeleton(
    template: Optional[ExtractionTemplateType],
) -> Optional[Dict[str, Any]]:
    """Return the expected field structure of a template, or None if unknown.

    Only object structure is relevant for merging; leaf descriptions are
    represented by None.
    """
    if template is None:
        return None
    if isinstance(template, str):
        try:
            parsed = json.loads(template)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
        return _template_skeleton(parsed)
    if isinstance(template, dict):
        return {
            key: _template_skeleton(value) if isinstance(value, dict) else None
            for key, value in template.items()
        }
    if isinstance(template, BaseModel):
        return _model_skeleton(type(template))
    if isclass(template) and issubclass(template, BaseModel):
        return _model_skeleton(template)
    return None


def _field_path(parent: str, key: str) -> str:
    return f"{parent}.{key}" if parent else str(key)


def _presence(candidates: _Candidates) -> FieldPresence:
    return (
        FieldPresence.FOUND
        if any(not _is_empty(value) for _, value in candidates)
        else FieldPresence.EMPTY
    )


def _missing_info(path: str) -> MergedFieldInfo:
    return MergedFieldInfo(name=path, presence=FieldPresence.MISSING)


def _register_missing_skeleton(
    skeleton: Optional[Dict[str, Any]],
    path: str,
    fields: Dict[str, MergedFieldInfo],
) -> None:
    """Record every template-expected path as MISSING when no page has it."""
    if path:
        fields.setdefault(path, _missing_info(path))
    if isinstance(skeleton, dict):
        for key, child_skeleton in skeleton.items():
            _register_missing_skeleton(child_skeleton, _field_path(path, key), fields)


def _choose_scalar(
    candidates: _Candidates, strategy: FieldMergeStrategy
) -> Tuple[int, Any]:
    return (
        candidates[0] if strategy == FieldMergeStrategy.FIRST_PAGE else candidates[-1]
    )


def _merge_scalar(
    candidates: _Candidates, path: str, strategy: FieldMergeStrategy
) -> Tuple[Any, MergedFieldInfo]:
    chosen_page, chosen_value = _choose_scalar(candidates, strategy)
    distinct = {_value_key(value) for _, value in candidates}
    info = MergedFieldInfo(
        name=path,
        presence=_presence(candidates),
        value=copy.deepcopy(chosen_value),
        chosen_page=chosen_page,
        source_pages=[page_no for page_no, _ in candidates],
        candidates=[
            FieldCandidate(page_no=page_no, value=copy.deepcopy(value))
            for page_no, value in candidates
        ],
        has_conflict=len(distinct) > 1,
    )
    return copy.deepcopy(chosen_value), info


def _merge_list(
    candidates: _Candidates, path: str
) -> Tuple[List[Any], MergedFieldInfo]:
    merged: List[Any] = []
    for _, value in candidates:
        if isinstance(value, list):
            merged.extend(copy.deepcopy(value))
    info = MergedFieldInfo(
        name=path,
        presence=_presence(candidates),
        value=merged,
        source_pages=[page_no for page_no, _ in candidates],
    )
    return merged, info


def _merge_dict(
    candidates: _Candidates,
    skeleton: Optional[Dict[str, Any]],
    path: str,
    strategy: FieldMergeStrategy,
    fields: Dict[str, MergedFieldInfo],
) -> Tuple[Dict[str, Any], MergedFieldInfo]:
    merged: Dict[str, Any] = {}
    dict_candidates = [(p, v) for p, v in candidates if isinstance(v, dict)]

    ordered_keys: List[Any] = []
    seen: set[Any] = set()
    if isinstance(skeleton, dict):
        for key in skeleton:
            if key not in seen:
                seen.add(key)
                ordered_keys.append(key)
    for _, value in dict_candidates:
        for key in value:
            if key not in seen:
                seen.add(key)
                ordered_keys.append(key)

    for key in ordered_keys:
        child_path = _field_path(path, key)
        child_candidates: _Candidates = [
            (page_no, value[key]) for page_no, value in dict_candidates if key in value
        ]
        child_skeleton = skeleton.get(key) if isinstance(skeleton, dict) else None
        if child_candidates:
            merged[key] = _merge_node(
                child_candidates, child_skeleton, child_path, strategy, fields
            )
        else:
            # Expected by the template but absent from every participating
            # page; the key is omitted from merged data and marked MISSING.
            _register_missing_skeleton(child_skeleton, child_path, fields)
        # Keys neither observed nor expected by the template are skipped.

    info = MergedFieldInfo(
        name=path,
        presence=_presence(candidates),
        value=copy.deepcopy(merged),
        source_pages=[page_no for page_no, _ in dict_candidates],
    )
    return merged, info


def _merge_node(
    candidates: _Candidates,
    skeleton: Optional[Dict[str, Any]],
    path: str,
    strategy: FieldMergeStrategy,
    fields: Dict[str, MergedFieldInfo],
) -> Any:
    """Merge the values of one field path and register its provenance.

    Returns the merged value. ``candidates`` is non-empty and ordered by page.
    """
    kinds = {
        "dict"
        if isinstance(value, dict)
        else "list"
        if isinstance(value, list)
        else "scalar"
        for _, value in candidates
    }

    if len(kinds) > 1:
        # Pages disagree on the field type: resolve by the same explicit
        # priority as scalars; every candidate stays auditable in the info.
        merged, info = _merge_scalar(candidates, path, strategy)
    elif "dict" in kinds:
        merged, info = _merge_dict(candidates, skeleton, path, strategy, fields)
    elif "list" in kinds:
        merged, info = _merge_list(candidates, path)
    else:
        merged, info = _merge_scalar(candidates, path, strategy)

    fields[path] = info
    return merged


def merge_page_results(
    pages: List[ExtractedPageData],
    template: Optional[ExtractionTemplateType] = None,
    options: Optional[ExtractionMergeOptions] = None,
) -> MergedExtractionData:
    """Merge independent per-page extraction data into a document-level result.

    Args:
        pages: The raw per-page results. They are never mutated.
        template: The extraction template, used to flag expected-but-missing
            fields. May be None or unparseable, in which case only observed
            fields are reported.
        options: Merge configuration (priority strategy). When None, the
            default options are used.

    Returns:
        A self-contained, serializable ``MergedExtractionData``. Pages that
        failed extraction or produced no JSON object are excluded from the
        merge and reported individually; the overall status is FAILURE only
        when no page contributed, PARTIAL_SUCCESS when some were excluded.
    """
    options = options or ExtractionMergeOptions()
    strategy = options.strategy
    skeleton = _template_skeleton(template)

    ordered_pages = sorted(pages, key=lambda page: page.page_no)
    page_infos: List[MergedPageInfo] = []
    participants: _Candidates = []
    for page in ordered_pages:
        if page.errors:
            # A failed page keeps a single error and only loses merge
            # participation; other pages continue normally.
            page_infos.append(
                MergedPageInfo(
                    page_no=page.page_no,
                    status=PageMergeStatus.FAILED,
                    attempts=page.attempts,
                    error=page.errors[-1],
                )
            )
        elif isinstance(page.extracted_data, dict):
            page_infos.append(
                MergedPageInfo(
                    page_no=page.page_no,
                    status=PageMergeStatus.MERGED,
                    attempts=page.attempts,
                )
            )
            participants.append((page.page_no, page.extracted_data))
        else:
            # Extraction ran but returned no parseable JSON object; there is
            # nothing to merge by template structure, but the raw page result
            # stays visible on ExtractionResult.pages.
            page_infos.append(
                MergedPageInfo(
                    page_no=page.page_no,
                    status=PageMergeStatus.UNPARSEABLE,
                    attempts=page.attempts,
                )
            )

    fields: Dict[str, MergedFieldInfo] = {}
    data: Dict[str, Any] = {}
    if participants:
        data = _merge_node(
            candidates=participants,
            skeleton=skeleton,
            path="",
            strategy=strategy,
            fields=fields,
        )
        # The document root itself is not a named field.
        fields.pop("", None)
    # Template fields absent from every page are reported as MISSING; this
    # never overwrites provenance registered for observed paths.
    _register_missing_skeleton(skeleton, "", fields)

    pages_merged = sum(info.status == PageMergeStatus.MERGED for info in page_infos)
    pages_excluded = len(page_infos) - pages_merged
    if pages_merged == 0:
        status = ConversionStatus.FAILURE
    elif pages_excluded > 0:
        status = ConversionStatus.PARTIAL_SUCCESS
    else:
        status = ConversionStatus.SUCCESS

    return MergedExtractionData(
        status=status,
        strategy=strategy,
        data=data,
        fields=fields,
        pages=page_infos,
        pages_total=len(page_infos),
        pages_merged=pages_merged,
        pages_failed=sum(info.status == PageMergeStatus.FAILED for info in page_infos),
    )
