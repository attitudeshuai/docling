# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Async client SDK for docling-serve."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
import sys
import time
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path, PurePath
from typing import IO, Any, Literal, TypeVar, cast, overload
from urllib.parse import urlencode, urlparse

import httpx
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.common.constants import CURRENT_VERSION
from docling_core.types.io import DocumentStream
from pydantic import ValidationError

from docling.backend.noop_backend import NoOpBackend
from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
    FormatToExtensions,
    InputFormat,
    OutputFormat,
)
from docling.datamodel.document import AssembledUnit, ConversionResult, InputDocument
from docling.datamodel.service.chunking import (
    HierarchicalChunkerOptions,
    HybridChunkerOptions,
)
from docling.datamodel.service.options import (
    ConvertDocumentsOptions as ConvertDocumentsRequestOptions,
)
from docling.datamodel.service.requests import (
    BatchConvertSourcesRequest,
    BatchSourceRequestInput,
    BatchTargetRequest,
    BatchTargetRequestInput,
    ConvertDocumentsRequest,
    HttpSourceRequest,
)
from docling.datamodel.service.responses import (
    ChunkDocumentResponse,
    ConvertDocumentResponse,
    HealthCheckResponse,
    PresignedUrlConvertDocumentResponse,
    PresignedUrlConvertResponse,
    TaskFailureResult,
    TaskStatusResponse,
    UsageLimitExceededResponse,
)
from docling.datamodel.service.targets import (
    InBodyTarget,
    PresignedUrlTarget,
    ZipTarget,
)
from docling.datamodel.settings import DocumentLimits, PageRange
from docling.service_client._scheduler import _run_bounded
from docling.service_client.client import (
    _STORAGE_TARGET_KINDS,
    DEFAULT_MAX_CONCURRENCY,
    BatchSubmitTarget,
    ChunkerKind,
    ConversionItem,
    RawServiceResult,
    SourceType,
    StatusWatcherKind,
    SubmitTarget,
    _BaseDoclingServiceClient,
    _convert_details,
    _descriptor_from_record,
    _is_storage_target,
    _ResolvedOptions,
    _sanitize_model,
    _SourceDescriptor,
)
from docling.service_client.exceptions import (
    ConversionError,
    ResponseSchemaMismatchError,
    ResultExpiredError,
    ResultNotReadyError,
    ServiceError,
    ServiceUnavailableError,
    TaskExecutionError,
    TaskNotFoundError,
    TaskTimeoutError,
    UsageLimitExceededError,
)
from docling.service_client.job import AsyncConversionJob, _AsyncJobHandlers
from docling.service_client.ledger import (
    JobLedgerConfig,
    _Existing,
    _LedgerRecord,
    _PeerWait,
    _TargetMarker,
)
from docling.service_client.watchers import (
    AsyncPollingWatcher,
    AsyncWebSocketWatcher,
    _poll_sleep_duration,
    is_terminal_task_status,
)

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class AsyncDoclingServiceClient(_BaseDoclingServiceClient):
    """Native async client for docling-serve."""

    def __init__(
        self,
        url: str,
        api_key: str = "",
        options: ConvertDocumentsRequestOptions | None = None,
        status_watcher: StatusWatcherKind = StatusWatcherKind.WEBSOCKET,
        ws_fallback_to_poll: bool = True,
        poll_server_wait: float = 5.0,
        poll_client_interval: float | None = None,
        job_timeout: float = 300.0,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        http_retries: int = 3,
        http_connect_timeout: float = 10.0,
        http_read_timeout: float = 60.0,
        ledger: JobLedgerConfig | str | Path | None = None,
    ) -> None:
        super().__init__(
            url=url,
            api_key=api_key,
            options=options,
            status_watcher=status_watcher,
            ws_fallback_to_poll=ws_fallback_to_poll,
            poll_server_wait=poll_server_wait,
            poll_client_interval=poll_client_interval,
            job_timeout=job_timeout,
            max_concurrency=max_concurrency,
            http_retries=http_retries,
            http_connect_timeout=http_connect_timeout,
            http_read_timeout=http_read_timeout,
            ledger=ledger,
        )
        self._async_client: httpx.AsyncClient | None = None
        self._polling_watcher: AsyncPollingWatcher | None = None
        self._ws_watcher: AsyncWebSocketWatcher | None = None

    async def __aenter__(self) -> AsyncDoclingServiceClient:
        timeout = httpx.Timeout(
            connect=self._http_connect_timeout,
            read=self._http_read_timeout,
            write=self._http_read_timeout,
            pool=self._http_read_timeout,
        )
        headers: dict[str, str] = {
            # Tell the server the newest DoclingDocument version this client's
            # installed docling-core can read; the server down-projects if needed.
            "Accept-Docling-Document-Version": CURRENT_VERSION,
        }
        if self._api_key:
            headers["X-Api-Key"] = self._api_key
        self._async_client = httpx.AsyncClient(timeout=timeout, headers=headers)

        self._polling_watcher = AsyncPollingWatcher(
            poll_status=self._poll_task_status,
            poll_server_wait=self._poll_server_wait,
            poll_client_interval=self._poll_client_interval,
            default_timeout=self._job_timeout,
        )

        ws_headers = {"X-Api-Key": self._api_key} if self._api_key else {}
        self._ws_watcher = AsyncWebSocketWatcher(
            ws_url_for_task=self._build_ws_status_url,
            poll_fallback=self._polling_watcher,
            fallback_to_poll=self._ws_fallback_to_poll,
            connect_timeout=self._http_connect_timeout,
            default_timeout=self._job_timeout,
            additional_headers=ws_headers,
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    @overload
    async def submit(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions | None = None,
        output_formats: list[OutputFormat] | None = None,
        headers: dict[str, str] | None = None,
        *,
        target: InBodyTarget = ...,
    ) -> AsyncConversionJob[ConversionResult]: ...

    @overload
    async def submit(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions | None = None,
        output_formats: list[OutputFormat] | None = None,
        headers: dict[str, str] | None = None,
        *,
        target: ZipTarget,
    ) -> AsyncConversionJob[RawServiceResult]: ...

    @overload
    async def submit(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions | None = None,
        output_formats: list[OutputFormat] | None = None,
        headers: dict[str, str] | None = None,
        *,
        target: PresignedUrlTarget | None = None,
    ) -> AsyncConversionJob[PresignedUrlConvertResponse | ConversionResult]: ...

    async def submit(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions | None = None,
        output_formats: list[OutputFormat] | None = None,
        headers: dict[str, str] | None = None,
        *,
        target: SubmitTarget | None = None,
    ) -> (
        AsyncConversionJob[ConversionResult]
        | AsyncConversionJob[RawServiceResult]
        | AsyncConversionJob[PresignedUrlConvertResponse]
    ):
        assert self._async_client is not None, "client not open — use async with"
        descriptor = self._describe_source(source)
        resolved = self._resolve_options(
            options=options,
            max_num_pages=None,
            max_file_size=None,
            page_range=None,
        )
        # Effective options/target are computed before the possible ledger
        # lookup so the fingerprint always reflects what is actually submitted.
        provisional_target = PresignedUrlTarget() if target is None else target
        submit_options = self._options_for_output_formats(
            resolved.options,
            output_formats=output_formats,
            target=provisional_target,
        )

        if target is None:
            fingerprint_payload = {
                "jk": "convert",
                "src": self._source_fingerprint(source),
                "opt": submit_options.model_dump(mode="json"),
                "tgt": ("auto",),
                "mat": False,
                "hdr": self._headers_fingerprint(headers),
            }
            actual_target_kind: dict[str, str] = {}

            async def do_submit() -> TaskStatusResponse:
                try:
                    status = await self._submit_convert_task(
                        source=source,
                        options=submit_options,
                        target=PresignedUrlTarget(),
                        async_client=self._async_client,
                        request_headers=headers,
                    )
                    actual_target_kind["tk"] = "presigned_url"
                    return status
                except ServiceError as exc:
                    if not self._should_fallback_from_presigned_target(exc):
                        raise
                inbody = InBodyTarget()
                fallback_options = self._options_for_output_formats(
                    resolved.options,
                    output_formats=output_formats,
                    target=inbody,
                )
                status = await self._submit_convert_task(
                    source=source,
                    options=fallback_options,
                    target=inbody,
                    async_client=self._async_client,
                    request_headers=headers,
                )
                actual_target_kind["tk"] = "inbody"
                return status

            def details_factory(initial: TaskStatusResponse) -> dict[str, Any]:
                return _convert_details(
                    descriptor=descriptor,
                    target_kind=actual_target_kind["tk"],
                    materialize=False,
                )
        else:
            fingerprint_payload = {
                "jk": "convert",
                "src": self._source_fingerprint(source),
                "opt": submit_options.model_dump(mode="json"),
                "tgt": ("explicit", _sanitize_model(target)),
                "mat": False,
                "hdr": self._headers_fingerprint(headers),
            }

            async def do_submit() -> TaskStatusResponse:
                assert target is not None
                return await self._submit_convert_task(
                    source=source,
                    options=submit_options,
                    target=target,
                    async_client=self._async_client,
                    request_headers=headers,
                )

            def details_factory(initial: TaskStatusResponse) -> dict[str, Any]:
                assert target is not None
                return _convert_details(
                    descriptor=descriptor,
                    target_kind=target.kind,
                    materialize=False,
                )

        if self._ledger is None:
            initial_status = await do_submit()
            # do_submit records the effective target kind for the auto path;
            # kind-based handler selection needs no credentials, so a marker is enough.
            effective_target = (
                _TargetMarker(actual_target_kind["tk"]) if target is None else target
            )
            return await self._assemble_conversion_job(
                initial_status=initial_status,
                descriptor=descriptor,
                limits=resolved.limits,
                target=effective_target,
            )

        fingerprint = self._fingerprint(fingerprint_payload)
        initial_status, record = await self._orchestrate_task(
            fingerprint=fingerprint,
            do_submit=do_submit,
            details_factory=details_factory,
        )
        return await self._assemble_after_orchestration(
            fingerprint=fingerprint,
            initial_status=initial_status,
            record=record,
            fresh_descriptor=descriptor,
            limits=resolved.limits,
        )

    async def submit_batch(
        self,
        sources: Sequence[BatchSourceRequestInput],
        target: BatchTargetRequestInput | None = None,
        output_formats: list[OutputFormat] | None = None,
        options: ConvertDocumentsRequestOptions | None = None,
        headers: dict[str, str] | None = None,
        *,
        targets: list[BatchTargetRequestInput] | None = None,
    ) -> (
        AsyncConversionJob[PresignedUrlConvertDocumentResponse]
        | AsyncConversionJob[PresignedUrlConvertResponse]
    ):
        assert self._async_client is not None, "client not open — use async with"
        if target is None and targets is None:
            raise ValueError("submit_batch() requires either 'target' or 'targets'.")
        if target is not None and targets is not None:
            raise ValueError(
                "submit_batch() received both 'target' and 'targets'; supply only one."
            )
        payload: dict[str, Any] = {"sources": sources}
        if targets is not None:
            payload["targets"] = targets
        else:
            payload["target"] = target
        request = BatchConvertSourcesRequest.model_validate(payload)
        resolved = self._resolve_options(
            options=options,
            max_num_pages=None,
            max_file_size=None,
            page_range=None,
        )
        # Use the first effective target for output-format hint.
        first_target = (request.targets or [request.target])[0]
        submit_options = self._options_for_output_formats(
            resolved.options,
            output_formats=output_formats,
            target=first_target,
        )
        all_targets = (
            request.targets
            if request.targets is not None
            else ([request.target] if request.target is not None else [])
        )
        storage_like = any(_is_storage_target(t) for t in all_targets)
        fingerprint: str | None = None
        if self._ledger is not None:
            fingerprint = self._fingerprint(
                {
                    "jk": "batch",
                    "src": [_sanitize_model(source) for source in request.sources],
                    "tgt": [_sanitize_model(t) for t in all_targets],
                    "opt": submit_options.model_dump(mode="json"),
                    "hdr": self._headers_fingerprint(headers),
                }
            )
            initial_status, record = await self._orchestrate_task(
                fingerprint=fingerprint,
                do_submit=lambda: self._submit_batch_task(
                    sources=request.sources,
                    options=submit_options,
                    target=request.target,
                    targets=request.targets,
                    async_client=self._async_client,
                    request_headers=headers,
                ),
                details_factory=lambda initial: {
                    "jk": "batch",
                    "storage": storage_like,
                },
            )
            if record is not None:
                storage_like = bool(record.details.get("storage", False))
        else:
            initial_status = await self._submit_batch_task(
                sources=request.sources,
                options=submit_options,
                target=request.target,
                targets=request.targets,
                async_client=self._async_client,
                request_headers=headers,
            )

        if storage_like:

            async def fetch_result(
                task_id: str,
                last_status: TaskStatusResponse | None,
            ) -> PresignedUrlConvertDocumentResponse:
                return await self._fetch_presigned_document_result(
                    task_id=task_id,
                    last_status=last_status,
                    async_client=self._async_client,
                )

        else:

            async def fetch_result(
                task_id: str,
                last_status: TaskStatusResponse | None,
            ) -> PresignedUrlConvertResponse:
                return await self._fetch_presigned_result(
                    task_id=task_id,
                    last_status=last_status,
                    async_client=self._async_client,
                )

        handlers = _AsyncJobHandlers[Any](
            poll=self._poll_task_status,
            watch=lambda tid, t: self._status_watcher().iter_updates(tid, t),
            wait=lambda tid, t: self._status_watcher().wait_for_terminal(tid, t),
            fetch_result=fetch_result,
        )
        if fingerprint is not None and self._ledger is not None:
            handlers = await self._ledered_handlers(fingerprint, handlers)
        return AsyncConversionJob(
            task_id=initial_status.task_id,
            submitted_at=datetime.now(tz=timezone.utc),
            handlers=handlers,
            initial_status=initial_status,
        )

    async def submit_chunk(
        self,
        source: SourceType,
        chunker: ChunkerKind,
        options: ConvertDocumentsRequestOptions | None = None,
    ) -> AsyncConversionJob[ChunkDocumentResponse]:
        resolved = self._resolve_options(
            options=options,
            max_num_pages=None,
            max_file_size=None,
            page_range=None,
        )
        fingerprint: str | None = None
        if self._ledger is not None:
            fingerprint = self._fingerprint(
                {
                    "jk": "chunk",
                    "ch": chunker.value,
                    "src": self._source_fingerprint(source),
                    "opt": resolved.options.model_dump(mode="json"),
                    "tgt": ("inbody",),
                    "hdr": self._headers_fingerprint(None),
                }
            )
            initial_status, _record = await self._orchestrate_task(
                fingerprint=fingerprint,
                do_submit=lambda: self._submit_chunk_task(
                    source=source,
                    chunker=chunker,
                    options=resolved.options,
                ),
                details_factory=lambda initial: {
                    "jk": "chunk",
                    "ch": chunker.value,
                },
            )
        else:
            initial_status = await self._submit_chunk_task(
                source=source,
                chunker=chunker,
                options=resolved.options,
            )
        handlers: _AsyncJobHandlers[ChunkDocumentResponse] = _AsyncJobHandlers(
            poll=self._poll_task_status,
            watch=lambda tid, t: self._status_watcher().iter_updates(tid, t),
            wait=lambda tid, t: self._status_watcher().wait_for_terminal(tid, t),
            fetch_result=lambda tid, last: self._fetch_chunk_result(
                task_id=tid,
                last_status=last,
            ),
        )
        if fingerprint is not None and self._ledger is not None:
            handlers = await self._ledered_handlers(fingerprint, handlers)
        return AsyncConversionJob(
            task_id=initial_status.task_id,
            submitted_at=datetime.now(tz=timezone.utc),
            handlers=handlers,
            initial_status=initial_status,
        )

    async def submit_and_retrieve_each(
        self,
        items: Iterable[ConversionItem],
        max_in_flight: int = DEFAULT_MAX_CONCURRENCY,
        ordered: bool = False,
        *,
        target: SubmitTarget | None = None,
    ) -> AsyncGenerator[
        tuple[
            ConversionItem,
            (
                ConvertDocumentResponse
                | PresignedUrlConvertDocumentResponse
                | PresignedUrlConvertResponse
                | Exception
            ),
        ],
        None,
    ]:
        assert self._async_client is not None, "client not open — use async with"
        max_in_flight = self._validate_concurrency(
            max_in_flight,
            name="max_in_flight",
        )

        async def process_one(
            _idx: int,
            item: ConversionItem,
            async_client: httpx.AsyncClient,
        ) -> (
            ConvertDocumentResponse
            | PresignedUrlConvertDocumentResponse
            | PresignedUrlConvertResponse
        ):
            resolved = self._resolve_options(
                options=item.options,
                max_num_pages=None,
                max_file_size=None,
                page_range=None,
            )
            effective_target = PresignedUrlTarget() if target is None else target
            submit_options = self._options_for_output_formats(
                resolved.options,
                output_formats=None,
                target=effective_target,
            )
            actual_target_kind: dict[str, str] = {}

            async def do_submit() -> TaskStatusResponse:
                nonlocal submit_options
                try:
                    status = await self._submit_convert_task(
                        source=item.source,
                        options=submit_options,
                        target=effective_target,
                        async_client=async_client,
                        request_headers=item.headers,
                    )
                    actual_target_kind["tk"] = effective_target.kind
                    return status
                except ServiceError as exc:
                    if (
                        target is not None
                        or not self._should_fallback_from_presigned_target(exc)
                    ):
                        raise
                fallback_target = InBodyTarget()
                submit_options = self._options_for_output_formats(
                    resolved.options,
                    output_formats=None,
                    target=fallback_target,
                )
                status = await self._submit_convert_task(
                    source=item.source,
                    options=submit_options,
                    target=fallback_target,
                    async_client=async_client,
                    request_headers=item.headers,
                )
                actual_target_kind["tk"] = fallback_target.kind
                return status

            def details_factory(initial: TaskStatusResponse) -> dict[str, Any]:
                descriptor = self._describe_source(item.source)
                return _convert_details(
                    descriptor=descriptor,
                    target_kind=actual_target_kind["tk"],
                    materialize=False,
                )

            fingerprint: str | None = None
            if self._ledger is not None:
                target_spec = (
                    ("auto",)
                    if target is None
                    else ("explicit", _sanitize_model(target))
                )
                fingerprint = self._fingerprint(
                    {
                        "jk": "convert",
                        "src": self._source_fingerprint(item.source),
                        "opt": submit_options.model_dump(mode="json"),
                        "tgt": target_spec,
                        "mat": False,
                        "hdr": self._headers_fingerprint(item.headers),
                    }
                )
                initial_status, record = await self._orchestrate_task(
                    fingerprint=fingerprint,
                    do_submit=do_submit,
                    details_factory=details_factory,
                )
                if record is None:
                    result_kind = actual_target_kind["tk"]
                else:
                    result_kind = record.details.get(
                        "tk", actual_target_kind.get("tk", "inbody")
                    )
            else:
                initial_status = await do_submit()
                result_kind = actual_target_kind["tk"]

            # Skip the wait when the reattachment probe already shows terminal.
            if not is_terminal_task_status(initial_status):
                terminal_status = (
                    await self._wait_for_terminal_status_for_submit_and_retrieve_many(
                        task_id=initial_status.task_id,
                        timeout=self._job_timeout,
                        async_client=async_client,
                        max_in_flight=max_in_flight,
                    )
                )
            else:
                terminal_status = initial_status

            # This path bypasses job handlers; persist terminal status explicitly.
            if fingerprint is not None and self._ledger is not None:
                await asyncio.to_thread(
                    self._ledger.note_status,
                    fingerprint,
                    initial_status.task_id,
                    terminal_status.task_status,
                )

            if result_kind == "presigned_url":
                return await self._fetch_presigned_result(
                    task_id=initial_status.task_id,
                    last_status=terminal_status,
                    async_client=async_client,
                )
            return await self._fetch_convert_result_payload(
                task_id=initial_status.task_id,
                last_status=terminal_status,
                async_client=async_client,
            )

        buffered_results: dict[
            int,
            tuple[
                ConversionItem,
                (
                    ConvertDocumentResponse
                    | PresignedUrlConvertDocumentResponse
                    | PresignedUrlConvertResponse
                    | Exception
                ),
            ],
        ] = {}
        next_ordered_index = 0

        async for idx, item, outcome in _run_bounded(
            items=items,
            process_one=process_one,
            async_client=self._async_client,
            max_in_flight=max_in_flight,
        ):
            normalized: (
                ConvertDocumentResponse
                | PresignedUrlConvertDocumentResponse
                | PresignedUrlConvertResponse
                | Exception
            )
            if isinstance(outcome, BaseException):
                normalized = self._normalize_exception(outcome)
            else:
                normalized = outcome

            if ordered:
                buffered_results[idx] = (item, normalized)
                while next_ordered_index in buffered_results:
                    yield buffered_results.pop(next_ordered_index)
                    next_ordered_index += 1
                continue

            yield item, normalized

    async def submit_and_retrieve_many(
        self,
        items: Iterable[ConversionItem],
        max_in_flight: int = DEFAULT_MAX_CONCURRENCY,
        ordered: bool = False,
        *,
        target: SubmitTarget | None = None,
    ) -> AsyncGenerator[
        tuple[
            ConversionItem,
            (
                ConvertDocumentResponse
                | PresignedUrlConvertDocumentResponse
                | PresignedUrlConvertResponse
                | Exception
            ),
        ],
        None,
    ]:
        warnings.warn(
            "submit_and_retrieve_many() is deprecated; use submit_and_retrieve_each().",
            DeprecationWarning,
            stacklevel=2,
        )
        async for item, outcome in self.submit_and_retrieve_each(
            items=items,
            max_in_flight=max_in_flight,
            ordered=ordered,
            target=target,
        ):
            yield item, outcome

    async def health(self) -> HealthCheckResponse:
        response = await self._request_with_retry("GET", "/health", retries=0)
        if response.status_code != 200:
            self._raise_for_generic_http_error(response, "Health check request failed.")
        return HealthCheckResponse.model_validate_json(response.text)

    async def version(self) -> dict[str, Any]:
        response = await self._request_with_retry("GET", "/version", retries=0)
        if response.status_code != 200:
            self._raise_for_generic_http_error(response, "Version request failed.")
        return response.json()

    def _status_watcher(self) -> AsyncPollingWatcher | AsyncWebSocketWatcher:
        assert self._polling_watcher is not None and self._ws_watcher is not None
        if self._status_watcher_kind == StatusWatcherKind.POLLING:
            return self._polling_watcher
        return self._ws_watcher

    def _make_convert_fetch_result_handler(
        self,
        descriptor: _SourceDescriptor,
        limits: DocumentLimits,
        target: SubmitTarget | _TargetMarker,
        async_client: httpx.AsyncClient,
    ) -> Any:
        kind = target.kind
        if kind == "zip":
            return lambda task_id, last_status: self._fetch_raw_result(
                task_id=task_id,
                last_status=last_status,
                async_client=async_client,
            )
        if kind == "presigned_url":
            return lambda task_id, last_status: self._fetch_presigned_result(
                task_id=task_id,
                last_status=last_status,
                async_client=async_client,
            )
        if kind in _STORAGE_TARGET_KINDS:
            return lambda task_id, last_status: self._fetch_presigned_document_result(
                task_id=task_id,
                last_status=last_status,
                async_client=async_client,
            )
        return lambda task_id, last_status: self._fetch_convert_result(
            task_id=task_id,
            descriptor=descriptor,
            limits=limits,
            last_status=last_status,
            async_client=async_client,
        )

    # ------------------------------------------------------------------
    # Ledger orchestration
    # ------------------------------------------------------------------

    async def _assemble_conversion_job(
        self,
        *,
        initial_status: TaskStatusResponse,
        descriptor: _SourceDescriptor,
        limits: DocumentLimits,
        target: SubmitTarget | _TargetMarker,
        fingerprint: str | None = None,
    ) -> AsyncConversionJob[Any]:
        assert self._async_client is not None
        fetch_result = self._make_convert_fetch_result_handler(
            descriptor=descriptor,
            limits=limits,
            target=target,
            async_client=self._async_client,
        )
        handlers: _AsyncJobHandlers[Any] = _AsyncJobHandlers(
            poll=self._poll_task_status,
            watch=lambda tid, t: self._status_watcher().iter_updates(tid, t),
            wait=lambda tid, t: self._status_watcher().wait_for_terminal(tid, t),
            fetch_result=fetch_result,
        )
        if fingerprint is not None and self._ledger is not None:
            handlers = await self._ledered_handlers(fingerprint, handlers)
        return AsyncConversionJob(
            task_id=initial_status.task_id,
            submitted_at=datetime.now(tz=timezone.utc),
            handlers=handlers,
            initial_status=initial_status,
        )

    async def _assemble_after_orchestration(
        self,
        *,
        fingerprint: str,
        initial_status: TaskStatusResponse,
        record: _LedgerRecord | None,
        fresh_descriptor: _SourceDescriptor,
        limits: DocumentLimits,
    ) -> AsyncConversionJob[Any]:
        assert self._ledger is not None
        stored = await asyncio.to_thread(self._ledger.get_record, fingerprint)
        assert stored is not None
        descriptor = (
            fresh_descriptor if record is None else _descriptor_from_record(stored)
        )
        return await self._assemble_conversion_job(
            initial_status=initial_status,
            descriptor=descriptor,
            limits=limits,
            target=_TargetMarker(stored.details["tk"]),
            fingerprint=fingerprint,
        )

    async def _orchestrate_task(
        self,
        *,
        fingerprint: str,
        do_submit: Any,
        details_factory: Any,
    ) -> tuple[TaskStatusResponse, _LedgerRecord | None]:
        """Async counterpart of the sync intent → submit → complete flow."""
        ledger = self._ledger
        assert ledger is not None
        await asyncio.to_thread(ledger.auto_purge_once)
        decision = await asyncio.to_thread(ledger.begin, fingerprint)
        while isinstance(decision, _PeerWait):
            await asyncio.sleep(ledger.peer_poll_interval)
            decision = await asyncio.to_thread(ledger.begin, fingerprint)
        if isinstance(decision, _Existing):
            record = decision.record
            if record.state == "orphaned":
                raise TaskNotFoundError(
                    f"Task {record.task_id} recorded in the local job ledger is "
                    "no longer known to the service and cannot be resumed."
                )
            return await self._probe_ledged_task(record), record

        token = decision.token
        try:
            initial_status = await do_submit()
        except BaseException:
            await asyncio.to_thread(ledger.abandon, fingerprint, token)
            raise
        completed = await asyncio.to_thread(
            ledger.complete,
            fingerprint,
            token,
            initial_status.task_id,
            initial_status.task_status,
            details_factory(initial_status),
        )
        if completed is None:
            peer = await asyncio.to_thread(ledger.get_record, fingerprint)
            assert peer is not None
            return await self._probe_ledged_task(peer), peer
        return initial_status, None

    async def _probe_ledged_task(self, record: _LedgerRecord) -> TaskStatusResponse:
        assert record.task_id is not None
        try:
            return await self._poll_task_status(record.task_id, 0.0)
        except TaskNotFoundError:
            assert self._ledger is not None
            await asyncio.to_thread(
                self._ledger.mark_orphaned,
                record.fingerprint,
                record.task_id,
            )
            raise TaskNotFoundError(
                f"Task {record.task_id} recorded in the local job ledger is "
                "no longer known to the service and cannot be resumed."
            )

    async def _ledered_handlers(
        self,
        fingerprint: str,
        handlers: _AsyncJobHandlers[Any],
    ) -> _AsyncJobHandlers[Any]:
        """Wrap async job handlers so status changes persist into the ledger."""
        ledger = self._ledger
        assert ledger is not None

        async def poll(task_id: str, wait: float) -> TaskStatusResponse:
            status = await handlers.poll(task_id, wait)
            await asyncio.to_thread(
                ledger.note_status, fingerprint, task_id, status.task_status
            )
            return status

        async def watch_iter(
            task_id: str, timeout: float | None
        ) -> AsyncIterator[TaskStatusResponse]:
            async for status in handlers.watch(task_id, timeout):
                await asyncio.to_thread(
                    ledger.note_status,
                    fingerprint,
                    task_id,
                    status.task_status,
                )
                yield status

        async def wait(task_id: str, timeout: float | None) -> TaskStatusResponse:
            status = await handlers.wait(task_id, timeout)
            await asyncio.to_thread(
                ledger.note_status, fingerprint, task_id, status.task_status
            )
            return status

        async def fetch_result(
            task_id: str, last_status: TaskStatusResponse | None
        ) -> Any:
            try:
                return await handlers.fetch_result(task_id, last_status)
            except TaskNotFoundError:
                await asyncio.to_thread(ledger.mark_orphaned, fingerprint, task_id)
                raise

        return _AsyncJobHandlers[Any](
            poll=poll,
            watch=watch_iter,
            wait=wait,
            fetch_result=fetch_result,
        )

    async def purge_expired_records(self) -> int:
        """Remove records past the configured TTL; no-op without a ledger."""
        if self._ledger is None:
            return 0
        return await asyncio.to_thread(self._ledger.purge_expired)

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        json: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        retries: int | None = None,
    ) -> httpx.Response:
        assert self._async_client is not None, "client not open — use async with"
        return await self._request_with_retry_using_client(
            async_client=self._async_client,
            method=method,
            path=path,
            json=json,
            data=data,
            files=files,
            params=params,
            headers=headers,
            retries=retries,
        )

    async def _request_with_retry_using_client(
        self,
        async_client: httpx.AsyncClient,
        method: str,
        path: str,
        json: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        retries: int | None = None,
    ) -> httpx.Response:
        url = self._url(path)
        method_name = method.upper()
        max_retries = self._http_retries if retries is None else retries
        for attempt in range(max_retries + 1):
            try:
                response = await async_client.request(
                    method=method_name,
                    url=url,
                    json=json,
                    data=data,
                    files=files,
                    params=params,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                delay = self._transport_retry_delay(
                    method=method_name,
                    exc=exc,
                    attempt=attempt,
                    max_retries=max_retries,
                )
                if delay is not None:
                    await asyncio.sleep(delay)
                    continue
                raise ServiceUnavailableError(
                    "Service transport request failed.",
                    detail=str(exc),
                ) from exc
            result, delay = self._check_retry(response, attempt, max_retries)
            if result is not None:
                return result
            if delay > 0:
                await asyncio.sleep(delay)

        raise ServiceUnavailableError("Service request failed after retry loop.")

    async def _submit_convert_task(
        self,
        source: SourceType,
        options: ConvertDocumentsRequestOptions,
        target: SubmitTarget,
        async_client: httpx.AsyncClient,
        request_headers: dict[str, str] | None = None,
    ) -> TaskStatusResponse:
        source = self._normalize_source(source)
        source_name = self._source_name(source)
        logger.info("Submitting convert task for source=%s", source_name)
        if isinstance(source, HttpSourceRequest):
            request = ConvertDocumentsRequest(
                options=options,
                sources=[source],
                target=target,
            )
            response = await self._request_with_retry_using_client(
                async_client=async_client,
                method="POST",
                path="/v1/convert/source/async",
                json=self._serialize_convert_request(request),
                headers=request_headers,
            )
        else:
            files = await self._source_to_upload_files(source)
            data = self._serialize_convert_options(options)
            data["target_type"] = target.kind
            response = await self._request_with_retry_using_client(
                async_client=async_client,
                method="POST",
                path="/v1/convert/file/async",
                data=self._form_encode_options(data),
                files=files,
                headers=request_headers,
            )

        if response.status_code != 200:
            self._raise_for_generic_http_error(response, "Task submission failed.")
        status = TaskStatusResponse.model_validate_json(response.text)
        logger.info(
            "Submitted convert task for source=%s task_id=%s status=%s position=%s",
            source_name,
            status.task_id,
            status.task_status,
            status.task_position,
        )
        return status

    async def _submit_batch_task(
        self,
        sources: Sequence[BatchSourceRequestInput],
        options: ConvertDocumentsRequestOptions,
        target: BatchSubmitTarget | None,
        async_client: httpx.AsyncClient,
        targets: list[BatchTargetRequest] | None = None,
        request_headers: dict[str, str] | None = None,
    ) -> TaskStatusResponse:
        payload: dict[str, Any] = {"options": options, "sources": sources}
        if targets is not None:
            payload["targets"] = targets
        else:
            payload["target"] = target
        request = BatchConvertSourcesRequest.model_validate(payload)
        response = await self._request_with_retry_using_client(
            async_client=async_client,
            method="POST",
            path="/v1/convert/source/batch",
            json=self._serialize_convert_request(request),
            headers=request_headers,
        )
        if response.status_code != 200:
            self._raise_for_generic_http_error(
                response,
                "Batch task submission failed.",
            )
        return TaskStatusResponse.model_validate_json(response.text)

    async def _submit_chunk_task(
        self,
        source: SourceType,
        chunker: ChunkerKind,
        options: ConvertDocumentsRequestOptions,
    ) -> TaskStatusResponse:
        source = self._normalize_source(source)
        if isinstance(source, HttpSourceRequest):
            chunking_options: HybridChunkerOptions | HierarchicalChunkerOptions
            if chunker == ChunkerKind.HYBRID:
                chunking_options = HybridChunkerOptions()
            else:
                chunking_options = HierarchicalChunkerOptions()
            payload = {
                "convert_options": self._serialize_convert_options(options),
                "chunking_options": chunking_options.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                "sources": [source.model_dump(mode="json", exclude_none=True)],
                "include_converted_doc": False,
                "target": InBodyTarget().model_dump(mode="json"),
                "callbacks": [],
            }
            response = await self._request_with_retry(
                method="POST",
                path=f"/v1/chunk/{chunker.value}/source/async",
                json=payload,
            )
        else:
            files = await self._source_to_upload_files(source)
            data: dict[str, Any] = {
                f"convert_{key}": value
                for key, value in self._serialize_convert_options(options).items()
            }
            chunk_model: HybridChunkerOptions | HierarchicalChunkerOptions
            if chunker == ChunkerKind.HYBRID:
                chunk_model = HybridChunkerOptions()
            else:
                chunk_model = HierarchicalChunkerOptions()
            chunk_payload = chunk_model.model_dump(mode="json", exclude_none=True)
            chunk_payload.pop("chunker", None)
            data.update(
                {f"chunking_{key}": value for key, value in chunk_payload.items()}
            )
            data["include_converted_doc"] = False
            data["target_type"] = InBodyTarget().kind
            response = await self._request_with_retry(
                method="POST",
                path=f"/v1/chunk/{chunker.value}/file/async",
                data=self._form_encode_options(data),
                files=files,
            )

        if response.status_code != 200:
            self._raise_for_generic_http_error(
                response, "Chunk task submission failed."
            )
        return TaskStatusResponse.model_validate_json(response.text)

    async def _poll_task_status(self, task_id: str, wait: float) -> TaskStatusResponse:
        response = await self._request_with_retry(
            method="GET",
            path=f"/v1/status/poll/{task_id}",
            params={"wait": wait},
        )
        if response.status_code == 404:
            raise TaskNotFoundError(f"Task {task_id} was not found.")
        if response.status_code != 200:
            self._raise_for_generic_http_error(
                response, f"Polling task {task_id} failed."
            )
        return TaskStatusResponse.model_validate_json(response.text)

    async def _poll_task_status_using_client(
        self,
        task_id: str,
        wait: float,
        async_client: httpx.AsyncClient,
    ) -> TaskStatusResponse:
        response = await self._request_with_retry_using_client(
            async_client=async_client,
            method="GET",
            path=f"/v1/status/poll/{task_id}",
            params={"wait": wait},
        )
        if response.status_code == 404:
            raise TaskNotFoundError(f"Task {task_id} was not found.")
        if response.status_code != 200:
            self._raise_for_generic_http_error(
                response, f"Polling task {task_id} failed."
            )
        return TaskStatusResponse.model_validate_json(response.text)

    async def _wait_for_terminal_status(
        self,
        task_id: str,
        timeout: float,
        async_client: httpx.AsyncClient,
    ) -> TaskStatusResponse:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TaskTimeoutError(
                    f"Timed out waiting for task {task_id} after {timeout:.2f}s."
                )
            wait = min(self._poll_server_wait, remaining)
            logger.info("Polling status for task_id=%s wait=%.2fs", task_id, wait)
            poll_started = time.monotonic()
            update = await self._poll_task_status_using_client(
                task_id=task_id,
                wait=wait,
                async_client=async_client,
            )
            logger.info(
                "Received status for task_id=%s status=%s position=%s",
                task_id,
                update.task_status,
                update.task_position,
            )
            if is_terminal_task_status(update):
                return update

            sleep_for = _poll_sleep_duration(
                poll_started=poll_started,
                poll_interval=self._poll_client_interval,
                deadline=deadline,
            )
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    async def _wait_for_terminal_status_for_submit_and_retrieve_many(
        self,
        task_id: str,
        timeout: float,
        async_client: httpx.AsyncClient,
        max_in_flight: int,
    ) -> TaskStatusResponse:
        if self._submit_and_retrieve_many_uses_websocket_wait(
            max_in_flight=max_in_flight
        ):
            assert self._ws_watcher is not None
            return await self._ws_watcher.wait_for_terminal(task_id, timeout)
        return await self._wait_for_terminal_status(
            task_id=task_id,
            timeout=timeout,
            async_client=async_client,
        )

    async def _fetch_convert_result(
        self,
        task_id: str,
        descriptor: _SourceDescriptor,
        limits: DocumentLimits,
        last_status: TaskStatusResponse | None,
        async_client: httpx.AsyncClient,
    ) -> ConversionResult:
        payload = await self._fetch_convert_result_payload(
            task_id=task_id,
            last_status=last_status,
            async_client=async_client,
        )
        return self._build_conversion_result(
            payload=payload,
            descriptor=descriptor,
            limits=limits,
        )

    async def _fetch_convert_result_payload(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
        async_client: httpx.AsyncClient,
    ) -> ConvertDocumentResponse:
        response = await self._fetch_result_response(
            async_client=async_client,
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching result for task {task_id} failed.",
        )
        return self._parse_result_model_response(response, ConvertDocumentResponse)

    async def _fetch_raw_result(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
        async_client: httpx.AsyncClient,
    ) -> RawServiceResult:
        response = await self._fetch_result_response(
            async_client=async_client,
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching result for task {task_id} failed.",
        )
        return self._decode_raw_result(response)

    async def _fetch_presigned_result(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
        async_client: httpx.AsyncClient,
    ) -> PresignedUrlConvertResponse:
        response = await self._fetch_result_response(
            async_client=async_client,
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching result for task {task_id} failed.",
        )
        return self._parse_result_model_response(response, PresignedUrlConvertResponse)

    async def _fetch_presigned_document_result(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
        async_client: httpx.AsyncClient,
    ) -> PresignedUrlConvertDocumentResponse:
        response = await self._fetch_result_response(
            async_client=async_client,
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching result for task {task_id} failed.",
        )
        return self._parse_result_model_response(
            response,
            PresignedUrlConvertDocumentResponse,
        )

    async def _fetch_chunk_result(
        self,
        task_id: str,
        last_status: TaskStatusResponse | None,
    ) -> ChunkDocumentResponse:
        response = await self._fetch_result_response(
            async_client=self._async_client,
            task_id=task_id,
            last_status=last_status,
            error_message=f"Fetching chunk result for task {task_id} failed.",
        )
        return self._parse_result_model_response(response, ChunkDocumentResponse)

    async def _fetch_result_response(
        self,
        async_client: httpx.AsyncClient | None,
        task_id: str,
        last_status: TaskStatusResponse | None,
        *,
        error_message: str,
    ) -> httpx.Response:
        assert async_client is not None, "client not open — use async with"
        response = await self._request_with_retry_using_client(
            async_client=async_client,
            method="GET",
            path=f"/v1/result/{task_id}",
        )
        if response.status_code == 404:
            self._raise_for_result_404(
                task_id=task_id,
                response=response,
                last_status=last_status,
            )
        if response.status_code != 200:
            self._raise_for_generic_http_error(response, error_message)
        self._raise_if_task_failure_result(response)
        return response

    async def _source_to_upload_files(
        self,
        source: Path | DocumentStream,
    ) -> dict[str, tuple[str, IO[bytes] | bytes, str]]:
        if isinstance(source, Path):
            filename = source.name
            content: IO[bytes] | bytes = await asyncio.to_thread(source.read_bytes)
        else:
            filename = source.name
            source.stream.seek(0)
            content = source.stream
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return {"files": (filename, content, mime)}
