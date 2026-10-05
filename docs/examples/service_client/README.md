# Client SDK Examples

These scripts use the `docling.service_client` SDK against an **already running**
`docling-serve` instance. They do not start a service.

## Setup

Point the client at your service. The client and these examples read the same
variables as `docling convert-remote` — from the environment or a `.env` file in
the working directory:

```
DOCLING_SERVICE_URL=https://your-docling-service.example.com
DOCLING_SERVICE_API_KEY=your-api-key   # omit if the service is unauthenticated
```

Install docling-slim with the `service-client` extra :

```
pip install "docling-slim[service-client]"
```

Run the examples **from the repository root** — they reference sample documents
under `tests/data/pdf/sources/` by relative path:

```
uv run python docs/examples/service_client/convert.py
```

## The basics

Convert one document — same call shape as a local `DocumentConverter`:

```python
from docling.service_client import DoclingServiceClient

client = DoclingServiceClient(url=..., api_key=...)
result = client.convert(source="path/to/report.pdf")  # or an http(s) URL
print(result.document.export_to_markdown())
```

Convert many concurrently:

```python
for result in client.convert_all(
    source=["a.pdf", "b.pdf", "https://.../c.pdf"],
    max_concurrency=4,
):
    print(result.input.file.name, result.status)
```

Defaults (OCR, table structure, Markdown output) match that of docling's `DocumentConverter`. Pass
`options=ConvertDocumentsOptions(...)` only when you need to override them.

## Resuming after a restart — the local job ledger

The remote service returns a task id, while the actual job handle only lives in
the process that submitted it. If that process exits or crashes, callers cannot
tell which tasks were already submitted and typically resubmit everything —
spending extra quota and potentially producing duplicate results.

Opt into a local **job ledger** to make `submit` / wait / result retrieval
resumable across restarts:

```python
from docling.service_client import DoclingServiceClient, JobLedgerConfig

client = DoclingServiceClient(
    url=...,
    api_key=...,
    ledger=JobLedgerConfig(path="./docling-jobs.jsonl"),
)
```

With a ledger configured:

- an intent record is written **before** each submission and completed with the
  server-assigned task id once the submission returns;
- the same source with the same settings is submitted only **once** — later
  calls (also from other processes sharing the ledger) reattach to the existing
  task; completed tasks return their result directly, running tasks keep waiting;
- if the service no longer recognizes a recorded task, a
  `TaskNotFoundError` is raised instead of waiting forever;
- corrupt or half-written records are skipped individually, records past the
  TTL are cleaned up (`purge_expired_records()`), and no credentials (API keys,
  request headers, storage secrets) are ever written to the ledger.

The ledger is **off by default**; without it the client behaves exactly as
before. Tune takeover timeout, record TTL and cleanup via `JobLedgerConfig`.

## Examples


| Script       | What it shows                                                      |
| ------------ | ------------------------------------------------------------------ |
| `convert.py` | `convert()` and `convert_all()` — the high-level API               |
| `tasks.py`   | the `submit*` API: job lifecycle, result targets, per-item fan-out |
| `batch.py`   | `submit_batch()` for built-in or plugin sources and artifact targets |
| `chunk.py`   | `chunk()` — split a document into retrieval-ready pieces           |
