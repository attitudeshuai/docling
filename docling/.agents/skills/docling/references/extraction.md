# Structured extraction (DocumentExtractor)

Beta feature. Conversion (`DocumentConverter`) turns a document into a full
`DoclingDocument`. **Extraction** (`DocumentExtractor`) does something different:
it pulls **specific, typed fields** out of a document according to a template —
e.g. invoice number and total from a scanned invoice, or a set of contract
fields. Use it when the user wants *values*, not the whole document.

Requires the `extract-core` extra (see [slim-packaging.md](slim-packaging.md)):

```bash
pip install "docling-slim[extract-core,format-pdf,models-vlm-inline]"
# (included in the full `docling` package)
```

## Entry point

```python
from docling.document_extractor import DocumentExtractor
from docling.datamodel.base_models import InputFormat

extractor = DocumentExtractor(allowed_formats=[InputFormat.PDF, InputFormat.IMAGE])
```

`extract(source, template, ...)` returns an `ExtractionResult`;
`extract_all(sources, template, ...)` returns an iterator of them. `source` is a
path, URL, or `DocumentStream`.

## Templates — four ways to describe what to pull

The `template` argument accepts a string, a dict, a Pydantic model **class**, or
a Pydantic model **instance** (`Union[str, dict, BaseModel, Type[BaseModel]]`).

```python
# 1. JSON-ish string
result = extractor.extract(source="invoice.pdf",
                           template='{"bill_no": "string", "total": "float"}')

# 2. dict template
result = extractor.extract(source="invoice.pdf",
                           template={"bill_no": "string", "total": "float"})

# 3. Pydantic model class (recommended — typed, self-documenting)
from pydantic import BaseModel

class Invoice(BaseModel):
    bill_no: str
    total: float

result = extractor.extract(source="invoice.pdf", template=Invoice)

# 4. Pydantic instance (fields double as examples / defaults)
result = extractor.extract(source="invoice.pdf",
                           template=Invoice(bill_no="INV-0001", total=0.0))
```

Prefer a **Pydantic model class** for durable schemas — it documents intent and
gives you validation on the way out.

## Reading the result

`ExtractionResult` has `status` (a `ConversionStatus`), `errors`, and `pages`
(one `ExtractedPageData` per page). Each page carries `extracted_data`
(the dict of pulled fields), `raw_text`, and per-page `errors`.

```python
from docling.datamodel.base_models import ConversionStatus

result = extractor.extract(source="invoice.pdf", template=Invoice)

if result.status in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
    for page in result.pages:
        print(page.page_no, page.extracted_data)   # e.g. {"bill_no": "...", "total": 42.0}
else:
    print("extraction failed:", result.errors)
```

## Many documents

```python
for result in extractor.extract_all(
    source=["a.pdf", "b.pdf", "https://example.com/c.pdf"],
    template=Invoice,
    raises_on_error=False,     # keep going past individual failures
):
    print(result.input.file.name, result.status)
```

See [python-sdk.md](python-sdk.md) for the same status/error handling pattern on
the conversion side, and [service-client.md](service-client.md) to run
extraction-style workloads against a remote service.

## Cross-page merging (document-level result)

By default each page is extracted independently and a single failing page fails
the whole document. Pass `ExtractionMergeOptions(enabled=True)` to additionally
receive a document-level result in `result.merged`, built from the per-page
results without changing or hiding `result.pages`:

```python
from docling.datamodel.extraction_options import (
    ExtractionMergeOptions,
    FieldMergeStrategy,
)

result = extractor.extract(
    source="multi_page_invoice.pdf",
    template=Invoice,
    merge_options=ExtractionMergeOptions(
        enabled=True,
        strategy=FieldMergeStrategy.LAST_PAGE,  # or FIRST_PAGE
        max_page_retries=2,                    # extra attempts per failed page
    ),
)

merged = result.merged
print(merged.status)   # success / partial_success / failure
print(merged.data)     # one template-shaped document dict
for name, info in merged.fields.items():
    print(name, info.presence, info.chosen_page, info.has_conflict)
    print([(c.page_no, c.value) for c in info.candidates])
```

Behavior:

- **Priority & conflicts.** A scalar field found on several pages takes the
  value from the lowest (`FIRST_PAGE`) or highest (`LAST_PAGE`) numbered page.
  Every candidate value and its source page is retained in
  `fields[...].candidates`, and `has_conflict` flags differing values.
- **Missing vs empty.** A field present only with a null/empty value
  (`null`, `""`, `[]`, `{}`) stays in `data` and is marked `empty`; a template
  field absent from every page is omitted from `data` and marked `missing`.
- **Nested objects and lists.** Nested objects merge key by key; list fields
  concatenate across pages in page order. Fields seen by the model but absent
  from the template are kept, so merging never drops a readable value.
- **Page failures.** A page is retried up to `max_page_retries` additional
  times on the same rendered image; if it still fails it is recorded once under
  `merged.pages` and excluded from the merge. Other pages keep merging and the
  result is `partial_success`; it is `failure` only when no page contributes.
- **Serialization.** `merged` is plain serializable data: use
  `merged.model_dump_json()` and `MergedExtractionData.model_validate_json(...)`
  for persistence. Old payloads without the merged section load with
  `merged=None`.
- **Opt-in.** Without `enabled=True`, results are identical to the per-page
  behavior described above: no retries, no `merged` section.

