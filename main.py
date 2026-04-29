"""
Stage 4 — Structured Extraction Service
Extracts structured data from court document images using GPT-4o.
Writes to signal_events, entities, and properties via extract_document() stored procedure.
"""

import asyncio
import base64
import json
import logging
import math
import os
import re
import threading
import time
from datetime import datetime
from typing import Any, Optional

import httpx
import openai
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
PORT = int(os.getenv("PORT", "8080"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "10"))

SUPABASE_HEADERS = {
    "apikey": SUPABASE_SERVICE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    "Content-Type": "application/json",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stage4")

# ---------------------------------------------------------------------------
# Street abbreviation table
# ---------------------------------------------------------------------------

STREET_ABBREVIATIONS: dict[str, str] = {
    "STREET": "ST",
    "AVENUE": "AVE",
    "ROAD": "RD",
    "DRIVE": "DR",
    "LANE": "LN",
    "COURT": "CT",
    "CIRCLE": "CIR",
    "BOULEVARD": "BLVD",
    "PARKWAY": "PKWY",
    "PLACE": "PL",
    "TERRACE": "TER",
    "NORTH": "N",
    "SOUTH": "S",
    "EAST": "E",
    "WEST": "W",
}

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def normalize_name(value: Optional[str]) -> Optional[str]:
    """Uppercase a name component; empty string → None."""
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    return value.upper()


def normalize_upper(value: Optional[str]) -> Optional[str]:
    """Generic uppercase; empty string → None."""
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    return value.upper()


def normalize_zip(value: Optional[str]) -> Optional[str]:
    """5-digit ZIP only — strip +4 extension."""
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    return value.split("-")[0][:5]


def normalize_street(value: Optional[str]) -> Optional[str]:
    """Uppercase + apply abbreviation table."""
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    value = value.upper()
    # Apply abbreviation table — match whole words only
    for long_form, abbr in STREET_ABBREVIATIONS.items():
        value = re.sub(rf"\b{long_form}\b", abbr, value)
    return value


def validate_date(value: Optional[str]) -> Optional[str]:
    """Validate an ISO 8601 date string. Return None if invalid."""
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
        # Confirm round-trip (catches impossible dates)
        if parsed.strftime("%Y-%m-%d") != value:
            logger.warning("Date validation failed (round-trip): %s", value)
            return None
        return value
    except ValueError:
        logger.warning("Date validation failed (parse): %s", value)
        return None


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------


def dedup_entities(
    entities: list[dict[str, Any]], properties: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Deduplicate primary entities by (first_name, last_name, suffix).
    Merge addresses from duplicates into the kept entity.
    """
    seen: dict[tuple, int] = {}  # (first, last, suffix) → index in deduped list
    deduped: list[dict[str, Any]] = []
    # Map old temp_index → new temp_index for property relinking
    index_remap: dict[str, str] = {}

    for entity in entities:
        if not entity.get("is_primary"):
            deduped.append(entity)
            continue

        key = (
            (entity.get("first_name") or "").upper(),
            (entity.get("last_name") or "").upper(),
            (entity.get("suffix") or "").upper(),
        )

        if key in seen:
            # Duplicate — remap properties to the kept entity
            kept_idx = seen[key]
            kept_temp = deduped[kept_idx]["temp_index"]
            index_remap[entity["temp_index"]] = kept_temp
        else:
            seen[key] = len(deduped)
            deduped.append(entity)

    # Remap properties whose entity was deduped away
    for prop in properties:
        eti = prop.get("entity_temp_index")
        if eti is not None and eti in index_remap:
            prop["entity_temp_index"] = index_remap[eti]

    return deduped, properties


def dedup_properties(properties: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicate properties with identical (street, city, state, zip)
    that belong to the same entity (or both standalone).
    """
    seen: set[tuple] = set()
    deduped: list[dict[str, Any]] = []

    for prop in properties:
        key = (
            (prop.get("street") or "").upper(),
            (prop.get("city") or "").upper(),
            (prop.get("state") or "").upper(),
            (prop.get("zip") or ""),
            prop.get("entity_temp_index"),  # None for standalone
        )
        if key not in seen:
            seen.add(key)
            deduped.append(prop)

    return deduped


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------


def compute_chunks(page_count: int, chunk_size: int) -> list[tuple[int, int]]:
    """
    Split page_count into balanced chunks.
    Returns list of (start_page_1indexed, end_page_1indexed) inclusive tuples.
    """
    if page_count <= chunk_size:
        return [(1, page_count)]

    num_chunks = math.ceil(page_count / chunk_size)
    base_size = page_count // num_chunks
    remainder = page_count % num_chunks

    chunks: list[tuple[int, int]] = []
    current = 1
    for i in range(num_chunks):
        size = base_size + (1 if i < remainder else 0)
        chunks.append((current, current + size - 1))
        current += size

    return chunks


def merge_chunked_results(chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Merge GPT-4o results from multiple chunks per the brief's merge rules.
    """
    if len(chunk_results) == 1:
        return chunk_results[0]

    merged = json.loads(json.dumps(chunk_results[0]))  # deep copy first chunk

    for chunk in chunk_results[1:]:
        # Concatenate primary_entities.people
        merged.setdefault("primary_entities", {}).setdefault("people", []).extend(
            chunk.get("primary_entities", {}).get("people", [])
        )
        # Concatenate primary_entities.properties
        merged.setdefault("primary_entities", {}).setdefault("properties", []).extend(
            chunk.get("primary_entities", {}).get("properties", [])
        )
        # Concatenate financial_obligations
        merged.setdefault("financial_obligations", []).extend(
            chunk.get("financial_obligations", [])
        )
        # Concatenate key_facts (do NOT deduplicate)
        merged.setdefault("document_narrative", {}).setdefault("key_facts", []).extend(
            chunk.get("document_narrative", {}).get("key_facts", [])
        )

    # Everything else (document_fingerprint, summary, primary_action,
    # service_entities, extraction_metadata, etc.) stays from first chunk.

    return merged


# ---------------------------------------------------------------------------
# Processing state
# ---------------------------------------------------------------------------

processing_lock = threading.Lock()
processing_status: dict[str, Any] = {
    "state": "idle",
    "documents_total": 0,
    "documents_processed": 0,
    "documents_failed": 0,
    "validation_failures": 0,
    "documents_remaining": 0,
    "errors": [],
}


def reset_status() -> None:
    processing_status.update(
        {
            "state": "idle",
            "documents_total": 0,
            "documents_processed": 0,
            "documents_failed": 0,
            "validation_failures": 0,
            "documents_remaining": 0,
            "errors": [],
        }
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Stage 4 — Structured Extraction Service")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
async def status() -> dict[str, Any]:
    return dict(processing_status)


@app.post("/process")
async def process(limit: Optional[int] = Query(default=None)) -> dict[str, str]:
    acquired = processing_lock.acquire(blocking=False)
    if not acquired:
        return JSONResponse(
            status_code=409,
            content={"error": "Processing already in progress"},
        )
    # Release will happen in the background thread
    reset_status()
    processing_status["state"] = "processing"
    thread = threading.Thread(
        target=run_extraction, args=(limit,), daemon=True
    )
    thread.start()
    return {"status": "started"}


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------


def run_extraction(limit: Optional[int]) -> None:
    """Main processing loop — runs in background thread."""
    try:
        with httpx.Client(timeout=30.0) as http:
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

            # Step 1: Load prompt
            prompt_text, schema, prompt_version = load_prompt(http)
            if prompt_text is None:
                processing_status["state"] = "error"
                processing_status["errors"].append(
                    "No active prompt found for stage_4_extractor"
                )
                return

            logger.info(
                "Loaded prompt version %d for stage_4_extractor", prompt_version
            )

            total_processed = 0
            total_failed = 0
            total_validation = 0
            effective_limit = limit  # Track remaining if limit was specified

            # Batch loop — re-query each batch
            while True:
                batch_size = BATCH_SIZE
                if effective_limit is not None:
                    batch_size = min(BATCH_SIZE, effective_limit - total_processed - total_failed - total_validation)
                    if batch_size <= 0:
                        break

                docs = fetch_documents(http, batch_size)
                if not docs:
                    break

                # Re-anchor remaining counter per batch
                processing_status["documents_remaining"] = len(docs)
                processing_status["documents_total"] = (
                    total_processed + total_failed + total_validation + len(docs)
                )

                for doc in docs:
                    doc_id = doc["id"]
                    document_id = doc["document_id"]
                    images_path = doc["images_path"]
                    page_count = doc["page_count"]
                    county = doc["county"]

                    try:
                        result = process_document(
                            http,
                            openai_client,
                            doc_id,
                            document_id,
                            images_path,
                            page_count,
                            county,
                            prompt_text,
                            schema,
                            prompt_version,
                        )
                        if result == "extracted":
                            total_processed += 1
                            processing_status["documents_processed"] = total_processed
                        elif result == "validation_failure":
                            total_validation += 1
                            processing_status["validation_failures"] = total_validation
                        else:
                            total_failed += 1
                            processing_status["documents_failed"] = total_failed

                    except Exception as e:
                        logger.error(
                            "Unexpected error processing %s: %s", document_id, e
                        )
                        total_failed += 1
                        processing_status["documents_failed"] = total_failed
                        processing_status["errors"].append(
                            f"{document_id}: {str(e)[:200]}"
                        )
                        # Mark as API error for retry
                        update_document_failure(
                            http,
                            doc_id,
                            "extraction_failed - API error",
                            is_valuable=True,
                        )

                    processing_status["documents_remaining"] -= 1

        logger.info(
            "Extraction complete: %d extracted, %d failed, %d validation failures",
            total_processed,
            total_failed,
            total_validation,
        )

    except Exception as e:
        logger.error("Fatal error in extraction loop: %s", e)
        processing_status["errors"].append(f"Fatal: {str(e)[:200]}")
        processing_status["state"] = "error"
    finally:
        if processing_status["state"] != "error":
            processing_status["state"] = "idle"
        processing_lock.release()


def load_prompt(http: httpx.Client) -> tuple[Optional[str], Optional[dict], Optional[int]]:
    """Load active prompt from prompts table."""
    url = (
        f"{SUPABASE_URL}/rest/v1/prompts"
        "?stage=eq.stage_4_extractor&is_active=eq.true"
        "&select=prompt_text,schema,version"
        "&limit=1"
    )
    resp = http.get(url, headers=SUPABASE_HEADERS)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return None, None, None
    row = rows[0]
    return row["prompt_text"], row["schema"], row["version"]


def fetch_documents(http: httpx.Client, batch_size: int) -> list[dict[str, Any]]:
    """Fetch next batch of documents to process."""
    url = f"{SUPABASE_URL}/rest/v1/documents"
    params = {
        "or": "(current_stage.eq.classified,current_stage.eq.extraction_failed - API error)",
        "is_valuable": "eq.true",
        "select": "id,document_id,images_path,page_count,county",
        "order": "created_at",
        "limit": str(batch_size),
    }
    resp = http.get(url, headers=SUPABASE_HEADERS, params=params)
    resp.raise_for_status()
    return resp.json()


def download_images(
    http: httpx.Client, images_path: str, page_count: int
) -> list[str]:
    """Download page images from Supabase Storage. Returns list of base64 strings."""
    images: list[str] = []
    for page_num in range(1, page_count + 1):
        filename = f"page_{page_num:03d}.png"
        storage_path = f"{images_path}{filename}"
        url = (
            f"{SUPABASE_URL}/storage/v1/object/documents/{storage_path}"
        )
        resp = http.get(url, headers={
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        })
        resp.raise_for_status()
        images.append(base64.b64encode(resp.content).decode("utf-8"))
    return images


def call_gpt4o(
    openai_client: openai.OpenAI,
    prompt_text: str,
    schema: dict[str, Any],
    image_data: list[str],
    document_id: str,
) -> dict[str, Any]:
    """
    Call GPT-4o with images and extraction prompt.
    Retries up to 3 times with exponential backoff for 429/500/timeout.
    """
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"Document: {document_id}\n\nAnalyze these {len(image_data)} page(s).",
        }
    ] + [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
        }
        for b64 in image_data
    ]

    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=16000,
                temperature=0.1,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction_result",
                        "strict": True,
                        "schema": schema,
                    },
                },
                timeout=120,
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except (
            openai.RateLimitError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            json.JSONDecodeError,
        ) as e:
            last_error = e
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "GPT-4o attempt %d failed for %s: %s. Retrying in %ds...",
                    attempt + 1,
                    document_id,
                    str(e)[:100],
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "GPT-4o exhausted retries for %s: %s", document_id, e
                )

    raise last_error  # type: ignore[misc]


def process_document(
    http: httpx.Client,
    openai_client: openai.OpenAI,
    doc_id: str,
    document_id: str,
    images_path: str,
    page_count: int,
    county: str,
    prompt_text: str,
    schema: dict[str, Any],
    prompt_version: int,
) -> str:
    """
    Process a single document through extraction.
    Returns: 'extracted', 'validation_failure', or 'api_error'.
    """
    doc_county = county.lower() if county else ""

    # Step 3: Download images
    logger.info("Downloading %d images for %s", page_count, document_id)
    image_data = download_images(http, images_path, page_count)

    # Step 4-5: GPT-4o call (with chunking if needed)
    chunks = compute_chunks(page_count, CHUNK_SIZE)
    is_chunked = len(chunks) > 1

    if is_chunked:
        logger.info(
            "Chunking %s: %d pages into %d chunks of sizes %s",
            document_id,
            page_count,
            len(chunks),
            [end - start + 1 for start, end in chunks],
        )

    try:
        chunk_results: list[dict[str, Any]] = []
        for chunk_start, chunk_end in chunks:
            chunk_images = image_data[chunk_start - 1 : chunk_end]
            result = call_gpt4o(
                openai_client, prompt_text, schema, chunk_images, document_id
            )
            chunk_results.append(result)
            # Rate limiting between API calls
            time.sleep(1.0)

        if is_chunked:
            result = merge_chunked_results(chunk_results)
        else:
            result = chunk_results[0]

    except Exception as e:
        logger.error("API failure for %s: %s", document_id, e)
        update_document_failure(
            http, doc_id, "extraction_failed - API error", is_valuable=True
        )
        processing_status["errors"].append(f"{document_id}: API error - {str(e)[:150]}")
        return "api_error"

    # Override extraction_method based on chunking
    result.setdefault("extraction_metadata", {})["extraction_method"] = (
        "chunked_vision" if is_chunked else "vision_images"
    )

    # Step 6: Code-level validation
    people = result.get("primary_entities", {}).get("people", [])
    props = result.get("primary_entities", {}).get("properties", [])
    has_people = len(people) > 0
    has_properties = len(props) > 0

    if not has_people and not has_properties:
        logger.info(
            "Validation failure for %s: no primary person or property", document_id
        )
        update_document_failure(
            http,
            doc_id,
            "extraction_failed - no primary person or property identified",
            is_valuable=False,
        )
        return "validation_failure"

    # Step 7-8: Build entities and properties, then dedup and normalize
    entities_payload, properties_payload = build_entity_property_payloads(
        result, doc_county
    )

    # Dedup entities (merges addresses from duplicates)
    entities_payload, properties_payload = dedup_entities(
        entities_payload, properties_payload
    )
    # Dedup properties
    properties_payload = dedup_properties(properties_payload)

    # Step 9: Build signal_event payload
    signal_event_payload = build_signal_event_payload(result, prompt_version)

    # Step 10: Call stored procedure
    entity_count = len(entities_payload)
    property_count = len(properties_payload)
    extraction_confidence = signal_event_payload.get("extraction_confidence", "unknown")
    doc_type = signal_event_payload.get("document_type", "unknown")

    logger.info(
        "Calling extract_document for %s: type=%s, entities=%d (primary=%d, service=%d), "
        "properties=%d, confidence=%s, method=%s%s",
        document_id,
        doc_type,
        entity_count,
        sum(1 for e in entities_payload if e.get("is_primary")),
        sum(1 for e in entities_payload if not e.get("is_primary")),
        property_count,
        extraction_confidence,
        "chunked_vision" if is_chunked else "vision_images",
        f", chunks={len(chunks)}" if is_chunked else "",
    )

    rpc_response = http.post(
        f"{SUPABASE_URL}/rest/v1/rpc/extract_document",
        headers=SUPABASE_HEADERS,
        json={
            "p_document_id": doc_id,
            "p_signal_event": signal_event_payload,
            "p_entities": entities_payload,
            "p_properties": properties_payload,
        },
    )

    if rpc_response.status_code >= 400:
        error_detail = rpc_response.text[:300]
        logger.error(
            "extract_document RPC failed for %s: %d — %s",
            document_id,
            rpc_response.status_code,
            error_detail,
        )
        update_document_failure(
            http, doc_id, "extraction_failed - API error", is_valuable=True
        )
        processing_status["errors"].append(
            f"{document_id}: RPC error {rpc_response.status_code}"
        )
        return "api_error"

    logger.info("Successfully extracted %s", document_id)
    return "extracted"


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def build_signal_event_payload(
    result: dict[str, Any], prompt_version: int
) -> dict[str, Any]:
    """Build the p_signal_event JSONB payload from GPT-4o response."""
    fp = result.get("document_fingerprint", {})
    narr = result.get("document_narrative", {})
    meta = result.get("extraction_metadata", {})
    obligations = result.get("financial_obligations", [])

    # Primary financial obligation (first item, or None)
    primary_fin = obligations[0] if obligations else {}
    additional = obligations[1:] if len(obligations) > 1 else None

    # Narrative details — combine into jsonb
    narrative_details = {
        "key_facts": narr.get("key_facts", []),
        "legal_descriptions": narr.get("legal_descriptions", []),
        "important_dates": narr.get("important_dates", []),
    }

    payload: dict[str, Any] = {
        # document_fingerprint
        "document_type": fp.get("document_type"),
        "document_status": fp.get("document_status"),
        "book_number": fp.get("book_number"),
        "page_number": fp.get("page_number"),
        "instrument_number": fp.get("instrument_number"),
        "case_number": fp.get("case_number"),
        "recording_date": fp.get("recording_date"),
        "recording_date_parsed": validate_date(fp.get("recording_date_parsed")),
        # document_narrative
        "narrative_summary": narr.get("summary"),
        "primary_action": narr.get("primary_action"),
        "relief_sought": narr.get("relief_sought"),
        "outcome": narr.get("outcome"),
        "narrative_details": narrative_details,
        # financial_obligations[0]
        "signal_type": primary_fin.get("obligation_type") if primary_fin else None,
        "financial_status": primary_fin.get("status") if primary_fin else None,
        "amount": primary_fin.get("amount") if primary_fin else None,
        "debtor": primary_fin.get("debtor") if primary_fin else None,
        "creditor": primary_fin.get("creditor") if primary_fin else None,
        "date_originated": primary_fin.get("date_originated") if primary_fin else None,
        "date_originated_parsed": validate_date(
            primary_fin.get("date_originated_parsed")
        )
        if primary_fin
        else None,
        "date_due": primary_fin.get("date_due") if primary_fin else None,
        "date_due_parsed": validate_date(primary_fin.get("date_due_parsed"))
        if primary_fin
        else None,
        "additional_obligations": additional,
        # extraction_metadata
        "extraction_confidence": meta.get("confidence"),
        "pages_processed": meta.get("pages_processed"),
        "extraction_method": meta.get("extraction_method"),
        "extraction_warnings": meta.get("warnings"),
        "prompt_version": prompt_version,
    }

    return payload


def build_entity_property_payloads(
    result: dict[str, Any], doc_county: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build p_entities and p_properties arrays from GPT-4o response.
    Applies normalization to all fields.
    """
    entities: list[dict[str, Any]] = []
    properties: list[dict[str, Any]] = []
    temp_idx = 0

    # --- Primary people ---
    for person in result.get("primary_entities", {}).get("people", []):
        entities.append(
            {
                "temp_index": str(temp_idx),
                "entity_type": "person",
                "is_primary": True,
                "role": person.get("role"),
                "raw_name": person.get("full_name"),
                "first_name": normalize_name(person.get("first_name")),
                "middle_name": normalize_name(person.get("middle_name")),
                "last_name": normalize_name(person.get("last_name")),
                "suffix": normalize_name(person.get("suffix")),
                "firm_name": None,
                "representing": None,
            }
        )

        for addr in person.get("addresses", []):
            properties.append(
                {
                    "entity_temp_index": str(temp_idx),
                    "raw_address": addr.get("full_address"),
                    "street": normalize_street(addr.get("street")),
                    "city": normalize_upper(addr.get("city")),
                    "state": normalize_upper(addr.get("state")),
                    "zip": normalize_zip(addr.get("zip")),
                    "county": doc_county,
                    "property_type": None,
                    "address_type": addr.get("address_type"),
                    "parcel_id": None,
                    "legal_description": None,
                }
            )

        temp_idx += 1

    # --- Service entities: attorneys ---
    for atty in result.get("service_entities", {}).get("attorneys", []):
        entities.append(
            {
                "temp_index": str(temp_idx),
                "entity_type": "person",
                "is_primary": False,
                "role": "attorney",
                "raw_name": atty.get("name"),
                "first_name": normalize_name(atty.get("first_name")),
                "middle_name": normalize_name(atty.get("middle_name")),
                "last_name": normalize_name(atty.get("last_name")),
                "suffix": normalize_name(atty.get("suffix")),
                "firm_name": atty.get("firm"),
                "representing": atty.get("representing"),
            }
        )
        temp_idx += 1

    # --- Service entities: creditors (bare strings) ---
    for cred_name in result.get("service_entities", {}).get("creditors", []):
        entities.append(
            {
                "temp_index": str(temp_idx),
                "entity_type": "business",
                "is_primary": False,
                "role": "creditor",
                "raw_name": cred_name,
                "first_name": None,
                "last_name": cred_name.upper() if cred_name else None,
                "middle_name": None,
                "suffix": None,
                "firm_name": None,
                "representing": None,
            }
        )
        temp_idx += 1

    # --- Service entities: courts (bare strings) ---
    for court_name in result.get("service_entities", {}).get("courts", []):
        entities.append(
            {
                "temp_index": str(temp_idx),
                "entity_type": "government",
                "is_primary": False,
                "role": "court",
                "raw_name": court_name,
                "first_name": None,
                "last_name": court_name.upper() if court_name else None,
                "middle_name": None,
                "suffix": None,
                "firm_name": None,
                "representing": None,
            }
        )
        temp_idx += 1

    # --- Service entities: others ---
    for other in result.get("service_entities", {}).get("others", []):
        entities.append(
            {
                "temp_index": str(temp_idx),
                "entity_type": "person",
                "is_primary": False,
                "role": other.get("role"),
                "raw_name": other.get("name"),
                "first_name": normalize_name(other.get("first_name")),
                "middle_name": normalize_name(other.get("middle_name")),
                "last_name": normalize_name(other.get("last_name")),
                "suffix": normalize_name(other.get("suffix")),
                "firm_name": None,
                "representing": None,
            }
        )
        temp_idx += 1

    # --- Standalone properties ---
    for prop in result.get("primary_entities", {}).get("properties", []):
        properties.append(
            {
                "entity_temp_index": None,
                "raw_address": prop.get("full_address"),
                "street": normalize_street(prop.get("street")),
                "city": normalize_upper(prop.get("city")),
                "state": normalize_upper(prop.get("state")),
                "zip": normalize_zip(prop.get("zip")),
                "county": doc_county,
                "property_type": prop.get("property_type"),
                "address_type": None,
                "parcel_id": prop.get("parcel_id"),
                "legal_description": prop.get("legal_description"),
            }
        )

    return entities, properties


# ---------------------------------------------------------------------------
# Document update helpers
# ---------------------------------------------------------------------------


def update_document_failure(
    http: httpx.Client,
    doc_id: str,
    stage: str,
    is_valuable: bool,
) -> None:
    """Direct UPDATE on documents table for failure cases."""
    url = f"{SUPABASE_URL}/rest/v1/documents?id=eq.{doc_id}"
    body: dict[str, Any] = {"current_stage": stage}
    if not is_valuable:
        body["is_valuable"] = False
    resp = http.patch(url, headers=SUPABASE_HEADERS, json=body)
    if resp.status_code >= 400:
        logger.error(
            "Failed to update document %s to stage '%s': %s",
            doc_id,
            stage,
            resp.text[:200],
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting Stage 4 Structured Extraction Service on port %d", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
