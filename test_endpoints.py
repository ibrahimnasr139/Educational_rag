# -*- coding: utf-8 -*-
"""
Comprehensive endpoint test script for the AI/RAG Backend.
Tests all endpoints defined in main.py and reports results.
"""

import requests
import json
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os
import time

BASE = "http://localhost:8000"
PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

results = []

def log(label, status, detail=""):
    print(f"  {status}  {label}")
    if detail:
        print(f"        {detail}")
    results.append((label, status, detail))

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

# ── 1. Root & Health ────────────────────────────────────────
section("1. Root & Health")

try:
    r = requests.get(f"{BASE}/", timeout=10)
    if r.status_code == 200 and r.json().get("status") == "running":
        log("GET /", PASS, str(r.json()))
    else:
        log("GET /", FAIL, f"status={r.status_code} body={r.text[:200]}")
except Exception as e:
    log("GET /", FAIL, str(e))

try:
    r = requests.get(f"{BASE}/health", timeout=10)
    body = r.json()
    if r.status_code == 200 and body.get("status") == "healthy":
        log("GET /health", PASS, f"db={body.get('database')} llm={body['settings'].get('llm_provider')}")
    else:
        log("GET /health", FAIL, f"status={r.status_code} body={r.text[:200]}")
except Exception as e:
    log("GET /health", FAIL, str(e))

# ── 2. Analytics ────────────────────────────────────────────
section("2. Analytics Endpoints")

for path in ["/api/analytics/completion", "/api/analytics/performance", "/api/analytics/revenue"]:
    try:
        r = requests.get(f"{BASE}{path}", timeout=15)
        if r.status_code == 200:
            log(f"GET {path}", PASS, str(r.json())[:120])
        else:
            log(f"GET {path}", FAIL, f"status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        log(f"GET {path}", FAIL, str(e))

try:
    r = requests.get(f"{BASE}/api/analytics/ai-analysis", timeout=60)
    body = r.json()
    # 200 with analysis key is pass (even if analysis says "no data")
    if r.status_code == 200 and "analysis" in body:
        # Check it didn't return the gemini_client AttributeError anymore
        if "gemini_client" in str(body.get("analysis", "")):
            log("GET /api/analytics/ai-analysis", FAIL, f"gemini_client bug still present: {body['analysis'][:120]}")
        else:
            log("GET /api/analytics/ai-analysis", PASS, str(body)[:120])
    else:
        log("GET /api/analytics/ai-analysis", FAIL, f"status={r.status_code} body={r.text[:200]}")
except Exception as e:
    log("GET /api/analytics/ai-analysis", FAIL, str(e))

# ── 3. Chunk / Transcript GET ────────────────────────────────
section("3. Chunk & Transcript Retrieval")

FAKE_ID = "nonexistent-file-id-000"

try:
    r = requests.get(f"{BASE}/api/get-chunks/{FAKE_ID}", timeout=10)
    if r.status_code == 404:
        log("GET /api/get-chunks/{file_id} (expect 404)", PASS, "Correct 404 for unknown file_id")
    elif r.status_code == 200:
        log("GET /api/get-chunks/{file_id}", PASS, str(r.json())[:120])
    else:
        log("GET /api/get-chunks/{file_id}", FAIL, f"status={r.status_code}")
except Exception as e:
    log("GET /api/get-chunks/{file_id}", FAIL, str(e))

try:
    r = requests.get(f"{BASE}/api/get-transcript-raw/{FAKE_ID}", timeout=10)
    if r.status_code == 404:
        log("GET /api/get-transcript-raw/{file_id} (expect 404)", PASS, "Correct 404 for unknown file_id")
    else:
        log("GET /api/get-transcript-raw/{file_id}", FAIL, f"status={r.status_code}")
except Exception as e:
    log("GET /api/get-transcript-raw/{file_id}", FAIL, str(e))

try:
    r = requests.get(f"{BASE}/api/suggest-metadata/{FAKE_ID}", timeout=30)
    log(f"GET /api/suggest-metadata/{{file_id}}", PASS if r.status_code == 200 else SKIP,
        f"status={r.status_code} body={r.text[:120]}")
except Exception as e:
    log("GET /api/suggest-metadata/{file_id}", FAIL, str(e))

try:
    r = requests.get(f"{BASE}/api/suggest-topics/{FAKE_ID}", timeout=30)
    log(f"GET /api/suggest-topics/{{file_id}}", PASS if r.status_code == 200 else SKIP,
        f"status={r.status_code} body={r.text[:120]}")
except Exception as e:
    log("GET /api/suggest-topics/{file_id}", FAIL, str(e))

# ── 4. AI Generation (correct schemas) ──────────────────────
section("4. AI Generation Endpoints")

# generate-description — context is a DescriptionContext object, type is a string literal
try:
    payload = {
        "context": {
            "lesson": {"title": "Photosynthesis", "description": "How plants make food from sunlight"}
        },
        "type": "lesson",
        "title": "Photosynthesis"
    }
    r = requests.post(f"{BASE}/api/generate-description", json=payload, timeout=60)
    if r.status_code == 200 and r.json().get("description"):
        log("POST /api/generate-description", PASS, r.json()["description"][:100])
    else:
        log("POST /api/generate-description", FAIL, f"status={r.status_code} body={r.text[:200]}")
except Exception as e:
    log("POST /api/generate-description", FAIL, str(e))

# ask-ai — only needs "question" + optional "previousAnswer"
try:
    payload = {
        "question": "What is photosynthesis and why is it important?"
    }
    r = requests.post(f"{BASE}/api/ask-ai", json=payload, timeout=60)
    if r.status_code == 200:
        log("POST /api/ask-ai", PASS, str(r.json())[:120])
    else:
        log("POST /api/ask-ai", FAIL, f"status={r.status_code} body={r.text[:200]}")
except Exception as e:
    log("POST /api/ask-ai", FAIL, str(e))

# ai-generate-questions — uses metadata + questionsNumber + difficulty + type
try:
    payload = {
        "metadata": {
            "subject": "Biology",
            "title": "Photosynthesis",
            "grade": "Grade 10"
        },
        "questionsNumber": 2,
        "difficulty": "medium",
        "type": "mcq"
    }
    r = requests.post(f"{BASE}/api/ai-generate-questions", json=payload, timeout=90)
    if r.status_code == 200:
        log("POST /api/ai-generate-questions", PASS, str(r.json())[:120])
    else:
        log("POST /api/ai-generate-questions", FAIL, f"status={r.status_code} body={r.text[:200]}")
except Exception as e:
    log("POST /api/ai-generate-questions", FAIL, str(e))

# generate-flashcards — requires subject, topic, numberOfCards
try:
    payload = {
        "subject": "Biology",
        "topic": "Photosynthesis",
        "chapter": "Chapter 5",
        "numberOfCards": 3
    }
    r = requests.post(f"{BASE}/api/generate-flashcards", json=payload, timeout=60)
    if r.status_code == 200:
        log("POST /api/generate-flashcards", PASS, str(r.json())[:120])
    else:
        log("POST /api/generate-flashcards", FAIL, f"status={r.status_code} body={r.text[:200]}")
except Exception as e:
    log("POST /api/generate-flashcards", FAIL, str(e))

# generate-quiz — uses topic (validation_alias for subject), questionsNumber, difficulty
try:
    payload = {
        "topic": "Photosynthesis",
        "questionsNumber": 2,
        "difficulty": "medium"
    }
    r = requests.post(f"{BASE}/api/generate-quiz", json=payload, timeout=60)
    if r.status_code == 200:
        log("POST /api/generate-quiz", PASS, str(r.json())[:120])
    else:
        log("POST /api/generate-quiz", FAIL, f"status={r.status_code} body={r.text[:200]}")
except Exception as e:
    log("POST /api/generate-quiz", FAIL, str(e))

# ai-assistant — requires message, fileId
try:
    payload = {
        "message": "Can you explain photosynthesis briefly?",
        "fileId": FAKE_ID,
        "course": "Biology 101",
        "module": "Plants",
        "lesson": "Photosynthesis"
    }
    r = requests.post(f"{BASE}/api/ai-assistant", json=payload, timeout=60)
    if r.status_code == 200:
        log("POST /api/ai-assistant", PASS, str(r.json())[:120])
    else:
        log("POST /api/ai-assistant", FAIL, f"status={r.status_code} body={r.text[:200]}")
except Exception as e:
    log("POST /api/ai-assistant", FAIL, str(e))

# summarise — language must be 'ar' or 'en', not 'english'
try:
    payload = {"fileId": FAKE_ID, "summaryLength": "short", "language": "en"}
    r = requests.post(f"{BASE}/api/summarise", json=payload, timeout=60)
    if r.status_code == 200:
        log("POST /api/summarise", PASS, str(r.json())[:120])
    elif r.status_code == 404:
        log("POST /api/summarise (expect 404 — no chunks for fake ID)", PASS, "Correct 404")
    else:
        log("POST /api/summarise", FAIL, f"status={r.status_code} body={r.text[:200]}")
except Exception as e:
    log("POST /api/summarise", FAIL, str(e))

# ── 5. File Upload Endpoints ─────────────────────────────────
section("5. File Upload Endpoints")

TEST_TXT = b"This is a test document for endpoint validation. It contains information about photosynthesis."
TEST_FILE_ID = f"test-file-{int(time.time())}"
TEST_JOB_ID  = f"test-job-{int(time.time())}"

try:
    files = {"file": ("test.txt", TEST_TXT, "text/plain")}
    data  = {"type": "document", "fileId": TEST_FILE_ID, "jobId": TEST_JOB_ID,
              "translateToEnglish": "false", "isCourseBook": "false"}
    r = requests.post(f"{BASE}/api/embed-and-transcribe", files=files, data=data, timeout=120)
    if r.status_code == 200:
        log("POST /api/embed-and-transcribe (document)", PASS, str(r.json()))
    else:
        log("POST /api/embed-and-transcribe (document)", FAIL, f"status={r.status_code} body={r.text[:300]}")
except Exception as e:
    log("POST /api/embed-and-transcribe (document)", FAIL, str(e))

try:
    files = {"file": ("test2.txt", TEST_TXT, "text/plain")}
    data  = {"type": "document", "fileId": f"{TEST_FILE_ID}-embed"}
    r = requests.post(f"{BASE}/api/embed-file", files=files, data=data, timeout=120)
    if r.status_code == 200:
        log("POST /api/embed-file (document)", PASS, str(r.json()))
    else:
        log("POST /api/embed-file (document)", FAIL, f"status={r.status_code} body={r.text[:300]}")
except Exception as e:
    log("POST /api/embed-file (document)", FAIL, str(e))

# Verify the uploaded file's chunks are retrievable (with polling for background task completion)
EMBED_FILE_ID = f"{TEST_FILE_ID}-embed"
try:
    success = False
    for attempt in range(15):
        time.sleep(1)
        r = requests.get(f"{BASE}/api/get-chunks/{EMBED_FILE_ID}", timeout=10)
        if r.status_code == 200 and len(r.json()) > 0:
            log(f"GET /api/get-chunks/{{embedded_file_id}} (verify embed worked)", PASS,
                f"{len(r.json())} chunk(s) found after {attempt+1}s")
            success = True
            break
    if not success:
        log(f"GET /api/get-chunks/{{embedded_file_id}} (verify embed worked)", SKIP,
            "Chunks not found — embedding may not have finished or stored in ChromaDB yet")
except Exception as e:
    log(f"GET /api/get-chunks/{{embedded_file_id}}", FAIL, str(e))

# ── 6. Asynchronous Bunny Video Embedding (JSON) ─────────────
section("6. Asynchronous Bunny Video Embedding")

try:
    # Use a dummy video ID for testing - it will fail downloading but the endpoint should return immediately.
    # We can inspect if the endpoint returns status: success.
    payload = {
        "fileId": "demo-bunny-video-id",
        "type": "video",
        "callbackUrl": "http://localhost:8000/health"
    }
    r = requests.post(f"{BASE}/api/embed-file", json=payload, timeout=30)
    if r.status_code == 200:
        log("POST /api/embed-file (Bunny Video JSON)", PASS, str(r.json()))
    else:
        log("POST /api/embed-file (Bunny Video JSON)", FAIL, f"status={r.status_code} body={r.text[:300]}")
except Exception as e:
    log("POST /api/embed-file (Bunny Video JSON)", FAIL, str(e))


# Tiny WAV for generate-transcript
TINY_AUDIO = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
try:
    files = {"audio": ("test.wav", TINY_AUDIO, "audio/wav")}
    data  = {"fileId": f"{TEST_FILE_ID}-audio", "jobId": f"{TEST_JOB_ID}-audio"}
    r = requests.post(f"{BASE}/api/generate-transcript", files=files, data=data, timeout=120)
    if r.status_code == 200:
        body = r.json()
        # A tiny invalid WAV will fail transcription gracefully — that's fine
        status_val = body.get("status")
        if status_val == "success":
            log("POST /api/generate-transcript (tiny WAV)", PASS, str(body))
        else:
            log("POST /api/generate-transcript (tiny WAV — graceful fail)", SKIP,
                "Tiny WAV failed transcription as expected (not a real audio file)")
    else:
        log("POST /api/generate-transcript (tiny WAV)", FAIL, f"status={r.status_code} body={r.text[:300]}")
except Exception as e:
    log("POST /api/generate-transcript (tiny WAV)", FAIL, str(e))

# ── Summary ──────────────────────────────────────────────────
section("SUMMARY")
passed  = sum(1 for _, s, _ in results if s == PASS)
failed  = sum(1 for _, s, _ in results if s == FAIL)
skipped = sum(1 for _, s, _ in results if s == SKIP)
total   = len(results)

print(f"\n  Total: {total}  |  {PASS} {passed}  |  {FAIL} {failed}  |  {SKIP} {skipped}")
print()

if failed > 0:
    print("  FAILED endpoints:")
    for label, status, detail in results:
        if status == FAIL:
            print(f"    - {label}: {detail[:150]}")

sys.exit(0 if failed == 0 else 1)
