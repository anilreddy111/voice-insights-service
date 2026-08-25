"""Integration tests for POST /analyze (hermetic: fake engine, real pipeline)."""

import numpy as np

from .conftest import SR, make_speech_like, make_wav_bytes


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_analyze_contract_matches_spec(client):
    wav = make_wav_bytes(make_speech_like(4.0))
    r = client.post(
        "/analyze",
        files={"file": ("call.wav", wav, "audio/wav")},
        params={"contact_id": "11111111-1111-1111-1111-111111111111"},
    )
    assert r.status_code == 200
    body = r.json()
    # Exact spec fields present with correct types
    assert body["contact_id"] == "11111111-1111-1111-1111-111111111111"
    assert body["gender"]["prediction"] in {"male", "female", "unknown"}
    assert 0.0 <= body["gender"]["confidence"] <= 1.0
    assert body["age_bracket"]["prediction"] in {"18-30", "31-45", "46-60", "60+", "unknown"}
    assert 0.0 <= body["age_bracket"]["confidence"] <= 1.0
    assert isinstance(body["processing_ms"], int) and body["processing_ms"] >= 0
    assert body["audio_quality"] in {"good", "degraded", "insufficient"}


def test_analyze_generates_contact_id_when_missing(client):
    wav = make_wav_bytes(make_speech_like(4.0))
    r = client.post("/analyze", files={"file": ("a.wav", wav, "audio/wav")})
    assert r.status_code == 200
    assert len(r.json()["contact_id"]) == 36  # uuid4


def test_insufficient_audio_returns_unknowns(client):
    # Digital silence: must NOT produce confident garbage.
    silence = np.zeros(SR * 3, dtype=np.float32)
    r = client.post("/analyze", files={"file": ("s.wav", make_wav_bytes(silence), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["audio_quality"] == "insufficient"
    assert body["gender"]["prediction"] == "unknown"
    assert body["age_bracket"]["prediction"] == "unknown"


def test_raw_audio_body_accepted(client):
    x = make_speech_like(4.0)
    pcm16 = (x * 32767).astype("<i2").tobytes()
    r = client.post(
        "/analyze",
        content=pcm16,
        headers={"Content-Type": "audio/raw"},
        params={"contact_id": "raw-test", "encoding": "pcm16"},
    )
    assert r.status_code == 200
    assert r.json()["contact_id"] == "raw-test"


def test_rejects_text_payload_with_415(client):
    r = client.post("/analyze", content=b"hello", headers={"Content-Type": "text/plain"})
    assert r.status_code == 415
    assert r.json()["error"] == "unsupported_media_type"


def test_rejects_oversized_upload(client, settings):
    big = b"\0" * (settings.max_upload_bytes + 1)
    r = client.post("/analyze", files={"file": ("big.wav", big, "audio/wav")})
    assert r.status_code == 413


def test_corrupt_audio_returns_400(client):
    r = client.post(
        "/analyze", files={"file": ("x.wav", b"not audio at all" * 100, "audio/wav")}
    )
    assert r.status_code == 400
    assert r.json()["error"] == "audio_decode_failed"


def test_request_id_header_present(client):
    wav = make_wav_bytes(make_speech_like(2.0))
    r = client.post("/analyze", files={"file": ("a.wav", wav, "audio/wav")})
    assert "x-request-id" in {k.lower() for k in r.headers.keys()}
