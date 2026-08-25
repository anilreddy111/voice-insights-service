"""Live tests against real model weights.

Opt-in (they download ~500MB on first run):
    VIS_LIVE_TESTS=1 pytest -m live
"""

import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.inference import AttributeEngine
from app.main import create_app

pytestmark = pytest.mark.live

requires_live = pytest.mark.skipif(
    not os.environ.get("VIS_LIVE_TESTS"), reason="VIS_LIVE_TESTS not set"
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Committed CC-BY LibriSpeech clips with documented speaker gender:
# these pin the gender head's column order AND sanity-check the full stack.
GENDER_SAMPLES = [
    ("samples/librispeech_female_1462.wav", "female"),
    ("samples/librispeech_male_3000.wav", "male"),
]


@pytest.fixture(scope="module")
def engine():
    s = Settings()
    eng = AttributeEngine(s)
    eng.load()
    return eng


@pytest.fixture(scope="module")
def live_client(engine):
    app = create_app(Settings(), engine=engine)
    with TestClient(app) as c:
        # Real Silero VAD - no overrides in live mode.
        yield c


@requires_live
def test_real_model_loads_and_predicts(engine):
    sr = 16000
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    x = np.sin(2 * np.pi * 150 * t).astype(np.float32)
    preds = engine.predict_windows([x])
    assert len(preds) == 1
    p = preds[0]
    assert abs(p.p_child + p.p_female + p.p_male - 1.0) < 1e-5
    assert 0 <= p.age_years <= 100


@requires_live
def test_rest_endpoint_with_real_model(live_client):
    from tests.conftest import make_speech_like, make_wav_bytes

    wav = make_wav_bytes(make_speech_like(4.0, f0=190.0))
    r = live_client.post("/analyze", files={"file": ("s.wav", wav, "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["processing_ms"] > 0
    assert set(body.keys()) >= {
        "contact_id", "gender", "age_bracket", "processing_ms", "audio_quality",
    }


@requires_live
def test_known_gender_speakers_classified_correctly(live_client):
    """The committed LibriSpeech clips must land on the right side.

    This is the executable version of scripts/verify_label_order.py: if the
    checkpoint's column order ever changed upstream, this fails loudly.
    """
    for rel_path, expected in GENDER_SAMPLES:
        path = REPO_ROOT / rel_path
        r = live_client.post(
            "/analyze",
            files={"file": (path.name, path.read_bytes(), "audio/wav")},
            params={"contact_id": f"live-{path.stem}"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["audio_quality"] != "insufficient", (
            f"{rel_path}: unexpectedly flagged insufficient ({body['audio_quality_reasons']})"
        )
        assert body["gender"]["prediction"] == expected, (
            f"{rel_path}: expected {expected}, got {body['gender']} "
            "(label order or model drift?)"
        )
        assert body["gender"]["confidence"] >= 0.6
        assert body["age_bracket"]["prediction"] in {"18-30", "31-45", "46-60", "60+"}
        assert 0.0 < body["processing_ms"] < 10_000  # generous CI bound
