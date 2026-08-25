"""Shared fixtures: hermetic app with a deterministic fake engine.

The FakeEngine makes integration tests fast, offline, and deterministic -
the real model path is exercised separately by @pytest.mark.live tests.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.inference import WindowPrediction  # noqa: E402
from app.main import create_app  # noqa: E402

SR = 16000


class FakeEngine:
    """Deterministic stand-in with the same surface as AttributeEngine."""

    model_id = "fake-engine"
    ready = True

    def predict_windows(self, windows):
        return [
            WindowPrediction(p_child=0.03, p_female=0.90, p_male=0.07, age_years=27.5)
            for _ in windows
        ]

    def aggregate(self, preds, f0_hz=None):
        from .fake_helpers import aggregate as fake_aggregate

        return fake_aggregate(preds, f0_hz)


@pytest.fixture(scope="session")
def settings():
    return Settings(
        max_upload_bytes=1_000_000,
        lang_id_enabled=False,
        quantize_dynamic=False,
    )


@pytest.fixture()
def app(settings):
    return create_app(settings, engine=FakeEngine())


@pytest.fixture()
def client(app):
    # Context manager ensures the FastAPI lifespan runs (state wiring).
    with TestClient(app) as c:
        # Deterministic VAD: mark everything voiced. Silero rejects synthetic
        # tones as non-speech; hermetic API tests only need stable regions.
        # Real Silero behaviour is exercised by quality unit tests + live tests.
        c.app.state.vad = lambda x, sr: [(0, len(x))]
        yield c


def make_wav_bytes(x: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def encode_audio(x: np.ndarray, sr: int = SR, format_name: str = "wav",
                 codec: str | None = None) -> bytes:
    """Encode float audio into a compressed container - in-process via PyAV.

    Used by codec roundtrip tests; needs no ffmpeg binary or network.
    Raises if the encoder is unavailable (tests skip on that).
    """
    import av

    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format=format_name)
    try:
        stream = container.add_stream(codec or "pcm_s16le", rate=sr)
        stream.layout = "mono"
        frame_size = 1600  # samples per input frame
        for i in range(0, max(pcm.size, 1), frame_size):
            piece = pcm[i : i + frame_size].reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(piece, format="s16", layout="mono")
            frame.sample_rate = sr
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):  # flush
            container.mux(packet)
    finally:
        container.close()
    return buf.getvalue()


def make_speech_like(seconds: float = 4.0, sr: int = SR, f0: float = 180.0) -> np.ndarray:
    """Synthetic speech-ish signal: harmonic tone + syllable envelope + noise.

    Not a real voice - just enough structure for VAD/decode plumbing tests.
    """
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    harmonics = sum(
        np.sin(2 * np.pi * f0 * n * t + 0.3 * n) / n for n in range(1, 6)
    )
    envelope = (np.sin(2 * np.pi * 3.0 * t) > -0.3).astype(np.float32)
    noise = np.random.default_rng(42).normal(0, 0.01, t.size)
    x = (0.15 * harmonics * envelope + noise).astype(np.float32)
    return np.clip(x, -1.0, 1.0)
