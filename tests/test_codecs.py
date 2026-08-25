"""Codec roundtrip tests for the decode chain (soundfile -> PyAV).

Logistics calls arrive as mp3, opus-in-webm, alaw telephony, etc. These tests
encode synthetic speech into each container in-process (PyAV) and push it
through the exact production decode path, so the matrix runs anywhere -
CI included - with no ffmpeg binary and no network.

The mp3 case skips gracefully if libmp3lame isn't in the installed PyAV build.
"""

import numpy as np
import pytest

from app.audio_io import AudioDecodeError, decode_to_pcm

from .conftest import SR, encode_audio, make_speech_like


def _encode(name: str) -> bytes:
    """Encode a 3s fixture for a named case; skip cleanly if encoder missing."""
    fmt, src_sr = CASES[name]
    x = make_speech_like(3.0, sr=src_sr)  # generate at the TRUE source rate
    try:
        return encode_audio(x, sr=src_sr, **fmt)
    except Exception as exc:
        pytest.skip(f"encoder unavailable: {exc}")


CASES = {
    "wav-pcm16": (dict(format_name="wav"), SR),
    "flac": (dict(format_name="flac", codec="flac"), SR),
    "ogg-vorbis": (dict(format_name="ogg", codec="vorbis"), SR),
    "webm-opus": (dict(format_name="matroska", codec="opus"), SR),
    "telephony-alaw-8k": (dict(format_name="wav", codec="pcm_alaw"), 8000),
}


@pytest.mark.parametrize("name", CASES.keys())
def test_decode_roundtrip(name):
    _, src_sr = CASES[name]
    x = decode_to_pcm(_encode(name), target_sr=SR)

    assert x.dtype == np.float32
    expected = 3.0 * SR  # 3 seconds at 16 kHz after resampling
    assert 0.9 * expected <= x.size <= 1.1 * expected, (
        f"{name}: decoded {x.size} samples, expected ~{expected}"
    )
    rms = float(np.sqrt(np.mean(x**2)))
    assert rms > 1e-4, f"{name}: decoded to near-silence"


def test_decode_roundtrip_mp3_optional():
    try:
        payload = encode_audio(make_speech_like(3.0, sr=SR), sr=SR,
                               format_name="mp3", codec="libmp3lame")
    except Exception as exc:
        pytest.skip(f"libmp3lame unavailable in this PyAV build: {exc}")
    x = decode_to_pcm(payload, target_sr=SR)
    assert abs(x.size - 3.0 * SR) < 0.1 * 3.0 * SR
    assert float(np.sqrt(np.mean(x**2))) > 1e-4


def test_corrupt_bytes_raise_decode_error():
    with pytest.raises(AudioDecodeError):
        decode_to_pcm(b"this is definitely not audio" * 512, target_sr=SR)


def test_truncated_container_raises_not_hangs():
    good = encode_audio(make_speech_like(2.0), sr=SR, format_name="flac", codec="flac")
    with pytest.raises(AudioDecodeError):
        decode_to_pcm(good[: len(good) // 3], target_sr=SR)


def test_compressed_codec_through_full_api(client):
    """End-to-end: an opus-in-webm upload through POST /analyze.

    This mirrors what browser/Twilio-style call platforms actually send.
    """
    payload = _encode("webm-opus")
    r = client.post(
        "/analyze",
        files={"file": ("call.webm", payload, "audio/webm")},
        params={"contact_id": "codec-e2e"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["contact_id"] == "codec-e2e"
    assert body["audio_quality"] != "insufficient"
