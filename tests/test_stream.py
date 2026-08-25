"""WebSocket streaming tests (hermetic)."""

import numpy as np

from .conftest import SR, make_speech_like


def _pcm_chunk(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def test_stream_ready_partial_final(client):
    x = make_speech_like(6.0)
    frames = []
    with client.websocket_connect("/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_text('{"type": "start", "contact_id": "ws-contact"}')
        assert ws.receive_json()["type"] == "ping"

        # 6s of speech in 0.5s chunks -> >=1 partial expected before stop.
        chunk = SR // 2
        for i in range(12):
            ws.send_bytes(_pcm_chunk(x[i * chunk : (i + 1) * chunk]))
        ws.send_text('{"type": "stop"}')
        # Server flushes queued partials, then the final frame.
        while True:
            msg = ws.receive_json()
            frames.append(msg)
            if msg["type"] == "final":
                break

    partials = [m for m in frames if m["type"] == "partial"]
    assert partials, f"expected at least one partial, got {[m['type'] for m in frames]}"
    final = frames[-1]
    assert final["contact_id"] == "ws-contact"
    for m in [*partials, final]:
        assert m["gender"]["prediction"] in {"male", "female", "unknown"}
        assert 0.0 <= m["gender"]["confidence"] <= 1.0
        assert m["age_bracket"]["prediction"] in {"18-30", "31-45", "46-60", "60+", "unknown"}
    assert partials[0]["sequence"] >= 1


def test_stream_binary_first_implicit_start(client):
    x = make_speech_like(3.0)
    with client.websocket_connect("/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(_pcm_chunk(x))
        ws.send_text('{"type": "stop"}')
        # A partial may flush before the final; drain until we see it.
        while True:
            msg = ws.receive_json()
            if msg["type"] == "final":
                break
    assert msg["audio_quality"] in {"good", "degraded"}


def test_stream_too_short_speech_is_insufficient(client):
    with client.websocket_connect("/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(_pcm_chunk(make_speech_like(0.3)))
        ws.send_text('{"type": "stop"}')
        final = ws.receive_json()
    assert final["type"] == "final"
    assert final["audio_quality"] == "insufficient"
