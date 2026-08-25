"""Concurrency behaviour tests.

Covers the two real bugs this suite exists to prevent:
1. The inference semaphore must actually SHED load (429) when saturated -
   not run inference anyway (the original bug: `acquired` was ignored).
2. Parallel requests must not corrupt shared state (VAD regions, engine).
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

from .conftest import FakeEngine, make_speech_like, make_wav_bytes


class SlowEngine(FakeEngine):
    """Simulates model latency so the semaphore actually saturates."""

    def __init__(self, delay: float = 0.6):
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0

    def predict_windows(self, windows):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self.delay)
            return super().predict_windows(windows)
        finally:
            self.in_flight -= 1


@pytest.fixture()
def busy_client():
    settings = Settings(max_concurrent_inference=1, max_upload_bytes=1_000_000)
    app = create_app(settings, engine=SlowEngine(delay=0.6))
    with TestClient(app) as c:
        c.app.state.vad = lambda x, sr: [(0, len(x))]
        yield c


def test_saturated_service_sheds_with_429(busy_client):
    """Second concurrent request must get 429 + Retry-After, not queue."""

    def call():
        wav = make_wav_bytes(make_speech_like(3.0))
        return busy_client.post(
            "/analyze", files={"file": ("a.wav", wav, "audio/wav")}
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1, f2 = [pool.submit(call) for _ in range(2)]
        codes = sorted([f1.result().status_code, f2.result().status_code])

    assert codes == [200, 429], f"expected one 200 and one 429, got {codes}"


def test_429_carries_retry_after_header():
    settings = Settings(max_concurrent_inference=1, max_upload_bytes=1_000_000)
    app = create_app(settings, engine=SlowEngine(delay=0.5))
    with TestClient(app) as client:
        client.app.state.vad = lambda x, sr: [(0, len(x))]
        wav = make_wav_bytes(make_speech_like(3.0))

        def call():
            return client.post("/analyze", files={"file": ("a.wav", wav, "audio/wav")})

        with ThreadPoolExecutor(max_workers=3) as pool:
            results = [f.result() for f in [pool.submit(call) for _ in range(3)]]

        statuses = [r.status_code for r in results]
        assert statuses.count(200) == 1
        assert statuses.count(429) == 2
        for r in results:
            if r.status_code == 429:
                assert r.headers.get("retry-after") == "1"
                assert r.json()["error"] == "server_busy"


def test_parallel_requests_no_corruption():
    """8 parallel requests all succeed with consistent predictions."""
    settings = Settings(max_concurrent_inference=4, max_upload_bytes=1_000_000)
    app = create_app(settings, engine=FakeEngine())
    with TestClient(app) as client:
        client.app.state.vad = lambda x, sr: [(0, len(x))]
        wav = make_wav_bytes(make_speech_like(3.0))

        def call(_):
            r = client.post(
                "/analyze",
                files={"file": ("a.wav", wav, "audio/wav")},
                params={"contact_id": f"c-{_}"},
            )
            assert r.status_code == 200, r.text
            b = r.json()
            assert b["contact_id"] == f"c-{_}"  # no cross-request bleed
            assert b["gender"]["prediction"] == "female"
            assert b["windows_analyzed"] >= 1
            return b

        with ThreadPoolExecutor(max_workers=8) as pool:
            bodies = list(pool.map(call, range(8)))

        assert len({b["contact_id"] for b in bodies}) == 8
