#!/usr/bin/env python3
"""End-to-end smoke test against a running service.

Usage:
    python scripts/smoke_test.py --url http://localhost:8000
    python scripts/smoke_test.py --file my_call.wav
    python scripts/smoke_test.py --ws            # also exercise streaming
"""

import argparse
import io
import json
import sys
import time

import numpy as np
import requests

SR = 16000

SPEC_KEYS = {"contact_id", "gender", "age_bracket", "processing_ms", "audio_quality"}


def synth_wav_bytes(seconds: float = 5.0, f0: float = 180.0) -> bytes:
    import soundfile as sf

    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    harmonics = sum(np.sin(2 * np.pi * f0 * n * t) / n for n in range(1, 7))
    env = (np.sin(2 * np.pi * 3.2 * t) > -0.25).astype(np.float32)
    x = 0.15 * harmonics * env + np.random.default_rng(1).normal(0, 0.01, t.size)
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def run_rest(url: str, wav: bytes) -> dict:
    t0 = time.perf_counter()
    r = requests.post(
        f"{url}/analyze",
        files={"file": ("smoke.wav", wav, "audio/wav")},
        params={"contact_id": "smoke-test-0001"},
        timeout=30,
    )
    wall_ms = int((time.perf_counter() - t0) * 1000)
    print(f"HTTP {r.status_code} in {wall_ms}ms (server reports {r.headers.get('X-Process-Time-Ms')}ms)")
    r.raise_for_status()
    body = r.json()
    print(json.dumps(body, indent=2))
    missing = SPEC_KEYS - set(body)
    assert not missing, f"spec fields missing: {missing}"
    return body


def run_ws(url: str, wav: bytes) -> None:
    import asyncio
    import io

    # Decode whatever container we were given into raw PCM16 for streaming.
    import soundfile as sf
    import websockets

    x, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import librosa  # only needed by the client, not the service
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()

    async def go():
        frames = []
        async with websockets.connect(url.replace("http", "ws", 1) + "/stream") as ws:
            frames.append(json.loads(await ws.recv()))
            await ws.send(json.dumps({"type": "start", "contact_id": "ws-smoke"}))
            frames.append(json.loads(await ws.recv()))  # ping ack
            chunk = SR // 2 * 2  # 0.5s of pcm16 in bytes
            for i in range(0, len(pcm), chunk):
                await ws.send(pcm[i : i + chunk])
                try:
                    frames.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=2)))
                except asyncio.TimeoutError:
                    pass
            await ws.send(json.dumps({"type": "stop"}))
            while True:
                frames.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=5)))
                if frames[-1]["type"] == "final":
                    break
        print("WS frames:")
        for f in frames:
            keep = {k: f.get(k) for k in ("type", "sequence", "speech_seconds",
                                          "audio_quality", "gender", "age_bracket")}
            print(" ", json.dumps(keep)[:170])
        assert frames[-1]["type"] == "final"
        assert any(f["type"] == "partial" for f in frames), "expected progressive partials"

    asyncio.run(go())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--file", help="audio file to send (default: synthetic)")
    ap.add_argument("--ws", action="store_true", help="also test WebSocket streaming")
    args = ap.parse_args()

    health = requests.get(f"{args.url}/healthz", timeout=10).json()
    print("healthz:", json.dumps(health))
    if not health.get("model_loaded"):
        print("model still loading; aborting")
        return 2

    wav = open(args.file, "rb").read() if args.file else synth_wav_bytes()
    run_rest(args.url, wav)
    if args.ws:
        run_ws(args.url, wav)
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
