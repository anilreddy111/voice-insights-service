#!/usr/bin/env python3
"""Reproducible latency benchmark for the full inference stack.

Builds 2s/5s/10s inputs from real speech (bundled LibriSpeech clips by
default), pushes them through the complete app (decode -> quality -> VAD ->
inference -> aggregate) and reports per-stage + end-to-end percentiles.

Usage:
    python scripts/benchmark.py                     # in-process, bundled clips
    python scripts/benchmark.py --repeats 30        # tighter percentiles
    python scripts/benchmark.py --url http://...    # against a running server
"""

import argparse
import io
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIPS = [
    REPO_ROOT / "samples" / "librispeech_female_1462.wav",
    REPO_ROOT / "samples" / "librispeech_male_3000.wav",
]
DURATIONS = [2.0, 5.0, 10.0]


def build_inputs(durations, clip_paths):
    """Tile real speech into fixed-duration wav payloads."""
    xs = []
    sr = None
    for p in clip_paths:
        x, sr = sf.read(p, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        xs.append(x)

    inputs = {}
    for d in durations:
        need = int(d * sr)
        parts = []
        while sum(p.size for p in parts) < need:
            parts.append(xs[len(parts) % len(xs)])
        y = np.concatenate(parts)[:need].astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
        inputs[d] = buf.getvalue()
    return inputs


def pct(values, q):
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q / 100 * (len(s) - 1)))))
    return s[idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--url", help="benchmark a running server instead of in-process")
    ap.add_argument("--out", help="also write results as JSON to this path")
    args = ap.parse_args()

    clips = [p for p in DEFAULT_CLIPS if p.exists()]
    if not clips:
        print("no sample clips found; run scripts/make_sample.py or restore samples/",
              file=sys.stderr)
        return 2
    inputs = build_inputs(DURATIONS, clips)

    if args.url:
        import requests

        session = requests.Session()

        def post(wav):
            t0 = time.perf_counter()
            r = session.post(f"{args.url}/analyze",
                             files={"file": ("b.wav", wav, "audio/wav")}, timeout=60)
            body = r.json()
            wall = (time.perf_counter() - t0) * 1000
            return wall, float(body["processing_ms"]), body.get("stages_ms", {})
    else:
        from fastapi.testclient import TestClient

        from app.config import Settings
        from app.inference import AttributeEngine
        from app.main import create_app

        eng = AttributeEngine(Settings())
        eng.load()
        app = create_app(Settings(), engine=eng)
        # Context manager is required: it runs the lifespan that wires
        # app.state (settings, VAD, semaphore).
        client_cm = TestClient(app)
        client_cm.__enter__()

        def post(wav):
            t0 = time.perf_counter()
            r = client_cm.post("/analyze", files={"file": ("b.wav", wav, "audio/wav")})
            assert r.status_code == 200, r.text
            body = r.json()
            wall = (time.perf_counter() - t0) * 1000
            return wall, float(body["processing_ms"]), body.get("stages_ms", {})

    results = {}
    print(f"\nbenchmark: {args.repeats} runs per duration "
          f"({'HTTP ' + args.url if args.url else 'in-process'})\n")
    print("| input | p50 | p95 | min | max | decode | quality | vad_trim | inference |")
    print("|---|---|---|---|---|---|---|---|---|")

    for d, wav in inputs.items():
        post(wav)  # warmup
        walls, servers, stages_acc = [], [], {}
        for _ in range(args.repeats):
            wall, server, stages = post(wav)
            walls.append(wall)
            servers.append(server)
            for k, v in stages.items():
                stages_acc.setdefault(k, []).append(v)

        label = f"{int(d)}s clip"
        row = [label,
               f"{pct(servers, 50):.0f}ms", f"{pct(servers, 95):.0f}ms",
               f"{min(servers):.0f}ms", f"{max(servers):.0f}ms"]
        for stage in ("decode", "quality", "vad_trim", "inference"):
            vals = stages_acc.get(stage)
            row.append(f"{statistics.median(vals):.0f}ms" if vals else "-")
        results[label] = {
            "server_p50": pct(servers, 50), "server_p95": pct(servers, 95),
            "server_min": min(servers), "server_max": max(servers),
            "wall_p95": pct(walls, 95),
            "stage_median_ms": {k: round(statistics.median(v), 1)
                                for k, v in stages_acc.items()},
        }
        print("| " + " | ".join(row) + " |")

    if args.out:
        Path(args.out).parent.mkdir(exist_ok=True, parents=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwritten {args.out}")
    print("\n(server-side processing_ms; add ~1-3ms HTTP overhead)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
