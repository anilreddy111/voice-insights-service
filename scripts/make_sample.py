#!/usr/bin/env python3
"""Generate synthetic sample audio for plumbing/smoke tests.

IMPORTANT: these are NOT real voices. They exercise decode/VAD/API paths.
For meaningful accuracy checks use real data - see scripts/evaluate_cv.py
(Mozilla Common Voice) or drop any WAV/MP3 into --file of smoke_test.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def synth(f0: float, seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    harmonics = sum(np.sin(2 * np.pi * f0 * n * t + 0.3 * n) / n for n in range(1, 8))
    vib = 1 + 0.02 * np.sin(2 * np.pi * 5 * t)
    syllables = (np.sin(2 * np.pi * 3.2 * t) > -0.25).astype(np.float32)
    breath = rng.normal(0, 0.008, t.size)
    x = 0.18 * harmonics * vib * syllables + breath
    return (x / max(1e-9, np.max(np.abs(x)))).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="samples", help="output directory")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, f0 in [("female_like_220hz.wav", 220.0), ("male_like_120hz.wav", 120.0)]:
        path = out / name
        sf.write(path, synth(f0, 5.0, seed=int(f0)), SR, subtype="PCM_16")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
