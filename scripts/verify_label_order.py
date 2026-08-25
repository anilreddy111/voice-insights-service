#!/usr/bin/env python3
"""Resolve the gender-head column order empirically.

The two public audEERING model cards contradict each other:
  -6  card prints columns as: child, female, male
  -24 card prints columns as: female, male, child
and config.json's id2label says female/male/child.

This script loads the model and scores short clips from LibriSpeech speakers
with *documented* gender (SPEAKERS.TXT) and reports which column fires for
which group. Run once after downloading weights; set VIS_GENDER_LABEL_ORDER
accordingly if it differs from the configured default.

Usage:
    python scripts/verify_label_order.py --librispeech-dir <dev-clean root>
        (expects <dir>/LibriSpeech/dev-clean and its SPEAKERS.TXT)

If no corpus is available it falls back to synthetic pitch probes, which are
indicative only.
"""

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.inference import AttributeEngine  # noqa: E402


def load_librispeech_clips(root: Path, per_speaker: int = 3, max_speakers: int = 12):
    """Return [(gender, np.ndarray)] from a LibriSpeech dev-clean tree."""
    import soundfile as sf

    speakers_file = None
    for cand in [root / "SPEAKERS.TXT", root.parent / "SPEAKERS.TXT",
                 root / "LibriSpeech" / "SPEAKERS.TXT"]:
        if cand.exists():
            speakers_file = cand
            break
    if speakers_file is None:
        return []
    gender_by_id = {}
    for line in speakers_file.read_text(errors="ignore").splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) > 4 and parts[0].isdigit() and parts[1] in {"M", "F"}:
            gender_by_id[parts[0]] = "female" if parts[1] == "F" else "male"

    clips = []
    audio_root = speakers_file.parent / next(
        d.name for d in speakers_file.parent.iterdir() if d.name.startswith("LibriSpeech")
    ) if any(d.name.startswith("LibriSpeech") for d in speakers_file.parent.iterdir()) else speakers_file.parent
    flacs = sorted(audio_root.rglob("*.flac"))
    by_speaker = collections.defaultdict(list)
    for f in flacs:
        spk = f.name.split("-")[0]
        if spk in gender_by_id and len(by_speaker[spk]) < per_speaker:
            by_speaker[spk].append(f)
    for i, (spk, files) in enumerate(sorted(by_speaker.items())):
        if i >= max_speakers:
            break
        for f in files[:per_speaker]:
            x, sr = sf.read(f, dtype="float32")
            clips.append((gender_by_id[spk], x, sr))
    return clips


def probe(engine: AttributeEngine, label_order: list[str]):
    sr = 16000
    t = np.linspace(0, 2.5, int(sr * 2.5), endpoint=False)
    probes = {
        "~110Hz tone (male-like F0)": np.sin(2 * np.pi * 110 * t).astype(np.float32) * 0.5,
        "~220Hz tone (female-like F0)": np.sin(2 * np.pi * 220 * t).astype(np.float32) * 0.5,
    }
    print(f"\ncolumns assumed = {label_order}\n")
    for name, x in probes.items():
        preds = engine.predict_windows([x])
        p = preds[0]
        cols = {label_order[i]: round(v, 3) for i, v in enumerate([p.p_child, p.p_female, p.p_male])}
        print(f"{name}: {cols}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--librispeech-dir", help="path containing SPEAKERS.TXT + dev-clean/")
    ap.add_argument("--model-id", default=None)
    args = ap.parse_args()

    settings = Settings(model_id=args.model_id) if args.model_id else Settings()
    engine = AttributeEngine(settings)
    engine.load()
    order = engine._labels

    clips = load_librispeech_clips(Path(args.librispeech_dir)) if args.librispeech_dir else []
    if not clips:
        print("No LibriSpeech clips found - running indicative synthetic probes only.")
        probe(engine, order)
        print("\nFor a definitive answer download LibriSpeech dev-clean (openslr.org/resources/12)")
        return 1

    wins = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for gender, x, _sr in clips:
        preds = engine.predict_windows([x])
        for p in preds:
            acc = wins[gender]
            acc[0] += p.p_child
            acc[1] += p.p_female
            acc[2] += p.p_male
            acc[3] += 1

    print(f"\nMean head outputs by documented speaker gender (columns={order}):\n")
    col_idx = {name: order.index(name) for name in ("child", "female", "male")}
    for gender in ("female", "male"):
        total, n = wins[gender][:3], wins[gender][3]
        means = {k: round(v / max(n, 1), 3) for k, v in zip(("child", "female", "male"), total, strict=False)}
        best = max(means, key=means.get)
        print(f"speakers={gender:>6}: {means} -> argmax column = '{best}' "
              f"(config maps this to index {col_idx[best]})")
    print("\nIf 'argmax column' matches the documented gender for both groups,")
    print(f"VIS_GENDER_LABEL_ORDER={','.join(order)} is correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
