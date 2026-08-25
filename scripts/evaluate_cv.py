#!/usr/bin/env python3
"""Eval harness: run the production pipeline against a public dataset.

Supports two datasets:

1. Mozilla Common Voice (age + gender labels):
      Download from https://commonvoice.mozilla.org/en/datasets (free account),
      extract, then:
      python scripts/evaluate_cv.py --cv-root /path/cv-corpus-XX.0/en --limit 300

2. LibriSpeech dev-clean (gender labels via SPEAKERS.TXT, zero setup):
      python scripts/evaluate_cv.py --librispeech-dir /path/LibriSpeech/dev-clean

Measures accuracy AND confidence calibration (ECE, reliability curve,
Brier-style summaries) - because a voice agent that says "female, 0.55" must
be honest about its uncertainty.

Notes on labels:
    * gender task: CV 'male'/'female' rows are used; unlabelled rows skipped.
    * age task (CV only): decade buckets map leniently onto API brackets:
      twenties->18-30, thirties+fourties->31-45, fifties->46-60, sixties+->60+.
      Boundary bleed is inherent to CV's decade labels.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio_io import decode_to_pcm  # noqa: E402
from app.config import Settings  # noqa: E402
from app.inference import AttributeEngine, build_windows  # noqa: E402
from app.quality import load_vad, voiced_concat  # noqa: E402

AGE_MAP = {
    "twenties": "18-30",
    "thirties": "31-45",
    "fourties": "31-45",
    "fifties": "46-60",
    "sixties": "60+",
    "seventies": "60+",
    "eighties": "60+",
    "nineties": "60+",
}


def load_rows(cv_root: Path, limit: int):
    tsv = cv_root / "validated.tsv"
    rows = []
    with open(tsv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            g = (row.get("gender") or "").strip()
            a = (row.get("age") or "").strip()
            if g not in {"male", "female"} and a not in AGE_MAP:
                continue
            path = cv_root / "clips" / row["path"]
            if path.exists():
                rows.append({"path": path, "gender": g if g in {"male", "female"} else None,
                             "age": AGE_MAP.get(a)})
            if len(rows) >= limit:
                break
    return rows


def load_librispeech_rows(root: Path, limit: int):
    """Gender-labelled rows from a LibriSpeech tree (SPEAKERS.TXT)."""
    speakers_file = root / "SPEAKERS.TXT"
    if not speakers_file.exists():  # allow pointing at the parent
        alt = root.parent / "SPEAKERS.TXT"
        if alt.exists():
            speakers_file = alt
        else:
            raise SystemExit(f"SPEAKERS.TXT not found under {root}")
    gender_by_id = {}
    for line in speakers_file.read_text(errors="ignore").splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) > 4 and parts[0].isdigit() and parts[1] in {"M", "F"}:
            gender_by_id[parts[0]] = "female" if parts[1] == "F" else "male"

    rows = []
    per_speaker = 3  # cap clips/speaker to avoid speaker domination
    count_by_spk: dict[str, int] = {}
    for flac in sorted(root.rglob("*.flac")):
        spk = flac.name.split("-")[0]
        g = gender_by_id.get(spk)
        if not g:
            continue
        if count_by_spk.get(spk, 0) >= per_speaker:
            continue
        count_by_spk[spk] = count_by_spk.get(spk, 0) + 1
        rows.append({"path": flac, "gender": g, "age": None})
        if len(rows) >= limit:
            break
    return rows


def predict(engine, vad, settings, x, sr):
    regions = vad(x, sr)
    voiced = voiced_concat(x, regions, settings.max_speech_seconds, sr)
    windows = build_windows(voiced, sr, settings.window_seconds, settings.hop_seconds,
                            settings.min_speech_seconds)
    if not windows:
        return None
    preds = engine.predict_windows(windows)
    return engine.aggregate(preds)


def ece(confidences: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    err = 0.0
    for i in range(bins):
        m = (confidences > edges[i]) & (confidences <= edges[i + 1])
        if m.sum() == 0:
            continue
        err += m.mean() * abs(correct[m].mean() - confidences[m].mean())
    return float(err)


def brier_multiclass(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def summarize(task: str, y_true: list[str], y_pred: list[str], confs: list[float]):
    assert len(y_true) == len(y_pred) == len(confs)
    n = len(y_true)
    if n == 0:
        print(f"[{task}] no usable samples")
        return
    classes = sorted(set(y_true) | set(y_pred))
    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred, strict=False)]))
    f1s = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred, strict=False))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred, strict=False))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred, strict=False))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    conf = np.array(confs)
    correct = np.array([t == p for t, p in zip(y_true, y_pred, strict=False)], dtype=float)
    print(f"\n[{task}] n={n}")
    print(f"  accuracy      : {acc:.3f}")
    print(f"  macro F1      : {float(np.mean(f1s)):.3f}  per-class: "
          + ", ".join(f"{c}={f:.3f}" for c, f in zip(classes, f1s, strict=False)))
    print(f"  mean confidence: {conf.mean():.3f}   ECE(10 bins): {ece(conf, correct):.3f}")
    print("  calibration curve (bin: avg_conf -> empirical_acc):")
    for lo in np.arange(0.5, 1.0, 0.05):
        m = (conf >= lo) & (conf < lo + 0.05)
        if m.sum():
            print(f"    {lo:.2f}-{lo+0.05:.2f}: {conf[m].mean():.3f} -> {correct[m].mean():.3f} (n={m.sum()})")
    cm = defaultdict(int)
    for t, p in zip(y_true, y_pred, strict=False):
        cm[(t, p)] += 1
    print("  confusion:", dict(sorted(cm.items())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cv-root", help="Common Voice dir containing validated.tsv + clips/")
    ap.add_argument("--librispeech-dir", help="LibriSpeech dir (e.g. dev-clean) with SPEAKERS.TXT")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", default="eval_out")
    args = ap.parse_args()
    if not args.cv_root and not args.librispeech_dir:
        ap.error("provide --cv-root or --librispeech-dir")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    settings = Settings()
    engine = AttributeEngine(settings)
    engine.load()
    vad = load_vad()

    rows = (
        load_rows(Path(args.cv_root), args.limit) if args.cv_root
        else load_librispeech_rows(Path(args.librispeech_dir), args.limit)
    )
    print(f"loaded {len(rows)} labelled clips")

    results = []
    for i, row in enumerate(rows):
        try:
            x = decode_to_pcm(row["path"].read_bytes(), settings.sample_rate)
        except Exception as exc:
            print(f"skip {row['path'].name}: {exc}", file=sys.stderr)
            continue
        agg = predict(engine, vad, settings, x, settings.sample_rate)
        if not agg or agg["n_windows"] == 0:
            continue
        results.append({
            "file": row["path"].name,
            "true_gender": row["gender"], "pred_gender": agg["gender_pred"],
            "gender_conf": agg["gender_conf"],
            "true_age": row["age"], "pred_age": agg["bracket_pred"],
            "age_conf": agg["bracket_conf"],
            "age_years": agg["age_median"],
        })
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(rows)}")

    (out_dir / "predictions.json").write_text(json.dumps(results, indent=2))

    g = [r for r in results if r["true_gender"]]
    a = [r for r in results if r["true_age"] and r["pred_age"] != "unknown"]
    summarize(
        "gender",
        [r["true_gender"] for r in g], [r["pred_gender"] for r in g],
        [r["gender_conf"] for r in g],
    )
    summarize(
        "age_bracket",
        [r["true_age"] for r in a], [r["pred_age"] for r in a],
        [r["age_conf"] for r in a],
    )
    ages_t = [r["age_years"] for r in results]
    if ages_t:
        print(f"\nage regression head: mean={np.mean(ages_t):.1f}y std={np.std(ages_t):.1f}y")
    print(f"\nper-file predictions written to {out_dir/'predictions.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
