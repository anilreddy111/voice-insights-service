"""Audio quality assessment + speech region detection.

Logistics calls happen next to forklifts and highways, so before trusting any
prediction we measure *how* degraded the signal is:

- speech_ratio : fraction of audio that Silero-VAD marks as speech
- snr_db       : proxy SNR from frame-energy percentiles
- clip_ratio   : fraction of samples at full scale (driver/mic distortion)
- rms_dbfs     : overall level (near-silence / muted mic)

These map to good | degraded | insufficient. On "insufficient" we skip model
inference entirely and return "unknown" - surfacing bad input honestly beats
silently returning confident nonsense.
"""

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from .obs import stage

logger = logging.getLogger(__name__)

VAD = None  # lazily loaded Silero model (or None -> energy fallback)


def load_vad():
    """Load Silero-VAD once; fall back to an energy gate if unavailable.

    Returns a callable ``regions(x, sr) -> [(start, end)]`` in samples.
    """
    global VAD
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad

        if VAD is None:
            VAD = load_silero_vad()
        return _make_silero_regions(VAD, get_speech_timestamps)
    except Exception as exc:  # pragma: no cover - depends on env
        logger.warning("silero-vad unavailable (%s); using energy fallback", exc)
        return energy_regions


def _make_silero_regions(model, get_speech_timestamps):
    # Silero's get_speech_timestamps() resets the model's internal state on
    # every call, so concurrent calls from worker threads would clobber each
    # other's hidden state -> corrupted speech regions under load. VAD is
    # ~20ms, so serializing it costs far less than re-running inference.
    lock = threading.Lock()

    def regions(x: np.ndarray, sr: int) -> list[tuple[int, int]]:
        with stage("vad"), lock:
            ts = get_speech_timestamps(
                torch_from(x),
                model,
                sampling_rate=sr,
                min_speech_duration_ms=200,
                min_silence_duration_ms=120,
                speech_pad_ms=30,
            )
        return [(t["start"], t["end"]) for t in ts]

    return regions


def torch_from(x: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(x))


def energy_regions(x: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """VAD fallback: frames >8dB above the 15th-percentile noise floor."""
    frame, hop = int(0.03 * sr), int(0.01 * sr)
    if x.size < frame:
        return []
    n = 1 + max(0, (x.size - frame)) // hop
    rms = np.array(
        [np.sqrt(np.mean(x[i * hop : i * hop + frame] ** 2) + 1e-12) for i in range(n)]
    )
    db = 20 * np.log10(rms + 1e-10)
    floor = np.percentile(db, 15)
    voiced = db > floor + 8.0
    regions: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            regions.append((start * hop, min(i * hop + frame, x.size)))
            start = None
    if start is not None:
        regions.append((start * hop, x.size))
    # merge tiny gaps
    merged: list[tuple[int, int]] = []
    gap = int(0.12 * sr)
    for s, e in regions:
        if merged and s - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s > int(0.1 * sr)]


@dataclass
class QualityReport:
    flag: str  # good | degraded | insufficient
    reasons: list[str] = field(default_factory=list)
    speech_ratio: float = 0.0
    snr_db: float = 0.0
    clip_ratio: float = 0.0
    rms_dbfs: float = -120.0
    regions: list[tuple[int, int]] = field(default_factory=list)  # reuse: don't re-run VAD


def assess(x: np.ndarray, sr: int, regions_fn=None) -> QualityReport:
    regions_fn = regions_fn or load_vad()
    reasons: list[str] = []

    with stage("quality"):
        dur = x.size / sr
        rms_dbfs = float(20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-10))
        clip_ratio = float(np.mean(np.abs(x) >= 0.985))

        regions = regions_fn(x, sr)
        speech_samples = sum(e - s for s, e in regions)
        speech_ratio = speech_samples / max(x.size, 1)

        # Proxy SNR: mean energy of the loudest 30% of frames vs quietest 20%.
        frame, hop = int(0.03 * sr), int(0.01 * sr)
        snr_db = 0.0
        if x.size >= frame:
            n = 1 + (x.size - frame) // hop
            energies = np.array(
                [
                    np.mean(x[i * hop : i * hop + frame] ** 2) + 1e-12
                    for i in range(n)
                ]
            )
            hi = np.sort(energies)[-max(1, int(0.30 * n)) :]
            lo = np.sort(energies)[: max(1, int(0.20 * n))]
            snr_db = float(10 * np.log10(hi.mean() / lo.mean()))

        flag = "good"
        if dur < 1.0:
            reasons.append("too_short")
            flag = "insufficient"
        if rms_dbfs < -45:
            reasons.append("low_level")
            flag = "insufficient"
        if speech_ratio < 0.06:
            reasons.append("no_speech")
            flag = "insufficient"
        elif speech_ratio < 0.18:
            reasons.append("low_speech_ratio")
            flag = "degraded"
        if snr_db < 3.0:
            reasons.append("very_noisy")
            flag = "insufficient"
        elif snr_db < 8.0:
            reasons.append("noisy")
            flag = "degraded" if flag == "good" else flag
        if clip_ratio > 0.05:
            reasons.append("clipping")
            flag = "degraded" if flag == "good" else flag

        return QualityReport(
            flag=flag,
            reasons=reasons,
            speech_ratio=round(speech_ratio, 4),
            snr_db=round(snr_db, 2),
            clip_ratio=round(clip_ratio, 5),
            rms_dbfs=round(rms_dbfs, 2),
            regions=regions,
        )


def voiced_concat(x: np.ndarray, regions: list[tuple[int, int]], cap_seconds: float, sr: int) -> np.ndarray:
    """Concatenate speech regions (dropping silence), capped for latency."""
    cap = int(cap_seconds * sr)
    parts: list[np.ndarray] = []
    total = 0
    for s, e in regions:
        part = x[s:e]
        if total + part.size > cap:
            part = part[: cap - total]
        parts.append(part)
        total += part.size
        if total >= cap:
            break
    return np.concatenate(parts) if parts else x[:cap]
