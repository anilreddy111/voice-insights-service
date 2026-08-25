"""Unit tests for the quality gate and aggregation math (no model, no VAD)."""

import numpy as np

from app.inference import BRACKETS, _bracket_mass, build_windows, estimate_f0
from app.quality import assess

from .conftest import SR, make_speech_like


def _fake_regions(ratio: float):
    def regions(x, sr):
        n = int(x.size * ratio)
        return [(0, n)] if n else []
    return regions


def test_silence_is_insufficient():
    x = np.zeros(SR * 3, dtype=np.float32)
    report = assess(x, SR, regions_fn=_fake_regions(0.0))
    assert report.flag == "insufficient"
    assert "low_level" in report.reasons or "no_speech" in report.reasons


def test_clean_speech_is_good():
    x = make_speech_like(4.0)
    report = assess(x, SR, regions_fn=_fake_regions(0.9))
    assert report.flag == "good", (report.flag, report.reasons)


def test_low_speech_ratio_degrades():
    x = make_speech_like(4.0)
    report = assess(x, SR, regions_fn=_fake_regions(0.10))
    assert report.flag == "degraded"
    assert "low_speech_ratio" in report.reasons


def test_clipping_flags_degraded():
    x = np.clip(make_speech_like(4.0) * 10.0, -1, 1)
    report = assess(x, SR, regions_fn=_fake_regions(0.9))
    assert "clipping" in report.reasons
    assert report.clip_ratio > 0.05


def test_build_windows_equal_length():
    x = make_speech_like(8.0)
    windows = build_windows(x, SR, window_s=3.0, hop_s=1.5, min_speech_s=0.8)
    assert len(windows) >= 4
    assert all(w.size == int(3.0 * SR) for w in windows)


def test_build_windows_short_clip_single_padded_window():
    x = make_speech_like(1.2)  # < one window
    windows = build_windows(x, SR, 3.0, 1.5, min_speech_s=0.8)
    assert len(windows) == 1
    assert windows[0].size == int(3.0 * SR)


def test_bracket_mass_behaviour():
    # Mass concentrates near the age and falls off with distance.
    near = _bracket_mass(25, 6.0, 18.0, 30.5)
    far = _bracket_mass(45, 6.0, 18.0, 30.5)
    assert near > 0.5
    assert far < 0.05
    # Away from range edges the bracket masses nearly sum to 1.
    total = sum(_bracket_mass(40, 6.0, lo, hi) for _, lo, hi in BRACKETS)
    assert 0.98 <= total <= 1.02


def test_estimate_f0_recovers_tone():
    t = np.linspace(0, 3, SR * 3, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    f0 = estimate_f0(tone, SR)
    assert f0 is not None and abs(f0 - 200) < 15
