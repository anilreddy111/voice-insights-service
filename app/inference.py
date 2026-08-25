"""Attribute inference engine: gender + age from raw speech.

Model
-----
`audeering/wav2vec2-large-robust-6-ft-age-gender` (public weights, CC BY-NC-SA).
A wav2vec2-large-robust backbone (first 6 transformer layers only) with two
heads trained jointly on aGender, Common Voice, TIMIT and VoxCeleb2:

- age head   : regression in ~[0, 1] == [0, 100] years
- gender head: 3-way softmax (child / female / male)

Why this model:
  * single forward pass yields BOTH attributes -> one latency budget;
  * 6-layer variant is ~3x faster than the 24-layer one, close in accuracy,
    which is what makes <500ms p95 on CPU realistic for real-time calls;
  * robust variant of the backbone was pre-trained on noisy/noisy-decoded
    speech, which suits warehouse/road-noise conditions.

Robustness layer (ours, not the model's):
  * VAD-trim -> sliding windows -> batched forward -> median aggregate.
    A single pass over a whole clip is brittle to music/ringing/beeps; the
    windowed median suppresses outlier windows.
  * Age uncertainty comes from across-window dispersion (MAD), and bracket
    confidence is the probability mass of the best bracket under that spread.
  * A pitch-based sanity check breaks near-ties between male/female.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from .config import Settings
from .obs import MODEL_INFO, stage

logger = logging.getLogger(__name__)

AGE_YEARS_MAX = 100.0  # card: age output ~[0,1] maps to [0,100] years


# --------------------------------------------------------------------------
# Model definition - must mirror the checkpoint's module names exactly
# (wav2vec2.*, age.*, gender.*) so from_pretrained maps weights correctly.
# --------------------------------------------------------------------------
class ModelHead(nn.Module):
    def __init__(self, config, num_labels: int):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):  # batched, equal-length inputs
        hidden = self.wav2vec2(input_values)[0]
        pooled = torch.mean(hidden, dim=1)
        age = self.age(pooled)  # (B,1) ~[0,1] == [0,100]y
        gender = torch.softmax(self.gender(pooled), dim=1)  # (B,3)
        return age, gender


@dataclass
class WindowPrediction:
    p_child: float
    p_female: float
    p_male: float
    age_years: float


BRACKETS: list[tuple[str, float | None, float | None]] = [
    ("18-30", 18.0, 30.5),
    ("31-45", 30.5, 45.5),
    ("46-60", 45.5, 60.5),
    ("60+", 60.5, None),
]


def _bracket_mass(age: float, sigma: float, lo: float | None, hi: float | None) -> float:
    def cdf(v):
        return 0.5 * (1 + math.erf((v - age) / (sigma * math.sqrt(2))))

    upper = cdf(hi) if hi is not None else 1.0
    lower = cdf(lo) if lo is not None else 0.0
    return max(0.0, upper - lower)


def build_windows(
    voiced: np.ndarray, sr: int, window_s: float, hop_s: float, min_speech_s: float
) -> list[np.ndarray]:
    """Slice voiced audio into equal-length windows (batch-friendly, no padding)."""
    win = int(window_s * sr)
    hop = int(hop_s * sr)
    if voiced.size < int(min_speech_s * sr):
        return []
    if voiced.size < win:
        pad = np.zeros(win - voiced.size, dtype=np.float32)  # short clip: pad tail
        return [np.concatenate([voiced, pad])]
    spans = range(0, voiced.size - win + 1, hop)
    windows = [voiced[i : i + win] for i in spans]
    if not windows:
        windows = [voiced[-win:]]
    return windows


def estimate_f0(x: np.ndarray, sr: int) -> float | None:
    """Median F0 over the clip via autocorrelation - used as a tie-breaker."""
    try:
        import torchaudio

        t = torch.from_numpy(np.ascontiguousarray(x[: sr * 8]))[None]
        f0 = torchaudio.functional.detect_pitch_frequency(
            t, sr, freq_low=70, freq_high=420
        )[0]
        f0 = f0[(f0 > 70) & (f0 < 420)]
        if f0.numel() < 5:
            return None
        return float(f0.median())
    except Exception as exc:  # pragma: no cover
        logger.debug("f0 estimation failed: %s", exc)
        return None


class AttributeEngine:
    """Thin wrapper around the HF checkpoint + our aggregation logic."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.ready = False
        self.model_id = settings.model_id
        self.quantized = False
        self._labels: list[str] = [
            s.strip() for s in settings.gender_label_order.split(",")
        ]
        assert sorted(self._labels) == ["child", "female", "male"], "bad label order"
        self._model: AgeGenderModel | None = None

    def load(self) -> None:
        from huggingface_hub import snapshot_download

        logger.info("loading model %s", self.model_id)
        path = snapshot_download(self.model_id)
        model = AgeGenderModel.from_pretrained(path)
        model.eval()
        if self.settings.device != "cpu":
            model.to(self.settings.device)
        if self.settings.quantize_dynamic and self.settings.device == "cpu":
            # int8 Linear layers speed up CPU inference on x86_64 (fbgemm),
            # but are ~2x SLOWER via qnnpack on Apple silicon, and fbgemm
            # on arm64-linux appears in supported_engines yet fails at
            # runtime ("unknown architecture"). So: probe the engine first,
            # quantize only when the probe passes, else stay fp32.
            engines = torch.backends.quantized.supported_engines
            if "fbgemm" in engines:
                torch.backends.quantized.engine = "fbgemm"
                try:
                    _probe = nn.Linear(4, 4)
                    torch.ao.quantization.quantize_dynamic(
                        _probe, {nn.Linear}, dtype=torch.qint8
                    )
                    model = torch.ao.quantization.quantize_dynamic(
                        model, {nn.Linear}, dtype=torch.qint8
                    )
                    self.quantized = True
                except Exception as exc:
                    logger.warning("int8 quantization unavailable (%s); running fp32", exc)
            else:
                logger.info("fbgemm unavailable (ARM/mac?); running fp32")
        torch.set_num_threads(max(1, self.settings.num_threads))
        self._model = model
        # Warmup compiles kernels / allocates buffers so first request isn't slow.
        dummy = torch.zeros(int(self.settings.sample_rate * self.settings.window_seconds))
        with torch.inference_mode():
            self._forward_batch([dummy])
        self.ready = True
        MODEL_INFO.labels(self.model_id, str(self.quantized)).set(1)
        logger.info(
            "model ready id=%s quantized=%s threads=%s device=%s",
            self.model_id, self.quantized, self.settings.num_threads, self.settings.device,
        )

    def _forward_batch(self, batch: list[torch.Tensor]) -> list[WindowPrediction]:
        assert self._model is not None
        x = torch.stack([t.reshape(-1) for t in batch])  # (B, T) guaranteed
        if self.settings.device != "cpu":
            x = x.to(self.settings.device)
        with torch.inference_mode():
            age, gender = self._model(x)
        idx = {name: self._labels.index(name) for name in ("child", "female", "male")}
        out = []
        for b in range(gender.shape[0]):
            out.append(
                WindowPrediction(
                    p_child=float(gender[b, idx["child"]]),
                    p_female=float(gender[b, idx["female"]]),
                    p_male=float(gender[b, idx["male"]]),
                    age_years=float(np.clip(float(age[b, 0]) * AGE_YEARS_MAX, 0, 100)),
                )
            )
        return out

    def predict_windows(self, windows: list[np.ndarray]) -> list[WindowPrediction]:
        if not windows:
            return []
        with stage("inference"):
            batch = [
                self._normalize(w) for w in windows
            ]
            tensors = [torch.from_numpy(w) for w in batch]
            # Bound memory: process in chunks of 8 windows.
            preds: list[WindowPrediction] = []
            for i in range(0, len(tensors), 8):
                preds.extend(self._forward_batch(tensors[i : i + 8]))
            return preds

    @staticmethod
    def _normalize(w: np.ndarray) -> np.ndarray:
        """Per-window zero-mean unit-variance (what Wav2Vec2FeatureExtractor does)."""
        s = w.std()
        if s < 1e-8:
            return w - w.mean()
        return (w - w.mean()) / (s + 1e-7)

    # ------------------------------------------------------------------
    # Aggregation: window predictions -> calibrated response fields
    # ------------------------------------------------------------------
    def aggregate(
        self, preds: list[WindowPrediction], f0_hz: float | None = None
    ) -> dict:
        if not preds:
            return {
                "gender_pred": "unknown", "gender_conf": 0.0,
                "bracket_pred": "unknown", "bracket_conf": 0.0,
                "age_median": None, "age_sigma": None,
                "n_windows": 0, "child_dominant": False,
            }

        ages = np.array([p.age_years for p in preds])
        pf = float(np.mean([p.p_female for p in preds]))
        pm = float(np.mean([p.p_male for p in preds]))
        pc = float(np.mean([p.p_child for p in preds]))

        child_dominant = pc >= 0.55 and pc >= max(pf, pm)

        # --- gender -------------------------------------------------------
        gender_pred, gender_conf = "unknown", 0.0
        reasons = []
        if child_dominant:
            reasons.append("child_voice")
        else:
            pf_adj, pm_adj = pf, pm
            if f0_hz is not None and abs(pf - pm) < 0.15:
                # Acoustic prior: adult female F0 ~165-255Hz, male ~85-155Hz.
                if f0_hz >= 175:
                    pf_adj += 0.08
                    reasons.append("pitch_tiebreak_f")
                elif f0_hz <= 135:
                    pm_adj += 0.08
                    reasons.append("pitch_tiebreak_m")
            total = pf_adj + pm_adj + 0.3 * pc
            gender_pred = "female" if pf_adj >= pm_adj else "male"
            gender_conf = round(min(0.99, max(pf_adj, pm_adj) / total), 3)

        # --- age bracket ----------------------------------------------------
        median_age = float(np.median(ages))
        mad = float(np.median(np.abs(ages - median_age))) * 1.4826
        sigma = float(np.clip(mad if len(ages) > 1 else 6.0, 3.0, 10.0))

        masses = {
            name: _bracket_mass(median_age, sigma, lo, hi)
            for name, lo, hi in BRACKETS
        }
        z = sum(masses.values()) or 1.0
        masses = {k: v / z for k, v in masses.items()}
        bracket_pred = max(masses, key=masses.get)
        bracket_conf = round(float(masses[bracket_pred]), 3)
        if median_age < 18.0 and not child_dominant:
            bracket_pred = "18-30"
            bracket_conf = round(bracket_conf * 0.6, 3)
            reasons.append("age_below_bracket_range")

        return {
            "gender_pred": gender_pred,
            "gender_conf": gender_conf,
            "bracket_pred": bracket_pred,
            "bracket_conf": bracket_conf,
            "age_median": round(median_age, 1),
            "age_sigma": round(sigma, 2),
            "n_windows": len(preds),
            "child_dominant": child_dominant,
            "reasons": reasons,
        }
