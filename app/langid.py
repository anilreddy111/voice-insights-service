"""Bonus: best-effort language identification via SpeechBrain VoxLingua107.

Disabled by default (VIS_LANG_ID_ENABLED=0):
  * adds a ~400MB model and an extra forward pass per request;
  * the assignment's latency budget belongs to age/gender first.

Loaded lazily on first use so the base service starts fast even with the
package installed. Any failure resolves to language=None - never to a
failed request.
"""

import logging

logger = logging.getLogger(__name__)

_classifier = None
_load_failed = False


def _get_classifier():
    global _classifier, _load_failed
    if _classifier is not None or _load_failed:
        return _classifier
    try:
        from speechbrain.inference.interfaces import EncoderClassifier

        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="/tmp/sb-langid",  # weights cache only; never audio
            run_opts={"device": "cpu"},
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("lang-id disabled: %s", exc)
        _load_failed = True
    return _classifier


def identify_language(x, sr: int) -> str | None:
    """Return a language code like 'en' or None. Best-effort by design."""
    import numpy as np
    import torch

    clf = _get_classifier()
    if clf is None:
        return None
    try:
        wav = torch.from_numpy(np.ascontiguousarray(x[: sr * 10]))[None]
        emb = clf.encode_batch(wav)  # (1, 1, 107) after softmax in mods
        probs = emb.squeeze()
        label_encoder = clf.hparams.label_encoder
        return str(label_encoder.ind2lab[int(probs.argmax())])
    except Exception as exc:
        logger.debug("language id failed: %s", exc)
        return None
