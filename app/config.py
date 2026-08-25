"""Central configuration.

Every knob is environment-driven with the ``VIS_`` prefix so the same image can
be tuned per deployment (dev laptop vs 4-vCPU container vs GPU node) without a
rebuild. Rationale for defaults lives next to each field.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIS_", env_file=".env", extra="ignore")

    # --- Model ---------------------------------------------------------------
    # 6-layer variant: ~3x faster than the 24-layer one with close accuracy,
    # which is what makes the <500ms p95 target realistic on CPU.
    model_id: str = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    # Empirically verified against known-gender LibriSpeech speakers
    # (scripts/verify_label_order.py): column order is female/male/child,
    # matching config.json id2label. NOTE: the -6 model card's printed
    # example header says "child,female,male" - that comment is wrong.
    gender_label_order: str = "female,male,child"
    device: str = "cpu"  # "cuda" also supported; CPU is the default target
    num_threads: int = 4  # matches typical k8s vCPU request; oversubscription hurts p95
    quantize_dynamic: bool = True  # int8 Linear layers: ~2x speedup on CPU

    # --- Audio pipeline --------------------------------------------------------
    sample_rate: int = 16000  # model-native rate
    window_seconds: float = 3.0  # model was trained on short utterances
    hop_seconds: float = 1.5  # 50% overlap -> more windows -> stabler aggregate
    min_speech_seconds: float = 0.8  # below this we refuse to guess (insufficient)
    max_speech_seconds: float = 15.0  # latency bound; extra speech adds little info
    max_upload_bytes: int = 25_000_000  # ~13min of 16kHz mono pcm16

    # --- Serving ---------------------------------------------------------------
    max_concurrent_inference: int = 2  # bounds RAM/CPU; excess requests queue
    lang_id_enabled: bool = False  # bonus feature; off by default for latency
    log_level: str = "INFO"
    version: str = "1.0.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
