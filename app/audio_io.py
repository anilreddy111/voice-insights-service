"""Audio decoding: arbitrary bytes -> mono float32 @16 kHz.

Two-stage decode chain, fully in-process:
1. libsndfile (via soundfile): fast path for wav/flac/ogg/mp3 containers.
2. PyAV (bundled FFmpeg libs): everything else - webm/opus, amr, alaw,
   exotic telephony containers. In-process bindings beat an ffmpeg
   subprocess: no fork/exec cost per request, no PATH dependency, no
   pipe-buffer edge cases, deterministic error handling.

Decode output is capped (MAX_DECODE_SECONDS) so a crafted small-but-huge
compressed payload can't pin the worker.

Privacy-critical module: everything happens in RAM (BytesIO / libav
buffers). There is no code path in this service that writes caller audio
to disk.
"""

import io

import numpy as np
import soundfile as sf
import torch
import torchaudio

# A compressed upload can expand enormously (25MB mp3 ~= hours of audio).
# We never need more than this much decoded audio per request.
MAX_DECODE_SECONDS = 90


class AudioDecodeError(ValueError):
    """Raised when bytes cannot be decoded as audio by any backend."""


def decode_to_pcm(data: bytes, target_sr: int = 16000) -> np.ndarray:
    """Decode compressed or raw audio to mono float32 at ``target_sr``."""
    x, sr = _decode_soundfile(data)
    if x is None:
        x, sr = _decode_pyav(data, target_sr)
    if x is None:
        raise AudioDecodeError("could not decode payload as audio")
    if sr != target_sr:
        t = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(x))[None], sr, target_sr
        )
        x = t[0].numpy()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.5:  # some codecs produce >1.0 floats; renormalize not clip
        x = x / peak
    if x.size < target_sr // 10:  # <100ms is useless for speaker attributes
        raise AudioDecodeError("audio too short")
    return np.ascontiguousarray(x, dtype=np.float32)


def _decode_soundfile(data: bytes):
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:
        return None, None
    if isinstance(x, np.ndarray) and x.ndim == 2:  # downmix to mono
        x = x.mean(axis=1)
    return x, sr


def _decode_pyav(data: bytes, target_sr: int):
    """In-process FFmpeg decode via PyAV. Returns (float32 mono, sr) or (None, None)."""
    try:
        import av
    except ImportError:  # pragma: no cover - pyav is a hard dependency
        return None, None

    container = None
    chunks: list[np.ndarray] = []
    total = 0
    cap = target_sr * MAX_DECODE_SECONDS
    try:
        container = av.open(io.BytesIO(data))
        streams = [s for s in container.streams if s.type == "audio"]
        if not streams:
            return None, None
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
        for packet in container.demux(streams[0]):
            for frame in packet.decode():
                out = resampler.resample(frame)
                for f in out if isinstance(out, list) else [out]:
                    pcm = f.to_ndarray().reshape(-1).astype(np.float32) / 32768.0
                    chunks.append(pcm)
                    total += pcm.size
                if total >= cap:  # bound hostile/expanding payloads
                    break
            if total >= cap:
                break
        if not chunks or total == 0:
            return None, None
        return np.concatenate(chunks)[:cap], target_sr
    except Exception:
        return None, None
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass


def pcm16_bytes_to_float(chunk: bytes) -> np.ndarray:
    """Streaming helper: little-endian s16 -> float32 in [-1, 1]."""
    usable = len(chunk) - (len(chunk) % 2)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(chunk[:usable], dtype="<i2").astype(np.float32) / 32768.0
