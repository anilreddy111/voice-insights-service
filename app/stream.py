"""Bonus: real-time streaming over WebSocket at /stream.

Protocol
--------
client -> server:
  text  {"type": "start", "contact_id"?: str}   (optional; defaults apply)
  bytes : raw PCM16-LE mono chunks @16kHz       (any chunking)
  text  {"type": "stop"}                        (flush + final)
server -> client:
  {"type":"ready"} on connect,
  {"type":"partial", ...attributes, sequence}   every ~2s of new speech,
  {"type":"final", ...attributes}               on stop/disconnect,
  {"type":"error", detail}                      on protocol/runtime errors.

Progressive predictions reuse the exact same engine + aggregation as /analyze
so REST and WS never drift. Partials are skipped (never queued) if inference
is saturated - stale partials are worthless in a live call.
"""

import asyncio
import logging
import uuid

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .audio_io import pcm16_bytes_to_float
from .inference import build_windows, estimate_f0
from .obs import WS_SESSIONS as ACTIVE_WS
from .obs import stage
from .quality import voiced_concat
from .schemas import AgeBracketOut, GenderOut, StreamFrame

logger = logging.getLogger(__name__)
router = APIRouter()

EMIT_SECONDS = 2.0        # emit a partial per this much *new* speech
CHECK_INTERVAL_S = 0.25   # how often we re-scan the buffer for speech growth


@router.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    settings = ws.app.state.settings
    session_id = str(uuid.uuid4())
    contact_id = ""
    sr = settings.sample_rate

    buffer = np.zeros(0, dtype=np.float32)
    emitted_speech_s = 0.0
    sequence = 0
    dropped_partials = 0
    lock = asyncio.Lock()

    async def send_frame(kind: str, agg: dict | None = None, quality: str | None = None,
                         speech_seconds: float = 0.0, detail: str | None = None) -> None:
        nonlocal sequence
        if kind == "partial":
            sequence += 1
        frame = StreamFrame(
            type=kind,  # type: ignore[arg-type]
            session_id=session_id,
            contact_id=contact_id,
            sequence=sequence if kind == "partial" else 0,
            gender=GenderOut(prediction=agg["gender_pred"], confidence=agg["gender_conf"])
            if agg else None,
            age_bracket=AgeBracketOut(prediction=agg["bracket_pred"], confidence=agg["bracket_conf"])
            if agg else None,
            audio_quality=quality,
            speech_seconds=round(speech_seconds, 2),
            dropped_partials=dropped_partials,
            detail=detail,
        )
        await ws.send_json(frame.model_dump())

    async def maybe_emit(final: bool = False) -> None:
        """Run inference on the newest speech and emit a partial."""
        nonlocal buffer, emitted_speech_s, dropped_partials
        regions = ws.app.state.vad(buffer, sr)
        speech_s = sum(e - s for s, e in regions) / sr
        if not final and speech_s - emitted_speech_s < EMIT_SECONDS:
            return
        if final and speech_s < settings.min_speech_seconds:
            await send_frame("final", agg=None, quality="insufficient",
                             speech_seconds=speech_s,
                             detail="not enough speech")
            return
        if lock.locked():
            dropped_partials += 1
            return
        async with lock:
            voiced = voiced_concat(buffer, regions, settings.max_speech_seconds, sr)
            windows = build_windows(
                voiced, sr, settings.window_seconds, settings.hop_seconds,
                settings.min_speech_seconds,
            )
            if windows:
                sem = ws.app.state.inference_sem
                acquired = sem.acquire(blocking=False)
                if not acquired:
                    dropped_partials += 1
                    return
                try:
                    with stage("ws_inference"):
                        preds = await asyncio.to_thread(
                            ws.app.state.engine.predict_windows,
                            windows,
                        )
                finally:
                    sem.release()
                f0 = estimate_f0(voiced, sr)
                agg = ws.app.state.engine.aggregate(preds, f0_hz=f0)
                await send_frame(
                    "final" if final else "partial",
                    agg=agg, quality="good", speech_seconds=speech_s,
                )

            elif final:
                await send_frame("final", agg=None, quality="insufficient",
                                 speech_seconds=speech_s)
            emitted_speech_s = speech_s

    ACTIVE_WS.inc()
    started = False
    try:
        await send_frame("ready")
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            text, data = msg.get("text"), msg.get("bytes")

            if text is not None:
                import json

                ctrl = json.loads(text)
                ctype = ctrl.get("type")
                if ctype == "start":
                    contact_id = str(ctrl.get("contact_id") or uuid.uuid4())
                    started = True
                    emitted_speech_s = 0.0
                    await send_frame("ping", detail="streaming")
                elif ctype == "stop":
                    await maybe_emit(final=True)
                    break
                continue

            if data:
                if not started:  # implicit start: binary-first clients
                    contact_id = contact_id or str(uuid.uuid4())
                    started = True
                buffer = np.concatenate([buffer, pcm16_bytes_to_float(data)])
                # Hard cap: force-final very long sessions to bound memory.
                if buffer.size > sr * 120:
                    await maybe_emit(final=True)
                    break
                if buffer.size >= int(sr * CHECK_INTERVAL_S):
                    await maybe_emit()

        await maybe_emit(final=True)  # flush on any exit path
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("ws session failed", extra={"session_id": session_id})
        try:
            await send_frame("error", detail=str(exc.__class__.__name__))
        except Exception:  # pragma: no cover
            pass
    finally:
        ACTIVE_WS.dec()
        try:
            await ws.close(code=1000)
        except Exception:  # pragma: no cover
            pass
        logger.info(
            "ws_session_end",
            extra={"session_id": session_id, "contact_id": contact_id,
                   "buffered_s": round(buffer.size / sr, 1)},
        )
