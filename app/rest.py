"""REST API: POST /analyze + health endpoint."""

import logging
import time
import uuid

import numpy as np
import torch
import torchaudio
from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from . import langid
from .audio_io import AudioDecodeError, decode_to_pcm, pcm16_bytes_to_float
from .config import Settings
from .inference import AttributeEngine, build_windows, estimate_f0
from .obs import (
    ERRORS_TOTAL,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
    StageTimer,
    request_id_ctx,
    stage,
)
from .quality import assess as assess_quality
from .quality import voiced_concat
from .schemas import AgeBracket, AgeBracketOut, AnalyzeResponse, Gender, GenderOut

logger = logging.getLogger(__name__)
router = APIRouter()


class ApiError(Exception):
    def __init__(self, status: int, code: str, detail: str | None = None,
                 headers: dict[str, str] | None = None):
        self.status, self.code, self.detail = status, code, detail
        self.headers = headers or {}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request,
    response: Response,
    contact_id: str | None = Query(default=None),
    contact_id_form: str | None = Form(default=None, alias="contact_id"),
    file: UploadFile | None = File(default=None),
    lang: bool = Query(default=False),
    encoding: str = Query(default="auto", description="auto | pcm16"),
    sample_rate: int = Query(default=16000, description="for encoding=pcm16"),
) -> AnalyzeResponse:
    settings: Settings = request.app.state.settings
    rid = str(uuid.uuid4())
    request_id_ctx.set(rid)
    response.headers["X-Request-ID"] = rid

    cid = contact_id or contact_id_form or str(uuid.uuid4())
    timer = StageTimer()
    status_code = 200
    try:
        payload = await _read_payload(request, file, settings)
        result = await run_in_threadpool(
            _run_pipeline, request.app, payload, cid, lang, timer,
            encoding.lower(), sample_rate,
        )
        response.headers["X-Process-Time-Ms"] = str(result.processing_ms)
        return result
    except ApiError as exc:
        status_code = exc.status
        ERRORS_TOTAL.labels(exc.code).inc()
        raise
    finally:
        REQUEST_LATENCY.labels("/analyze", str(status_code)).observe(timer.total_ms / 1000)
        REQUESTS_TOTAL.labels("/analyze", str(status_code)).inc()
        logger.info(
            "analyze_done",
            extra={
                "contact_id": cid,
                "status": status_code,
                "processing_ms": timer.total_ms,
                "stages_ms": timer.stages,
            },
        )


async def _read_payload(request: Request, file: UploadFile | None, settings: Settings) -> bytes:
    if file is not None:
        data = await file.read()
    else:
        content_type = request.headers.get("content-type", "")
        # Raw streaming uploads: any audio/* or opaque octet-stream body.
        if not content_type.startswith(("audio/", "application/octet-stream")):
            raise ApiError(
                415,
                "unsupported_media_type",
                "send multipart/form-data with a 'file' field, or raw audio bytes",
            )
        data = await request.body()
    if len(data) == 0:
        raise ApiError(400, "empty_body")
    if len(data) > settings.max_upload_bytes:
        raise ApiError(413, "payload_too_large")
    return data


def _run_pipeline(
    app,
    data: bytes,
    contact_id: str,
    want_lang: bool,
    timer: StageTimer,
    encoding: str = "auto",
    raw_sr: int = 16000,
) -> AnalyzeResponse:
    """CPU-bound pipeline; executed on the threadpool to keep the loop free."""
    settings: Settings = app.state.settings
    engine: AttributeEngine = app.state.engine
    if engine is None or not engine.ready:
        raise ApiError(503, "engine_unavailable", "model is still loading")

    with timer.measure("decode"):
        try:
            if encoding == "pcm16":
                # Headerless telephony streams (e.g. Twilio Media Streams dumps):
                # caller must declare rate; we assume mono little-endian s16.
                x = pcm16_bytes_to_float(data)
                if x.size < settings.sample_rate // 10:
                    raise AudioDecodeError("audio too short")
            else:
                x = decode_to_pcm(data, settings.sample_rate)
        except AudioDecodeError as exc:
            raise ApiError(400, "audio_decode_failed", str(exc)) from exc
        if encoding == "pcm16" and raw_sr != settings.sample_rate:
            x = torchaudio.functional.resample(
                torch.from_numpy(np.ascontiguousarray(x))[None], raw_sr, settings.sample_rate
            )[0].numpy()

    with timer.measure("quality"):
        report = assess_quality(x, settings.sample_rate, regions_fn=app.state.vad)

    if report.flag == "insufficient":
        # Honest refusal beats confident garbage on unusable audio.
        return _respond(timer, contact_id, report, agg=None, language=None, settings=settings)

    with timer.measure("vad_trim"):
        voiced = voiced_concat(
            x,
            report.regions,  # single VAD pass: quality + trim share regions
            settings.max_speech_seconds,
            settings.sample_rate,
        )
    windows = build_windows(
        voiced,
        settings.sample_rate,
        settings.window_seconds,
        settings.hop_seconds,
        settings.min_speech_seconds,
    )

    preds = []
    sem = app.state.inference_sem
    acquired = bool(windows) and sem.acquire(blocking=False)
    try:
        if windows and not acquired:
            # Backpressure contract: saturated -> fast 429, never queue.
            # A slow success is worse than a fast retry for a live call.
            raise ApiError(
                429, "server_busy",
                "inference at capacity; retry shortly",
                headers={"Retry-After": "1"},
            )
        if windows:
            with timer.measure("inference"):
                preds = engine.predict_windows(windows)
    finally:
        if acquired:
            sem.release()

    f0 = estimate_f0(voiced, settings.sample_rate)
    agg = engine.aggregate(preds, f0_hz=f0)

    language = None
    if want_lang and settings.lang_id_enabled:
        with stage("langid"):
            language = langid.identify_language(voiced, settings.sample_rate)

    return _respond(timer, contact_id, report, agg=agg, language=language, settings=settings)


def _respond(timer, contact_id, report, *, agg, language, settings) -> AnalyzeResponse:
    if agg is None:
        agg = {
            "gender_pred": "unknown", "gender_conf": 0.0,
            "bracket_pred": "unknown", "bracket_conf": 0.0,
            "age_median": None, "n_windows": 0,
        }
    return AnalyzeResponse(
        contact_id=contact_id,
        gender=GenderOut(prediction=Gender(agg["gender_pred"]), confidence=agg["gender_conf"]),
        age_bracket=AgeBracketOut(
            prediction=AgeBracket(agg["bracket_pred"]), confidence=agg["bracket_conf"]
        ),
        processing_ms=timer.total_ms,
        audio_quality=report.flag,
        audio_quality_reasons=report.reasons,
        language=language,
        windows_analyzed=agg["n_windows"],
        age_years_estimate=agg["age_median"],
        stages_ms=timer.stages,
        model_version=settings.model_id,
    )


@router.get("/healthz")
async def healthz(request: Request):
    engine = getattr(request.app.state, "engine", None)
    return {
        "status": "ok" if engine and engine.ready else "degraded",
        "model_loaded": bool(engine and engine.ready),
        "model_id": getattr(engine, "model_id", None),
        "device": request.app.state.settings.device,
        "version": request.app.state.settings.version,
        "uptime_s": int(time.time() - request.app.state.started_at),
    }
