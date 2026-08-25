"""App factory + lifespan wiring.

``create_app(engine=None)`` accepts an engine override so tests can run fully
hermetic (no weights, no network) against a deterministic fake - see
tests/conftest.py.
"""

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from .config import Settings, get_settings
from .inference import AttributeEngine
from .obs import ERRORS_TOTAL, configure_logging, request_id_ctx
from .quality import load_vad
from .rest import ApiError
from .rest import router as rest_router
from .schemas import ErrorOut
from .stream import router as stream_router

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, engine: AttributeEngine | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.started_at = time.time()
        app.state.vad = await asyncio.to_thread(load_vad)
        # Bounded engine access from worker threads; excess work is shed.
        app.state.inference_sem = threading.BoundedSemaphore(settings.max_concurrent_inference)
        if engine is not None:
            app.state.engine = engine
        else:
            app.state.engine = None
            import uuid

            rid = str(uuid.uuid4())
            request_id_ctx.set(rid)

            def _load():
                eng = AttributeEngine(settings)
                try:
                    eng.load()
                    app.state.engine = eng
                except Exception:
                    logger.exception("model load failed; serving 503 until restart")
                    ERRORS_TOTAL.labels("model_load_failed").inc()

            # Load in background so /healthz answers immediately during boots.
            threading.Thread(target=_load, name="model-loader", daemon=True).start()
        logger.info("startup complete version=%s", settings.version)
        yield
        logger.info("shutdown")

    app = FastAPI(
        title="Voice Insights Service",
        description="Real-time caller attribute inference (gender + age bracket) for voice agents",
        version=settings.version,
        lifespan=lifespan,
    )
    configure_logging(settings.log_level)

    app.include_router(rest_router)
    app.include_router(stream_router)
    app.mount("/metrics", make_asgi_app())

    @app.middleware("http")
    async def observability(request: Request, call_next):
        rid = str(__import__("uuid").uuid4())[:8]
        request_id_ctx.set(rid)
        t0 = time.perf_counter()
        try:
            resp = await call_next(request)
        except ApiError as exc:
            payload = ErrorOut(error=exc.code, detail=exc.detail, request_id=rid)
            resp = JSONResponse(
                status_code=exc.status, content=payload.model_dump(), headers=exc.headers
            )
        except Exception:
            logger.exception("unhandled error")
            payload = ErrorOut(error="internal_error", request_id=rid)
            resp = JSONResponse(status_code=500, content=payload.model_dump())
        resp.headers["X-Request-ID"] = rid
        resp.headers["X-Process-Time-Ms"] = str(int((time.perf_counter() - t0) * 1000))
        return resp

    return app


app = create_app()
