"""API schemas.

The wire contract matches the assignment spec exactly; extra fields are
additive (documented in README) so clients written against the spec keep
working.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Gender(str, Enum):
    male = "male"
    female = "female"
    unknown = "unknown"


class AgeBracket(str, Enum):
    b18_30 = "18-30"
    b31_45 = "31-45"
    b46_60 = "46-60"
    b60_plus = "60+"
    unknown = "unknown"


Quality = Literal["good", "degraded", "insufficient"]


class GenderOut(BaseModel):
    prediction: Gender
    confidence: float = Field(ge=0.0, le=1.0)


class AgeBracketOut(BaseModel):
    prediction: AgeBracket
    confidence: float = Field(ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    contact_id: str
    gender: GenderOut
    age_bracket: AgeBracketOut
    processing_ms: int
    audio_quality: Quality
    # --- additive fields below ---
    audio_quality_reasons: list[str] = []
    language: str | None = None  # populated when VIS_LANG_ID_ENABLED=1 & lang=true
    windows_analyzed: int = 0
    age_years_estimate: float | None = None  # raw regression output, for tuning
    stages_ms: dict[str, int] = {}  # per-stage latency breakdown (observability)
    model_version: str = ""
    schema_version: str = "1.0"


class StreamFrame(BaseModel):
    """Envelope for WebSocket frames (partials and finals share the shape)."""

    type: Literal["ready", "partial", "final", "error", "ping"]
    session_id: str
    contact_id: str = ""
    sequence: int = 0
    gender: GenderOut | None = None
    age_bracket: AgeBracketOut | None = None
    audio_quality: Quality | None = None
    speech_seconds: float = 0.0
    dropped_partials: int = 0
    detail: str | None = None


class ErrorOut(BaseModel):
    error: str
    detail: str | None = None
    request_id: str
