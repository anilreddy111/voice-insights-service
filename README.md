# Voice Insights Service

Real-time caller attribute inference for voice AI agents: given call audio,
estimate the speaker's **gender** and **age bracket** with calibrated
confidence estimates, an **audio-quality gate**, and (optionally) **language** — over a
low-latency REST API and a progressive WebSocket stream.

Built for logistics contact centers: noisy warehouses, road noise, compressed
telephony codecs. When audio is unusable the service says so (`insufficient`)
instead of returning confident nonsense.

```
POST /analyze        multipart file OR raw bytes  -> attributes + confidence
WS   /stream         PCM16 chunks                 -> progressive predictions
GET  /healthz        liveness + model status
GET  /metrics        Prometheus
```

---

## Quickstart

### Docker (recommended)

```bash
docker compose up --build
# first build downloads and caches the model weights; afterwards it runs offline
curl -F "file=@samples/female_like_220hz.wav" http://localhost:8000/analyze
```

### Local (Python 3.10+)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt

python scripts/make_sample.py            # synthetic samples (NOT real voices)
uvicorn app.main:app --port 8000 &       # weights download on first start
python scripts/smoke_test.py             # end-to-end check incl. schema asserts
```

## Sample audio

- `samples/librispeech_*.wav` — two real CC-BY-4.0 LibriSpeech clips
  (known-gender speakers) for meaningful smoke tests.
- `python scripts/make_sample.py` generates `samples/*_like.wav` — synthetic,
  harmonic babble for plumbing tests only (the quality gate correctly flags
  them `insufficient`: Silero-VAD rejects non-speech).
- Any Common Voice clip works too: `--file /path/to/clip.mp3`.

---

## API

### `POST /analyze`

Accepts, in order of preference:

1. `multipart/form-data` with a `file` field (any container libsndfile or
   PyAV reads: wav, mp3, flac, ogg, webm/opus, amr…)
2. raw body with `Content-Type: audio/*` (self-describing containers)
3. headerless telephony PCM: `?encoding=pcm16&sample_rate=16000` with raw s16le bytes

Query params: `contact_id` (echoed back; generated if absent), `lang=true`
(language ID, requires `VIS_LANG_ID_ENABLED=1`), `encoding`, `sample_rate`.

```bash
curl -F "file=@call.wav" "http://localhost:8000/analyze?contact_id=d0c7...&lang=true"
```

Response (spec fields plus documented additive fields):

```json
{
  "contact_id": "d0c7...",
  "gender": {"prediction": "female", "confidence": 0.86},
  "age_bracket": {"prediction": "31-45", "confidence": 0.58},
  "processing_ms": 187,
  "audio_quality": "good",
  "audio_quality_reasons": [],
  "language": null,
  "windows_analyzed": 3,
  "age_years_estimate": 36.2,
  "model_version": "audeering/wav2vec2-large-robust-6-ft-age-gender",
  "schema_version": "1.0"
}
```

Error envelope: `{"error": "<code>", "detail": "...", "request_id": "..."}` with
400 (undecodable), 413 (too large), 415 (wrong content type), 503 (model loading).

### `WS /stream` (bonus: real-time progressive predictions)

Send binary PCM16-LE mono @16kHz chunks; receive JSON frames.

| client → server                          | server → client                                   |
|------------------------------------------|---------------------------------------------------|
| `{"type":"start","contact_id":"..."}`    | `{"type":"ready"}` on connect                     |
| *(binary)* raw PCM16 chunk               | `{"type":"partial", gender…, age_bracket…, seq}` every ~2s of speech |
| `{"type":"stop"}`                        | `{"type":"final", …}` aggregated over the session |
|                                          | `{"type":"error","detail":…}` on failure          |

Partials reuse the identical engine/aggregation as REST (no drift). If
inference is saturated, stale partials are *skipped* and counted in
`dropped_partials` rather than queued — freshest data wins in a live call.

---

## Architecture

```
bytes ──► decode ──► quality gate ──► VAD trim ──► windows ──► wav2vec2 ──► aggregate ──► JSON
          (ram only)  snr/rms/vad     silero-vad   3s/hop1.5s  batched fwd   median+mass
                                                            │
                                     pitch sanity-check ────┘ (tie-break only)
```

Latency (reproducible: `python scripts/benchmark.py`, fp32 CPU, 4 threads):

| input    | p50   | p95   | inference stage |
|----------|-------|-------|-----------------|
| 2s clip  | 85ms  | 87ms  | 76ms            |
| **5s clip**  | **95ms**  | **98ms**  | 76ms            |
| 10s clip | 334ms | 396ms | 298ms           |

Target: <500ms end-to-end for a 5-second chunk — met with ~5× margin natively.
Note that Docker Desktop on macOS inflates torch CPU ops ~3× via its VM;
production Linux hosts behave like the native numbers. x86_64 Linux
additionally gets fbgemm int8 (~1.5–2× faster); ARM/mac intentionally stays
fp32 because qnnpack dynamic-int8 measured ~2× *slower*.

### Measured accuracy (LibriSpeech dev-clean, 120 clips / 40 speakers)

Run it yourself: `python scripts/evaluate_cv.py --librispeech-dir <dev-clean>`

| metric            | gender task |
|-------------------|-------------|
| accuracy          | **0.950**   |
| macro F1          | 0.949       |
| mean confidence   | 0.966       |
| ECE (10 bins)     | **0.025** (well calibrated) |
| confusion         | F→F 60, M→M 53, M→F 6, F→M 0 |

Age has no ground truth in LibriSpeech; the age head's continuous estimate
averaged 41.6y ± 8.7y across these speakers. Use Common Voice
(`--cv-root`) for bracket-level age evaluation.

## Design decisions (why each piece exists)

**Model: audEERING `wav2vec2-large-robust-6-ft-age-gender`.**
One forward pass yields both attributes — age (0–100 regression) and a 3-way
head over child/female/male — so one latency budget covers both tasks. The
6-layer variant is ~3× faster than the 24-layer sibling with close accuracy
(per audEERING's paper, arXiv 2306.16962), which is what makes CPU-only
real-time feasible. The *robust* backbone was pre-trained on noisy/degraded
speech — a good prior for warehouse and road noise. Weights are public under
CC BY-NC-SA 4.0 (fine for this exercise; commercial use would need
audEERING's commercial model or a re-trained head).

**Windowed inference + median aggregation, not one pass per clip.**
Ringing, DTMF, hold music and crosstalk poison whole-clip embeddings. Sliding
3s windows (50% overlap) let the median reject poisoned segments; equal-length
windows also batch cleanly with no padding/masking complexity.

**Confidence and uncertainty estimates.**
Age is a point regression, so we estimate uncertainty from across-window
dispersion (MAD σ) and report bracket confidence as Gaussian probability mass
under that spread. Gender confidence is derived from the aggregated class
probabilities, with a pitch-based tie-breaker near 50/50. Gender confidence
calibration is evaluated separately using ECE on the labeled evaluation set.

**Pitch tie-breaker (hybrid pipeline).**
Near-ties (Δp < 0.15) get nudged by median F0 (>175Hz → female prior,
<135Hz → male). F0 is nearly free to compute and decorrelated from what
wav2vec2 might have overfit to.

**Quality gate before inference.**
SNR proxy, clipping ratio, level, and Silero-VAD speech-ratio map to
good/degraded/insufficient. `insufficient` short-circuits to `unknown`
predictions — surfacing bad input beats silent bad output in a routing system.

**Honest child handling.** The head can say *child*, which is outside our
brackets and outside male/female. We return `unknown` (+reason) instead of
forcing a wrong adult label.

**Explicit backpressure, not invisible queues.** The inference semaphore
*actually sheds*: when at capacity, /analyze returns 429 + `Retry-After` (a
fast retry beats a slow success in a live call), and WS sessions skip stale
partials (`dropped_partials` counter) instead of queueing them. A test pins
this behaviour (`tests/test_concurrency.py`) — the original implementation
acquired the semaphore and then ran inference anyway.

**In-process decode (PyAV), no subprocesses.** libsndfile handles
wav/flac fast-path; PyAV (bundled FFmpeg libs) covers webm/opus, amr,
alaw telephony. No ffmpeg binary, no PATH dependency, no fork/exec cost,
and decoded output is capped so crafted payloads can't pin a worker.
Covered by an offline codec matrix test (`tests/test_codecs.py`).

**VAD serialization.** Silero resets its internal state on every call, so
concurrent VAD invocations from worker threads would corrupt each other's
hidden states. VAD calls are serialized by a lock (~20ms each) and the
regions are computed once per request and shared between quality and trim.

**Privacy by construction.** Audio lives in RAM only (BytesIO / in-process
libav buffers); there is no code path that writes caller audio to disk; logs
carry durations and ids, never audio; nothing leaves the process at runtime;
weights are baked into the image. Inferred attributes are special-category
data under GDPR Art. 9 — production would need consent flow, DPIA, and short
retention upstream; this service retains nothing.

**Column-order verification (documented gotcha).** The head's classes are
child/female/male, but the public docs are inconsistent about the *column*
order of the output vector (the two model cards' example headers disagree).
We pinned it empirically against `config.json` id2label plus known-gender
LibriSpeech speakers (`scripts/verify_label_order.py`, and a permanent live
test over committed CC-BY clips). Lesson: trust config + data over doc prose.

**Reliability.** Model loads in background (healthz reports degraded until
ready), bounded semaphore sheds excess inference instead of queueing forever,
bounded decode caps hostile inputs, structured JSON logs carry request ids,
Prometheus histograms per stage, graceful 503s while loading.

## Bonus features included

- **WebSocket streaming** with progressive partial predictions (see above).
- **Language ID** (best-effort, VoxLingua107 via SpeechBrain): off by default
  to protect the latency budget; enable with
  `pip install -r requirements-langid.txt` + `VIS_LANG_ID_ENABLED=1`, then
  `POST /analyze?lang=true`.
- **Eval harness**: `scripts/evaluate_cv.py` runs the exact production
  pipeline against Mozilla Common Voice and reports accuracy, macro-F1,
  confusion matrix, ECE and a calibration curve for both tasks.

## Testing

```bash
pytest -q                        # hermetic suite (fake engine, no downloads)
VIS_LIVE_TESTS=1 pytest -m live  # real weights end-to-end (bundled clips)
python scripts/smoke_test.py --ws          # black-box smoke vs running server
python scripts/benchmark.py                # reproducible latency table
```

Coverage highlights:
- **Codec matrix** (`tests/test_codecs.py`): wav/flac/ogg/webm-opus/mp3/alaw-8k
  roundtrips through the production decode chain — fixtures are encoded
  in-process with PyAV, so it runs offline and in CI.
- **Concurrency** (`tests/test_concurrency.py`): saturated service must shed
  with 429+`Retry-After`; 8 parallel requests show no state bleed.
- **Live known-gender check** (`tests/test_live.py`): committed LibriSpeech
  clips must classify correctly — catches label-order/model drift loudly.
- The integration tests run the full HTTP/WebSocket stack against a
  deterministic fake engine: fast, offline, CI-safe.

## Scaling to 1,000 concurrent calls

Per-call cost here is one ~150–250ms CPU burst per analysis. Rough plan:

1. **Stateless horizontal scale**: replicas behind LB (the service keeps no
   state); K8s HPA on RPS × p95 latency or queue depth.
2. **Right-size per pod**: 4 vCPU / ~2.5GB; int8 dynamic quantization already
   halves the burst; ONNX Runtime or torch.compile for another ~30%.
3. **Batching at the edge**: micro-batch concurrent requests server-side
   (windows from different calls in one forward) — wav2vec2 batches almost
   linearly to saturation.
4. **GPU pool for spikes**: one A10G serves dozens of streams; Triton with
   dynamic batching if volume justifies it.
5. **Backpressure as contract**: 429 + `Retry-After` when saturated (call
   agents degrade to "unknown attributes", not timeouts).
6. **Streaming fan-in**: WS sessions emit every ~2s of *speech* (not audio),
   so 1,000 concurrent calls ≈ a few hundred inferences/sec worst case —
   sized fleet ≈ 40–80 vCPUs with margin.

## Known limitations

- Age is a noisy signal even for humans; treat brackets as soft routing hints,
  not facts. MAE of the underlying model is ~7–11 years (paper).
- NC license on weights → non-commercial use only as-is.
- Trained mostly on read/interview speech; heavy accents and non-English calls
  may skew (eval harness exists precisely to quantify this per locale).
- Single-node concurrency guard is per-process; multi-worker deployments need
  pod-level limits instead.

## Repo layout

```
app/
  main.py       app factory, lifespan, middleware, error handlers
  rest.py       POST /analyze, GET /healthz
  stream.py     WS /stream protocol + session loop
  inference.py  model wrapper, windowing, aggregation/calibration
  quality.py    SNR/clipping/speech-ratio gate + VAD
  audio_io.py   in-memory decode (libsndfile fast-path → PyAV fallback)
  langid.py     optional language ID
  obs.py        JSON logs + Prometheus metrics
  schemas.py    pydantic wire models
  config.py     env-driven settings (VIS_*)
scripts/        make_sample, smoke_test, benchmark, evaluate_cv, verify_label_order
tests/          hermetic integration/unit + codec matrix + concurrency + opt-in live
DESIGN.md       200-word write-up + deeper rationale
INTERVIEW_NOTES.md  decision table + likely follow-ups
```
