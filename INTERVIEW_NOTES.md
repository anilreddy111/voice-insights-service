# Interview Notes — every choice, and why

A quick reference for defending each decision. Grouped by pipeline stage.

## Model choice

| Decision | Why | Follow-up they may ask |
|---|---|---|
| audEERING wav2vec2 age-gender, 6-layer | One pass → both attributes (one latency budget). Robust backbone = pretrained on noisy speech. 6 layers ≈ 3× faster than 24 with close accuracy (their paper reports the gap is small) | "Why not openSMILE + classifier?" → hand-crafted features (eGeMAPS) are interpretable but ~4% UAR worse on gender per the paper; also two-stage pipelines have two latency budgets and more drift surface |
| CPU-first, int8 dynamic quantization | Voice infra is mostly CPU; GPU adds ops burden. Dynamic quantization of Linear layers ≈ ~1.5–2× speedup for transformer FFN-heavy nets | "Why not ONNX?" → next step listed in roadmap; torch dynamic quant gets most of the win with zero export risk |
| CC BY-NC-SA license accepted | Assignment/research context; documented that production needs commercial model or retrained head | Shows licensing awareness |
| Label-order verified empirically | The two public model cards contradict each other; config.json id2label matched data, card comment didn't. `scripts/verify_label_order.py` proves it against LibriSpeech known-gender speakers | Great war story: trust config+data over docs; make it configurable (`VIS_GENDER_LABEL_ORDER`) |

## Pipeline

| Decision | Why |
|---|---|
| Silero-VAD trim before inference | Silence/music/ringing wastes window budget and skews embeddings; VAD gives speech-only concatenation capped at 15s (latency bound — extra speech adds little info) |
| 3s windows / 1.5s hop, batched | Matches short-utterance training regime; 50% overlap doubles vote count; equal lengths batch without padding masks; median across windows rejects poisoned segments (DTMF, beeps) |
| Per-window zero-mean/unit-variance | Replicates Wav2Vec2FeatureExtractor normalization without hauling the processor into the hot path |
| Age σ from MAD of window predictions | Regression has no native uncertainty; cross-window dispersion is an honest, cheap proxy; bracket confidence = Gaussian mass under that σ (so 0.63 means something) |
| F0 tie-breaker only near ties (Δp<0.15) | Pitch alone is brittle (falsetto, smokers, children); as a nudge it's decorrelated evidence; autocorrelation via torchaudio is nearly free |
| Child → unknown (+reason) | Contract has no child bracket; forcing adult labels would be a silent lie downstream |
| Quality gate: SNR proxy, clip ratio, RMS floor, speech ratio | Cheap frame stats catch the failure modes logistics calls actually have; `insufficient` skips inference entirely — honest refusal beats confident garbage |

## API / serving

| Decision | Why |
|---|---|
| FastAPI + uvicorn, single worker | Model is shared in-process; scale by replicas not workers; asyncio loop stays free because inference runs on threadpool |
| threading.BoundedSemaphore(2), shed-not-queue | Queueing compounds tail latency in real-time contexts. At capacity: 429 + Retry-After (REST) / skip stale partials (WS). Test-pinned — the first version acquired the semaphore but ignored the result and ran inference anyway; the concurrency tests exist precisely to catch that class of bug |
| Multipart OR raw OR headerless-PCM16 | Covers file uploads, raw streams from telephony gateways, and Twilio-style dumps |
| PyAV instead of ffmpeg subprocess | In-process decode: no fork/exec per request, no PATH dependency, deterministic errors; decoded output capped so crafted payloads can't pin a worker; codec matrix test encodes fixtures in-process so it runs offline/CI |
| Silero VAD serialized by a lock | get_speech_timestamps() resets model state on each call -> concurrent calls clobber hidden states and corrupt speech regions under load. VAD is ~20ms so serializing is cheap; regions computed once per request and reused for quality + trim |
| Additive response fields | Spec fields verbatim; extras (windows_analyzed, age_years_estimate, reasons) aid debugging without breaking contract clients |
| Background model load + degraded healthz | Pod becomes ready fast; LB won't route until healthy; no thundering-herd cold starts |
| JSON logs + request_id ContextVar | One grep joins all lines of a request incl. threadpool work; audio never logged (PII) |

## Privacy

| Decision | Why |
|---|---|
| Zero disk writes for audio | BytesIO + in-process libav buffers; nothing to leak, nothing to GC-later-delete |
| No runtime network calls | Weights baked at build; inference local; GDPR-friendly by construction |
| Logs carry durations/ids only | Audio bytes are special-category biometric data under GDPR Art.9; even transient logs must not capture them |
| Documented upstream duties | Consent flow, DPIA, retention policy belong to the calling system; service retains nothing |

## Testing strategy

| Decision | Why |
|---|---|
| Hermetic integration tests w/ fake engine | Full HTTP/WS stack exercised offline, deterministic, CI-safe |
| FakeEngine shares aggregation impl | REST and WS tested against the same math as prod path (minus weights) |
| Live tests opt-in (`VIS_LIVE_TESTS=1`) | Real-model assertions run locally/release, not every commit (500MB download) |
| Eval harness measures calibration too | ECE/Brier matter for routing thresholds; accuracy alone hides overconfidence |

## Scaling story (1,000 concurrent calls)

- Burst cost ≈ one 150–250ms CPU forward per analysis; streaming emits per ~2s of *speech*, so worst case ≈ few hundred forwards/sec fleet-wide.
- Stateless replicas + HPA on latency/RPS; 4 vCPU pods (thread-pinned).
- Server-side micro-batching across requests (wav2vec2 scales near-linearly with batch to saturation).
- Triton + A10G pool for spikes; gRPC between gateway and model tier if RPC overhead shows.
- Backpressure contract: 429 + Retry-After; agents degrade gracefully to unknown.

## Known weaknesses I'd fix with time

1. Fine-tune head on 8kHz telephony domain shift (model trained on wideband).
2. ONNX Runtime int8 + graph fusion (~30% more).
3. Temperature scaling per locale fitted on CV (harness already computes ECE).
4. Fairness slice: report metrics per accent/language bucket, not just aggregate.
5. Two-stage VAD→embeddings could cache speaker embeddings across a call to stabilize predictions.
