# Design Write-up (200 words)

I chose audEERING's `wav2vec2-large-robust-6-ft-age-gender`: a single forward
pass yields both gender and continuous age, so one latency budget serves both
tasks, and its *robust* backbone was pre-trained on noisy speech — the right
prior for warehouse/road-noise calls. The 6-layer variant trades little
accuracy for ~3× speed, making <500ms p95 realistic on CPU. On top of the raw
model I added robustness the checkpoint lacks: Silero-VAD trimming, sliding
3-second windows batched into one forward pass with median aggregation to
reject poisoned segments (ringing, hold music), an SNR/clipping/speech-ratio
quality gate that refuses to guess on unusable audio, and calibration — age-bracket
confidence derived from across-window variance, gender confidence from class
share with an F0 tie-breaker. A quality gate plus honest `unknown` outputs
matters more than +1% accuracy for downstream routing.

With more time: fine-tune the head on in-domain telephony audio (8kHz band-limited,
far-field), swap to ONNX Runtime int8 with per-channel quantization, add
temperature scaling fitted per locale, and evaluate fairness across accents.

At 1,000 concurrent calls (~150–250ms CPU bursts): stateless replicas behind
an HPA at 4 vCPU each, server-side micro-batching of windows across requests,
Triton+GPU pool for spikes, and explicit 429 backpressure so agents degrade
to "unknown" instead of timing out.
