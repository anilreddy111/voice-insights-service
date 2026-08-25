# Design Write-up

I chose audEERING's `wav2vec2-large-robust-6-ft-age-gender`: a single forward pass yields both gender and continuous age, so one latency budget serves both tasks. Its robust backbone was trained on noisy speech, making it a good fit for warehouse and road-noise calls. The 6-layer variant provides a useful speed/accuracy trade-off for CPU inference.

I added robustness around the checkpoint: Silero VAD removes non-speech, sliding 3-second windows with 50% overlap reduce the impact of corrupted segments, and an SNR/clipping/speech-ratio quality gate refuses to guess on unusable audio. Window predictions are aggregated to produce stable gender and age-bracket estimates. Child predictions are mapped to `unknown` because children are outside the requested adult brackets.

With more time, I would fine-tune on in-domain 8 kHz telephony and far-field speech, calibrate confidence separately by locale, evaluate performance across accents and languages, and investigate ONNX Runtime/int8 inference.

For 1,000 concurrent calls, I would keep the API tier stateless and place a bounded inference pool behind it. Horizontal replicas could scale based on CPU and queue pressure, while GPU workers with dynamic batching could handle bursts. Explicit backpressure would prevent unbounded queues and allow the service to return `unknown` quickly when inference capacity is exhausted.
