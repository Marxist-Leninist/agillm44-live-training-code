# AGILLM attention backend validation

Validated on 11 September 2026 against the exact retained AGILLM-4.4 runtime snapshot.

## Scope

This branch introduces an isolated attention backend and a deterministic builder that folds it into a generated single-file runtime. It does **not** modify checkpoints, retarget the canonical runtime, restart the trainer, or alter supervisor/watchdog state.

The implementation routes:

- square, zero-offset causal attention with no additive bias to native SDPA using `is_causal=True` and no dense mask;
- unrestricted/NAT attention to ordinary unmasked SDPA;
- fixed-block SAT and variable-block SAT to FlexAttention when a compatible CUDA/PyTorch runtime is available;
- every unsupported or explicitly disabled case to an exact dense-mask SDPA fallback.

Variable SAT uses the supplied per-token block IDs. It is not approximated as fixed-stride SAT. Explicit AGILLM score scaling and additive bias semantics are retained.

## Source and generated candidate

| Item | SHA-256 |
| --- | --- |
| Retention-v2 source runtime | `714ee71c440ba9e9699034d7ea5f1de6f73d65cc80caf9f9350cf30f11a33e28` |
| Folded attention backend source | `0c3bc5c40b54dce70a3b2063a6a082116d2a3a500862480b422f438f31ff4892` |
| Generated single-file candidate | `ca583d724ae859d2120255f1c1b76e9dc6c6f4ba08f9894d88630f95e7e7d26e` |

Generated candidate path during validation:

```text
/workspace/agillm_attention_v1_validation/evidence/runtime_agillm_attention_v1.py
```

The builder refuses in-place modification and requires the expected source SHA before emitting a candidate.

## Correctness evidence

### Full CPU suite

The backend test suite completed with **18 passing tests**. It covers:

- native causal forward and backward parity;
- cached/non-square causal fallback;
- fixed SAT;
- variable SAT with actual block assignments;
- unrestricted/NAT attention;
- additive score bias and ALiBi-style bias;
- boolean visibility masks;
- explicit scale handling;
- forced rollback/backend selection;
- generated-runtime structure and receipt verification.

### Production-host compatibility

A CPU-only smoke test ran under the production host's installed **PyTorch 2.2.1** with CUDA hidden. Native causal, fixed SAT fallback, cached causal fallback, and variable SAT fallback all passed.

For native causal versus the previous dense additive mask on float64 CPU:

| Measurement | Maximum absolute difference |
| --- | ---: |
| Output | `6.661338147750939e-16` |
| Gradients | `5.551115123125783e-16` |

The folded backend was then extracted from the generated single-file candidate and executed independently under PyTorch 2.2.1. It selected `native_causal`, matched the dense reference within `6.661338147750939e-16`, and produced finite gradients for Q, K, and V.

### Runtime-builder verification

All structural checks passed:

- one folded backend copy;
- one attention-dispatch insertion;
- `TuneableAttentionMHA.forward` delegates through the AGILLM backend;
- no direct SDPA call remains in that model attention method;
- DBlock causal producers use the symbolic causal helper;
- DBlock SAT producers use the symbolic SAT helper;
- generated candidate SHA matches its receipt.

## Performance evidence and limits

On the real RTX 3090 workload shape, the earlier isolated forward-plus-backward probe measured approximately:

| Path | Mean time |
| --- | ---: |
| Dense causal additive-mask SDPA | `2.559 ms` |
| Native causal SDPA | `1.026 ms` |

That is about **2.49x faster inside the isolated attention operation**. It is not an end-to-end trainer speedup claim.

No additional GPU benchmark was launched while the live trainer occupied the RTX 3090. Promotion still requires a conflict-free GPU window, repeated candidate-versus-baseline measurements, committed-step throughput comparison, and short checkpoint-safe continuation quality checks.

## Rollback controls

```text
AGILLM_ATTENTION_FORCE_BACKEND=dense_sdpa
AGILLM_ATTENTION_PREFER_FLEX=0
```

These controls keep the generated candidate reversible without changing checkpoint structure.