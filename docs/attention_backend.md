# AGILLM attention backend v1

This branch owns the attention dispatch used by AGILLM without forking all of
PyTorch. Native PyTorch SDPA remains the correctness reference. The project is
small enough to audit and specific enough to improve the model's real AR, SAT,
NAT and cached-decoding shapes.

## Current evidence

On the production-class RTX 3090 with PyTorch 2.2.1, an isolated
`B=1, H=20, T=2048, D=64, fp16` forward-plus-backward probe measured:

| Route | Mean time |
|---|---:|
| Dense additive causal mask through SDPA | 2.559 ms |
| Native causal SDPA, `attn_mask=None`, `is_causal=True` | 1.026 ms |

That is about a **2.49x improvement inside the isolated attention operation**.
It is not a claim of a 2.49x end-to-end trainer speedup. The end-to-end gain is
bounded by the fraction of each committed training step spent in attention.

The production host currently has PyTorch 2.2.1, which has no FlexAttention
API. Therefore v1 behaves as follows:

| Mask | PyTorch 2.2 route |
|---|---|
| Square, zero-offset causal | Native causal SDPA |
| Cached/non-square causal | Exact dense SDPA fallback |
| Fixed-block SAT | Exact dense SDPA fallback |
| Variable-block SAT | Exact dense SDPA fallback |
| NAT/unrestricted | Unmasked SDPA |
| Any structural mask plus ALiBi/score bias | Exact dense combined fallback |

On a separately pinned newer PyTorch environment, fixed and variable SAT may
use FlexAttention. That route is never silently selected on a build where the
API is unavailable.

## Design

`agillm_attention/backend.py` separates three concerns:

1. `MaskSpec` describes visibility without allocating a quadratic tensor.
2. `select_backend` makes a reversible, inspectable routing decision.
3. `attention` executes native causal SDPA, optional FlexAttention, or the
   exact dense fallback.

Boolean masks preserve PyTorch's `True = visible` semantics. Explicit score
biases such as ALiBi are never discarded merely to qualify a fused kernel.
Forward and backward correctness are checked against dense SDPA.

## Building a single-file runtime candidate

The live trainer is intentionally not edited in place. Generate an isolated
candidate from the checked-in retention runtime:

```bash
python tools/build_runtime_candidate.py \
  runtime_retain_activations_v2.py \
  runtime_agillm_attention_v1.py \
  --source-sha256 714ee71c440ba9e9699034d7ea5f1de6f73d65cc80caf9f9350cf30f11a33e28

python tools/verify_runtime_candidate.py \
  runtime_agillm_attention_v1.py \
  --receipt runtime_agillm_attention_v1.py.receipt.json
```

The builder folds the backend source into the generated file so the result
remains a genuine single-file trainer. It also changes the DBlock causal and
fixed-SAT mask producers to pass symbolic rules to the dispatcher. Changing
only the consumer would create a fast path that never receives a symbolic
mask.

The builder refuses in-place modification, requires exact patch counts,
compiles the complete result, writes atomically, and emits SHA256 provenance.
It does not transform a checkpoint, signal a process, edit a supervisor or
activate production.

## Tests

For a host without pytest:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python tests/run_cpu_smoke.py
```

For the full suite:

```bash
PYTHONPATH=. python -m pytest -q
```

For an isolated GPU window:

```bash
PYTHONPATH=. python benchmarks/bench_sdpa.py \
  --batch 1 --heads 20 --seq 2048 --dim 64 \
  --phase forward-backward --include-mask-build
```

Do not run the GPU benchmark beside the production trainer. Co-located probes
have already distorted throughput measurements and helped turn a valid kernel
experiment into a small administrative opera.

## Runtime controls

`AGILLM_ATTENTION_FORCE_BACKEND=dense_sdpa` forces the correctness fallback
without regenerating the candidate. `AGILLM_ATTENTION_PREFER_FLEX=0` disables
automatic FlexAttention selection while retaining native causal SDPA.

## Promotion gate

A candidate should not replace the live runtime until all of these hold:

- source and generated SHA256 values match the receipt;
- CPU forward/backward parity passes on the production PyTorch version;
- one isolated RTX 3090 benchmark passes correctness before timing;
- end-to-end committed-step throughput improves over an uncontended matched
  retention-v2 window;
- AR, fixed SAT, SAT-variable gate inputs, NAT, cached decoding and ALiBi
  semantics remain unchanged;
- a short continuation shows finite gradients and no held-out regression;
- only one checkpoint-safe cutover is performed, with an immediate rollback
  path and no autonomous restart loop.
