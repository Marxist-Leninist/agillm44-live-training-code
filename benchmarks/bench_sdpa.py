#!/usr/bin/env python3
"""Benchmark AGILLM attention routes on a CUDA device.

The default shape matches the isolated RTX 3090 qualification probe used for
AGILLM-4.4.  This script does not modify a trainer or checkpoint.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import statistics
import time
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from agillm_attention import MaskSpec, attention, build_dense_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=20)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--sat-block", type=int, default=2)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--phase", choices=("forward", "forward-backward"), default="forward-backward"
    )
    parser.add_argument(
        "--include-mask-build",
        action="store_true",
        help="also time dense mask materialization inside each call",
    )
    parser.add_argument(
        "--skip-flex",
        action="store_true",
        help="do not compile or run the optional SAT FlexAttention route",
    )
    return parser.parse_args()


def cuda_time_ms(fn: Callable[[], None], warmup: int, iters: int) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: List[float] = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))

    ordered = sorted(samples)
    p50 = statistics.median(ordered)
    p90 = ordered[min(len(ordered) - 1, max(0, math.ceil(0.90 * len(ordered)) - 1))]
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": p50,
        "p90_ms": p90,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    device = torch.device("cuda")
    torch.manual_seed(20260911)

    shape = (args.batch, args.heads, args.seq, args.dim)
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    scale = 1.0 / math.sqrt(args.dim)

    causal = MaskSpec("causal", q_len=args.seq)
    sat = MaskSpec("sat", q_len=args.seq, block_size=args.sat_block)
    causal_dense = build_dense_mask(causal, device=device, dtype=dtype)
    sat_dense = build_dense_mask(sat, device=device, dtype=dtype)

    def timed(call: Callable[[], torch.Tensor]) -> Callable[[], None]:
        if args.phase == "forward":
            def run_forward() -> None:
                with torch.no_grad():
                    call()
            return run_forward

        def run_forward_backward() -> None:
            for tensor in (q, k, v):
                tensor.grad = None
            out = call()
            out.float().square().mean().backward()
        return run_forward_backward

    calls: Dict[str, Callable[[], torch.Tensor]] = {
        "causal_dense_sdpa_reused_mask": lambda: F.scaled_dot_product_attention(
            q, k, v, attn_mask=causal_dense, dropout_p=0.0, scale=scale
        ),
        "causal_native_sdpa": lambda: attention(
            q, k, v, mask_spec=causal, scale=scale, prefer_flex=False
        ),
        "sat_dense_sdpa_reused_mask": lambda: F.scaled_dot_product_attention(
            q, k, v, attn_mask=sat_dense, dropout_p=0.0, scale=scale
        ),
    }

    if args.include_mask_build:
        calls["causal_dense_sdpa_rebuild_mask"] = lambda: attention(
            q,
            k,
            v,
            mask_spec=causal,
            scale=scale,
            prefer_flex=False,
            force_backend="dense_sdpa",
        )
        calls["sat_dense_sdpa_rebuild_mask"] = lambda: attention(
            q,
            k,
            v,
            mask_spec=sat,
            scale=scale,
            prefer_flex=False,
            force_backend="dense_sdpa",
        )

    if not args.skip_flex:
        calls["sat_flex_attention"] = lambda: attention(
            q,
            k,
            v,
            mask_spec=sat,
            scale=scale,
            force_backend="flex",
            compile_flex=True,
        )

    # Correctness before timing.  Flex compilation also happens here, outside
    # measured iterations.  Different fused kernels need not be bitwise equal.
    reference_causal = calls["causal_dense_sdpa_reused_mask"]().detach()
    reference_sat = calls["sat_dense_sdpa_reused_mask"]().detach()
    correctness: Dict[str, Dict[str, object]] = {}
    runnable: Dict[str, Callable[[], torch.Tensor]] = {}
    for name, call in calls.items():
        try:
            out = call().detach()
            ref = reference_sat if name.startswith("sat_") else reference_causal
            diff = (out.float() - ref.float()).abs()
            correctness[name] = {
                "ok": bool(torch.allclose(out.float(), ref.float(), atol=3e-3, rtol=3e-3)),
                "max_abs": float(diff.max()),
                "mean_abs": float(diff.mean()),
            }
            runnable[name] = call
        except Exception as exc:
            correctness[name] = {
                "ok": False,
                "error": "%s: %s" % (type(exc).__name__, exc),
            }

    results: Dict[str, Dict[str, float]] = {}
    for name, call in runnable.items():
        if not correctness[name]["ok"]:
            continue
        results[name] = cuda_time_ms(timed(call), args.warmup, args.iters)

    native_mean = results.get("causal_native_sdpa", {}).get("mean_ms")
    dense_mean = results.get("causal_dense_sdpa_reused_mask", {}).get("mean_ms")
    sat_flex_mean = results.get("sat_flex_attention", {}).get("mean_ms")
    sat_dense_mean = results.get("sat_dense_sdpa_reused_mask", {}).get("mean_ms")

    speedups: Dict[str, Optional[float]] = {
        "causal_native_vs_dense": (
            dense_mean / native_mean if dense_mean and native_mean else None
        ),
        "sat_flex_vs_dense": (
            sat_dense_mean / sat_flex_mean
            if sat_dense_mean and sat_flex_mean
            else None
        ),
    }

    report = {
        "schema": "agillm.attention.benchmark.v1",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "shape": {
            "batch": args.batch,
            "heads": args.heads,
            "seq": args.seq,
            "dim": args.dim,
            "dtype": args.dtype,
            "sat_block": args.sat_block,
        },
        "phase": args.phase,
        "warmup": args.warmup,
        "iters": args.iters,
        "correctness": correctness,
        "timings": results,
        "speedups": speedups,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
