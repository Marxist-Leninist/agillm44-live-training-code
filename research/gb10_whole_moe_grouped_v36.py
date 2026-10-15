#!/usr/bin/env python3
"""V36: broad-coverage grouped-MoE training benchmark for GB10/SM121.

This is deliberately isolated from the production trainer.  It tests the lane that
can plausibly matter at the whole-update gate: collapse the two routed experts'
up projection, activation, down projection, dX and dW into grouped GEMMs.

The benchmark reports both:
  1. pre-packed expert execution, and
  2. dispatch-inclusive execution (sort -> grouped expert -> inverse permutation).

The reference is the ordinary per-expert BF16 loop.  No sparse/NVFP4 path is used
here: the first question is whether broad grouped execution has enough wall-clock
coverage to justify a precision-specific SM12x kernel afterwards.

Default AGILLM4.4 routed expert shape:
    E=2, d_model=1280, d_ff=5120, top-k=1, ReLU

PyTorch grouped_mm uses the same three-GEMM training identity for each linear:
    y      = grouped_mm(x, W^T)
    dX     = grouped_mm(dY, W)
    dW     = grouped_mm(dY^T, X)

Nothing in this file mutates checkpoints or production state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.amp import custom_bwd, custom_fwd


DENSE_48_S = 83.54877431201749
SPARSE_V11_48_S = 84.56960714003071
STRICT_SPEEDUP = 1.15


def grouped_mm(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    public = getattr(F, "grouped_mm", None)
    if public is not None:
        return public(a, b, offs=offs, bias=None)
    private = getattr(torch, "_grouped_mm", None)
    if private is not None:
        return private(a, b, offs=offs)
    raise RuntimeError(
        "No grouped_mm primitive is available. This lane requires a PyTorch build "
        "with torch.nn.functional.grouped_mm or torch._grouped_mm."
    )


class GroupedMM(torch.autograd.Function):
    """Autograd wrapper with explicit grouped forward, dX, and dW GEMMs.

    W is stored in normal expert-linear layout [E, out_features, in_features].
    The 2-D/2-D grouped_mm in dW partitions the shared reduction dimension using
    the same CUDA int32 offsets and returns one [out_features, in_features] matrix
    per expert.
    """

    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor):
        ctx.save_for_backward(x, w, offs)
        return grouped_mm(x, w.transpose(-2, -1), offs)

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_y: torch.Tensor):
        grad_y = grad_y.contiguous()
        x, w, offs = ctx.saved_tensors
        grad_x = grad_w = None
        if ctx.needs_input_grad[0]:
            grad_x = grouped_mm(grad_y, w, offs)
        if ctx.needs_input_grad[1]:
            grad_w = grouped_mm(grad_y.transpose(0, 1), x, offs)
        return grad_x, grad_w, None


def activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    raise ValueError(name)


def _ends(lengths: list[int]) -> list[int]:
    out = []
    total = 0
    for value in lengths:
        total += int(value)
        out.append(total)
    return out


def sequential_packed(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    lengths: list[int],
    act: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    pieces = []
    start = 0
    for expert, end in enumerate(_ends(lengths)):
        h = F.linear(x[start:end], w_up[expert])
        pieces.append(F.linear(act(h), w_down[expert]))
        start = end
    return torch.cat(pieces, dim=0)


def grouped_packed(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    offs: torch.Tensor,
    act: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    h = GroupedMM.apply(x, w_up, offs)
    return GroupedMM.apply(act(h), w_down, offs)


def sequential_dispatch(
    x: torch.Tensor,
    route: torch.Tensor,
    gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    experts: int,
    act: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    out = x.new_zeros((x.size(0), w_down.size(1)))
    for expert in range(experts):
        idx = torch.nonzero(route == expert, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        h = F.linear(x.index_select(0, idx), w_up[expert])
        y = F.linear(act(h), w_down[expert])
        y = y * gate.index_select(0, idx).unsqueeze(-1)
        out = out.index_copy(0, idx, y)
    return out


def grouped_dispatch(
    x: torch.Tensor,
    order: torch.Tensor,
    inverse: torch.Tensor,
    gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    offs: torch.Tensor,
    act: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    packed = x.index_select(0, order)
    h = GroupedMM.apply(packed, w_up, offs)
    y = GroupedMM.apply(act(h), w_down, offs)
    y = y * gate.index_select(0, order).unsqueeze(-1)
    return y.index_select(0, inverse)


def reset_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def one_backward(
    fn: Callable[[], torch.Tensor],
    tensors: tuple[torch.Tensor, ...],
    grad_seed: torch.Tensor,
) -> torch.Tensor:
    reset_grads(*tensors)
    out = fn()
    (out * grad_seed).sum().backward()
    return out


def tensor_delta(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.detach()
    b = b.detach()
    diff = (a.float() - b.float()).abs()
    denom = b.float().abs().clamp_min(1e-12)
    return {
        "shape": list(a.shape),
        "exact_fraction": float((a == b).float().mean().item()),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_rel": float((diff / denom).max().item()) if diff.numel() else 0.0,
    }


def pct(samples: list[float], q: float) -> float:
    if not samples:
        return float("nan")
    values = torch.tensor(samples, dtype=torch.float64)
    return float(torch.quantile(values, q).item())


def timing_stats(samples: list[float]) -> dict:
    return {
        "n": len(samples),
        "min_ms": min(samples),
        "p10_ms": pct(samples, 0.10),
        "p20_ms": pct(samples, 0.20),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p80_ms": pct(samples, 0.80),
        "p90_ms": pct(samples, 0.90),
    }


def time_backward(
    fn: Callable[[], torch.Tensor],
    tensors: tuple[torch.Tensor, ...],
    grad_seed: torch.Tensor,
    warmup: int,
    samples: int,
) -> list[float]:
    for _ in range(warmup):
        one_backward(fn, tensors, grad_seed)
    torch.cuda.synchronize()
    out: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        reset_grads(*tensors)
        start.record()
        value = fn()
        (value * grad_seed).sum().backward()
        end.record()
        end.synchronize()
        out.append(float(start.elapsed_time(end)))
    return out


def minimum_component_fraction(local_speedup: float, target_speedup: float) -> float | None:
    if not math.isfinite(local_speedup) or local_speedup <= 1.0:
        return None
    numerator = 1.0 - 1.0 / target_speedup
    denominator = 1.0 - 1.0 / local_speedup
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=1280)
    parser.add_argument("--d-ff", type=int, default=5120)
    parser.add_argument(
        "--tokens-per-expert",
        type=str,
        default="128,128",
        help="Comma-separated routed token counts; count must equal --experts.",
    )
    parser.add_argument("--activation", choices=("relu", "silu"), default="relu")
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=360915)
    parser.add_argument("--out", type=Path, default=Path("v36_grouped_moe_receipt.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("V36 requires CUDA")
    lengths = [int(v) for v in args.tokens_per_expert.split(",") if v.strip()]
    if len(lengths) != args.experts or any(v <= 0 for v in lengths):
        raise ValueError("--tokens-per-expert must contain one positive count per expert")
    if args.experts < 2:
        raise ValueError("This benchmark is intended to test grouped multi-expert execution")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    act = activation(args.activation)
    total = sum(lengths)

    # Pre-packed tensors used to isolate expert execution.
    x0 = torch.randn(total, args.d_model, device=device, dtype=torch.bfloat16)
    up0 = torch.randn(
        args.experts, args.d_ff, args.d_model, device=device, dtype=torch.bfloat16
    ) / math.sqrt(args.d_model)
    down0 = torch.randn(
        args.experts, args.d_model, args.d_ff, device=device, dtype=torch.bfloat16
    ) / math.sqrt(args.d_ff)
    offs = torch.tensor(_ends(lengths), device=device, dtype=torch.int32)

    # Dispatch-inclusive case starts deliberately ungrouped.
    route_sorted = torch.repeat_interleave(
        torch.arange(args.experts, device=device, dtype=torch.long),
        torch.tensor(lengths, device=device, dtype=torch.long),
    )
    shuffle = torch.randperm(total, device=device)
    route = route_sorted.index_select(0, shuffle)
    x_dispatch0 = x0.index_select(0, shuffle).detach().clone()
    gate = (0.25 + 0.75 * torch.rand(total, device=device, dtype=torch.float32)).to(
        torch.bfloat16
    )
    order = torch.argsort(route, stable=True)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(total, device=device)

    # Exactness / numerical-equivalence census for pre-packed path.
    x_s = x0.detach().clone().requires_grad_(True)
    up_s = up0.detach().clone().requires_grad_(True)
    down_s = down0.detach().clone().requires_grad_(True)
    x_g = x0.detach().clone().requires_grad_(True)
    up_g = up0.detach().clone().requires_grad_(True)
    down_g = down0.detach().clone().requires_grad_(True)
    grad_seed = torch.randn(total, args.d_model, device=device, dtype=torch.bfloat16)

    out_s = one_backward(
        lambda: sequential_packed(x_s, up_s, down_s, lengths, act),
        (x_s, up_s, down_s),
        grad_seed,
    )
    out_g = one_backward(
        lambda: grouped_packed(x_g, up_g, down_g, offs, act),
        (x_g, up_g, down_g),
        grad_seed,
    )
    torch.cuda.synchronize()
    packed_equivalence = {
        "output": tensor_delta(out_g, out_s),
        "dx": tensor_delta(x_g.grad, x_s.grad),
        "dweight_up": tensor_delta(up_g.grad, up_s.grad),
        "dweight_down": tensor_delta(down_g.grad, down_s.grad),
    }

    # Exactness / numerical-equivalence census including routing pack/unpack.
    xd_s = x_dispatch0.detach().clone().requires_grad_(True)
    upd_s = up0.detach().clone().requires_grad_(True)
    downd_s = down0.detach().clone().requires_grad_(True)
    xd_g = x_dispatch0.detach().clone().requires_grad_(True)
    upd_g = up0.detach().clone().requires_grad_(True)
    downd_g = down0.detach().clone().requires_grad_(True)
    dispatch_grad_seed = torch.randn(total, args.d_model, device=device, dtype=torch.bfloat16)
    dout_s = one_backward(
        lambda: sequential_dispatch(
            xd_s, route, gate, upd_s, downd_s, args.experts, act
        ),
        (xd_s, upd_s, downd_s),
        dispatch_grad_seed,
    )
    dout_g = one_backward(
        lambda: grouped_dispatch(
            xd_g, order, inverse, gate, upd_g, downd_g, offs, act
        ),
        (xd_g, upd_g, downd_g),
        dispatch_grad_seed,
    )
    torch.cuda.synchronize()
    dispatch_equivalence = {
        "output": tensor_delta(dout_g, dout_s),
        "dx": tensor_delta(xd_g.grad, xd_s.grad),
        "dweight_up": tensor_delta(upd_g.grad, upd_s.grad),
        "dweight_down": tensor_delta(downd_g.grad, downd_s.grad),
    }

    # Timings use fresh leaves but identical numerical values.
    def leaves(x_base: torch.Tensor):
        return x_base.detach().clone().requires_grad_(True)

    p_xs, p_us, p_ds = leaves(x0), leaves(up0), leaves(down0)
    p_xg, p_ug, p_dg = leaves(x0), leaves(up0), leaves(down0)
    packed_seq_ms = time_backward(
        lambda: sequential_packed(p_xs, p_us, p_ds, lengths, act),
        (p_xs, p_us, p_ds),
        grad_seed,
        args.warmup,
        args.samples,
    )
    packed_group_ms = time_backward(
        lambda: grouped_packed(p_xg, p_ug, p_dg, offs, act),
        (p_xg, p_ug, p_dg),
        grad_seed,
        args.warmup,
        args.samples,
    )

    d_xs, d_us, d_ds = leaves(x_dispatch0), leaves(up0), leaves(down0)
    d_xg, d_ug, d_dg = leaves(x_dispatch0), leaves(up0), leaves(down0)
    dispatch_seq_ms = time_backward(
        lambda: sequential_dispatch(
            d_xs, route, gate, d_us, d_ds, args.experts, act
        ),
        (d_xs, d_us, d_ds),
        dispatch_grad_seed,
        args.warmup,
        args.samples,
    )
    dispatch_group_ms = time_backward(
        lambda: grouped_dispatch(
            d_xg, order, inverse, gate, d_ug, d_dg, offs, act
        ),
        (d_xg, d_ug, d_dg),
        dispatch_grad_seed,
        args.warmup,
        args.samples,
    )

    p_seq = timing_stats(packed_seq_ms)
    p_grp = timing_stats(packed_group_ms)
    d_seq = timing_stats(dispatch_seq_ms)
    d_grp = timing_stats(dispatch_group_ms)
    packed_speedup_p20 = p_seq["p20_ms"] / p_grp["p20_ms"]
    dispatch_speedup_p20 = d_seq["p20_ms"] / d_grp["p20_ms"]

    strict_target_s = DENSE_48_S / STRICT_SPEEDUP
    receipt = {
        "schema": "agillm44.gb10.whole-moe-grouped-v36.v1",
        "production_mutated": False,
        "contract": {
            "scope": "two-routed-expert up+activation+down forward+dX+dW",
            "shared_expert_included": False,
            "precision": "bf16",
            "activation": args.activation,
            "top_k": 1,
            "parameter_layout": "[experts,out_features,in_features]",
            "grouped_backward": "three-grouped-gemm identity",
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "grouped_mm_public": hasattr(F, "grouped_mm"),
            "grouped_mm_private": hasattr(torch, "_grouped_mm"),
        },
        "shape": {
            "experts": args.experts,
            "d_model": args.d_model,
            "d_ff": args.d_ff,
            "tokens_per_expert": lengths,
            "total_routed_tokens": total,
        },
        "equivalence": {
            "prepacked": packed_equivalence,
            "dispatch_inclusive": dispatch_equivalence,
        },
        "timing_ms": {
            "prepacked_sequential": p_seq,
            "prepacked_grouped": p_grp,
            "dispatch_sequential": d_seq,
            "dispatch_grouped": d_grp,
        },
        "speedup": {
            "prepacked_p20": packed_speedup_p20,
            "prepacked_median": p_seq["median_ms"] / p_grp["median_ms"],
            "dispatch_p20": dispatch_speedup_p20,
            "dispatch_median": d_seq["median_ms"] / d_grp["median_ms"],
        },
        "whole_step_gate": {
            "dense_48_update_s": DENSE_48_S,
            "sparse_v11_48_update_s": SPARSE_V11_48_S,
            "required_speedup_vs_dense": STRICT_SPEEDUP,
            "target_48_update_s": strict_target_s,
            "required_reduction_from_sparse_v11_s": SPARSE_V11_48_S - strict_target_s,
            "required_reduction_per_update_ms":
                (SPARSE_V11_48_S - strict_target_s) * 1000.0 / 48.0,
            "minimum_dense_wall_fraction_at_observed_dispatch_p20_speedup":
                minimum_component_fraction(dispatch_speedup_p20, STRICT_SPEEDUP),
            "interpretation": (
                "Advance only if dispatch-inclusive grouped expert execution is faster and "
                "profiling shows routed expert execution owns at least the reported minimum "
                "dense wall fraction. Otherwise this lane cannot reach the 1.15x gate."
            ),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"[v36] wrote {args.out}")


if __name__ == "__main__":
    main()
