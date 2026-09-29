#!/usr/bin/env python3
"""Dependency-free CPU correctness smoke test for PyTorch 2.2+ hosts."""

from __future__ import annotations

import json
import math
import sys
from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F

from agillm_attention import MaskSpec, attention, build_dense_mask, select_backend


def tensors(q_len: int, kv_len: int | None = None) -> Tuple[torch.Tensor, ...]:
    kv_len = q_len if kv_len is None else kv_len
    generator = torch.Generator(device="cpu").manual_seed(20260911)
    shape_q = (2, 3, q_len, 8)
    shape_kv = (2, 3, kv_len, 8)
    return (
        torch.randn(shape_q, generator=generator, dtype=torch.float64),
        torch.randn(shape_kv, generator=generator, dtype=torch.float64),
        torch.randn(shape_kv, generator=generator, dtype=torch.float64),
    )


def run_grads(fn: Callable[..., torch.Tensor], values: Tuple[torch.Tensor, ...]):
    q, k, v = (item.clone().requires_grad_(True) for item in values)
    out = fn(q, k, v)
    weights = torch.linspace(-0.5, 0.7, out.numel(), dtype=out.dtype).reshape_as(out)
    loss = (out * weights).sum() + 0.01 * out.square().sum()
    grads = torch.autograd.grad(loss, (q, k, v))
    return out.detach(), tuple(item.detach() for item in grads)


def assert_same(a, b, name: str) -> Dict[str, float | str | bool]:
    out_a, grad_a = a
    out_b, grad_b = b
    torch.testing.assert_close(out_a, out_b, atol=1e-9, rtol=1e-8)
    for lhs, rhs in zip(grad_a, grad_b):
        torch.testing.assert_close(lhs, rhs, atol=1e-9, rtol=1e-8)
    return {
        "name": name,
        "ok": True,
        "max_output_abs": float((out_a - out_b).abs().max()),
        "max_grad_abs": max(float((x - y).abs().max()) for x, y in zip(grad_a, grad_b)),
    }


def main() -> None:
    results = []
    scale = 1.0 / math.sqrt(8)

    values = tensors(9)
    spec = MaskSpec("causal", 9)
    dense = build_dense_mask(spec, device=torch.device("cpu"), dtype=torch.float64)
    actual = run_grads(
        lambda q, k, v: attention(q, k, v, mask_spec=spec, scale=scale),
        values,
    )
    reference = run_grads(
        lambda q, k, v: F.scaled_dot_product_attention(
            q, k, v, attn_mask=dense, dropout_p=0.0, scale=scale
        ),
        values,
    )
    result = assert_same(actual, reference, "native_causal")
    result["route"] = select_backend(*values, mask_spec=spec).backend
    results.append(result)

    values = tensors(9)
    spec = MaskSpec("sat", 9, block_size=2)
    dense = build_dense_mask(spec, device=torch.device("cpu"), dtype=torch.float64)
    results.append(assert_same(
        run_grads(
            lambda q, k, v: attention(
                q, k, v, mask_spec=spec, scale=scale, prefer_flex=False
            ),
            values,
        ),
        run_grads(
            lambda q, k, v: F.scaled_dot_product_attention(
                q, k, v, attn_mask=dense, dropout_p=0.0, scale=scale
            ),
            values,
        ),
        "fixed_sat_dense_fallback",
    ))

    values = tensors(3, 8)
    spec = MaskSpec("causal", 3, kv_len=8, query_base=5)
    dense = build_dense_mask(spec, device=torch.device("cpu"), dtype=torch.float64)
    results.append(assert_same(
        run_grads(
            lambda q, k, v: attention(
                q, k, v, mask_spec=spec, scale=scale, prefer_flex=False
            ),
            values,
        ),
        run_grads(
            lambda q, k, v: F.scaled_dot_product_attention(
                q, k, v, attn_mask=dense, dropout_p=0.0, scale=scale
            ),
            values,
        ),
        "cached_causal_dense_fallback",
    ))

    values = tensors(7)
    ids = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 2], [0, 0, 0, 1, 1, 2, 3]],
        dtype=torch.long,
    )
    spec = MaskSpec("variable_sat", 7)
    dense = build_dense_mask(
        spec,
        device=torch.device("cpu"),
        dtype=torch.float64,
        batch=2,
        q_block_ids=ids,
        kv_block_ids=ids,
    )
    results.append(assert_same(
        run_grads(
            lambda q, k, v: attention(
                q,
                k,
                v,
                mask_spec=spec,
                q_block_ids=ids,
                kv_block_ids=ids,
                scale=scale,
                prefer_flex=False,
            ),
            values,
        ),
        run_grads(
            lambda q, k, v: F.scaled_dot_product_attention(
                q, k, v, attn_mask=dense, dropout_p=0.0, scale=scale
            ),
            values,
        ),
        "variable_sat_dense_fallback",
    ))

    report = {
        "schema": "agillm.attention.cpu-smoke.v1",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "tests": results,
        "passed": all(bool(item["ok"]) for item in results),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
