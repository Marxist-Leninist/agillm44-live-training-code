import math
import warnings

import pytest
import torch
import torch.nn.functional as F

from agillm_attention import (
    MaskSpec,
    attention,
    build_dense_mask,
    flex_attention_available,
    select_backend,
)


def _tensors(*, batch=2, heads=3, q_len=8, kv_len=None, dim=4, dtype=torch.float64):
    kv_len = q_len if kv_len is None else kv_len
    generator = torch.Generator(device="cpu").manual_seed(20260911)
    q = torch.randn(batch, heads, q_len, dim, generator=generator, dtype=dtype)
    k = torch.randn(batch, heads, kv_len, dim, generator=generator, dtype=dtype)
    v = torch.randn(batch, heads, kv_len, dim, generator=generator, dtype=dtype)
    return q, k, v


def _run_with_grads(fn, q, k, v):
    q = q.clone().requires_grad_(True)
    k = k.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)
    out = fn(q, k, v)
    weight = torch.linspace(
        -0.8, 1.1, out.numel(), dtype=out.dtype, device=out.device
    ).reshape_as(out)
    loss = (out * weight).sum() + 0.03 * out.square().sum()
    grads = torch.autograd.grad(loss, (q, k, v))
    return out.detach(), tuple(g.detach() for g in grads)


def _assert_forward_backward_close(lhs, rhs, atol=1e-9, rtol=1e-8):
    out_a, grads_a = lhs
    out_b, grads_b = rhs
    torch.testing.assert_close(out_a, out_b, atol=atol, rtol=rtol)
    for grad_a, grad_b in zip(grads_a, grads_b):
        torch.testing.assert_close(grad_a, grad_b, atol=atol, rtol=rtol)


def test_mask_spec_validation():
    assert MaskSpec("SAT", 8, block_size=2).kind == "sat"
    assert MaskSpec("causal", 3).kv_len == 3
    with pytest.raises(ValueError):
        MaskSpec("invented", 8)
    with pytest.raises(ValueError):
        MaskSpec("sat", 8, block_size=0)
    with pytest.raises(ValueError):
        MaskSpec("causal", -1)


def test_fixed_sat_dense_rule():
    spec = MaskSpec("sat", q_len=6, kv_len=6, block_size=2)
    mask = build_dense_mask(spec, device=torch.device("cpu"), dtype=torch.float32)
    allow = torch.isfinite(mask[0, 0])
    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(allow, expected)


def test_cached_causal_dense_rule():
    spec = MaskSpec("causal", q_len=2, kv_len=6, query_base=4)
    mask = build_dense_mask(spec, device=torch.device("cpu"), dtype=torch.float32)
    allow = torch.isfinite(mask[0, 0])
    expected = torch.tensor(
        [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]],
        dtype=torch.bool,
    )
    torch.testing.assert_close(allow, expected)


def test_variable_sat_dense_rule_per_batch():
    spec = MaskSpec("variable_sat", q_len=5, kv_len=5)
    q_ids = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 1, 2, 2]])
    kv_ids = q_ids.clone()
    mask = build_dense_mask(
        spec,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch=2,
        q_block_ids=q_ids,
        kv_block_ids=kv_ids,
    )
    allow = torch.isfinite(mask[:, 0])
    for batch in range(2):
        expected = kv_ids[batch][None, :] <= q_ids[batch][:, None]
        torch.testing.assert_close(allow[batch], expected)


def test_native_causal_route_and_forward_backward_parity():
    q, k, v = _tensors()
    spec = MaskSpec("causal", q_len=8)
    scale = 1.0 / math.sqrt(q.size(-1))

    decision = select_backend(q, k, v, mask_spec=spec)
    assert decision.backend == "native_causal"

    actual = _run_with_grads(
        lambda q_, k_, v_: attention(
            q_, k_, v_, mask_spec=spec, scale=scale
        ),
        q,
        k,
        v,
    )
    dense = build_dense_mask(spec, device=q.device, dtype=q.dtype)
    reference = _run_with_grads(
        lambda q_, k_, v_: F.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=dense, dropout_p=0.0, scale=scale
        ),
        q,
        k,
        v,
    )
    _assert_forward_backward_close(actual, reference)


def test_cached_causal_uses_dense_and_matches_reference():
    q, k, v = _tensors(q_len=3, kv_len=8)
    spec = MaskSpec("causal", q_len=3, kv_len=8, query_base=5)
    scale = 0.37

    decision = select_backend(q, k, v, mask_spec=spec, prefer_flex=False)
    assert decision.backend == "dense_sdpa"

    dense = build_dense_mask(spec, device=q.device, dtype=q.dtype)
    actual = _run_with_grads(
        lambda q_, k_, v_: attention(
            q_,
            k_,
            v_,
            mask_spec=spec,
            scale=scale,
            prefer_flex=False,
        ),
        q,
        k,
        v,
    )
    reference = _run_with_grads(
        lambda q_, k_, v_: F.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=dense, dropout_p=0.0, scale=scale
        ),
        q,
        k,
        v,
    )
    _assert_forward_backward_close(actual, reference)


@pytest.mark.parametrize("block_size", [1, 2, 3, 4])
def test_fixed_sat_dense_fallback_forward_backward(block_size):
    q, k, v = _tensors(q_len=9)
    spec = MaskSpec("sat", q_len=9, block_size=block_size)
    scale = 0.29
    dense = build_dense_mask(spec, device=q.device, dtype=q.dtype)

    actual = _run_with_grads(
        lambda q_, k_, v_: attention(
            q_,
            k_,
            v_,
            mask_spec=spec,
            scale=scale,
            prefer_flex=False,
        ),
        q,
        k,
        v,
    )
    reference = _run_with_grads(
        lambda q_, k_, v_: F.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=dense, dropout_p=0.0, scale=scale
        ),
        q,
        k,
        v,
    )
    _assert_forward_backward_close(actual, reference)


def test_variable_sat_dense_fallback_forward_backward():
    q, k, v = _tensors(batch=2, q_len=7)
    q_ids = torch.tensor(
        [[0, 0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 1, 1, 2]]
    )
    kv_ids = q_ids.clone()
    spec = MaskSpec("variable_sat", q_len=7)
    scale = 0.41
    dense = build_dense_mask(
        spec,
        device=q.device,
        dtype=q.dtype,
        batch=2,
        q_block_ids=q_ids,
        kv_block_ids=kv_ids,
    )

    actual = _run_with_grads(
        lambda q_, k_, v_: attention(
            q_,
            k_,
            v_,
            mask_spec=spec,
            q_block_ids=q_ids,
            kv_block_ids=kv_ids,
            scale=scale,
            prefer_flex=False,
        ),
        q,
        k,
        v,
    )
    reference = _run_with_grads(
        lambda q_, k_, v_: F.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=dense, dropout_p=0.0, scale=scale
        ),
        q,
        k,
        v,
    )
    _assert_forward_backward_close(actual, reference)


def test_unrestricted_nat_route_matches_unmasked_sdpa():
    q, k, v = _tensors(q_len=5)
    spec = MaskSpec("nat", q_len=5)
    out, decision = attention(
        q, k, v, mask_spec=spec, return_decision=True
    )
    assert decision.backend == "unmasked_sdpa"
    ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    torch.testing.assert_close(out, ref)


def test_score_bias_forces_dense_and_is_preserved():
    q, k, v = _tensors(q_len=6)
    spec = MaskSpec("causal", q_len=6)
    bias = torch.linspace(0.0, -0.5, 6, dtype=q.dtype).view(1, 1, 1, 6)
    out, decision = attention(
        q,
        k,
        v,
        mask_spec=spec,
        score_bias=bias,
        return_decision=True,
    )
    assert decision.backend == "dense_sdpa"
    dense = build_dense_mask(spec, device=q.device, dtype=q.dtype)
    ref = F.scaled_dot_product_attention(
        q, k, v, attn_mask=dense + bias, dropout_p=0.0
    )
    torch.testing.assert_close(out, ref)


def test_forced_native_rejects_ineligible_cached_call():
    q, k, v = _tensors(q_len=2, kv_len=6)
    spec = MaskSpec("causal", q_len=2, kv_len=6, query_base=4)
    with pytest.raises(ValueError):
        attention(q, k, v, mask_spec=spec, force_backend="native_causal")


@pytest.mark.skipif(not flex_attention_available(), reason="FlexAttention unavailable")
def test_flex_fixed_sat_cpu_debug_forward_backward_parity():
    # CPU direct FlexAttention is deliberately unfused and used only to validate
    # the mask semantics and autograd before the CUDA benchmark.
    q, k, v = _tensors(
        batch=1, heads=2, q_len=9, dim=8, dtype=torch.float32
    )
    spec = MaskSpec("sat", q_len=9, block_size=2)
    scale = 1.0 / math.sqrt(8)
    dense = build_dense_mask(spec, device=q.device, dtype=q.dtype)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = _run_with_grads(
            lambda q_, k_, v_: attention(
                q_,
                k_,
                v_,
                mask_spec=spec,
                scale=scale,
                force_backend="flex",
                compile_flex=False,
            ),
            q,
            k,
            v,
        )
    reference = _run_with_grads(
        lambda q_, k_, v_: F.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=dense, dropout_p=0.0, scale=scale
        ),
        q,
        k,
        v,
    )
    _assert_forward_backward_close(actual, reference, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not flex_attention_available(), reason="FlexAttention unavailable")
def test_flex_variable_sat_cpu_debug_forward_parity():
    q, k, v = _tensors(
        batch=2, heads=1, q_len=7, dim=8, dtype=torch.float32
    )
    ids = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 2], [0, 0, 0, 1, 1, 1, 2]]
    )
    spec = MaskSpec("variable_sat", q_len=7)
    dense = build_dense_mask(
        spec,
        device=q.device,
        dtype=q.dtype,
        batch=2,
        q_block_ids=ids,
        kv_block_ids=ids,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = attention(
            q,
            k,
            v,
            mask_spec=spec,
            q_block_ids=ids,
            kv_block_ids=ids,
            force_backend="flex",
            compile_flex=False,
        )
    ref = F.scaled_dot_product_attention(
        q, k, v, attn_mask=dense, dropout_p=0.0
    )
    torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)


def test_boolean_attention_mask_preserves_sdpa_semantics():
    q, k, v = _tensors(batch=1, heads=2, q_len=5, dtype=torch.float64)
    allow = torch.tril(torch.ones(5, 5, dtype=torch.bool)).view(1, 1, 5, 5)
    scale = 0.31

    actual = _run_with_grads(
        lambda q_, k_, v_: attention(
            q_, k_, v_, attn_mask=allow, scale=scale
        ),
        q,
        k,
        v,
    )
    reference = _run_with_grads(
        lambda q_, k_, v_: F.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=allow, dropout_p=0.0, scale=scale
        ),
        q,
        k,
        v,
    )
    _assert_forward_backward_close(actual, reference)


def test_boolean_mask_plus_score_bias_is_combined_additively():
    q, k, v = _tensors(batch=1, heads=2, q_len=5, dtype=torch.float64)
    allow = torch.tril(torch.ones(5, 5, dtype=torch.bool)).view(1, 1, 5, 5)
    score_bias = torch.linspace(-0.4, 0.3, 25, dtype=q.dtype).view(1, 1, 5, 5)
    additive = torch.zeros_like(allow, dtype=q.dtype).masked_fill(~allow, float("-inf"))

    actual = _run_with_grads(
        lambda q_, k_, v_: attention(
            q_, k_, v_, attn_mask=allow, score_bias=score_bias
        ),
        q,
        k,
        v,
    )
    reference = _run_with_grads(
        lambda q_, k_, v_: F.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=additive + score_bias, dropout_p=0.0
        ),
        q,
        k,
        v,
    )
    _assert_forward_backward_close(actual, reference)
