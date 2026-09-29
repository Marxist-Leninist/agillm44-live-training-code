"""AGILLM attention backend selection.

This module keeps model semantics explicit and makes optimized attention paths
reversible.  Native PyTorch SDPA remains the reference implementation.

Supported structured masks:
* ``causal``: token-causal attention.
* ``sat`` / ``block_causal``: fixed-size block-causal attention.
* ``variable_sat``: block-causal attention described by per-token block ids.
* ``none`` / ``nat`` / ``unrestricted``: no visibility restriction.

The production-safe path on PyTorch 2.2 is native causal SDPA.  FlexAttention
is optional and is only selected automatically on CUDA when it is available.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import os
import threading
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F


_UNRESTRICTED_KINDS = frozenset({"none", "nat", "bidirectional", "unrestricted"})
_CAUSAL_KINDS = frozenset({"causal"})
_FIXED_SAT_KINDS = frozenset({"sat", "block_causal", "block-causal"})
_VARIABLE_SAT_KINDS = frozenset(
    {"variable_sat", "sat_variable", "variable-block-causal"}
)
_ALL_KINDS = (
    _UNRESTRICTED_KINDS
    | _CAUSAL_KINDS
    | _FIXED_SAT_KINDS
    | _VARIABLE_SAT_KINDS
)


@dataclass(frozen=True)
class MaskSpec:
    """A compact description of token visibility.

    ``query_base`` is the absolute position represented by query index zero.
    It matters for cached decoding, where Q can be shorter than K/V.
    """

    kind: str
    q_len: int
    kv_len: Optional[int] = None
    query_base: int = 0
    block_size: int = 1

    def __post_init__(self) -> None:
        kind = str(self.kind or "none").strip().lower()
        if kind not in _ALL_KINDS:
            raise ValueError("unsupported attention mask kind: %s" % kind)
        q_len = int(self.q_len)
        kv_len = q_len if self.kv_len is None else int(self.kv_len)
        query_base = int(self.query_base)
        block_size = int(self.block_size)
        if q_len < 0 or kv_len < 0:
            raise ValueError("q_len and kv_len must be non-negative")
        if query_base < 0:
            raise ValueError("query_base must be non-negative")
        if block_size < 1:
            raise ValueError("block_size must be at least one")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "q_len", q_len)
        object.__setattr__(self, "kv_len", kv_len)
        object.__setattr__(self, "query_base", query_base)
        object.__setattr__(self, "block_size", block_size)


@dataclass(frozen=True)
class AttentionDecision:
    """The backend chosen for one attention call."""

    backend: str
    reason: str
    mask_kind: str
    q_len: int
    kv_len: int


def _shape_check(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> Tuple[int, int]:
    if query.ndim < 3 or key.ndim < 3 or value.ndim < 3:
        raise ValueError("query, key and value must have at least three dimensions")
    if query.shape[:-2] != key.shape[:-2] or key.shape[:-2] != value.shape[:-2]:
        raise ValueError(
            "query, key and value batch/head prefixes must match: %r, %r, %r"
            % (query.shape, key.shape, value.shape)
        )
    if query.size(-1) != key.size(-1):
        raise ValueError("query and key head dimensions must match")
    if key.size(-2) != value.size(-2):
        raise ValueError("key and value sequence lengths must match")
    return int(query.size(-2)), int(key.size(-2))


def _expand_block_ids(
    ids: torch.Tensor,
    batch: int,
    length: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    ids = torch.as_tensor(ids, device=device, dtype=torch.long)
    if ids.ndim == 1:
        if ids.numel() != length:
            raise ValueError("%s length mismatch" % name)
        ids = ids.unsqueeze(0).expand(batch, -1)
    elif ids.ndim == 2:
        if ids.shape != (batch, length):
            raise ValueError(
                "%s shape must be (%d, %d), got %r"
                % (name, batch, length, tuple(ids.shape))
            )
    else:
        raise ValueError("%s must be rank one or two" % name)
    return ids


def build_dense_mask(
    spec: MaskSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch: int = 1,
    q_block_ids: Optional[torch.Tensor] = None,
    kv_block_ids: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Materialize the exact additive mask used as the correctness fallback."""

    if spec.kind in _UNRESTRICTED_KINDS:
        return None

    if spec.kind in _VARIABLE_SAT_KINDS:
        if q_block_ids is None or kv_block_ids is None:
            raise ValueError("variable SAT requires q_block_ids and kv_block_ids")
        q_ids = _expand_block_ids(
            q_block_ids, batch, spec.q_len, device, "q_block_ids"
        )
        kv_ids = _expand_block_ids(
            kv_block_ids, batch, int(spec.kv_len), device, "kv_block_ids"
        )
        allow = kv_ids[:, None, :] <= q_ids[:, :, None]
        zeros = torch.zeros(
            (batch, 1, spec.q_len, int(spec.kv_len)),
            device=device,
            dtype=dtype,
        )
        return zeros.masked_fill(~allow[:, None, :, :], float("-inf"))

    q_pos = torch.arange(
        spec.query_base,
        spec.query_base + spec.q_len,
        device=device,
        dtype=torch.long,
    ).view(spec.q_len, 1)
    kv_pos = torch.arange(
        int(spec.kv_len), device=device, dtype=torch.long
    ).view(1, int(spec.kv_len))

    if spec.kind in _CAUSAL_KINDS:
        allow = kv_pos <= q_pos
    elif spec.kind in _FIXED_SAT_KINDS:
        allow = (kv_pos // spec.block_size) <= (q_pos // spec.block_size)
    else:  # guarded by MaskSpec validation
        raise AssertionError(spec.kind)

    zeros = torch.zeros(
        (1, 1, spec.q_len, int(spec.kv_len)),
        device=device,
        dtype=dtype,
    )
    return zeros.masked_fill(~allow.view(1, 1, spec.q_len, int(spec.kv_len)), float("-inf"))


def _call_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
) -> torch.Tensor:
    kwargs = {
        "attn_mask": attn_mask,
        "dropout_p": float(dropout_p),
        "is_causal": bool(is_causal),
    }
    if scale is None:
        return F.scaled_dot_product_attention(query, key, value, **kwargs)

    try:
        return F.scaled_dot_product_attention(
            query, key, value, scale=float(scale), **kwargs
        )
    except TypeError as exc:
        # PyTorch versions without ``scale=`` use 1/sqrt(query_dim).  Multiplying
        # Q preserves the requested score scale without changing gradients.
        if "scale" not in str(exc):
            raise
        q_scale = float(scale) * math.sqrt(float(query.size(-1)))
        return F.scaled_dot_product_attention(
            query * q_scale, key, value, **kwargs
        )


def _native_causal_eligible(
    spec: MaskSpec,
    q_len: int,
    kv_len: int,
    *,
    has_score_bias: bool,
) -> bool:
    return bool(
        spec.kind in _CAUSAL_KINDS
        and spec.query_base == 0
        and spec.q_len == q_len
        and int(spec.kv_len) == kv_len
        and q_len == kv_len
        and not has_score_bias
    )


_FLEX_IMPORT_LOCK = threading.Lock()
_FLEX_API: Optional[Tuple[Callable[..., Any], Callable[..., Any]]] = None
_FLEX_IMPORT_ATTEMPTED = False
_FLEX_COMPILED: Optional[Callable[..., torch.Tensor]] = None
_FLEX_COMPILE_LOCK = threading.Lock()

_BLOCK_MASK_CACHE: "OrderedDict[Tuple[Any, ...], Any]" = OrderedDict()
_BLOCK_MASK_CACHE_LOCK = threading.Lock()
_BLOCK_MASK_CACHE_MAX = max(
    1, int(os.environ.get("AGILLM_ATTENTION_BLOCK_MASK_CACHE", "32"))
)


def _load_flex_api() -> Optional[Tuple[Callable[..., Any], Callable[..., Any]]]:
    global _FLEX_API, _FLEX_IMPORT_ATTEMPTED
    if _FLEX_IMPORT_ATTEMPTED:
        return _FLEX_API
    with _FLEX_IMPORT_LOCK:
        if _FLEX_IMPORT_ATTEMPTED:
            return _FLEX_API
        try:
            from torch.nn.attention.flex_attention import (
                create_block_mask,
                flex_attention,
            )

            _FLEX_API = (flex_attention, create_block_mask)
        except (ImportError, AttributeError):
            _FLEX_API = None
        _FLEX_IMPORT_ATTEMPTED = True
    return _FLEX_API


def flex_attention_available() -> bool:
    return _load_flex_api() is not None


def _compiled_flex() -> Callable[..., torch.Tensor]:
    global _FLEX_COMPILED
    api = _load_flex_api()
    if api is None:
        raise RuntimeError("FlexAttention is not available in this PyTorch build")
    if _FLEX_COMPILED is not None:
        return _FLEX_COMPILED
    with _FLEX_COMPILE_LOCK:
        if _FLEX_COMPILED is None:
            _FLEX_COMPILED = torch.compile(api[0], dynamic=False)
    return _FLEX_COMPILED


def _fixed_mask_mod(spec: MaskSpec) -> Callable[..., torch.Tensor]:
    query_base = int(spec.query_base)
    block_size = int(spec.block_size)
    kind = spec.kind

    if kind in _CAUSAL_KINDS:

        def mask_mod(
            _batch: torch.Tensor,
            _head: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            return kv_idx <= (q_idx + query_base)

        return mask_mod

    if kind in _FIXED_SAT_KINDS:

        def mask_mod(
            _batch: torch.Tensor,
            _head: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            return (kv_idx // block_size) <= (
                (q_idx + query_base) // block_size
            )

        return mask_mod

    raise ValueError("fixed FlexAttention mask is unsupported for %s" % kind)


def _get_fixed_block_mask(
    spec: MaskSpec,
    *,
    device: torch.device,
    compile_mask: bool,
) -> Any:
    api = _load_flex_api()
    if api is None:
        raise RuntimeError("FlexAttention is not available")
    _, create_block_mask = api
    cache_key = (
        spec.kind,
        spec.q_len,
        int(spec.kv_len),
        spec.query_base,
        spec.block_size,
        device.type,
        device.index,
        bool(compile_mask),
    )
    with _BLOCK_MASK_CACHE_LOCK:
        cached = _BLOCK_MASK_CACHE.get(cache_key)
        if cached is not None:
            _BLOCK_MASK_CACHE.move_to_end(cache_key)
            return cached

    kwargs = {
        "B": None,
        "H": None,
        "Q_LEN": spec.q_len,
        "KV_LEN": int(spec.kv_len),
        "device": device,
    }
    try:
        block_mask = create_block_mask(
            _fixed_mask_mod(spec), _compile=bool(compile_mask), **kwargs
        )
    except TypeError:
        block_mask = create_block_mask(_fixed_mask_mod(spec), **kwargs)

    with _BLOCK_MASK_CACHE_LOCK:
        _BLOCK_MASK_CACHE[cache_key] = block_mask
        _BLOCK_MASK_CACHE.move_to_end(cache_key)
        while len(_BLOCK_MASK_CACHE) > _BLOCK_MASK_CACHE_MAX:
            _BLOCK_MASK_CACHE.popitem(last=False)
    return block_mask


def _variable_block_mask(
    spec: MaskSpec,
    *,
    query: torch.Tensor,
    q_block_ids: torch.Tensor,
    kv_block_ids: torch.Tensor,
    compile_mask: bool,
) -> Any:
    api = _load_flex_api()
    if api is None:
        raise RuntimeError("FlexAttention is not available")
    _, create_block_mask = api

    batch = int(query.shape[0])
    q_ids = _expand_block_ids(
        q_block_ids, batch, spec.q_len, query.device, "q_block_ids"
    )
    kv_ids = _expand_block_ids(
        kv_block_ids, batch, int(spec.kv_len), query.device, "kv_block_ids"
    )

    def mask_mod(
        batch_idx: torch.Tensor,
        _head_idx: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        return kv_ids[batch_idx, kv_idx] <= q_ids[batch_idx, q_idx]

    kwargs = {
        "B": batch,
        "H": None,
        "Q_LEN": spec.q_len,
        "KV_LEN": int(spec.kv_len),
        "device": query.device,
    }
    try:
        return create_block_mask(
            mask_mod, _compile=bool(compile_mask), **kwargs
        )
    except TypeError:
        return create_block_mask(mask_mod, **kwargs)


def _call_flex(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    spec: MaskSpec,
    scale: Optional[float],
    q_block_ids: Optional[torch.Tensor],
    kv_block_ids: Optional[torch.Tensor],
    compile_flex: bool,
) -> torch.Tensor:
    api = _load_flex_api()
    if api is None:
        raise RuntimeError("FlexAttention is not available")

    compile_mask = bool(compile_flex and query.is_cuda)
    if spec.kind in _VARIABLE_SAT_KINDS:
        if q_block_ids is None or kv_block_ids is None:
            raise ValueError("variable SAT requires q_block_ids and kv_block_ids")
        block_mask = _variable_block_mask(
            spec,
            query=query,
            q_block_ids=q_block_ids,
            kv_block_ids=kv_block_ids,
            compile_mask=compile_mask,
        )
    else:
        block_mask = _get_fixed_block_mask(
            spec, device=query.device, compile_mask=compile_mask
        )

    flex_fn = _compiled_flex() if compile_flex else api[0]
    return flex_fn(
        query,
        key,
        value,
        block_mask=block_mask,
        scale=scale,
    )


def select_backend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask_spec: Optional[MaskSpec] = None,
    attn_mask: Optional[torch.Tensor] = None,
    score_bias: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    prefer_flex: bool = True,
    force_backend: Optional[str] = None,
) -> AttentionDecision:
    """Select a backend without executing attention."""

    q_len, kv_len = _shape_check(query, key, value)
    if mask_spec is not None:
        if mask_spec.q_len != q_len or int(mask_spec.kv_len) != kv_len:
            raise ValueError(
                "MaskSpec lengths (%d, %d) do not match tensors (%d, %d)"
                % (mask_spec.q_len, int(mask_spec.kv_len), q_len, kv_len)
            )

    force = str(force_backend or "auto").strip().lower()
    aliases = {
        "native": "native_causal",
        "causal": "native_causal",
        "sdpa": "dense_sdpa",
        "dense": "dense_sdpa",
        "none": "unmasked_sdpa",
    }
    force = aliases.get(force, force)
    allowed = {
        "auto",
        "native_causal",
        "flex",
        "dense_sdpa",
        "unmasked_sdpa",
    }
    if force not in allowed:
        raise ValueError("unknown force_backend: %s" % force)

    kind = mask_spec.kind if mask_spec is not None else "tensor_or_none"
    has_bias = attn_mask is not None or score_bias is not None

    if force != "auto":
        return AttentionDecision(
            backend=force,
            reason="forced by caller",
            mask_kind=kind,
            q_len=q_len,
            kv_len=kv_len,
        )

    if mask_spec is not None and _native_causal_eligible(
        mask_spec, q_len, kv_len, has_score_bias=has_bias
    ):
        return AttentionDecision(
            backend="native_causal",
            reason="square zero-offset causal mask is representable by is_causal=True",
            mask_kind=kind,
            q_len=q_len,
            kv_len=kv_len,
        )

    if (
        mask_spec is not None
        and mask_spec.kind not in _UNRESTRICTED_KINDS
        and not has_bias
        and float(dropout_p) == 0.0
        and bool(prefer_flex)
        and query.is_cuda
        and flex_attention_available()
    ):
        return AttentionDecision(
            backend="flex",
            reason="structured non-native mask on CUDA with FlexAttention available",
            mask_kind=kind,
            q_len=q_len,
            kv_len=kv_len,
        )

    if (
        (mask_spec is None or mask_spec.kind in _UNRESTRICTED_KINDS)
        and attn_mask is None
        and score_bias is None
    ):
        return AttentionDecision(
            backend="unmasked_sdpa",
            reason="no visibility mask or score bias",
            mask_kind=kind,
            q_len=q_len,
            kv_len=kv_len,
        )

    return AttentionDecision(
        backend="dense_sdpa",
        reason="correctness fallback for unsupported or biased mask",
        mask_kind=kind,
        q_len=q_len,
        kv_len=kv_len,
    )


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask_spec: Optional[MaskSpec] = None,
    attn_mask: Optional[torch.Tensor] = None,
    score_bias: Optional[torch.Tensor] = None,
    q_block_ids: Optional[torch.Tensor] = None,
    kv_block_ids: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    prefer_flex: bool = True,
    force_backend: Optional[str] = None,
    compile_flex: bool = True,
    return_decision: bool = False,
) -> Any:
    """Run attention through a validated backend.

    ``score_bias`` is an additive tensor such as ALiBi.  V1 deliberately routes
    score-biased calls to dense SDPA rather than silently dropping the bias.
    """

    q_len, kv_len = _shape_check(query, key, value)
    decision = select_backend(
        query,
        key,
        value,
        mask_spec=mask_spec,
        attn_mask=attn_mask,
        score_bias=score_bias,
        dropout_p=dropout_p,
        prefer_flex=prefer_flex,
        force_backend=force_backend,
    )

    if decision.backend == "native_causal":
        if mask_spec is None or not _native_causal_eligible(
            mask_spec,
            q_len,
            kv_len,
            has_score_bias=(attn_mask is not None or score_bias is not None),
        ):
            raise ValueError("native_causal was forced for an ineligible call")
        output = _call_sdpa(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
            scale=scale,
        )
    elif decision.backend == "unmasked_sdpa":
        if attn_mask is not None or score_bias is not None:
            raise ValueError("unmasked_sdpa cannot accept a mask or score bias")
        if mask_spec is not None and mask_spec.kind not in _UNRESTRICTED_KINDS:
            raise ValueError("unmasked_sdpa was forced for a restricted mask")
        output = _call_sdpa(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False,
            scale=scale,
        )
    elif decision.backend == "flex":
        if float(dropout_p) != 0.0:
            raise ValueError("FlexAttention v1 route requires dropout_p=0")
        if attn_mask is not None or score_bias is not None:
            raise ValueError("FlexAttention v1 route does not accept dense bias")
        if mask_spec is None or mask_spec.kind in _UNRESTRICTED_KINDS:
            raise ValueError("FlexAttention requires a structured mask")
        output = _call_flex(
            query,
            key,
            value,
            spec=mask_spec,
            scale=scale,
            q_block_ids=q_block_ids,
            kv_block_ids=kv_block_ids,
            compile_flex=compile_flex,
        )
    elif decision.backend == "dense_sdpa":
        if attn_mask is None and mask_spec is not None:
            dense = build_dense_mask(
                mask_spec,
                device=query.device,
                dtype=query.dtype,
                batch=int(query.shape[0]),
                q_block_ids=q_block_ids,
                kv_block_ids=kv_block_ids,
            )
        else:
            dense = attn_mask

        if dense is not None:
            dense = dense.to(device=query.device)

        if score_bias is not None:
            bias = score_bias.to(device=query.device, dtype=query.dtype)
            if dense is None:
                dense = bias
            elif dense.dtype == torch.bool:
                # SDPA boolean masks use True for visible positions.  Convert to
                # an additive mask before combining with a score bias.
                additive = torch.zeros_like(dense, dtype=query.dtype)
                additive = additive.masked_fill(~dense, float("-inf"))
                dense = additive + bias
            else:
                dense = dense.to(dtype=query.dtype) + bias
        elif dense is not None and dense.dtype != torch.bool:
            dense = dense.to(dtype=query.dtype)

        output = _call_sdpa(
            query,
            key,
            value,
            attn_mask=dense,
            dropout_p=dropout_p,
            is_causal=False,
            scale=scale,
        )
    else:
        raise AssertionError(decision.backend)

    if return_decision:
        return output, decision
    return output
