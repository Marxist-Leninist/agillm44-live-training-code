from __future__ import annotations

import argparse
import copy
import ctypes
import json
import math
import os
import queue
import random
import signal
import statistics
import shutil
import sys
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path

_PROC_T0 = time.time()  # v2 telemetry: process start, for restart-cost accounting

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.ops.triton.layer_norm import rms_norm_fn as _flash_rms_norm_fn

# AGILLM-GB10-1PF v0.1: deliberately hardware-locked for one NVIDIA GB10.
# Conventional parameter count target: ~2B. Training target: 600B tokens.
NAME = "AGILLM-GB10-1PF-targetfix"
TARGET_ALIGNMENT = "next_token_shifted_labels_fullseq_v1"
VOCAB = 129_280
TOKEN_DIM = 256
D = 1_280
STAGES = 16
EXPERTS = 6
FFN = 5_120
Q_HEADS = 20
KV_HEADS = 5
HEAD_DIM = 64
CONTEXT = 2_048
TOTAL_TRAIN_TOKENS = 600_000_000_000
VOCAB_GROUPS = 505
VOCAB_LOW = 256
VOCAB_RANK = 16
VOCAB_GROUP_PAD = 512
VOCAB_AFFINE_A = (1,3,7,9,11,13,17,19,21,23,27,29,31,33,37,39)
VOCAB_AFFINE_B = tuple((i * 7919) % VOCAB for i in range(VOCAB_RANK))
SPARSE_PF_TARGET = 1_000_000_000_000_000.0
VERIFIED_SPARSE_TFLOPS = 969.794609
WINDOWS = (256,256,256,256,512,512,512,512,1024,1024,1024,1024,-1,-1,-1,-1)
GB10_MEM_BYTES = 128_427_982_848

ROOT = Path(__file__).resolve().parent
SPARSE_RUNTIME = Path(os.environ.get("AGILLM_GB10_SPARSE_RUNTIME", str(ROOT / "runtime")))
TOKENIZER_JSON = Path(os.environ.get("AGILLM_GB10_TOKENIZER", str(ROOT / "tokenizer_bundle.json")))

FINEWEB_EDU_REVISION = os.environ.get(
    "AGILLM_FINEWEB_EDU_REVISION",
    "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
)

HF_SOURCES = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", 1.60),
    "fineweb": ("HuggingFaceFW/fineweb", "CC-MAIN-2024-10", 0.50),
    "wikipedia": ("wikimedia/wikipedia", "20231101.en", 1.10),
    "c4": ("allenai/c4", "en", 0.40),
    "openwebtext": ("Skylion007/openwebtext", None, 0.35),
    "cosmopedia": ("HuggingFaceTB/cosmopedia", "web_samples_v2", 1.30),
    "smollm-cosmopedia": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 1.30),
    "proof-pile-2": ("EleutherAI/proof-pile-2", "all", 1.50),
    "dolma": ("allenai/dolma", "v1_6-sample", 0.50),
    "codeparrot": ("codeparrot/codeparrot-clean", None, 1.25),
}


def parameter_accounting():
    emb = VOCAB * TOKEN_DIM
    factors = 2 * TOKEN_DIM * D
    expert = STAGES * EXPERTS * 3 * D * FFN
    attn_per = D * D + 2 * D * (KV_HEADS * HEAD_DIM) + D * D
    attn = STAGES * attn_per
    routers = STAGES * D * EXPERTS
    norms = (2 * STAGES + 1) * D
    vocab_factor_head = TOKEN_DIM * (VOCAB_RANK * (VOCAB_GROUP_PAD + VOCAB_LOW) + VOCAB_RANK)
    total = emb + factors + expert + attn + routers + norms + vocab_factor_head
    return {
        "token_embedding": emb,
        "token_factor_projections": factors,
        "experts": expert,
        "attention": attn,
        "routers": routers,
        "norms": norms,
        "rank16_product_vocab_head": vocab_factor_head,
        "total": total,
        "target": 2_000_000_000,
        "deviation_percent": 100.0 * (total / 2_000_000_000 - 1.0),
    }


class SparseRuntime:
    def __init__(self, root: Path):
        self.root = root
        self.native = ctypes.CDLL(str(root / "libsparse_native.so"))
        self.quant = ctypes.CDLL(str(root / "libfused_quant.so"))
        P = ctypes.c_void_p
        n = self.native
        n.sg_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]; n.sg_create.restype = P
        n.sg_output_layout.argtypes = []; n.sg_output_layout.restype = ctypes.c_int
        if n.sg_output_layout() != 1:
            raise RuntimeError("sparse runtime requires column-major native output ABI")
        n.sg_pack.argtypes = [P,P,P,P]; n.sg_pack.restype = ctypes.c_int
        n.sg_run_device.argtypes = [P,P,P,P,P,P]; n.sg_run_device.restype = ctypes.c_int
        n.sg_destroy.argtypes = [P]; n.sg_destroy.restype = None
        n.sg_error.argtypes = []; n.sg_error.restype = ctypes.c_char_p
        q = self.quant
        q.sg_global_scale.argtypes = [P,P,P]; q.sg_global_scale.restype = ctypes.c_int
        q.sg_quant32_device_bf16.argtypes = [P,ctypes.c_int,P,P,P,ctypes.c_int64,P,P]
        q.sg_quant32_device_bf16.restype = ctypes.c_int
        q.sg_pair48.argtypes = [P,ctypes.c_int,P,ctypes.c_int64,P]; q.sg_pair48.restype = ctypes.c_int
        q.sg_cuda_error.argtypes = [ctypes.c_int]; q.sg_cuda_error.restype = ctypes.c_char_p

    def check_native(self, rc):
        if rc:
            raise RuntimeError(self.native.sg_error().decode())

    def check_quant(self, rc):
        if rc:
            raise RuntimeError(self.quant.sg_cuda_error(rc).decode())


_RT = None

def rt():
    global _RT
    if _RT is None:
        _RT = SparseRuntime(SPARSE_RUNTIME)
    return _RT


def dtype_kind(x):
    if not x.is_cuda:
        raise ValueError("CUDA tensor required")
    if x.dtype == torch.float32: return 0
    if x.dtype == torch.bfloat16: return 1
    raise ValueError(f"unsupported dtype {x.dtype}")


@torch.no_grad()
def pair48_mask(w):
    w = w.contiguous()
    mask = torch.empty_like(w, dtype=torch.bool)
    rt().check_quant(rt().quant.sg_pair48(w.data_ptr(), dtype_kind(w), mask.data_ptr(), w.numel(), torch.cuda.current_stream().cuda_stream))
    return mask


@torch.no_grad()
def quantize32_bf16(x, materialize=True):
    if x.ndim != 2 or x.shape[-1] % 32 or x.numel() == 0:
        raise ValueError("quantize32 requires nonempty [rows,K], K divisible by 32")
    x = x.contiguous()
    amax = x.abs().amax().to(torch.float64)
    g = torch.empty_like(amax)
    rt().check_quant(rt().quant.sg_global_scale(amax.data_ptr(), g.data_ptr(), torch.cuda.current_stream().cuda_stream))
    packed = torch.empty((x.shape[0], x.shape[1] // 2), device=x.device, dtype=torch.uint8)
    sf = torch.empty((x.shape[0], x.shape[1] // 32), device=x.device, dtype=torch.float8_e4m3fn)
    dq = torch.empty_like(x, dtype=torch.bfloat16) if materialize else None
    rt().check_quant(rt().quant.sg_quant32_device_bf16(
        x.data_ptr(), dtype_kind(x), packed.data_ptr(), sf.data_ptr(),
        dq.data_ptr() if dq is not None else None, x.numel(), g.data_ptr(),
        torch.cuda.current_stream().cuda_stream))
    return packed, sf, dq, g


_SILU_PACK_LIB = None
def silu_pack_lib():
    global _SILU_PACK_LIB
    if _SILU_PACK_LIB is None:
        lib = ctypes.CDLL('/workspace/rpv16_direct_packed_siluquant_v1/libfused_quant.so')
        P = ctypes.c_void_p
        lib.rpv_silu_mul_amax.argtypes = [P,P,P,ctypes.c_int64,P]; lib.rpv_silu_mul_amax.restype = ctypes.c_int
        lib.rpv_silu_mul_scale.argtypes = [P,P,P]; lib.rpv_silu_mul_scale.restype = ctypes.c_int
        lib.rpv_quant32_silu_mul.argtypes = [P,P,P,P,ctypes.c_int64,P,P]; lib.rpv_quant32_silu_mul.restype = ctypes.c_int
        lib.sg_cuda_error.argtypes = [ctypes.c_int]; lib.sg_cuda_error.restype = ctypes.c_char_p
        _SILU_PACK_LIB = lib
    return _SILU_PACK_LIB


def _silu_pack_check(lib, rc):
    if rc:
        raise RuntimeError(lib.sg_cuda_error(rc).decode())


@torch.no_grad()
def quantize_silu_mul_packed(gate, up):
    if gate.shape != up.shape or gate.ndim != 2 or gate.shape[-1] % 32 or gate.numel() == 0:
        raise ValueError('direct SiLU*up pack requires equal nonempty [rows,K], K divisible by 32')
    if gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16 or not gate.is_cuda or not up.is_cuda:
        raise ValueError('direct SiLU*up pack requires CUDA BF16 inputs')
    gate = gate.contiguous(); up = up.contiguous()
    lib = silu_pack_lib(); stream = torch.cuda.current_stream().cuda_stream
    amax = torch.empty((), device=gate.device, dtype=torch.float32)
    g = torch.empty((), device=gate.device, dtype=torch.float64)
    packed = torch.empty((gate.shape[0], gate.shape[1] // 2), device=gate.device, dtype=torch.uint8)
    sf = torch.empty((gate.shape[0], gate.shape[1] // 32), device=gate.device, dtype=torch.float8_e4m3fn)
    _silu_pack_check(lib, lib.rpv_silu_mul_amax(gate.data_ptr(), up.data_ptr(), amax.data_ptr(), gate.numel(), stream))
    _silu_pack_check(lib, lib.rpv_silu_mul_scale(amax.data_ptr(), g.data_ptr(), stream))
    _silu_pack_check(lib, lib.rpv_quant32_silu_mul(gate.data_ptr(), up.data_ptr(), packed.data_ptr(), sf.data_ptr(), gate.numel(), g.data_ptr(), stream))
    return packed, sf, g


class NativeContext:
    def __init__(self, m, n, k):
        self.handle = rt().native.sg_create(m, n, k)
        if not self.handle:
            raise RuntimeError(rt().native.sg_error().decode())
        self.m, self.n, self.k = m, n, k
    def close(self):
        if getattr(self, "handle", None):
            rt().native.sg_destroy(self.handle); self.handle = None
    def __del__(self):
        try: self.close()
        except Exception: pass
    def pack(self, packed_w, sf_w):
        rt().check_native(rt().native.sg_pack(self.handle, packed_w.data_ptr(), sf_w.data_ptr(), torch.cuda.current_stream().cuda_stream))
    def run(self, packed_x, sf_x, alpha):
        out = torch.empty((self.n, self.m), device=packed_x.device, dtype=torch.bfloat16)
        alpha = alpha.to(device=packed_x.device, dtype=torch.float32).contiguous()
        rt().check_native(rt().native.sg_run_device(
            self.handle, packed_x.data_ptr(), sf_x.data_ptr(), out.data_ptr(),
            alpha.data_ptr(), torch.cuda.current_stream().cuda_stream))
        return out


class _SparseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, owner):
        shape = x.shape
        rows = x.numel() // shape[-1]
        k = shape[-1]
        n = (rows + 127) // 128 * 128
        flat = x.reshape(rows, k)
        xp = flat if n == rows else F.pad(flat, (0,0,0,n-rows))
        xpack, xsf, _, xg = quantize32_bf16(xp, materialize=False)
        native, wq, wg = owner.shadow(n)
        y = native.run(xpack, xsf, xg * wg)[:rows]
        ctx.owner = owner
        ctx.shape = shape
        ctx.xdtype = x.dtype
        ctx.save_for_backward(flat)
        return y.reshape(*shape[:-1], owner.out_features).to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        (x,) = ctx.saved_tensors
        o = ctx.owner
        g = gy.reshape(-1, o.out_features).to(torch.bfloat16)
        gx = (g @ o._wq).reshape(ctx.shape).to(ctx.xdtype) if ctx.needs_input_grad[0] else None
        if ctx.needs_input_grad[1]:
            gw = (g.t() @ x.to(torch.bfloat16)).to(o.weight.dtype)
            gw.mul_(o._fixed_mask)
        else:
            gw = None
        return gx, gw, None


class _SparsePairFn(torch.autograd.Function):
    """Two independent sparse linears sharing one activation quantization.

    Gate and up receive the identical expert input. The production path currently
    quantizes that tensor twice. This function quantizes it once, then executes
    the two ordinary native sparse GEMMs with independent packed weights, weight
    scales, and native contexts. Weight semantics are unchanged.
    """
    @staticmethod
    def forward(ctx, x, w1, w2, owner1, owner2):
        shape = x.shape
        rows = x.numel() // shape[-1]
        k = shape[-1]
        n = (rows + 127) // 128 * 128
        flat = x.reshape(rows, k)
        xp = flat if n == rows else F.pad(flat, (0,0,0,n-rows))
        xpack, xsf, _, xg = quantize32_bf16(xp, materialize=False)
        native1, _, wg1 = owner1.shadow(n)
        native2, _, wg2 = owner2.shadow(n)
        y1 = native1.run(xpack, xsf, xg * wg1)[:rows]
        y2 = native2.run(xpack, xsf, xg * wg2)[:rows]
        ctx.owner1 = owner1
        ctx.owner2 = owner2
        ctx.shape = shape
        ctx.xdtype = x.dtype
        ctx.save_for_backward(flat)
        outshape = (*shape[:-1], owner1.out_features)
        return y1.reshape(outshape).to(x.dtype), y2.reshape(outshape).to(x.dtype)

    @staticmethod
    def backward(ctx, gy1, gy2):
        (x,) = ctx.saved_tensors
        o1, o2 = ctx.owner1, ctx.owner2
        g1 = gy1.reshape(-1, o1.out_features).to(torch.bfloat16)
        g2 = gy2.reshape(-1, o2.out_features).to(torch.bfloat16)
        gx = None
        if ctx.needs_input_grad[0]:
            gx1 = g1 @ o1._wq
            gx2 = g2 @ o2._wq
            gx = (gx1 + gx2).reshape(ctx.shape).to(ctx.xdtype)
        gw1 = None
        if ctx.needs_input_grad[1]:
            gw1 = (g1.t() @ x.to(torch.bfloat16)).to(o1.weight.dtype)
            gw1.mul_(o1._fixed_mask)
        gw2 = None
        if ctx.needs_input_grad[2]:
            gw2 = (g2.t() @ x.to(torch.bfloat16)).to(o2.weight.dtype)
            gw2.mul_(o2._fixed_mask)
        return gx, gw1, gw2, None, None


# ---------------------------------------------------------------------------
# v2b_stagefwd: exact (bitwise) stage-forward restructuring. Each switch set to
# False restores the v1 code path verbatim. main() maps CLI flags onto this.
# ---------------------------------------------------------------------------
STAGEFWD = {
    "syncfree_routing": True,       # --stagefwd-v1-routing restores v1
    "active_fused_silupack": True,  # --stagefwd-v1-active-silu restores v1
    "inplace_repack": True,         # --stagefwd-v1-repack restores v1
}


class _GatherRowsFn(torch.autograd.Function):
    """One expert-major row gather feeding all experts.

    `order` is the stable sort permutation of the route, so block i holds exactly the
    rows `(route == i).nonzero()` selected in v1, in the same ascending source
    order. Backward reproduces v1's accumulated index_select gradients: every
    source row receives exactly one expert gradient added onto a zero buffer.
    """
    @staticmethod
    def forward(ctx, flat, order, groups):
        ctx.save_for_backward(order)
        ctx.groups = int(groups)
        ctx.rows = flat.shape[0]
        xs = flat.index_select(0, order)
        return tuple(xs.narrow(0, i * ctx.groups, ctx.groups) for i in range(xs.shape[0] // ctx.groups))

    @staticmethod
    def backward(ctx, *gs):
        (order,) = ctx.saved_tensors
        ref = next(g for g in gs if g is not None)
        gflat = torch.zeros((ctx.rows, ref.shape[-1]), device=ref.device, dtype=ref.dtype)
        for i, g in enumerate(gs):
            if g is not None:
                gflat.index_add_(0, order.narrow(0, i * ctx.groups, ctx.groups), g.reshape(ctx.groups, -1))
        return gflat, None, None


class _ScatterRowsFn(torch.autograd.Function):
    """Scatter the expert outputs back to source-row order (pure copies).

    v1 used six in-place index_copy_ calls whose autograd backward cloned the
    full [rows, D] gradient five times (index_fill) before each index_select.
    The expert row sets are disjoint, so one gather of the incoming gradient is
    value- and bit-identical.
    """
    @staticmethod
    def forward(ctx, order, groups, *ys):
        groups = int(groups)
        ctx.save_for_backward(order)
        ctx.groups = groups
        y = torch.empty((groups * len(ys), ys[0].shape[-1]), device=ys[0].device, dtype=ys[0].dtype)
        for i, yi in enumerate(ys):
            y.index_copy_(0, order.narrow(0, i * groups, groups), yi.reshape(groups, -1))
        return y

    @staticmethod
    def backward(ctx, gy):
        (order,) = ctx.saved_tensors
        gs = gy.index_select(0, order)
        return (None, None) + tuple(gs.narrow(0, i * ctx.groups, ctx.groups) for i in range(gs.shape[0] // ctx.groups))


class _SiluDownFn(torch.autograd.Function):
    """Active-stage down projection fed by the fused SiLU*up -> NVFP4 pack.

    Forward emits the identical block32 NVFP4 down-projection input the frozen
    path already uses (no BF16 SiLU*up materialisation, no second read for
    abs/amax/quant). Backward recomputes SiLU(gate)*up with the same ATen
    kernels autograd used in v1 and then evaluates exactly v1's BF16 formulas:
    dW = (g^T @ act) * mask, d_act = g @ Wq, d_up = d_act * silu(gate),
    d_gate = silu_backward(d_act * up, gate).
    """
    @staticmethod
    def forward(ctx, gate, up, w, owner):
        shape = gate.shape
        gf = gate.reshape(-1, owner.in_features)
        uf = up.reshape(-1, owner.in_features)
        rows = gf.shape[0]
        packed, sf, xg = quantize_silu_mul_packed(gf, uf)
        native, _, wg = owner.shadow(rows)
        y = native.run(packed, sf, xg * wg)[:rows]
        ctx.owner = owner
        ctx.shape = shape
        ctx.xdtype = gate.dtype
        ctx.save_for_backward(gf, uf)
        return y.reshape(*shape[:-1], owner.out_features).to(gate.dtype)

    @staticmethod
    def backward(ctx, gy):
        gf, uf = ctx.saved_tensors
        o = ctx.owner
        g = gy.reshape(-1, o.out_features).to(torch.bfloat16)
        s = F.silu(gf)
        gw = None
        if ctx.needs_input_grad[2]:
            act = s * uf
            gw = (g.t() @ act.to(torch.bfloat16)).to(o.weight.dtype)
            gw.mul_(o._fixed_mask)
            del act
        ggate = gup = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ga = (g @ o._wq).reshape(gf.shape).to(ctx.xdtype)
            if ctx.needs_input_grad[1]:
                gup = (ga * s).reshape(ctx.shape)
            if ctx.needs_input_grad[0]:
                ggate = torch.ops.aten.silu_backward(ga * uf, gf).reshape(ctx.shape)
        return ggate, gup, gw, None


class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        if in_features % 256 or out_features % 128:
            raise ValueError(f"sparse shape must satisfy K%256=0 and M%128=0, got {out_features}x{in_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device="cuda", dtype=torch.float32))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self._fixed_mask = None
        self._cache_key = None
        self._wq = None
        self._wg = None
        self._packed_w = None
        self._sf_w = None
        self._contexts = OrderedDict()
        self.pack_count = 0
        self.kernel_calls = 0

    @torch.no_grad()
    def shadow(self, n):
        if self._fixed_mask is None:
            self._fixed_mask = pair48_mask(self.weight)
        key = (self.weight.data_ptr(), self.weight._version)
        if self._cache_key != key:
            masked = self.weight * self._fixed_mask
            self._packed_w, self._sf_w, self._wq, self._wg = quantize32_bf16(masked, materialize=True)
            self._cache_key = key
            if STAGEFWD["inplace_repack"]:
                # The native State::pack is re-callable: it rewrites the packed
                # weight, the sparsity metadata and the weight block scales in
                # the buffers sg_create allocated. Re-packing in place replaces
                # v1's sg_destroy (6 cudaFree = 6 device syncs) + sg_create
                # (4 cudaMalloc + 2 cudaMemset) + 2 lazy workspace cudaMalloc
                # per layer per weight version. Shapes never change, so the
                # buffers, layouts and workspaces are identical.
                for c in self._contexts.values():
                    c.pack(self._packed_w, self._sf_w)
                    self.pack_count += 1
            else:
                for c in self._contexts.values(): c.close()
                self._contexts.clear()
        c = self._contexts.get(n)
        if c is None:
            # One native context per sparse layer: the production router uses a
            # fixed 8192-row expert shape, and this hard cap prevents accidental
            # shape drift from leaking GBs of native workspace.
            for old_ctx in self._contexts.values():
                old_ctx.close()
            self._contexts.clear()
            c = NativeContext(self.out_features, n, self.in_features)
            c.pack(self._packed_w, self._sf_w)
            self._contexts[n] = c
            self.pack_count += 1
        self.kernel_calls += 1
        return c, self._wq, self._wg

    def forward(self, x):
        return _SparseFn.apply(x, self.weight, self)

    @torch.no_grad()
    def forward_packed(self, packed_x, sf_x, xg, rows):
        n = (int(rows) + 127) // 128 * 128
        if n != int(rows):
            raise ValueError('direct packed sparse path currently requires rows divisible by 128')
        native, _, wg = self.shadow(n)
        return native.run(packed_x, sf_x, xg * wg)[:int(rows)]

    def invalidate(self):
        self._cache_key = None


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d, device="cuda", dtype=torch.bfloat16))
    def forward(self, x):
        return _flash_rms_norm_fn(x, self.weight, None, eps=1e-5)


def alibi_slopes(n):
    def slopes_pow2(k):
        start = 2 ** (-2 ** -(math.log2(k) - 3))
        ratio = start
        return [start * ratio ** i for i in range(k)]
    if math.log2(n).is_integer(): vals = slopes_pow2(n)
    else:
        p = 2 ** math.floor(math.log2(n))
        vals = slopes_pow2(p) + slopes_pow2(2*p)[0::2][:n-p]
    return torch.tensor(vals, device="cuda", dtype=torch.float32)


class Attention(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.q = nn.Linear(D, D, bias=False, device="cuda", dtype=torch.bfloat16)
        self.k = nn.Linear(D, KV_HEADS*HEAD_DIM, bias=False, device="cuda", dtype=torch.bfloat16)
        self.v = nn.Linear(D, KV_HEADS*HEAD_DIM, bias=False, device="cuda", dtype=torch.bfloat16)
        self.o = nn.Linear(D, D, bias=False, device="cuda", dtype=torch.bfloat16)
    def forward(self, x):
        from flash_attn import flash_attn_func
        b,s,_ = x.shape
        q = self.q(x).view(b,s,Q_HEADS,HEAD_DIM)
        k = self.k(x).view(b,s,KV_HEADS,HEAD_DIM)
        v = self.v(x).view(b,s,KV_HEADS,HEAD_DIM)
        if _AH_ATTN.mode == "ar":
            ws = (-1,-1) if self.window < 0 else (self.window, 0)
            y = flash_attn_func(q,k,v,dropout_p=0.0,causal=True,window_size=ws)
        else:
            y = ah_attention(q,k,v,self.window)
        return self.o(y.reshape(b,s,D))


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = SparseLinear(D, FFN)
        self.up = SparseLinear(D, FFN)
        self.down = SparseLinear(FFN, D)
    def forward(self, x):
        gate, up = _SparsePairFn.apply(x, self.gate.weight, self.up.weight, self.gate, self.up)
        # Frozen prefix/head-anchor stages execute under no_grad. Emit the exact
        # block32 NVFP4 down-projection input directly, avoiding the 8192x5120
        # BF16 SiLU*up materialization and the subsequent reread/requantize.
        # The active trainable stage keeps the original autograd path unchanged.
        if not gate.requires_grad and not up.requires_grad:
            gf = gate.reshape(-1, FFN)
            uf = up.reshape(-1, FFN)
            packed, sf, xg = quantize_silu_mul_packed(gf, uf)
            y = self.down.forward_packed(packed, sf, xg, gf.shape[0])
            return y.reshape(*x.shape[:-1], D).to(x.dtype)
        if (STAGEFWD["active_fused_silupack"] and gate.dtype == torch.bfloat16 and up.dtype == torch.bfloat16
                and gate.numel() and (gate.numel() // FFN) % 128 == 0):
            return _SiluDownFn.apply(gate, up, self.down.weight, self.down)
        return self.down(F.silu(gate) * up)


class Stage(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.n1 = RMSNorm(D); self.n2 = RMSNorm(D)
        self.attn = Attention(window)
        self.router = nn.Linear(D, EXPERTS, bias=False, device="cuda", dtype=torch.bfloat16)
        self.experts = nn.ModuleList([Expert() for _ in range(EXPERTS)])
        # 12 affine permutations of six experts. Every candidate assigns one
        # token in each six-token hardware group to every expert, guaranteeing
        # exact 8192-row expert GEMMs for the 24x2048 production geometry.
        perms = [[(a*j + b) % EXPERTS for j in range(EXPERTS)]
                 for a in (1, EXPERTS - 1) for b in range(EXPERTS)]
        self.register_buffer("_route_perms",
            torch.tensor(perms, device="cuda", dtype=torch.long), persistent=False)
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        h = self.n2(x)
        flat = h.reshape(-1, D)
        logits = self.router(flat).float()
        probs = logits.softmax(-1)
        if flat.shape[0] % EXPERTS == 0:
            groups = flat.shape[0] // EXPERTS
            # Group tokens spaced evenly through the flattened batch, then pick
            # the highest-scoring balanced affine assignment per group. This is
            # exact-capacity top-1 routing with zero expert-shape variance.
            gl = logits.view(EXPERTS, groups, EXPERTS).permute(1, 0, 2).contiguous()
            perms = self._route_perms
            pick = perms.view(1, perms.shape[0], EXPERTS, 1).expand(groups, -1, -1, 1)
            scores = gl.unsqueeze(1).expand(-1, perms.shape[0], -1, -1).gather(3, pick).squeeze(3).sum(2)
            best = scores.argmax(1)
            route = perms.index_select(0, best).transpose(0, 1).reshape(-1)
        else:
            route = probs.argmax(-1)
        residual_included = STAGEFWD["syncfree_routing"] and flat.shape[0] % EXPERTS == 0
        if residual_included:
            # Balanced affine routing assigns every expert exactly one row of
            # every six-row group, so each expert owns exactly `groups` rows by
            # construction and no host-side count is needed. A stable sort of
            # the route is expert-major and preserves ascending source order
            # inside each expert block, i.e. block i == (route == i).nonzero()
            # of v1 element for element: expert GEMM inputs and the BF16 dW
            # reduction order are unchanged. Zero host syncs. (uint8 keys: one
            # radix pass; measured ~3x cheaper than the int64 argsort and ~5x
            # cheaper than v1's six nonzero scans, identical permutation.)
            order = torch.sort(route.to(torch.uint8), stable=True)[1]
            xs = _GatherRowsFn.apply(flat, order, groups)
            ys = [e(xi) for e, xi in zip(self.experts, xs)]
            from sg_moe_residual_d803c31d import scatter_add
            y = scatter_add(x, order, groups, ys, _ScatterRowsFn.apply, emit=_emit)
            counts = [groups] * EXPERTS
        else:
            y = torch.empty_like(flat)
            counts = []
            for i,e in enumerate(self.experts):
                idx = (route == i).nonzero(as_tuple=False).flatten()
                counts.append(int(idx.numel()))
                if idx.numel():
                    yi = e(flat.index_select(0, idx))
                    y.index_copy_(0, idx, yi)
        # Routing is exactly balanced by construction, so train the router by
        # reinforcing the selected balanced assignment rather than a balance loss
        # that would be constant under exact capacities.
        selected = probs.gather(1, route[:, None]).squeeze(1).clamp_min(1e-9)
        aux = -selected.log().mean()
        return (y if residual_included else x + y.view_as(x)), aux, counts




# >>> FUSED_CE_EMBED BEGIN (source: fused_ce/rpv16_fused_ce.py)
# rpv16_fused_ce.py -- chunked FUSED LINEAR + rank-16 product-vocab CE (Liger-style) for AGILLM-GB10-1PF.
#
# Drop-in replacement for the inline RPV16 CE block of the production trainer
# (agillm_gb10_1pf.rpv16_affine_sharedquant_phaseclock_datacursor_targetfix_prod_v1.py, Model.forward,
# "if labels is not None:" branch).  Production semantics reproduced exactly (up to float rounding):
#
#   hh = h.reshape(-1, 256) (bf16), yy = labels.reshape(-1) (int64, every row counts, NO ignore mask)
#   per chunk of ce_chunk rows:  raw = F.linear(z, W_factor)  [bf16 GEMM, no bias]  -> FP32
#       group logits raw[:, r, :512] with classes 505..511 forced to -1e9, low logits raw[:, r, 512:768]
#       code_r = (token*a_r + b_r) mod V ; group_r = code_r // 256 ; low_r = code_r % 256
#       lm = log_softmax(F.linear(z, W_mix).float())
#       nll_row = -logsumexp_r(lm_r + log_softmax(group_r)[group_r] + log_softmax(low_r)[low_r])
#   ce = (sum over chunks of chunk-sums) / yy.numel()            (FP32 scalar)
#
# Integration (later step, NOT done here) is one line in Model.forward:
#   ce = rpv16_fused_ce(h.reshape(-1, TOKEN_DIM), labels.reshape(-1),
#                       self.factor_head.weight, self.mix_head.weight,
#                       ce_chunk=int(getattr(self, "ce_chunk", 4096)))      # (+ grad_scale_hint=1/grad_accum if != 1)
#   loss = ce + 0.01 * aux_total / STAGES
#
# Implementation: per chunk one BF16 GEMM for the 12288 factor logits + one for the 16 mix logits, then ONE
# Triton kernel (one program per row) that derives the 16 affine targets with integer math, does the masked
# group log-softmax (505 of 512), the low log-softmax (256), the target gathers and the rank logsumexp with
# FP32 accumulators, writes the per-row NLL and overwrites the logits / mix-logits buffers IN PLACE with
# d(ce)/dlogits (already scaled by 1/N in FP32 *before* the cast to the logits dtype, exactly where
# production's autograd does its FP32->BF16 cast).  dW / dh are accumulated with GEMMs of the same dtype and
# the same shapes production's autograd uses.  No FP32 [chunk,16,768] tensor, no checkpoint recompute, no
# slice copies.  backward() only scales the stored grads by grad_output.
#
# Gradients are only computed for inputs that require grad (stage steps: hidden rows only; head-anchor
# step: hidden rows + factor_head.weight + mix_head.weight; no_grad/eval: loss only).

import os

import torch
import triton
import triton.language as tl

VOCAB = 129_280
VOCAB_GROUPS = 505
VOCAB_LOW = 256
VOCAB_RANK = 16
VOCAB_GROUP_PAD = 512
VOCAB_AFFINE_A = (1, 3, 7, 9, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 37, 39)
VOCAB_AFFINE_B = tuple((i * 7919) % VOCAB for i in range(VOCAB_RANK))
assert VOCAB == VOCAB_GROUPS * VOCAB_LOW

# Kernel launch config chosen by tune_rpv16_fused_ce.py on the GB10 (sm_121); see the JSON receipt.
# Sweep (4096-row chunk, production sharing the GPU, best-of-15): num_warps 1 -> 13.1 ms, 2 -> 2.3 ms,
# 4 / 8 / 16 -> 0.87-0.89 ms (indistinguishable, ~memory-bandwidth bound); num_stages 1..4 makes no difference.
# Overridable for tuning (mutate the dict, or RPV16_CE_NUM_WARPS / RPV16_CE_NUM_STAGES in the environment).
KERNEL_CONFIG = {"num_warps": int(os.environ.get("RPV16_CE_NUM_WARPS", "4")),
                 "num_stages": int(os.environ.get("RPV16_CE_NUM_STAGES", "2"))}

_AFFINE_CACHE = {}


def _affine_tensors(device):
    key = (device.type, device.index)
    t = _AFFINE_CACHE.get(key)
    if t is None:
        t = (torch.tensor(VOCAB_AFFINE_A, device=device, dtype=torch.long),
             torch.tensor(VOCAB_AFFINE_B, device=device, dtype=torch.long))
        _AFFINE_CACHE[key] = t
    return t


@triton.jit
def _rpv16_ce_row_kernel(
    LOGITS, MIX, TGT, AFF_A, AFF_B, NLL,
    stride_lrow, stride_mrow,
    scale,
    V: tl.constexpr, NG: tl.constexpr, GP: tl.constexpr, LOW: tl.constexpr, R: tl.constexpr,
    WITH_GRAD: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, R)
    gi = tl.arange(0, GP)
    li = tl.arange(0, LOW)

    # (i) affine relabel with integer math; python-style remainder like torch.remainder
    tok = tl.load(TGT + row)
    a = tl.load(AFF_A + r)
    b = tl.load(AFF_B + r)
    code = (tok * a + b) % V
    code = (code + V) % V
    tgt_g = code // LOW
    tgt_l = code % LOW

    lbase = LOGITS + row * stride_lrow
    gptr = lbase + r[:, None] * (GP + LOW) + gi[None, :]
    lptr = lbase + r[:, None] * (GP + LOW) + GP + li[None, :]

    # (ii) Match the reference finite padding sentinel, including extreme-logit inputs.
    g = tl.load(gptr).to(tl.float32)
    g = tl.where(gi[None, :] < NG, g, -1.0e9)
    g_hit = gi[None, :] == tgt_g[:, None]
    gmax = tl.max(g, axis=1)
    ge = tl.exp(g - gmax[:, None])
    gsum = tl.sum(ge, axis=1)
    g_t = tl.sum(tl.where(g_hit, g, 0.0), axis=1)
    lg = (g_t - gmax) - tl.log(gsum)

    l = tl.load(lptr).to(tl.float32)
    l_hit = li[None, :] == tgt_l[:, None]
    lmax = tl.max(l, axis=1)
    le = tl.exp(l - lmax[:, None])
    lsum = tl.sum(le, axis=1)
    l_t = tl.sum(tl.where(l_hit, l, 0.0), axis=1)
    lo = (l_t - lmax) - tl.log(lsum)

    mptr = MIX + row * stride_mrow + r
    m = tl.load(mptr).to(tl.float32)
    mmax = tl.max(m, axis=0)
    me = tl.exp(m - mmax)
    msum = tl.sum(me, axis=0)
    lm = (m - mmax) - tl.log(msum)

    # Keep ordinary trained logits on the original FP32 path. Rare extreme
    # logits need wider rank-score accumulation to retain small mixture offsets.
    extreme = (tl.max(tl.abs(g_t - gmax), axis=0) > 1.0e4) | (tl.max(tl.abs(l_t - lmax), axis=0) > 1.0e4) | (tl.max(tl.abs(m - mmax), axis=0) > 1.0e4)
    # Triton requires variables used after a runtime branch to exist on every path.
    # This neutral FP64 scalar is overwritten on the extreme path and unused otherwise.
    base64 = tl.full((), 0.0, tl.float64)
    if extreme:
        t64 = ((g_t.to(tl.float64) - gmax.to(tl.float64)) - tl.log(gsum).to(tl.float64)
               + (l_t.to(tl.float64) - lmax.to(tl.float64)) - tl.log(lsum).to(tl.float64)
               + (m.to(tl.float64) - mmax.to(tl.float64)) - tl.log(msum).to(tl.float64))
        base64 = tl.max(t64, axis=0)
        t = (t64 - base64).to(tl.float32)
    else:
        t = lm + lg + lo
    tmax = tl.max(t, axis=0)
    te = tl.exp(t - tmax)
    tsum = tl.sum(te, axis=0)
    if extreme:
        nll_value = -(base64 + (tmax + tl.log(tsum)).to(tl.float64))
    else:
        # Keep branch result types identical for Triton SSA; store casts to FP32 NLL.
        nll_value = -(tmax + tl.log(tsum)).to(tl.float64)
    tl.store(NLL + row, nll_value)

    # (iii) gradients written in place; scale (= 1/N) applied in FP32 before the cast to the logits dtype
    if WITH_GRAD:
        w = te / tsum                      # posterior over ranks
        ws = w * scale
        dg = ws[:, None] * (ge / gsum[:, None] - tl.where(g_hit, 1.0, 0.0))
        # Reference overwrites padding with a constant, so its input derivative is zero.
        dg = tl.where(gi[None, :] < NG, dg, 0.0)
        tl.store(gptr, dg.to(LOGITS.dtype.element_ty))
        dl = ws[:, None] * (le / lsum[:, None] - tl.where(l_hit, 1.0, 0.0))
        tl.store(lptr, dl.to(LOGITS.dtype.element_ty))
        dm = (me / msum - w) * scale
        tl.store(mptr, dm.to(MIX.dtype.element_ty))


def rpv16_kernel_launch(logits, mix, targets, nll, scale, with_grad):
    """logits [n,12288] and mix [n,16] (row-contiguous, same float dtype) are OVERWRITTEN with grads if with_grad."""
    n = logits.shape[0]
    assert logits.shape[1] == VOCAB_RANK * (VOCAB_GROUP_PAD + VOCAB_LOW) and logits.stride(1) == 1
    assert mix.shape == (n, VOCAB_RANK) and mix.stride(1) == 1
    assert targets.dtype == torch.long and targets.is_contiguous() and targets.shape[0] == n
    assert nll.dtype == torch.float32 and nll.is_contiguous() and nll.shape[0] == n
    aff_a, aff_b = _affine_tensors(logits.device)
    _rpv16_ce_row_kernel[(n,)](
        logits, mix, targets, aff_a, aff_b, nll,
        logits.stride(0), mix.stride(0),
        float(scale),
        V=VOCAB, NG=VOCAB_GROUPS, GP=VOCAB_GROUP_PAD, LOW=VOCAB_LOW, R=VOCAB_RANK,
        WITH_GRAD=bool(with_grad),
        num_warps=KERNEL_CONFIG["num_warps"], num_stages=KERNEL_CONFIG["num_stages"],
    )


class _RPV16FusedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hh, yy, factor_weight, mix_weight, chunk, mean, hint):
        need_h, need_fw, need_mw = ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3]
        with_grad = bool(need_h or need_fw or need_mw)
        n, d = hh.shape
        dt = hh.dtype
        dev = hh.device
        nlogit = factor_weight.shape[0]
        scale = (hint / n) if mean else hint       # d(loss_total)/d(nll_row), folded in before the BF16 rounding
        ctx.hint = hint
        chunk = max(1, min(int(chunk), n))
        with torch.cuda.device(dev):
            logits_buf = torch.empty((chunk, nlogit), device=dev, dtype=dt)
            mix_buf = torch.empty((chunk, VOCAB_RANK), device=dev, dtype=dt)
            nll = torch.empty((n,), device=dev, dtype=torch.float32)
            dh = torch.empty_like(hh, memory_format=torch.contiguous_format) if need_h else None
            dfw = torch.zeros_like(factor_weight, memory_format=torch.contiguous_format) if need_fw else None
            dmw = torch.zeros_like(mix_weight, memory_format=torch.contiguous_format) if need_mw else None
            tmp_fw = torch.empty_like(dfw) if need_fw else None
            tmp_mw = torch.empty_like(dmw) if need_mw else None
            tmp_h = torch.empty((chunk, d), device=dev, dtype=dt) if need_h else None
            fw_t = factor_weight.t()
            mw_t = mix_weight.t()
            offs = list(range(0, n, chunk))
            # Reverse chunk order: autograd runs production's per-chunk checkpoint nodes last-chunk-first, so
            # the low-precision dW accumulation order (dW_last + dW_prev + ...) is the same as production's.
            for off in reversed(offs):
                end = min(off + chunk, n)
                m = end - off
                z = hh[off:end]
                lg = logits_buf[:m]
                mx = mix_buf[:m]
                torch.mm(z, fw_t, out=lg)          # same GEMM as F.linear(z, W) for 2-D z, bias=None
                torch.mm(z, mw_t, out=mx)
                rpv16_kernel_launch(lg, mx, yy[off:end], nll[off:end], scale, with_grad)
                if need_h:
                    torch.mm(lg, factor_weight, out=dh[off:end])
                    th = tmp_h[:m]
                    torch.mm(mx, mix_weight, out=th)
                    dh[off:end].add_(th)
                if need_fw:
                    torch.mm(lg.t(), z, out=tmp_fw)
                    dfw.add_(tmp_fw)
                if need_mw:
                    torch.mm(mx.t(), z, out=tmp_mw)
                    dmw.add_(tmp_mw)
            # loss reduction exactly like production: FP32 chunk sums accumulated in forward chunk order
            nll_sum = torch.zeros((), device=dev, dtype=torch.float32)
            for off in offs:
                nll_sum = nll_sum + nll[off:min(off + chunk, n)].sum()
            out = nll_sum / n if mean else nll_sum
        ctx.save_for_backward(dh, dfw, dmw)
        # NLL is already computed by the fused kernel; expose it for detached SAT regret targets.
        ctx.mark_non_differentiable(nll)
        return out, nll

    @staticmethod
    def backward(ctx, grad_out, grad_nll):
        dh, dfw, dmw = ctx.saved_tensors
        g = grad_out.to(torch.float32)
        if ctx.hint != 1.0:
            g = g / ctx.hint                        # == 1.0 when the caller's hint was right

        def sc(x):
            # scale in FP32 then round once (exact no-op for grad_out == 1.0 and any power of two)
            return None if x is None else (x.to(torch.float32) * g).to(x.dtype)

        return sc(dh), None, sc(dfw), sc(dmw), None, None, None


def rpv16_fused_ce(hidden_rows, labels, factor_weight, mix_weight, ce_chunk=4096, reduction="mean",
                   grad_scale_hint=1.0, return_token_nll=False):
    """Rank-16 product-vocab CE, production-equivalent.

    hidden_rows : [N,256] (or [...,256]) float tensor (production: bf16) -- out_proj(norm(x)) rows
    labels      : [N] (or [...]) int64 next-token labels, already shifted; every row counts (no ignore mask,
                  same as production; out-of-range labels wrap like torch.remainder, same as production)
    factor_weight: [16*(512+256), 256] = model.factor_head.weight (no bias in production)
    mix_weight  : [16, 256]            = model.mix_head.weight    (no bias in production)
    returns     : FP32 scalar ce = sum(nll)/N  (reduction="mean", production) or sum(nll) (reduction="sum").
                  With return_token_nll=True, returns (ce, detached FP32 NLL [N]) in original row order.
                  NLL is an auxiliary target/metric only; backpropagate through ce.
    grad_scale_hint: the upstream gradient the caller WILL feed to this loss, i.e. 1/grad_accum for
                  (loss / grad_accum).backward().  Production runs --grad-accum 1, so the default 1.0 is exact.
                  The grads are computed in forward (Liger-style) and rounded to BF16 there; production rounds
                  to BF16 AFTER multiplying by the upstream gradient.  For upstream gradients that are a power
                  of two the two orders are bit-identical; for others (e.g. 1/3) pass the hint so the factor is
                  folded in before the rounding (otherwise a second BF16 rounding adds ~4e-3 relative noise).
                  The result is always mathematically correct: backward rescales by grad_output / hint.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")
    hh = hidden_rows.reshape(-1, hidden_rows.shape[-1])
    yy = labels.reshape(-1)
    if yy.dtype != torch.long:
        yy = yy.long()
    yy = yy.contiguous()
    if hh.shape[0] != yy.shape[0]:
        raise ValueError(f"rows {hh.shape[0]} != labels {yy.shape[0]}")
    if factor_weight.shape != (VOCAB_RANK * (VOCAB_GROUP_PAD + VOCAB_LOW), hh.shape[1]) or mix_weight.shape != (VOCAB_RANK, hh.shape[1]):
        raise ValueError("unexpected head weight shapes")
    if not (hh.is_cuda and hh.dtype == factor_weight.dtype == mix_weight.dtype):
        raise ValueError("hidden rows and head weights must be CUDA tensors of one dtype")
    if hh.stride(1) != 1:
        hh = hh.contiguous()
    if not torch.is_grad_enabled():
        hh, factor_weight, mix_weight = hh.detach(), factor_weight.detach(), mix_weight.detach()
    hint = float(grad_scale_hint)
    if not (hint > 0.0 and hint < float("inf")):
        raise ValueError("grad_scale_hint must be a positive finite float")
    ce, nll = _RPV16FusedCE.apply(hh, yy, factor_weight, mix_weight, int(ce_chunk), reduction == "mean", hint)
    return (ce, nll) if return_token_nll else ce

# <<< FUSED_CE_EMBED END

def training_topology(model):
    """Cache references only for the fixed trainer topology; never checkpoint them.

    Model device/dtype transforms and state restoration clear this cache. Call
    clear_training_topology_cache() after intentional module/parameter surgery.
    Parameter order and per-module alias deduplication match nn.Module exactly.
    """
    cache = getattr(model, "_training_topology_cache", None)
    if cache is None:
        modules = tuple(model.modules())
        params = {m: tuple(m.parameters()) for m in modules}
        sparse = tuple(m for m in modules if isinstance(m, SparseLinear))
        cache = {
            "parameters": params,
            "all": params[model],
            "sparse": sparse,
            "stage_sparse": tuple(tuple(m for m in st.modules()
                                        if isinstance(m, SparseLinear))
                                  for st in model.stages),
            # Do not deduplicate across stages: the original telemetry counted
            # a shared parameter once for each stage that contains it.
            "body": tuple((p, p.numel()) for st in model.stages for p in params[st]),
        }
        model._training_topology_cache = cache
        print(json.dumps({"event":"host_topology_cache_enabled",
                          "deployment":"pf2-topology-live-20260920",
                          "parameter_objects":len(cache["all"]),
                          "sparse_layers":len(cache["sparse"]),
                          "source":__file__}), flush=True)
    return cache

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, TOKEN_DIM, device="cuda", dtype=torch.bfloat16)
        self.in_proj = nn.Linear(TOKEN_DIM, D, bias=False, device="cuda", dtype=torch.bfloat16)
        self.stages = nn.ModuleList([Stage(w) for w in WINDOWS])
        self.norm = RMSNorm(D)
        self.out_proj = nn.Linear(D, TOKEN_DIM, bias=False, device="cuda", dtype=torch.bfloat16)
        self.factor_head = nn.Linear(TOKEN_DIM, VOCAB_RANK * (VOCAB_GROUP_PAD + VOCAB_LOW), bias=False, device="cuda", dtype=torch.bfloat16)
        self.mix_head = nn.Linear(TOKEN_DIM, VOCAB_RANK, bias=False, device="cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def init_factor_head_from_embedding(self):
        # Warm-start each normalized product component from the trained token
        # embedding under a different bijective affine vocabulary layout.
        # For each rank, E[token] ~= group_weight[g] + low_weight[l].
        E = self.embed.weight.detach().float()
        W = self.factor_head.weight.view(VOCAB_RANK, VOCAB_GROUP_PAD + VOCAB_LOW, TOKEN_DIM)
        W.zero_()
        ids = torch.arange(VOCAB, device=E.device, dtype=torch.long)
        for r,(a,b) in enumerate(zip(VOCAB_AFFINE_A, VOCAB_AFFINE_B)):
            # p(token)=(a*token+b) mod V; scatter trained embeddings into code order.
            code = torch.remainder(ids * int(a) + int(b), VOCAB)
            grid = torch.empty((VOCAB, TOKEN_DIM), device=E.device, dtype=torch.float32)
            grid.index_copy_(0, code, E)
            grid = grid.view(VOCAB_GROUPS, VOCAB_LOW, TOKEN_DIM)
            gw = grid.mean(dim=1)
            lw = (grid - gw[:,None,:]).mean(dim=0)
            W[r, :VOCAB_GROUPS].copy_(gw.to(W.dtype))
            W[r, VOCAB_GROUP_PAD:].copy_(lw.to(W.dtype))
        self.mix_head.weight.zero_()

    def clear_training_topology_cache(self):
        self.__dict__.pop("_training_topology_cache", None)

    def _apply(self, fn, recurse=True):
        self.clear_training_topology_cache()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_training_topology_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def set_trainable_stage(self, stage_idx=None, head_anchor=False):
        # GB10-native local-update schedule. Exactly one Transformer stage owns
        # gradients on stage updates. A head anchor keeps all stages frozen and
        # updates only the tied token interface, avoiding a 2B-parameter global
        # gradient spike while still training embeddings/input/output/norm.
        topology = training_topology(self)
        for p in topology["all"]:
            p.requires_grad_(False)
        if head_anchor:
            # Output-side tied embedding + final projection only. The Transformer
            # body is evaluated under no_grad during this phase.
            for mod in (self.norm, self.out_proj, self.factor_head, self.mix_head):
                for p in topology["parameters"][mod]: p.requires_grad_(True)
        elif stage_idx is not None:
            for p in topology["parameters"][self.stages[int(stage_idx)]]:
                p.requires_grad_(True)
            # Stage 0 already requires an end-to-end gradient path, so train the
            # input token interface there at essentially no extra graph depth.
            if int(stage_idx) == 0:
                for mod in (self.embed, self.in_proj):
                    for p in topology["parameters"][mod]: p.requires_grad_(True)

    def forward(self, ids, labels=None, train_stage=None, head_anchor=False):
        # Rotating stage updates use a local tied-token objective: the prefix is
        # frozen, exactly one stage is trainable, and its representation is decoded
        # immediately by the shared final token head. This deliberately removes
        # downstream-stage backward to maximize GB10 sparse-NVFP4 wall-time share.
        aux_total = torch.zeros((), device=ids.device, dtype=torch.float32)
        route_counts = []
        if head_anchor:
            # Head-only phase: do not retain a 16-stage activation graph merely
            # to train the tied output vocabulary interface. The embedding still
            # receives its exact output-side CE gradient because it is the tied
            # LM-head weight below. Input-side embedding/in_proj gradients are
            # handled on the rotating stage-0 phase.
            with torch.no_grad():
                x = self.in_proj(self.embed(ids))
                for j in range(STAGES):
                    x, aux, counts = self.stages[j](x)
                    aux_total = aux_total + aux
                    route_counts.append(counts)
            x = x.detach()
        elif train_stage is None:
            x = self.in_proj(self.embed(ids))
            for j in range(STAGES):
                x, aux, counts = self.stages[j](x)
                aux_total = aux_total + aux
                route_counts.append(counts)
        else:
            # GB10-local objective: frozen prefix, exactly one trainable stage,
            # then jump directly to the shared tied-token decoder. There is no
            # downstream-stage forward/backward on a rotating stage update.
            ts = int(train_stage)
            if ts == 0:
                # Stage 0 also owns the input token interface, so keep this path
                # in autograd. Later stages consume a frozen prefix.
                x = self.in_proj(self.embed(ids))
            else:
                with torch.no_grad():
                    x = self.in_proj(self.embed(ids))
                    for j in range(ts):
                        x, aux, counts = self.stages[j](x)
                        aux_total = aux_total + aux
                        route_counts.append(counts)
            x, aux, counts = self.stages[ts](x)
            aux_total = aux_total + aux
            route_counts.append(counts)
        h = self.out_proj(self.norm(x))
        loss = None
        logits = None
        if labels is not None:
            if getattr(self, "fused_ce", False):
                hh = h.reshape(-1, TOKEN_DIM)
                yy = labels.reshape(-1)
                ce = rpv16_fused_ce(hh, yy, self.factor_head.weight, self.mix_head.weight,
                                    ce_chunk=int(getattr(self, "ce_chunk", 4096)),
                                    grad_scale_hint=float(getattr(self, "fused_ce_grad_scale_hint", 1.0)))
            else:
                # Rank-16 product vocabulary: 129280 == 505 * 256 exactly.
                # This is an exact normalized model distribution (not sampled CE):
                # p(token)=sum_r p(r)*p(group|r)*p(low|r). Seven padded group
                # classes are masked. The 256->12288 projection is GB10-friendly.
                from torch.utils.checkpoint import checkpoint
                # HFStream.batch already supplies labels shifted by one token, so every
                # hidden position i predicts its matching next-token label i.
                hh = h.reshape(-1, TOKEN_DIM)
                yy = labels.reshape(-1)
                affine_a = torch.tensor(VOCAB_AFFINE_A, device=yy.device, dtype=torch.long)
                affine_b = torch.tensor(VOCAB_AFFINE_B, device=yy.device, dtype=torch.long)
                chunk = int(getattr(self, "ce_chunk", 4096))
                nll_sum = torch.zeros((), device=h.device, dtype=torch.float32)
                def rpv_chunk(z, target, factor_weight, mix_weight, aa, bb):
                    raw = F.linear(z, factor_weight).float().view(-1, VOCAB_RANK, VOCAB_GROUP_PAD + VOCAB_LOW)
                    gl = raw[:, :, :VOCAB_GROUP_PAD]
                    gl[:, :, VOCAB_GROUPS:] = -1.0e9
                    ll = raw[:, :, VOCAB_GROUP_PAD:]
                    code = torch.remainder(target[:,None] * aa[None,:] + bb[None,:], VOCAB)
                    target_group = torch.div(code, VOCAB_LOW, rounding_mode="floor")
                    target_low = torch.remainder(code, VOCAB_LOW)
                    lg = gl.log_softmax(-1).gather(2, target_group.unsqueeze(2)).squeeze(2)
                    lo = ll.log_softmax(-1).gather(2, target_low.unsqueeze(2)).squeeze(2)
                    lm = F.linear(z, mix_weight).float().log_softmax(-1)
                    return -torch.logsumexp(lm + lg + lo, dim=1).sum()
                for off in range(0, hh.shape[0], chunk):
                    end = min(off + chunk, hh.shape[0])
                    nll_sum = nll_sum + checkpoint(rpv_chunk, hh[off:end], yy[off:end], self.factor_head.weight, self.mix_head.weight, affine_a, affine_b, use_reentrant=False)
                ce = nll_sum / yy.numel()
            loss = ce + 0.01 * aux_total / STAGES
        else:
            logits = self.factor_head(h)
        return loss, logits, aux_total / STAGES, route_counts
    def invalidate_sparse(self):
        for m in training_topology(self)["sparse"]:
            m.invalidate()
    def invalidate_stage(self, stage_idx):
        for m in training_topology(self)["stage_sparse"][int(stage_idx)]:
            m.invalidate()
    def sparse_telemetry(self):
        xs=training_topology(self)["sparse"]
        return {"layers":len(xs),"kernel_calls":sum(m.kernel_calls for m in xs),"packs":sum(m.pack_count for m in xs)}


_LOG_LOCK = threading.Lock()

def _emit(obj):
    # v2 runs a data producer thread next to the loop. print() issues the text and the
    # newline as two writes; emit each JSON line as one locked write instead. Bytes on the
    # wire are identical to print(json.dumps(obj), flush=True).
    line = json.dumps(obj) + "\n"
    with _LOG_LOCK:
        sys.stdout.write(line); sys.stdout.flush()


class PrefetchAborted(Exception):
    """A stop signal arrived while the loop was starved for data; no batch was consumed."""


class HFStream:
    """Restart-exact streaming corpus mixer.

    Hugging Face IterableDataset.state_dict() resumes the underlying stream, but
    HF's built-in shuffle intentionally drops its in-memory shuffle buffer on
    resume. For a 600B-token run that makes every restart silently alter/replay
    corpus coverage. We therefore keep the small row shuffle buffer ourselves and
    checkpoint it alongside the underlying dataset state, source RNG, and leftover
    token buffer.
    """
    STATE_SCHEMA = "agillm.gb10.1pf.data-cursor.v1"
    ROW_SHUFFLE = 200

    def __init__(self, names, seed=42):
        from datasets import load_dataset
        from tokenizers import Tokenizer
        self.load_dataset = load_dataset
        raw = TOKENIZER_JSON.read_text()
        try:
            obj = json.loads(raw)
        except Exception:
            obj = None
        if isinstance(obj, dict) and isinstance(obj.get("tokenizer_bundle"), dict) and isinstance(obj["tokenizer_bundle"].get("tokenizer.json"), str):
            self.tok = Tokenizer.from_str(obj["tokenizer_bundle"]["tokenizer.json"])
        else:
            self.tok = Tokenizer.from_str(raw)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.sources = [n for n in names if n in HF_SOURCES]
        if not self.sources: raise ValueError("no valid HF sources")
        self.weights = [HF_SOURCES[n][2] for n in self.sources]
        self.datasets = {}
        self.iters = {}
        self.row_buffers = {}
        self.disabled = set()
        self.buffer = []
        self.rows_emitted = {n: 0 for n in self.sources}
        # v2 prefetch (inert until start_prefetch): producer thread, bounded queue, and the
        # cursor snapshot belonging to the last batch the training loop actually consumed.
        self._pf_thread = None
        self._pf_queue = None
        self._pf_stop = None
        self._consumed = None
        self._loaded_dataset_state = None  # HF cursor handed to load_state_dict(); see start_prefetch()
        # Per-source exact iterator state used only to reconstruct a stream after a
        # transient remote FileNotFoundError / expired signed URL. This is not a
        # second cursor: after every successful yield it is replaced with the
        # dataset's own current state.
        self._transport_resume_state = {}
        self.abort_check = None
        self.last_qsize = None
        self._pf_gilfree = False
        self._pf_verify = 0

    def _open(self, name):
        repo,cfg,_=HF_SOURCES[name]
        kw=dict(split="train",streaming=True)
        if name == "fineweb-edu":
            kw["revision"] = FINEWEB_EDU_REVISION
        ds=self.load_dataset(repo,cfg,**kw) if cfg else self.load_dataset(repo,**kw)
        # Do NOT call HF IterableDataset.shuffle here: its shuffle buffer is not
        # included in state_dict(). Our explicit row_buffers below are.
        self.datasets[name]=ds
        self.iters[name]=iter(ds)
        return self.iters[name]

    def _reopen_at_transport_state(self, name, state):
        """Rebuild one HF iterator at an exact datasets state.

        Used only after a transport-resolution failure.  The source is reopened
        at the pinned revision, then the saved iterator state is installed before
        any row is consumed.
        """
        self.iters.pop(name, None)
        self.datasets.pop(name, None)
        self.disabled.discard(name)
        self._open(name)
        if state is not None:
            ds=self.datasets[name]
            if not hasattr(ds, "load_state_dict"):
                raise RuntimeError(f"stream source {name} lacks load_state_dict() during transport retry")
            ds.load_state_dict(copy.deepcopy(state))
            self.iters[name]=iter(ds)

    def _text(self,row):
        for k in ("text","content","code"):
            v=row.get(k) if isinstance(row,dict) else None
            if isinstance(v,str) and v.strip(): return v
        return ""

    def _raw_ids(self,name):
        # Finite sample configs eventually hit StopIteration and intentionally
        # wrap.  A remote FileNotFoundError is different: FineWeb's Xet/LFS
        # object still exists, but a signed URL / fsspec resolution may fail
        # transiently.  In that case rebuild the SAME iterator state and retry.
        wraps = 0
        transport_retries = 0
        while True:
            if name not in self.iters:
                self._open(name)
            ds=self.datasets.get(name)
            retry_state=self._transport_resume_state.get(name)
            if retry_state is None and ds is not None and hasattr(ds,"state_dict"):
                retry_state=copy.deepcopy(ds.state_dict())
            try:
                row=next(self.iters[name])
            except FileNotFoundError as e:
                transport_retries += 1
                if transport_retries > 5:
                    raise
                delay=min(8.0,0.5*(2**(transport_retries-1)))
                _emit({"event":"hf_transport_retry","source":name,
                       "attempt":transport_retries,"delay_s":delay,
                       "rows_emitted":int(self.rows_emitted.get(name,0)),
                       "error":type(e).__name__,"detail":str(e)[:240],
                       "cursor_action":"reopen_same_revision_same_state"})
                time.sleep(delay)
                self._reopen_at_transport_state(name,retry_state)
                continue
            except StopIteration:
                wraps += 1
                if wraps > 3:
                    raise RuntimeError(f"HF source {name} still empty after {wraps} wraps")
                _emit({"event":"hf_source_wrap","source":name,"wrap":wraps,
                       "rows_emitted":int(self.rows_emitted.get(name,0)),
                       "reason":"StopIteration on finite streaming sample; reopening from start"})
                self._transport_resume_state.pop(name,None)
                self.iters.pop(name, None)
                self.datasets.pop(name, None)
                self.disabled.discard(name)
                self._open(name)
                continue
            # Successful next(): datasets state is now authoritative for a retry
            # of the following row/chunk.
            transport_retries = 0
            ds=self.datasets.get(name)
            if ds is not None and hasattr(ds,"state_dict"):
                self._transport_resume_state[name]=copy.deepcopy(ds.state_dict())
            text=self._text(row)
            if not text:
                continue
            ids=self._encode_ids(text) if self._pf_gilfree else self.tok.encode(text).ids
            if len(ids)>4095:
                ids=ids[:4095]
            if ids:
                return ids

    def _prime_rows(self,name):
        buf=self.row_buffers.setdefault(name,[])
        while len(buf)<self.ROW_SHUFFLE:
            try: buf.append(self._raw_ids(name))
            except StopIteration: break
        return buf

    def _next_ids_from_source(self,name):
        buf=self._prime_rows(name)
        if not buf: raise StopIteration
        j=self.rng.randrange(len(buf))
        out=buf[j]
        try:
            buf[j]=self._raw_ids(name)
        except StopIteration:
            buf.pop(j)
        self.rows_emitted[name]=self.rows_emitted.get(name,0)+1
        return out

    def next_ids(self):
        available=[n for n in self.sources if n not in self.disabled]
        if not available: raise RuntimeError("all HF sources disabled")
        for _ in range(max(4,len(available)*2)):
            weights=[HF_SOURCES[n][2] for n in available]
            n=self.rng.choices(available,weights=weights,k=1)[0]
            try:
                return n,self._next_ids_from_source(n)
            except StopIteration:
                # Prefer wrap/reopen (finite sample configs) over permanent disable.
                try:
                    self.iters.pop(n,None); self.datasets.pop(n,None); self.disabled.discard(n)
                    self._open(n)
                    _emit({"event":"hf_source_wrap","source":n,"wrap":"next_ids",
                           "rows_emitted":int(self.rows_emitted.get(n,0)),
                           "reason":"StopIteration in next_ids; reopened from start"})
                    return n, self._next_ids_from_source(n)
                except Exception as wrap_e:
                    _emit({"event":"hf_source_wrap_failed","source":n,"error":type(wrap_e).__name__,"detail":str(wrap_e)[:240]})
                    self.disabled.add(n); self.iters.pop(n,None); self.datasets.pop(n,None)
                    available=[x for x in available if x!=n]
                    if not available: raise RuntimeError("all HF sources exhausted")
            except Exception as e:
                _emit({"dataset_disable":n,"error":type(e).__name__,"detail":str(e)[:240]})
                self.disabled.add(n); self.iters.pop(n,None); self.datasets.pop(n,None)
                available=[x for x in available if x!=n]
                if not available: raise
        raise RuntimeError("could not fetch nonempty HF row")

    @staticmethod
    def _pack_rows(rows):
        lens=torch.tensor([len(x) for x in rows],dtype=torch.int32)
        flat=torch.tensor([v for x in rows for v in x],dtype=torch.int32) if rows else torch.empty(0,dtype=torch.int32)
        return {"lengths":lens,"tokens":flat}

    @staticmethod
    def _unpack_rows(obj):
        lens=[int(x) for x in obj.get("lengths",torch.empty(0,dtype=torch.int32)).tolist()]
        flat=obj.get("tokens",torch.empty(0,dtype=torch.int32)).tolist()
        out=[]; off=0
        for n in lens:
            out.append(flat[off:off+n]); off+=n
        if off!=len(flat): raise RuntimeError("corrupt data cursor row buffer")
        return out

    def state_dict(self):
        ds_state={}
        for name,ds in self.datasets.items():
            if not hasattr(ds,"state_dict"):
                raise RuntimeError(f"stream source {name} lacks state_dict()")
            ds_state[name]=ds.state_dict()
        return {
            "schema":self.STATE_SCHEMA,
            "seed":self.seed,
            "sources":list(self.sources),
            "rng_state":self.rng.getstate(),
            "disabled":sorted(self.disabled),
            "token_buffer":torch.tensor(self.buffer,dtype=torch.int32),
            "row_buffers":{n:self._pack_rows(rows) for n,rows in self.row_buffers.items()},
            "dataset_state":ds_state,
            "rows_emitted":dict(self.rows_emitted),
        }

    def load_state_dict(self,state):
        if state.get("schema")!=self.STATE_SCHEMA:
            raise RuntimeError(f"unsupported data cursor schema {state.get('schema')!r}")
        if list(state.get("sources",[]))!=list(self.sources):
            raise RuntimeError(f"data source mismatch checkpoint={state.get('sources')} runtime={self.sources}")
        self.rng.setstate(state["rng_state"])
        self.disabled=set(state.get("disabled",[]))
        tb=state.get("token_buffer",torch.empty(0,dtype=torch.int32))
        self.buffer=[int(x) for x in tb.tolist()]
        self.row_buffers={n:self._unpack_rows(v) for n,v in state.get("row_buffers",{}).items()}
        self.rows_emitted={n:int(v) for n,v in state.get("rows_emitted",{}).items()}
        self.datasets={}; self.iters={}
        import copy  # function-local on purpose: this hunk needs no module-level import
        self._loaded_dataset_state=copy.deepcopy(state.get("dataset_state",{}))  # v2: used by start_prefetch() only
        self._transport_resume_state=copy.deepcopy(state.get("dataset_state",{}))
        for name,ds_state in state.get("dataset_state",{}).items():
            if name in self.disabled: continue
            self._open(name)
            ds=self.datasets[name]
            if not hasattr(ds,"load_state_dict"):
                raise RuntimeError(f"stream source {name} lacks load_state_dict()")
            ds.load_state_dict(ds_state)
            self.iters[name]=iter(ds)

    # ---- v2 prefetch: background producer + consumed-cursor snapshots -------------------
    DEVICE="cuda"

    def _snapshot(self):
        # Cheap capture of the restart-exact cursor at the producer's current position. Row
        # lists are never mutated in place (only replaced or popped), so shallow copies of the
        # row buffers are exact; HF state_dict() already returns a deep copy.
        ds_state={}
        for name,ds in self.datasets.items():
            if not hasattr(ds,"state_dict"):
                raise RuntimeError(f"stream source {name} lacks state_dict()")
            ds_state[name]=ds.state_dict()
        return {"rng_state":self.rng.getstate(),"disabled":sorted(self.disabled),"buffer":list(self.buffer),
                "row_buffers":{n:list(rows) for n,rows in self.row_buffers.items()},
                "dataset_state":ds_state,"rows_emitted":dict(self.rows_emitted)}

    def _materialize(self,snap):
        # Same keys, order, dtypes and values as state_dict() taken at the snapshot position.
        return {
            "schema":self.STATE_SCHEMA,
            "seed":self.seed,
            "sources":list(self.sources),
            "rng_state":snap["rng_state"],
            "disabled":snap["disabled"],
            "token_buffer":torch.tensor(snap["buffer"],dtype=torch.int32),
            "row_buffers":{n:self._pack_rows(rows) for n,rows in snap["row_buffers"].items()},
            "dataset_state":snap["dataset_state"],
            "rows_emitted":snap["rows_emitted"],
        }

    def consumed_state_dict(self):
        """Cursor as of the last batch handed to the training loop, never the read-ahead position."""
        if self._pf_thread is None:
            return self.state_dict()
        return self._materialize(self._consumed)

    GILFREE_VERIFY_ROWS=256

    def _encode_ids(self,text):
        # Producer thread only. Tokenizer.encode() holds the GIL for the whole Rust call (measured
        # on the box with the production tokenizer: a thread that needs the GIL back drops from
        # ~19,000 to ~400 reacquires/s next to it, ~5 ms per reacquire), so a producer using it
        # would stall the kernel-launch thread for most of the host work prefetch is meant to hide.
        # encode_batch([text]) runs the same pipeline on one input with the GIL released. It is
        # never trusted blindly: the first GILFREE_VERIFY_ROWS rows are encoded both ways, and any
        # difference or error switches this process back to encode() for good. The ids that are
        # used are v1's encode() ids in every such case.
        try:
            ids=self.tok.encode_batch([text])[0].ids
        except Exception as e:
            self._pf_gilfree=False
            _emit({"event":"prefetch_gilfree_disabled","ts":time.time(),"reason":"encode_batch failed: "+type(e).__name__})
            return self.tok.encode(text).ids
        if self._pf_verify>0:
            self._pf_verify-=1
            ref=self.tok.encode(text).ids
            if ids!=ref:
                self._pf_gilfree=False
                _emit({"event":"prefetch_gilfree_disabled","ts":time.time(),"reason":"encode_batch ids differ from encode ids; using encode()"})
                return ref
        return ids

    def start_prefetch(self,batch,seq,depth=3,pin=True,gil_release=True):
        if self._pf_thread is not None: raise RuntimeError("prefetch already started")
        self._pf_shape=(int(batch),int(seq)); self._pf_pin=bool(pin)
        self._pf_gilfree=bool(gil_release) and hasattr(self.tok,"encode_batch")
        self._pf_verify=self.GILFREE_VERIFY_ROWS
        if self._pf_gilfree: os.environ.setdefault("TOKENIZERS_PARALLELISM","false")  # one input per call: no rayon pool wanted
        self._pf_queue=queue.Queue(maxsize=max(1,int(depth)))
        self._pf_stop=threading.Event()
        self._consumed=self._snapshot()  # nothing consumed yet: the loaded/initial cursor
        if self._loaded_dataset_state is not None:
            # real `datasets` reports a ROW-0 cursor between load_state_dict() and the first next(): keep the loaded one
            import copy
            self._consumed["dataset_state"]=copy.deepcopy(self._loaded_dataset_state)
        self._pf_thread=threading.Thread(target=self._pf_run,name="hfstream-prefetch",daemon=True)
        self._pf_thread.start()

    def _pf_run(self):
        # Sole owner of the stream state from here on. One producer + FIFO queue keeps the v1
        # batch order; each batch travels with the cursor snapshot taken right after it.
        batch,seq=self._pf_shape
        need=batch*(seq+1)
        try:
            while not self._pf_stop.is_set():
                source_counts={}
                while len(self.buffer)<need:
                    src,ids=self.next_ids(); source_counts[src]=source_counts.get(src,0)+1
                    self.buffer.extend(ids); self.buffer.append(1)
                chunk=self.buffer[:need]; del self.buffer[:need]
                t=torch.tensor(chunk,dtype=torch.long).view(batch,seq+1)
                if self._pf_pin:
                    try: t=t.pin_memory()
                    except Exception: self._pf_pin=False
                self._pf_queue.put(("batch",t,source_counts,self._snapshot()))  # blocks while full
        except BaseException as e:
            self._pf_queue.put(("error",e,None,None))  # surfaces in the main thread, in order

    def _pf_next(self,batch,seq):
        if (int(batch),int(seq))!=self._pf_shape: raise RuntimeError("prefetch batch geometry changed")
        self.last_qsize=self._pf_queue.qsize()
        while True:
            try:
                kind,t,source_counts,snap=self._pf_queue.get(timeout=1.0)
                break
            except queue.Empty:
                if not self._pf_thread.is_alive() and self._pf_queue.empty():
                    raise RuntimeError("data prefetch producer exited without a result")
                if self.abort_check is not None and self.abort_check():
                    raise PrefetchAborted()
        if kind=="error": raise t
        self._consumed=snap
        t=t.to(self.DEVICE,non_blocking=True)
        return t[:,:seq],t[:,1:],source_counts

    def stop_prefetch(self,timeout=5.0):
        th=self._pf_thread
        if th is None or self._pf_stop.is_set(): return
        self._pf_stop.set()
        try:
            while True: self._pf_queue.get_nowait()  # release a producer blocked on a full queue
        except queue.Empty:
            pass
        th.join(timeout)  # daemon thread: a producer stuck in a network read cannot block exit

    def batch(self,batch,seq):
        if self._pf_thread is not None:
            return self._pf_next(batch,seq)
        need=batch*(seq+1)
        source_counts={}
        while len(self.buffer)<need:
            src,ids=self.next_ids(); source_counts[src]=source_counts.get(src,0)+1
            self.buffer.extend(ids); self.buffer.append(1)
        chunk=self.buffer[:need]; del self.buffer[:need]
        t=torch.tensor(chunk,dtype=torch.long,device="cuda").view(batch,seq+1)
        return t[:,:seq],t[:,1:],source_counts


# >>> ALLHEADS BEGIN  (generated by make_v3_allheads.py -- do not hand-edit)
# AR + SAT-fixed + SAT-var + NAT on the RPV16 DBlock-stagewise trainer.
#
# Core contract: NO new BF16/FP32 GEMM is added to the trainable transformer core.  The only new
# trainable tensors live OUTSIDE the core in AllHeadsAux: shift_emb [2,256] (an add on the 256-d head
# input) and the SAT-var regret MLP 256->256->2 (fp32, DETACHED input).  SAT adds one extra
# flash-attn call per stage (attention kernel, not a GEMM).  Model/Stage/Expert are byte-identical to v1.
ALLHEADS_SCHEMA = "agillm.gb10.1pf.allheads.v3"
ALLHEADS_CKPT_SCHEMA = "agillm.gb10.1pf.recovery.v6"
ALLHEADS_OBJECTIVES = ("ar", "sat", "nat")
ALLHEADS_PROB_FLOOR = 0.05
NAT_MASK_ID = 2                      # <pad> of the production tokenizer; same id as natmaskedhead_v2 / S
SATVAR_KMAX = 2                      # owner contract: SAT-var chooses between 1 and 2 tokens only
SATVAR_REGRET_HIDDEN = 256
SATVAR_REGRET_PRIOR_CE = (6.3, 6.9)  # S: SATVAR_REGRET_PRIOR_CE[:2]
ALLHEADS_HOT_DEFAULTS = {
    "dblock_ar_prob": 0.60, "dblock_sat_prob": 0.25, "dblock_nat_prob": 0.15,
    "sat_fixed_share": 0.5, "satvar_block_probs": "1:0.5,2:0.5",
    "satvar_lambda": 1.0, "satvar_merge_cost": None, "dblock_satvar_regret_weight": 0.05,
    "satvar_log_every": 20, "satvar_regret_max_blocks": 4096,
    "nat_mask_ratio": "uniform:0.1:1.0", "nat_span_mask_prob": 0.35, "nat_suffix_mask_prob": 0.20,
    "nat_region_partial_prob": 0.5,
}
_AH_M64 = (1 << 64) - 1


def _ah_mix64(x):
    x = (int(x) + 0x9E3779B97F4A7C15) & _AH_M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _AH_M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _AH_M64
    return x ^ (x >> 31)


def ah_hash64(seed, clock, stream=0, micro=0):
    """Counter-based 64-bit hash of (seed, update clock, stream, micro). No stored RNG state."""
    z = _ah_mix64(int(seed) & _AH_M64)
    z = _ah_mix64(z ^ (int(clock) & _AH_M64))
    z = _ah_mix64(z ^ ((int(stream) & 0xFFFFFFFF) << 32) ^ (int(micro) & 0xFFFFFFFF))
    return z


def ah_u01(seed, clock, stream=0, micro=0):
    return (ah_hash64(seed, clock, stream, micro) >> 11) / float(1 << 53)


def ah_generator(seed, clock, stream, micro=0):
    g = torch.Generator(device="cpu")
    g.manual_seed(ah_hash64(seed, clock, stream, micro) & ((1 << 63) - 1))
    return g


def ah_floor_probs(raw, floor=ALLHEADS_PROB_FLOOR):
    """Normalise objective probabilities and enforce p >= floor for EVERY objective (water-filling)."""
    names = ALLHEADS_OBJECTIVES
    vals = []
    for n in names:
        try:
            v = float(raw.get(n, 0.0))
        except Exception:
            v = 0.0
        vals.append(v if (v == v and v > 0.0 and v != float("inf")) else 0.0)
    if sum(vals) <= 0.0:
        vals = [1.0] * len(names)
    fixed = set()
    p = {}
    for _ in range(len(names) + 1):
        free = [i for i in range(len(names)) if i not in fixed]
        mass = 1.0 - floor * len(fixed)
        s = sum(vals[i] for i in free)
        p = {i: floor for i in fixed}
        for i in free:
            p[i] = (vals[i] / s * mass) if s > 0.0 else mass / len(free)
        low = [i for i in free if p[i] < floor - 1e-12]
        if not low:
            break
        fixed.update(low)
    return {names[i]: float(p[i]) for i in range(len(names))}


def ah_draw_objective(seed, clock, probs):
    """Deterministic objective for optimizer update `clock`: pure function of (seed, clock, probs)."""
    u = ah_u01(seed, clock, 0)
    acc = 0.0
    for n in ALLHEADS_OBJECTIVES:
        acc += float(probs[n])
        if u < acc:
            return n
    return ALLHEADS_OBJECTIVES[-1]


def ah_parse_block_probs(spec, floor=ALLHEADS_PROB_FLOOR):
    """'1:0.5,2:0.5' -> P(block size 2); both sizes are floored so neither stride can be configured away."""
    p = {1: 0.0, 2: 0.0}
    try:
        for part in str(spec).split(","):
            k, v = part.split(":")
            k = int(k)
            if k in p:
                p[k] = max(0.0, float(v))
    except Exception:
        p = {1: 0.5, 2: 0.5}
    tot = p[1] + p[2]
    p2 = 0.5 if tot <= 0.0 else p[2] / tot
    return min(1.0 - floor, max(floor, p2))


def ah_parse_mask_rate(spec):
    """'uniform:lo:hi' (default, per-sequence sampled rate) or a fixed float."""
    s = str(spec).strip().lower()
    try:
        if s.startswith("uniform"):
            parts = s.split(":")
            lo = float(parts[1]) if len(parts) > 1 else 0.1
            hi = float(parts[2]) if len(parts) > 2 else 1.0
        else:
            lo = hi = float(s)
    except Exception:
        lo, hi = 0.1, 1.0
    lo = min(1.0, max(0.01, lo)); hi = min(1.0, max(lo, hi))
    return lo, hi


# ---------------------------------------------------------------- SAT partition + attention
def ah_sat_partition(B, T, p2, fixed2, generator):
    """Per-row random partition of T positions into blocks of size {1,2} (CPU tensors).

    Returns first2 [B,T] bool (position is the FIRST token of a size-2 block), slot [B,T] (0/1),
    bsize [B,T] (1/2), bend [B,T] (last position of the block).  fixed2 -> all-2 blocks (fixed SAT).
    """
    B = int(B); T = int(T)
    if fixed2:
        sizes = torch.full((B, T), 2, dtype=torch.long)
    else:
        sizes = 1 + (torch.rand((B, T), generator=generator) < float(p2)).to(torch.long)
    starts = torch.cumsum(sizes, 1) - sizes
    ok = starts < T
    eff = torch.minimum(sizes, T - starts)
    rows = torch.arange(B).view(B, 1).expand(B, T)
    size_at = torch.zeros((B, T), dtype=torch.long)
    size_at[rows[ok], starts[ok]] = eff[ok]
    first2 = size_at == 2
    second2 = torch.zeros((B, T), dtype=torch.bool)
    second2[:, 1:] = first2[:, :-1]
    slot = second2.to(torch.long)
    bsize = 1 + (first2 | second2).to(torch.long)
    bend = torch.arange(T).view(1, T) + first2.to(torch.long)
    return first2, slot, bsize, bend


def ah_routing_alignment(B, T):
    """The balanced affine router couples the 6 tokens at flat indices g + k*(B*T/6). When B*T/6 is a multiple of
    T (production 24x2048) those are the SAME position in different rows, which cannot carry a row's own future
    back to it. Any other coupled geometry can, so block-causal / bidirectional objectives refuse to run on it."""
    n = int(B) * int(T)
    if n % EXPERTS:
        return "per_token"
    return "aligned" if (n // EXPERTS) % int(T) == 0 else "misaligned"


def _ah_sdpa_masked(q, k, v, allow):
    """Dense-mask reference attention. q [B,S,Hq,Dh], k/v [B,Sk,Hkv,Dh], allow [B or 1,S,Sk] bool."""
    hq, hkv = q.shape[2], k.shape[2]
    qq = q.transpose(1, 2)
    kk = k.transpose(1, 2); vv = v.transpose(1, 2)
    if hq != hkv:
        kk = kk.repeat_interleave(hq // hkv, dim=1); vv = vv.repeat_interleave(hq // hkv, dim=1)
    y = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=allow[:, None, :, :])
    return y.transpose(1, 2)


def ah_flash_semantics_dense(q, k, v, window, causal=True):
    """Dense emulation of flash_attn_func(causal, window_size=(window,0)) INCLUDING its bottom-right
    alignment when seqlen_k != seqlen_q: query i sees keys j in [i+Sk-Sq-window, i+Sk-Sq]."""
    S, Sk = q.shape[1], k.shape[1]
    i = torch.arange(S, device=q.device).view(S, 1) + (Sk - S)
    j = torch.arange(Sk, device=q.device).view(1, Sk)
    allow = torch.ones((S, Sk), dtype=torch.bool, device=q.device)
    if causal:
        allow = allow & (j <= i)
    if window is not None and int(window) >= 0:
        allow = allow & (j >= i - int(window))
    return _ah_sdpa_masked(q, k, v, allow[None])


def ah_block_causal_true_dense(q, k, v, window, bend):
    """TRUE partition block-causal attention: q at i attends k at j iff j <= block_end(i)
    (bidirectional inside a block, causal across blocks) and, for sliding-window stages, i-j <= window."""
    S = q.shape[1]
    i = torch.arange(S, device=q.device).view(1, S, 1)
    j = torch.arange(S, device=q.device).view(1, 1, S)
    allow = j <= bend.to(q.device).view(-1, S, 1)
    if window is not None and int(window) >= 0:
        allow = allow & ((i - j) <= int(window))
    return _ah_sdpa_masked(q, k, v, allow)


def ah_block_causal_twocall(q, k, v, window, first2, attn_fn):
    """Two-call block-causal attention for {1,2} partitions, exact at sequence edges and under windows.

    call A = the ordinary causal call (stage window w).
    call B = the same causal call against K/V given ONE extra trailing slot, so the (bottom-right
             aligned) causal frontier of every query moves one key to the right; window w+1 keeps the
             left edge where call A has it.  A query that is the FIRST token of a size-2 block takes
             call B (it sees exactly one extra key: its block partner), everything else takes call A.
    The literal "shift K/V left by one" variant drops key 0 (and key i-w under a window) from call B;
    padding on the right instead is the edge-exact form of the same idea.  The pad slot is only
    visible to the last query, which can never be the first token of a size-2 block.
    """
    w = -1 if window is None else int(window)
    yA = attn_fn(q, k, v, w)
    kB = F.pad(k, (0, 0, 0, 0, 0, 1)); vB = F.pad(v, (0, 0, 0, 0, 0, 1))
    yB = attn_fn(q, kB, vB, -1 if w < 0 else w + 1)
    return torch.where(first2.to(q.device)[:, :, None, None], yB, yA)


def _ah_flash_causal(q, k, v, w):
    from flash_attn import flash_attn_func
    return flash_attn_func(q, k, v, dropout_p=0.0, causal=True, window_size=((-1, -1) if w < 0 else (w, 0)))


class _AHAttnState:
    mode = "ar"          # "ar" | "sat" | "nat_bidir"
    first2 = None


_AH_ATTN = _AHAttnState()


class ah_attn_mode:
    def __init__(self, mode, first2=None):
        self.mode, self.first2 = mode, first2
    def __enter__(self):
        _AH_ATTN.mode, _AH_ATTN.first2 = self.mode, self.first2
        return self
    def __exit__(self, *exc):
        _AH_ATTN.mode, _AH_ATTN.first2 = "ar", None
        return False


def ah_attention(q, k, v, window):
    """Non-AR attention; SAT uses pinned, reversible one-key merge dispatch."""
    mode = _AH_ATTN.mode
    if mode == "sat":
        fn = _ah_flash_causal if q.is_cuda else (lambda a, b, c, w: ah_flash_semantics_dense(a, b, c, w))
        from sg_sat_partner_0ac2c32d import dispatch
        return dispatch(q, k, v, window, _AH_ATTN.first2,
                        lambda: ah_block_causal_twocall(q, k, v, window, _AH_ATTN.first2, fn),
                        emit=_emit)
    if mode == "nat_bidir":
        if q.is_cuda:
            from flash_attn import flash_attn_func
            return flash_attn_func(q, k, v, dropout_p=0.0, causal=False, window_size=(-1, -1))
        return ah_flash_semantics_dense(q, k, v, -1, causal=False)
    raise ValueError(f"unknown attention mode {mode!r}")


# ---------------------------------------------------------------- NAT corruption
def _ah_nat_span_len(T, rate, u):
    target = max(1, min(T, int(round(T * max(0.01, min(0.95, float(rate)))))))
    hi = min(T, max(target, target * 2))
    lo = max(1, min(hi, target // 2 if target > 1 else 1))
    return lo + min(hi - lo, int(float(u) * (hi - lo + 1)))


def ah_nat_corrupt(ids, cfg, generator):
    """S-style corruption mix (suffix / span / Bernoulli) with a per-sequence SAMPLED mask rate so
    iterative mask-predict ("discrete diffusion") decoding is in-distribution.  The mask never
    depends on token values.  Returns (corrupted ids, mask [B,T] bool on ids.device, stats)."""
    B, T = ids.shape
    lo, hi = ah_parse_mask_rate(cfg["nat_mask_ratio"])
    suffix_p = min(1.0, max(0.0, float(cfg["nat_suffix_mask_prob"])))
    span_p = min(1.0 - suffix_p, max(0.0, float(cfg["nat_span_mask_prob"])))
    partial_p = min(1.0, max(0.0, float(cfg["nat_region_partial_prob"])))
    u = torch.rand((B, 6), generator=generator)
    bern = torch.rand((B, T), generator=generator)
    mask = torch.zeros((B, T), dtype=torch.bool)
    kinds = {"suffix": 0, "span": 0, "bernoulli": 0}
    rates = []
    for b in range(B):
        rate = lo + (hi - lo) * float(u[b, 1])
        kind = float(u[b, 0])
        if kind < suffix_p or kind < suffix_p + span_p:
            n = _ah_nat_span_len(T, rate, u[b, 2])
            start = (T - n) if kind < suffix_p else min(T - n, int(float(u[b, 3]) * (T - n + 1)))
            kinds["suffix" if kind < suffix_p else "span"] += 1
            region = torch.zeros(T, dtype=torch.bool); region[start:start + n] = True
            if float(u[b, 4]) < partial_p:
                inner = lo + (hi - lo) * float(u[b, 5])      # a partially refined region (later passes)
                region = region & (bern[b] < inner)
            mask[b] = region
        else:
            kinds["bernoulli"] += 1
            mask[b] = bern[b] < rate
        if not bool(mask[b].any()):
            mask[b, min(T - 1, int(float(u[b, 3]) * T))] = True
        rates.append(rate)
    dmask = mask.to(ids.device)
    corrupted = torch.where(dmask, torch.full_like(ids, int(NAT_MASK_ID)), ids)
    stats = {"mask_frac": float(mask.float().mean()), "kinds": kinds,
             "rate_min": min(rates), "rate_max": max(rates)}
    return corrupted, dmask, stats


# ---------------------------------------------------------------- aux heads (outside the core)
class AllHeadsAux(nn.Module):
    """shift_emb: learned per-slot shift added to the 256-d head input (zero-init: slot 0 == AR head).
    stride_regret: SAT-var policy head regressing log per-slot CE from the block's last hidden (detached)."""
    def __init__(self, device):
        super().__init__()
        self.shift_emb = nn.Parameter(torch.zeros((SATVAR_KMAX, TOKEN_DIM), device=device, dtype=torch.float32))
        self.stride_regret = nn.Sequential(
            nn.Linear(TOKEN_DIM, SATVAR_REGRET_HIDDEN), nn.GELU(), nn.Linear(SATVAR_REGRET_HIDDEN, SATVAR_KMAX)).to(device=device, dtype=torch.float32)
        self.register_buffer("regret_updates", torch.zeros((), dtype=torch.long, device=device), persistent=True)
        self.init_()

    @torch.no_grad()
    def init_(self):
        # deterministic fresh init on a private generator: never consumes the global torch RNG
        self.shift_emb.zero_()
        lin0, lin1 = self.stride_regret[0], self.stride_regret[2]
        g = torch.Generator(device="cpu").manual_seed(20260912)
        bound = 1.0 / math.sqrt(float(TOKEN_DIM))
        lin0.weight.copy_(((torch.rand(lin0.weight.shape, generator=g) * 2.0 - 1.0) * bound).to(lin0.weight.device))
        lin0.bias.zero_()
        lin1.weight.zero_()
        lin1.bias.copy_(torch.log(torch.tensor(SATVAR_REGRET_PRIOR_CE, dtype=torch.float32)).to(lin1.bias.device))
        self.regret_updates.zero_()

    def slot_hidden(self, h, slot):
        return h + self.shift_emb.index_select(0, slot).to(h.dtype)

    def regret_log_ce(self, h_last):
        return self.stride_regret(h_last.detach().float()).float()


def ah_rpv_token_nll(model, hh, yy):
    """Per-token NLL under the exact rank-16 product-vocab distribution (same math as v1 rpv_chunk)."""
    from torch.utils.checkpoint import checkpoint
    affine_a = torch.tensor(VOCAB_AFFINE_A, device=yy.device, dtype=torch.long)
    affine_b = torch.tensor(VOCAB_AFFINE_B, device=yy.device, dtype=torch.long)
    chunk = int(getattr(model, "ce_chunk", 4096))
    def rpv_chunk_tok(z, target, factor_weight, mix_weight, aa, bb):
        raw = F.linear(z, factor_weight).float().view(-1, VOCAB_RANK, VOCAB_GROUP_PAD + VOCAB_LOW)
        gl = raw[:, :, :VOCAB_GROUP_PAD].clone()
        gl[:, :, VOCAB_GROUPS:] = -1.0e9
        ll = raw[:, :, VOCAB_GROUP_PAD:]
        code = torch.remainder(target[:, None] * aa[None, :] + bb[None, :], VOCAB)
        target_group = torch.div(code, VOCAB_LOW, rounding_mode="floor")
        target_low = torch.remainder(code, VOCAB_LOW)
        lg = gl.log_softmax(-1).gather(2, target_group.unsqueeze(2)).squeeze(2)
        lo = ll.log_softmax(-1).gather(2, target_low.unsqueeze(2)).squeeze(2)
        lm = F.linear(z, mix_weight).float().log_softmax(-1)
        return -torch.logsumexp(lm + lg + lo, dim=1)
    outs = []
    for off in range(0, hh.shape[0], chunk):
        end = min(off + chunk, hh.shape[0])
        outs.append(checkpoint(rpv_chunk_tok, hh[off:end], yy[off:end], model.factor_head.weight, model.mix_head.weight, affine_a, affine_b, use_reentrant=False))
    return torch.cat(outs) if len(outs) != 1 else outs[0]


def ah_trunk(model, ids, ts):
    """v1's rotating-stage branch verbatim: frozen no_grad prefix 0..ts-1, trainable stage ts, no downstream."""
    aux_total = torch.zeros((), device=ids.device, dtype=torch.float32)
    route_counts = []
    ts = int(ts)
    if ts == 0:
        x = model.in_proj(model.embed(ids))
    else:
        with torch.no_grad():
            x = model.in_proj(model.embed(ids))
            for j in range(ts):
                x, aux, counts = model.stages[j](x)
                aux_total = aux_total + aux
                route_counts.append(counts)
    x, aux, counts = model.stages[ts](x)
    aux_total = aux_total + aux
    route_counts.append(counts)
    return x, aux_total, route_counts


def ah_satvar_gate_stats(ce1, ce2, pred_log, lam, mu):
    """Gate telemetry on size-2 blocks. Teacher: stride 2 acceptable iff lam - max(0, CE2-CE1) > 0 on the
    REALISED per-slot CE; policy: the same rule on the regret head's CEhat.  Base rate and lift are always
    reported next to precision/recall."""
    gap = (ce2 - ce1).float()
    teacher = (float(lam) - gap.clamp_min(0.0)) > 0.0
    chat = pred_log.float().exp()
    gap_hat = chat[:, 1] - chat[:, 0]
    policy = (float(lam) - gap_hat.clamp_min(0.0)) > 0.0
    policy_merge = (float(lam) - gap_hat.clamp_min(0.0) - float(mu)) > 0.0
    n = int(gap.numel())
    tp = int((teacher & policy).sum()); npos = int(teacher.sum()); npred = int(policy.sum())
    base = npos / max(1, n)
    prec = (tp / npred) if npred > 0 else None
    rec = (tp / npos) if npos > 0 else None
    lift = (prec / base) if (prec is not None and base > 0.0) else None
    qs = torch.quantile(gap[: min(n, 4_000_000)], torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=gap.device)).tolist() if n > 0 else []
    return {"n_blocks2": n, "lam": float(lam), "mu": float(mu),
            "policy_stride_hist": {"1": n - npred, "2": npred},
            "teacher_stride_hist": {"1": n - npos, "2": npos},
            "policy_decision_basis": "learned_ce_regret_plus_explicit_speed_reward",
            "policy_counts_are_emitted_tokens": False,
            "base_rate": round(base, 5), "pred_rate": round(npred / max(1, n), 5),
            "precision": None if prec is None else round(prec, 5), "recall": None if rec is None else round(rec, 5),
            "lift": None if lift is None else round(lift, 4),
            "ce_gap_q10_25_50_75_90": [round(float(x), 4) for x in qs],
            "mean_stride_teacher": round(1.0 + base, 5),
            "mean_stride_policy": round(1.0 + npred / max(1, n), 5),
            "mean_stride_policy_if_merge_needed": round(1.0 + int(policy_merge.sum()) / max(1, n), 5),
            "ce1_mean": round(float(ce1.float().mean()), 4), "ce2_mean": round(float(ce2.float().mean()), 4),
            "cehat1_mean": round(float(chat[:, 0].mean()), 4), "cehat2_mean": round(float(chat[:, 1].mean()), 4)}


def ah_fused_ce_fn(model):
    """The composed stack embeds rpv16_fused_ce (fused_ce patch set) and switches it on with model.fused_ce.
    The v1-based candidate has neither, so this returns None there and the reference CE below is used."""
    fn = globals().get("rpv16_fused_ce")
    return fn if (callable(fn) and bool(getattr(model, "fused_ce", False))) else None


def ah_ce_mean(model, hh, yy):
    """Mean RPV CE and all per-row NLL. Fused NLL is detached, matching its SAT regret-target use."""
    fn = ah_fused_ce_fn(model)
    if fn is not None:
        hint = float(getattr(model, "fused_ce_grad_scale_hint", getattr(model, "ce_grad_scale", 1.0)))
        return fn(hh, yy, model.factor_head.weight, model.mix_head.weight, ce_chunk=int(getattr(model, "ce_chunk", 4096)), grad_scale_hint=hint, return_token_nll=True)
    nll = ah_rpv_token_nll(model, hh, yy)
    return nll.mean(), nll


class AllHeadsRuntime:
    """All non-AR training state. Absent (None) under --objectives ar, which leaves v1 behaviour untouched."""
    def __init__(self, args, model, head_params, device):
        self.args = args
        self.seed = int(args.seed)
        self.device = device
        self.head_params = list(head_params)
        self.nat_attn = str(args.nat_attn)
        self.head_grad = str(args.allheads_head_grad)
        self.aux_lr_mult = float(args.allheads_aux_lr_mult)
        self.aux = AllHeadsAux(device)
        self.aux_opt = torch.optim.AdamW(list(self.aux.parameters()), lr=float(args.lr), betas=(0.9, 0.95), weight_decay=0.0)
        self.hot_path = Path(args.allheads_hot_config) if args.allheads_hot_config else Path(args.save_dir) / "allheads_hot.json"
        self._hot_mtime = None
        self.cli_cfg = dict(ALLHEADS_HOT_DEFAULTS)
        self.cli_cfg.update({
            "dblock_ar_prob": args.dblock_ar_prob, "dblock_sat_prob": args.dblock_sat_prob, "dblock_nat_prob": args.dblock_nat_prob,
            "sat_fixed_share": args.sat_fixed_share, "satvar_block_probs": args.satvar_block_probs,
            "satvar_lambda": args.satvar_lambda, "satvar_merge_cost": (None if args.satvar_merge_cost < 0 else args.satvar_merge_cost),
            "dblock_satvar_regret_weight": args.satvar_regret_weight, "satvar_log_every": args.satvar_log_every,
            "satvar_regret_max_blocks": getattr(args, "satvar_regret_max_blocks", 4096),
            "nat_mask_ratio": args.nat_mask_rate, "nat_span_mask_prob": args.nat_span_mask_prob,
            "nat_suffix_mask_prob": args.nat_suffix_mask_prob, "nat_region_partial_prob": args.nat_region_partial_prob})
        self.cfg = dict(self.cli_cfg)
        self._clock = None
        self._objective = "ar"
        self._fixed2 = False
        self.probs = ah_floor_probs(self._raw_probs())
        self.counts = {n: 0 for n in ALLHEADS_OBJECTIVES}
        self.sat_steps = 0
        self.info = {}
        self.step_info = {}
        self.debug = False
        self.debug_out = {}
        # AR-protective norm matching (make_v3_4_normmatch): per-stage log-EMA of the pre-clip AR grad norm
        self.nonar_clip_mult = float(getattr(args, "allheads_nonar_clip_mult", 1.0))
        self.nonar_clip_default = 0.02
        self.ar_gn_logema = {}
        self.ar_gn_n = {}
        self._last_nonar_max_norm = None
        n_aux = sum(p.numel() for p in self.aux.parameters())
        print(json.dumps({"event": "allheads_ready", "schema": ALLHEADS_SCHEMA, "objectives": list(ALLHEADS_OBJECTIVES),
                          "probs": self.probs, "prob_floor": ALLHEADS_PROB_FLOOR, "hot_config": str(self.hot_path),
                          "nat_attn": self.nat_attn, "nat_mask_id": NAT_MASK_ID, "head_grad": self.head_grad,
                          "aux_parameters_outside_core": n_aux,
                          "core_contract": "no new BF16/FP32 GEMM in the trainable transformer core; outside-core additions: shift_emb[2,256] fp32 add on the 256-d head input, SAT-var regret MLP 256-256-2 fp32 on a DETACHED input; SAT adds one extra flash-attn call per stage",
                          "satvar_contract": "always enabled; block sizes {1,2}; policy = learned CE regret + explicit speed reward lam; no disable switch, no forced constant stride"}), flush=True)

    # ---- configuration
    def _raw_probs(self):
        return {"ar": self.cfg["dblock_ar_prob"], "sat": self.cfg["dblock_sat_prob"], "nat": self.cfg["dblock_nat_prob"]}

    def reload_hot(self):
        try:
            mt = self.hot_path.stat().st_mtime_ns
        except OSError:
            mt = None
        if mt == self._hot_mtime:
            return False
        self._hot_mtime = mt
        cfg = dict(self.cli_cfg)
        applied = {}
        if mt is not None:
            try:
                obj = json.loads(self.hot_path.read_text())
                if not isinstance(obj, dict):
                    raise ValueError("hot config must be a JSON object")
                if "satvar_regret_weight" in obj and "dblock_satvar_regret_weight" not in obj:
                    obj["dblock_satvar_regret_weight"] = obj["satvar_regret_weight"]
                for k in ALLHEADS_HOT_DEFAULTS:
                    if k in obj:
                        cfg[k] = obj[k]; applied[k] = obj[k]
            except Exception as e:
                print(json.dumps({"event": "allheads_hot_config_rejected", "path": str(self.hot_path), "error": type(e).__name__, "detail": str(e)[:200], "kept": "previous values"}), flush=True)
                return False
        self.cfg = cfg
        self.probs = ah_floor_probs(self._raw_probs())
        print(json.dumps({"event": "allheads_hot_config", "path": str(self.hot_path), "present": mt is not None, "applied": applied, "probs": self.probs, "effective": self.effective()}), flush=True)
        return True

    def effective(self):
        c = self.cfg
        def _f(key, default, lo, hi):
            try:
                v = float(c.get(key, default))
                if v != v: v = default
            except Exception:
                v = default
            return min(hi, max(lo, v))
        lam = _f("satvar_lambda", 1.0, 1.0e-3, 100.0)              # explicit speed reward, never zero
        mc = c.get("satvar_merge_cost", None)
        try:
            mu = 0.5 * lam if mc is None else max(0.0, float(mc))
        except Exception:
            mu = 0.5 * lam
        return {"lam": lam, "mu": mu,
                "sat_fixed_share": _f("sat_fixed_share", 0.5, 0.0, 0.9),   # variable partitions always keep >= 10% of SAT updates
                "p2": ah_parse_block_probs(c.get("satvar_block_probs", "1:0.5,2:0.5")),
                "regret_weight": _f("dblock_satvar_regret_weight", 0.05, 1.0e-4, 1.0),
                "log_every": int(_f("satvar_log_every", 20, 1, 1_000_000)),
                "regret_max_blocks": int(_f("satvar_regret_max_blocks", 4096, 256, 1 << 20))}

    # ---- objective schedule
    def objective_for(self, clock, micro_in_update=0):
        clock = int(clock)
        if clock != self._clock:
            self.reload_hot()
            self._clock = clock
            self._objective = ah_draw_objective(self.seed, clock, self.probs)
            self._fixed2 = ah_u01(self.seed, clock, 1) < self.effective()["sat_fixed_share"]
        return self._objective

    def wants_head_grad(self, objective):
        return (objective == "sat" and self.head_grad in ("sat", "sat+nat")) or (objective == "nat" and self.head_grad == "sat+nat")

    # ---- forward
    def forward(self, objective, model, ids, labels, train_stage, clock, micro):
        if train_stage is None:
            raise RuntimeError("non-AR objectives run only on rotating stage updates; the head anchor stays AR")
        if (objective == "sat" or self.nat_attn == "bidir") and ah_routing_alignment(*ids.shape) == "misaligned":
            raise RuntimeError(f"batch x seq = {tuple(ids.shape)} couples DIFFERENT positions through the balanced router; non-causal objectives need batch*seq/{EXPERTS} to be a multiple of seq (production 24x2048 is) or a per-token-routed shape")
        if self.wants_head_grad(objective):
            for p in self.head_params:
                p.requires_grad_(True)       # model.set_trainable_stage() resets this at the next update
        if objective == "sat":
            return self._forward_sat(model, ids, labels, train_stage, clock, micro)
        if objective == "nat":
            return self._forward_nat(model, ids, labels, train_stage, clock, micro)
        raise ValueError(f"unknown objective {objective!r}")

    def _forward_sat(self, model, ids, labels, ts, clock, micro):
        eff = self.effective()
        B, T = ids.shape
        dev = ids.device
        gen = ah_generator(self.seed, clock, 2, micro)
        # ONE partition shared by every row (as S does). With per-row partitions the router's cross-row coupling at
        # equal positions opens a path  token p -> row r' (pairs p-1,p) -> routing group p-1 -> row r position p-1,
        # whose target can be token p. A shared partition closes it: every position that can be influenced by token p
        # has block_end >= p in EVERY row, and targets always lie beyond block_end.
        first2, slot, bsize, bend = (x.expand(B, T) for x in ah_sat_partition(1, T, eff["p2"], self._fixed2, gen))
        first2_d = first2.to(dev)
        with ah_attn_mode("sat", first2_d):
            x, aux_total, route_counts = ah_trunk(model, ids, ts)
        h = model.out_proj(model.norm(x))
        slot_d = slot.to(dev).reshape(-1); bsize_d = bsize.to(dev)
        # slot j of a block predicts the token at pos + blocksize (first token the block cannot see).
        # labels[:, i] is token i+1, so token pos+bsize is labels[:, pos+bsize-1].
        tgt_idx = torch.arange(T, device=dev).view(1, T) + bsize_d - 1
        valid = (tgt_idx <= T - 1).reshape(-1)
        sel = valid.nonzero(as_tuple=False).flatten()
        hflat = h.reshape(-1, TOKEN_DIM)
        hh = self.aux.slot_hidden(hflat.index_select(0, sel), slot_d.index_select(0, sel))
        y_all = labels.gather(1, tgt_idx.clamp(max=T - 1)).reshape(-1)
        yy = y_all.index_select(0, sel)
        ce, nll = ah_ce_mean(model, hh, yy)
        # ---- SAT-var regret head: regress log per-slot CE from the block's LAST hidden state (detached)
        nll_grid = torch.zeros(B * T, device=dev, dtype=torch.float32)
        second2_d = (slot_d == 1)
        last = ((bsize_d.reshape(-1) == 1) | second2_d) & valid          # every slot of the block has a target
        last_idx = last.nonzero(as_tuple=False).flatten()
        if nll is not None:
            nll_grid.index_copy_(0, sel, nll.detach().float())
        else:
            # Compatibility fallback for external CE adapters without row NLL. The embedded full-NLL kernel
            # never takes this path. Like S (which samples 256 blocks), take regret targets from an
            # exact no-grad reference CE on a sampled subset of blocks, drawn from the same dedicated generator.
            cap = int(eff["regret_max_blocks"])
            if int(last_idx.numel()) > cap:
                pick = torch.randperm(int(last_idx.numel()), generator=gen)[:cap].sort().values.to(dev)
                last_idx = last_idx.index_select(0, pick)
            _is2 = second2_d.index_select(0, last_idx)
            rows = torch.cat([last_idx, last_idx[_is2] - 1])
            with torch.no_grad():
                hr = self.aux.slot_hidden(hflat.index_select(0, rows), slot_d.index_select(0, rows)).detach()
                nll_grid.index_copy_(0, rows, ah_rpv_token_nll(model, hr, y_all.index_select(0, rows)).float())
        pred = self.aux.regret_log_ce(hflat.index_select(0, last_idx))   # [M,2]
        is2 = second2_d.index_select(0, last_idx)
        idx2 = last_idx[is2]
        idx1 = last_idx[~is2]
        tlog = lambda v: torch.log(v.clamp_min(1.0e-3))
        pred_parts = [pred[~is2, 0], pred[is2, 0], pred[is2, 1]]
        tgt_parts = [tlog(nll_grid.index_select(0, idx1)), tlog(nll_grid.index_select(0, idx2 - 1)), tlog(nll_grid.index_select(0, idx2))]
        pred_sel = torch.cat(pred_parts); target_log = torch.cat(tgt_parts)
        regret = F.smooth_l1_loss(pred_sel, target_log, beta=0.5) if pred_sel.numel() else torch.zeros((), device=dev)
        loss = ce + 0.01 * aux_total / STAGES + eff["regret_weight"] * regret
        n1 = int((bsize == 1).sum()); n2 = int(first2.sum())
        self.sat_steps += 1
        info = {"objective": "sat", "n_targets": int(sel.numel()), "sat_ce": float(ce.detach()), "sat_regret_loss": float(regret.detach()),
                "ce_path": "fused_full_nll" if ah_fused_ce_fn(model) is not None else "reference", "regret_blocks": int(last_idx.numel()),
                "sat": {"fixed": bool(self._fixed2), "blocks1": n1, "blocks2": n2, "mean_block": round((n1 + 2 * n2) / max(1, n1 + n2), 4),
                        "frac_tokens_in_2": round(2 * n2 / max(1, B * T), 4), "p2": eff["p2"], "lam": eff["lam"], "regret_weight": eff["regret_weight"]}}
        if self.sat_steps % eff["log_every"] == 0 or self.sat_steps == 1:
            with torch.no_grad():
                if int(idx2.numel()) > 0:
                    gate = ah_satvar_gate_stats(nll_grid.index_select(0, idx2 - 1), nll_grid.index_select(0, idx2), pred[is2].detach(), eff["lam"], eff["mu"])
                else:
                    gate = {"n_blocks2": 0}
                # Symbolic baseline for the learned cost: the best CONSTANT per slot (this batch's own mean log-CE, i.e. an
                # oracle constant, so the comparison is conservative).  Predictions are made BEFORE this batch is trained
                # on, so regret_mae_log is an online held-out error; skill > 0 means the head beats the constant.
                mae = mae_const = skill = None
                if pred_sel.numel():
                    n_a, n_b = int(idx1.numel()), int(idx2.numel())
                    col0 = target_log[: n_a + n_b]; col1 = target_log[n_a + n_b:]
                    dev_abs = torch.cat([(c - c.mean()).abs() for c in (col0, col1) if c.numel()])
                    mae = float((pred_sel - target_log).abs().mean()); mae_const = float(dev_abs.mean())
                    skill = (1.0 - mae / mae_const) if mae_const > 0.0 else None
                gate.update({"event": "satvar_gate", "sat_step": self.sat_steps, "optimizer_update_clock": int(clock), "train_stage": int(ts),
                             "partition": "fixed2" if self._fixed2 else "variable", "regret_updates": int(self.aux.regret_updates),
                             "regret_mae_log": None if mae is None else round(mae, 4),
                             "regret_mae_log_const_baseline": None if mae_const is None else round(mae_const, 4),
                             "regret_skill_vs_const": None if skill is None else round(skill, 4)})
            print(json.dumps(gate), flush=True)
            info["satvar_gate"] = {k: gate.get(k) for k in ("base_rate", "pred_rate", "precision", "recall", "lift", "mean_stride_policy", "regret_skill_vs_const")}
        if self.debug:
            self.debug_out = {"h": h.detach(), "sel": sel, "yy": yy, "first2": first2, "slot": slot, "bsize": bsize, "bend": bend, "nll": (None if nll is None else nll.detach()), "tgt_idx": tgt_idx, "nll_grid": nll_grid, "last_idx": last_idx}
        self.info = info
        return loss, aux_total / STAGES, route_counts

    def _forward_nat(self, model, ids, labels, ts, clock, micro):
        gen = ah_generator(self.seed, clock, 3, micro)
        model_ids, mask, stats = ah_nat_corrupt(ids, self.cfg, gen)
        if self.nat_attn == "bidir":
            with ah_attn_mode("nat_bidir"):
                x, aux_total, route_counts = ah_trunk(model, model_ids, ts)
        else:
            # causal denoiser: identical attention kernel/windows to AR. The network has no positional
            # encoding, so causality is its only position signal (see allheads_ready / self-tests).
            x, aux_total, route_counts = ah_trunk(model, model_ids, ts)
        h = model.out_proj(model.norm(x))
        # true denoising target: ORIGINAL token i at masked position i; rows gathered BEFORE the
        # 256->12288 projection (natmaskedhead_v2 head compaction).
        sel = mask.reshape(-1).nonzero(as_tuple=False).flatten()
        hh = h.reshape(-1, TOKEN_DIM).index_select(0, sel)
        yy = ids.reshape(-1).index_select(0, sel)
        ce, nll = ah_ce_mean(model, hh, yy)
        loss = ce + 0.01 * aux_total / STAGES
        self.info = {"objective": "nat", "n_targets": int(sel.numel()), "nat_ce": float(ce.detach()), "ce_path": "fused_full_nll" if ah_fused_ce_fn(model) is not None else "reference", "nat": dict(stats, attn=self.nat_attn)}
        if self.debug:
            self.debug_out = {"h": h.detach(), "sel": sel, "yy": yy, "mask": mask, "model_ids": model_ids}
        return loss, aux_total / STAGES, route_counts

    # ---- AR-protective gradient norm matching
    def note_ar_grad_norm(self, stage, grad_norm):
        """Record the PRE-clip grad norm of an AR stage update (never called for the head anchor or non-AR updates)."""
        if stage is None:
            return
        g = float(grad_norm)
        if not (g == g) or g <= 0.0 or g == float("inf"):
            return
        s = int(stage)
        lg = math.log(g)
        n = self.ar_gn_n.get(s, 0)
        self.ar_gn_logema[s] = lg if n == 0 else 0.9 * self.ar_gn_logema[s] + 0.1 * lg
        self.ar_gn_n[s] = n + 1

    def nonar_max_norm(self, stage):
        """Clip threshold for a SAT/NAT stage update: mult x the stage's typical AR grad norm, never above v1's 1.0."""
        if self.nonar_clip_mult <= 0.0:
            self._last_nonar_max_norm = 1.0
            return 1.0
        s = None if stage is None else int(stage)
        ref = math.exp(self.ar_gn_logema[s]) if self.ar_gn_n.get(s, 0) >= 4 else self.nonar_clip_default
        m = min(1.0, max(1.0e-4, self.nonar_clip_mult * ref))
        self._last_nonar_max_norm = m
        return m

    # ---- optimizer side
    def after_step(self, objective, lr_now, head_opt):
        self.step_info = {}
        self.counts[objective] = self.counts.get(objective, 0) + 1
        if objective == "ar":
            return
        ps = [p for p in self.aux.parameters() if p.grad is not None]
        if ps:
            gn = torch.nn.utils.clip_grad_norm_(ps, 1.0)
            if not torch.isfinite(gn): raise RuntimeError("nonfinite allheads aux grad norm")
            for pg in self.aux_opt.param_groups: pg["lr"] = float(lr_now) * self.aux_lr_mult
            self.aux_opt.step()
            self.aux.regret_updates += 1
            self.step_info["aux_grad_norm"] = float(gn)
        self.aux_opt.zero_grad(set_to_none=True)
        if self.wants_head_grad(objective):
            gh = torch.nn.utils.clip_grad_norm_(self.head_params, 1.0)
            if not torch.isfinite(gh): raise RuntimeError("nonfinite head grad norm on a non-AR update")
            for pg in head_opt.param_groups: pg["lr"] = float(lr_now)
            head_opt.step()
            head_opt.zero_grad(set_to_none=True)
            self.step_info["head_grad_norm"] = float(gh)

    def telemetry(self, objective, n_tokens):
        if objective == "ar":
            out = {"objective": "ar", "n_targets": int(n_tokens)}
        else:
            out = dict(self.info)
            out["nonar_max_norm"] = self._last_nonar_max_norm
        out.update(self.step_info)
        out["objective_probs"] = self.probs
        out["objective_counts"] = dict(self.counts)
        self.step_info = {}
        return out

    # ---- checkpoint (schema v6 = v5 + this blob)
    def ckpt_extra(self):
        return {"allheads": {"schema": ALLHEADS_SCHEMA, "aux": self.aux.state_dict(), "aux_optimizer": self.aux_opt.state_dict(),
                             "objective_counts": dict(self.counts), "sat_steps": int(self.sat_steps),
                             "config": {"hot": {k: self.cfg.get(k) for k in ALLHEADS_HOT_DEFAULTS}, "effective": self.effective(), "probs": self.probs,
                                        "nat_attn": self.nat_attn, "nat_mask_id": NAT_MASK_ID, "head_grad": self.head_grad,
                                        "satvar_kmax": SATVAR_KMAX, "shift_on": "head_input_256", "regret_input": "head_input_256_detached_block_last",
                                        "sat_attention": "block-causal {1,2}: call A causal(window w) + call B causal(window w+1) over K/V right-padded by one; first-of-pair rows take call B",
                                        "sat_target": "slot j at pos predicts token pos+blocksize", "sat_partition": "one {1,2} partition per micro-batch, shared by all rows"}}}

    def load_from_ckpt(self, ck):
        """Only a genuinely pre-SAT v5 checkpoint may initialize a new policy.
    
        An existing allheads checkpoint must restore its learned weights, optimizer,
        counters and effective policy configuration. Corruption is never interpreted
        as permission to silently reset the gate or discard its training state.
        """
        if not isinstance(ck, dict):
            raise RuntimeError("allheads resume requires a checkpoint dictionary")
        blob = ck.get("allheads")
        if blob is None:
            if ck.get("schema") != "agillm.gb10.1pf.recovery.v5":
                raise RuntimeError("only legacy v5 may initialize a missing allheads policy")
            print(json.dumps({"event": "allheads_fresh_init", "from_schema": ck["schema"],
                "to_schema": ALLHEADS_CKPT_SCHEMA, "fresh_modules": [
                    "allheads.aux.shift_emb (zeros)", "allheads.aux.stride_regret (prior-CE bias)",
                    "allheads.aux_optimizer (empty AdamW)"],
                "reason": "explicit v5-to-allheads migration; existing core state is not reset"}), flush=True)
            return "fresh"
        if not isinstance(blob, dict) or blob.get("schema") != ALLHEADS_SCHEMA:
            raise RuntimeError("allheads checkpoint schema mismatch")
        for key in ("aux", "aux_optimizer", "objective_counts", "config"):
            if not isinstance(blob.get(key), dict):
                raise RuntimeError("allheads checkpoint missing or malformed " + key)
        cfg = blob["config"]
        if cfg.get("satvar_kmax") != 2 or cfg.get("nat_mask_id") != NAT_MASK_ID:
            raise RuntimeError("allheads checkpoint stride or NAT mask identity changed")
        if cfg.get("nat_attn") not in ("causal", "bidir"):
            raise RuntimeError("allheads checkpoint invalid NAT attention mode")
        if cfg.get("head_grad") not in ("none", "sat", "sat+nat"):
            raise RuntimeError("allheads checkpoint invalid head gradient ownership")
        if not isinstance(cfg.get("hot"), dict):
            raise RuntimeError("allheads checkpoint lacks its saved objective configuration")
        unknown = set(cfg["hot"]) - set(ALLHEADS_HOT_DEFAULTS)
        if unknown:
            raise RuntimeError("allheads checkpoint has unsupported configuration keys: " + str(sorted(unknown)))
        counts = blob["objective_counts"]
        if set(counts) != set(ALLHEADS_OBJECTIVES) or any(type(v) is not int or v < 0 for v in counts.values()):
            raise RuntimeError("allheads checkpoint invalid objective counters")
        if type(blob.get("sat_steps")) is not int or blob["sat_steps"] < 0:
            raise RuntimeError("allheads checkpoint invalid SAT-step counter")
        def finite(x):
            if torch.is_tensor(x):
                return bool(torch.isfinite(x).all())
            if isinstance(x, dict):
                return all(finite(v) for v in x.values())
            if isinstance(x, (list, tuple)):
                return all(finite(v) for v in x)
            return not isinstance(x, float) or math.isfinite(x)
        if not finite(blob):
            raise RuntimeError("allheads checkpoint contains nonfinite learned state")
        # AdamW accepts some malformed state tensors at load time and fails only
        # during the next update. Validate parameter identity and geometry first.
        saved_aux = blob["aux"]
        live_aux = self.aux.state_dict()
        if set(saved_aux) != set(live_aux):
            raise RuntimeError("allheads auxiliary state keys mismatch")
        for key, target in live_aux.items():
            value = saved_aux[key]
            if not torch.is_tensor(value) or value.shape != target.shape or value.dtype != target.dtype:
                raise RuntimeError("allheads auxiliary tensor shape/dtype mismatch: " + key)
        updates = saved_aux["regret_updates"]
        if updates.numel() != 1 or int(updates) < 0:
            raise RuntimeError("allheads invalid learned-policy update counter")
        opt = blob["aux_optimizer"]
        saved_groups, saved_state = opt.get("param_groups"), opt.get("state")
        if not isinstance(saved_groups, list) or not isinstance(saved_state, dict):
            raise RuntimeError("allheads optimizer state/groups missing")
        if len(saved_groups) != len(self.aux_opt.param_groups):
            raise RuntimeError("allheads optimizer group count mismatch")
        mapped = {}
        for saved, current in zip(saved_groups, self.aux_opt.param_groups):
            ids = saved.get("params") if isinstance(saved, dict) else None
            if not isinstance(ids, list) or len(ids) != len(current["params"]):
                raise RuntimeError("allheads optimizer parameter count mismatch")
            for pid, param in zip(ids, current["params"]):
                if type(pid) is not int or pid in mapped:
                    raise RuntimeError("allheads optimizer duplicate/invalid parameter identity")
                mapped[pid] = param
            betas = saved.get("betas")
            if not isinstance(betas, (list, tuple)) or len(betas) != 2 or not all(0 <= float(b) < 1 for b in betas):
                raise RuntimeError("allheads optimizer invalid betas")
            if float(saved.get("lr", -1)) < 0 or float(saved.get("eps", 0)) <= 0 or float(saved.get("weight_decay", -1)) < 0:
                raise RuntimeError("allheads optimizer invalid learning hyperparameters")
        if any(pid not in mapped for pid in saved_state):
            raise RuntimeError("allheads optimizer state has unknown parameter identity")
        if int(updates) > 0 and not saved_state:
            raise RuntimeError("trained allheads policy has no optimizer moments")
        for pid, state in saved_state.items():
            if not isinstance(state, dict) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
                raise RuntimeError("allheads optimizer incomplete Adam state")
            step = state["step"]
            if torch.is_tensor(step):
                if step.numel() != 1: raise RuntimeError("allheads optimizer step is not scalar")
                step = float(step)
            if not isinstance(step, (int, float)) or step < 0 or step != int(step):
                raise RuntimeError("allheads optimizer step is negative or fractional")
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key not in state: continue
                value = state[key]
                if not torch.is_tensor(value) or value.shape != mapped[pid].shape or not value.is_floating_point():
                    raise RuntimeError("allheads optimizer moment geometry mismatch: " + key)
        # Rejection must preserve the already installed policy, including when a
        # later saved-configuration check fails after state_dict loading succeeds.
        import copy as _ah_copy
        _aux_before = _ah_copy.deepcopy(self.aux.state_dict())
        _opt_before = _ah_copy.deepcopy(self.aux_opt.state_dict())
        _fields = ("counts", "sat_steps", "cfg", "cli_cfg", "nat_attn", "head_grad", "probs", "_clock", "_hot_mtime")
        _before = {key: _ah_copy.deepcopy(getattr(self, key)) for key in _fields}
        try:
            self.aux.load_state_dict(blob["aux"], strict=True)
            self.aux_opt.load_state_dict(blob["aux_optimizer"])
            self.counts = dict(counts)
            self.sat_steps = blob["sat_steps"]
            # Saved policy is the restart baseline. A subsequently read hot-config file
            # remains the explicit mechanism to change it after restoration.
            restored_cfg = dict(self.cli_cfg)
            restored_cfg.update(cfg["hot"])
            self.cfg = restored_cfg
            self.cli_cfg = dict(restored_cfg)
            self.nat_attn = cfg["nat_attn"]
            self.head_grad = cfg["head_grad"]
            self.probs = ah_floor_probs(self._raw_probs())
            effective = self.effective()
            old_effective = cfg.get("effective")
            if isinstance(old_effective, dict) and effective != old_effective:
                raise RuntimeError("saved allheads policy cannot be restored exactly")
            self._clock = None
            self._hot_mtime = None
            print(json.dumps({"event": "allheads_resumed", "schema": blob["schema"],
                "aux_optimizer": "resumed", "policy_config": "restored_from_checkpoint",
                "regret_updates": int(self.aux.regret_updates), "objective_counts": dict(self.counts),
                "effective": effective}), flush=True)
            return "resumed"
        except Exception:
            self.aux.load_state_dict(_aux_before, strict=True)
            self.aux_opt.load_state_dict(_opt_before)
            for key, value in _before.items(): setattr(self, key, value)
            raise


def allheads_setup(args, model, head_params, device="cuda"):
    spec = sorted(x.strip().lower() for x in str(args.objectives).split(",") if x.strip())
    if spec == ["ar"]:
        print(json.dumps({"event": "allheads_disabled_operator_flag", "objectives": "ar", "note": "v1 AR-only behaviour reproduced exactly; this violates the all-objectives owner contract and is meant for A/B and bisecting only"}), flush=True)
        return None
    if spec != sorted(ALLHEADS_OBJECTIVES):
        raise SystemExit("--objectives must be 'ar,sat,nat' (owner contract: all objectives stay active) or 'ar' (exact v1 reproduction)")
    return AllHeadsRuntime(args, model, head_params, device)
# <<< ALLHEADS END


def build_model():
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GB10":
        raise RuntimeError("AGILLM-GB10-1PF is intentionally GB10-only")
    return Model()


def profile():
    out={
        "name":NAME,"architecture":"hardware_locked_v2","parameters":parameter_accounting(),
        "training_tokens":TOTAL_TRAIN_TOKENS,"tokens_per_parameter":TOTAL_TRAIN_TOKENS/parameter_accounting()["total"],
        "vocab":VOCAB,"token_dim":TOKEN_DIM,"model_dim":D,"stages":STAGES,"experts_per_stage":EXPERTS,
        "expert_shape":[D,FFN],"attention":{"q_heads":Q_HEADS,"kv_heads":KV_HEADS,"head_dim":HEAD_DIM,"windows":WINDOWS},
        "sparse_target_tflops":1000.0,"verified_sparse_tflops":VERIFIED_SPARSE_TFLOPS,
        "tokenizer":str(TOKENIZER_JSON),"hf_sources":HF_SOURCES,
    }
    if torch.cuda.is_available(): out["device"]={"name":torch.cuda.get_device_name(0),"memory":torch.cuda.get_device_properties(0).total_memory}
    print(json.dumps(out,indent=2))


def dataset_probe(names):
    s=HFStream(names)
    src,text=s.next_text()
    ids=s.tok.encode(text).ids
    print(json.dumps({"ok":True,"source":src,"chars":len(text),"tokens":len(ids),"head_ids":ids[:16]},indent=2))


# ---- v2 full-stack host helpers ---------------------------------------------------------
_STOP = {"n": 0, "sig": None}
_BACKGROUND = {"stream": None, "writer": None, "ckpt_state": None, "flush": None}

def _on_signal(signum, frame):
    # Flag only. The loop finishes the current update, saves, and exits 0. Repeated signals
    # just count, so a second SIGTERM/SIGINT can never interrupt the save it triggered.
    _STOP["n"] += 1; _STOP["sig"] = int(signum)


def _mem_available_bytes():
    """Host bytes that can be taken without swapping: min(MemAvailable, cgroup-v2 headroom). None if unknown."""
    avail = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024
                break
    except Exception:
        pass
    try:
        mx = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if mx != "max":
            head = int(mx) - int(Path("/sys/fs/cgroup/memory.current").read_text())
            for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines():
                if line.startswith("inactive_file "):
                    head += int(line.split()[1])
                    break
            avail = head if avail is None else min(avail, head)
    except Exception:
        pass
    return avail


def _host_snapshot(obj, memo):
    """Deep host copy of a checkpoint payload for the background writer.

    Keeps container types and key order, OrderedDict attributes (state_dict _metadata), object
    identity sharing, storage sharing and full storage extents (torch.save writes whole
    storages) and tensor python attributes (bitsandbytes is_paged/page_deviceid), so
    torch.save(snapshot) pickles the same structure and tensor bytes as torch.save(live).
    Fails closed (TypeError) on anything it cannot mirror; the caller then saves synchronously.
    """
    if isinstance(obj, torch.Tensor):
        if type(obj) is not torch.Tensor or obj.layout is not torch.strided:
            raise TypeError(f"cannot host-snapshot tensor type {type(obj).__name__}/{obj.layout}")
        st = obj.untyped_storage(); nbytes = st.nbytes(); key = ("storage", st.data_ptr(), nbytes)
        host = memo.get(key)
        if host is None:
            host = torch.empty(nbytes, dtype=torch.uint8)
            if nbytes:
                host.copy_(torch.empty(0, dtype=torch.uint8, device=obj.device).set_(st))
                memo[key] = host
        out = torch.empty(0, dtype=obj.dtype).set_(host.untyped_storage(), obj.storage_offset(), obj.size(), obj.stride())
        if obj.requires_grad: out.requires_grad_(True)
        if obj.__dict__: out.__dict__.update(obj.__dict__)
        return out
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return obj
    key = ("id", id(obj))
    if key in memo:
        return memo[key][1]
    if isinstance(obj, dict):
        out = obj.__class__()
        memo[key] = (obj, out)
        for k, v in obj.items(): out[k] = _host_snapshot(v, memo)
        if getattr(obj, "__dict__", None): out.__dict__.update(copy.deepcopy(obj.__dict__))
        return out
    if isinstance(obj, list):
        out = []
        memo[key] = (obj, out)
        out.extend(_host_snapshot(v, memo) for v in obj)
        return out
    if isinstance(obj, tuple):
        items = [_host_snapshot(v, memo) for v in obj]
        if all(a is b for a, b in zip(items, obj)):
            out = obj  # immutable all the way down (rng state, betas): share it
        else:
            out = tuple(items) if type(obj) is tuple else obj.__class__(*items)
        memo[key] = (obj, out)
        return out
    out = copy.deepcopy(obj)
    memo[key] = (obj, out)
    return out


def _fsync_best_effort(path):
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    except OSError:
        pass


def _write_checkpoint(payload, tmp, dst, fsync):
    # v1 sequence (torch.save to latest.pt.tmp, atomic os.replace). With fsync the data is
    # durable before the rename and the directory entry after it.
    torch.save(payload, tmp)
    if fsync and os.name == "posix":
        fd = os.open(str(tmp), os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    os.replace(tmp, dst)
    if fsync: _fsync_best_effort(Path(dst).parent)


class _CkptWriter:
    """Single-slot background checkpoint writer: never more than one write in flight.

    Non-daemon on purpose: on any orderly interpreter exit the in-flight write completes, so
    latest.pt.tmp is not left behind (the keeper refuses to launch while it exists). A failed
    write removes its own tmp. Only a hard kill (SIGKILL/OOM/power) mid-write can strand
    latest.pt.tmp, exactly as it can during v1's synchronous save.
    """
    def __init__(self):
        self._thread = None
        self._result = None

    def start(self, payload, tmp, dst, event, fsync, extra, t_req):
        if self._thread is not None: raise RuntimeError("checkpoint write already in flight")
        box = [payload]
        self._thread = threading.Thread(target=self._run, args=(box, tmp, dst, dict(event), fsync, extra, t_req), name="ckpt-writer", daemon=False)
        self._thread.start()

    def _run(self, box, tmp, dst, event, fsync, extra, t_req):
        t0 = time.perf_counter()
        try:
            _write_checkpoint(box[0], tmp, dst, fsync)
            if extra is not None:
                now = time.perf_counter()
                event.update({"ts": time.time(), **extra, "write_s": now - t0, "save_s": now - t_req})
            self._result = {"ok": True, "event": event}
        except BaseException as e:
            try: os.unlink(tmp)
            except OSError: pass
            self._result = {"ok": False, "event": event, "error": f"{type(e).__name__}: {e}"[:240]}
        finally:
            box.clear()  # release the host snapshot as soon as the file is in place

    def finish(self, block):
        th = self._thread
        if th is None or (not block and th.is_alive()): return None
        th.join(); self._thread = None
        r, self._result = self._result, None
        return r


def _report_ckpt(result, ckpt_state):
    # Main thread only. The "checkpoint" event is published after os.replace, so a log
    # follower never sees it before latest.pt is the new file.
    if result is None: return
    if result["ok"]:
        _emit(result["event"])
        return
    if ckpt_state is not None:
        ckpt_state["async_disabled"] = True; ckpt_state["retry"] = True; ckpt_state["saved_clock"] = -1
    ev = result["event"]
    _emit({"event": "checkpoint_async_failed", "ts": time.time(), "path": ev.get("path"), "step": ev.get("step"), "error": result["error"], "action": "tmp removed; async disabled for this process; synchronous retry at the next update boundary"})


def _shutdown_background():
    # Any exit path from train(): stop the producer and let an in-flight write finish.
    stream = _BACKGROUND["stream"]; writer = _BACKGROUND["writer"]; flush = _BACKGROUND["flush"]
    if flush is not None:
        try: flush(True)  # best effort: deferred step records of a run that is dying for another reason
        except Exception: pass
    if stream is not None: stream.stop_prefetch()
    if writer is not None: _report_ckpt(writer.finish(block=True), _BACKGROUND["ckpt_state"])


def train(args):
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sources = list(HF_SOURCES) if args.dataset == "mix" else [args.dataset]
    stream=HFStream(sources,args.seed)
    print(json.dumps({"event":"dataset_ready","sources":sources,"tokenizer":str(TOKENIZER_JSON)}),flush=True)
    t0=time.time(); model=build_model(); model.ce_chunk=args.ce_chunk; model.fused_ce=bool(args.fused_ce); model.fused_ce_grad_scale_hint=1.0/float(max(1,args.grad_accum)); torch.cuda.synchronize()
    params=sum(p.numel() for p in model.parameters())
    print(json.dumps({"event":"model_ready","parameters":params,"expected":parameter_accounting()["total"],"build_s":time.time()-t0,"cuda_alloc":torch.cuda.memory_allocated(),"target_alignment":TARGET_ALIGNMENT}),flush=True)
    if params != parameter_accounting()["total"]: raise RuntimeError("parameter count drift")
    def _uniq_params(xs):
        out=[]; seen_ids=set()
        for q in xs:
            if id(q) not in seen_ids:
                seen_ids.add(id(q)); out.append(q)
        return out
    stage_param_sets=[]
    for si in range(STAGES):
        ps=list(model.stages[si].parameters())
        if si == 0:
            ps += list(model.embed.parameters()) + list(model.in_proj.parameters())
        stage_param_sets.append(_uniq_params(ps))
    head_params=_uniq_params(list(model.norm.parameters()) + list(model.out_proj.parameters()) + list(model.factor_head.parameters()) + list(model.mix_head.parameters()))
    try:
        import bitsandbytes as bnb
        def _make_opt(ps):
            return bnb.optim.PagedAdamW8bit(ps,lr=args.lr,betas=(0.9,0.95),weight_decay=0.1)
        optimizer_name="paged_adamw8bit_bank"
    except Exception:
        def _make_opt(ps):
            return torch.optim.AdamW(ps,lr=args.lr,betas=(0.9,0.95),weight_decay=0.1,foreach=True)
        optimizer_name="adamw_bank"
    stage_opts=[_make_opt(ps) for ps in stage_param_sets]
    head_opt=_make_opt(head_params)
    AH=allheads_setup(args,model,head_params)
    seen=0; step_times=[]; start_step=1; optimizer_update_clock=0; micro_in_update=0
    if args.resume and Path(args.resume).exists():
        t_resume0=time.perf_counter(); stream_restore_s=None
        ck=torch.load(args.resume,map_location="cpu",weights_only=False)
        ckpt_load_s=time.perf_counter()-t_resume0
        missing, unexpected = model.load_state_dict(ck["model"],strict=False)
        if missing or unexpected:
            print(json.dumps({"event":"model_resume_non_strict","missing":missing[:16],"unexpected":unexpected[:16]}),flush=True)
        if "factor_head.weight" in missing or "mix_head.weight" in missing:
            with torch.no_grad():
                model.factor_head.weight.mul_(0.10)
                model.mix_head.weight.zero_()
            print(json.dumps({"event":"rpv16_affine_near_uniform_init","factor_scale":0.10,"mix":"zero"}),flush=True)
        resume_target_alignment=ck.get("target_alignment")
        bank=ck.get("optimizer_bank")
        if resume_target_alignment != TARGET_ALIGNMENT:
            bank=None
            print(json.dumps({"event":"optimizer_target_migration_reset","from":resume_target_alignment or "legacy_double_shift_i_plus_2","to":TARGET_ALIGNMENT,"reason":"objective alignment changed; stale Adam moments rejected"}),flush=True)
        if isinstance(bank,dict) and isinstance(bank.get("stages"),list) and len(bank["stages"]) == STAGES:
            try:
                for o,sd in zip(stage_opts,bank["stages"]): o.load_state_dict(sd)
                try:
                    head_opt.load_state_dict(bank["head"])
                    head_state = "resumed"
                except Exception as e:
                    raise RuntimeError("head optimizer resume failed; refusing reset on exact production lineage") from e
                print(json.dumps({"event":"optimizer_bank_resumed","optimizer":optimizer_name,"head_state":head_state}),flush=True)
            except Exception as e:
                raise RuntimeError("optimizer bank resume failed; refusing partial-state training: " + str(e)[:240]) from e
        elif "optimizer" in ck:
            print(json.dumps({"event":"optimizer_migration_reset","from":ck.get("optimizer_name","legacy"),"to":optimizer_name,"reason":"static optimizer bank migration"}),flush=True)
        if AH is not None: AH.load_from_ckpt(ck)
        seen=int(ck.get("seen_tokens",0)); ck_step=int(ck.get("step",0)); start_step=ck_step+1
        if "optimizer_update_clock" in ck:
            optimizer_update_clock=int(ck["optimizer_update_clock"])
            micro_in_update=int(ck.get("micro_in_update",0))
            if micro_in_update != 0:
                raise RuntimeError("checkpoint must be saved at an optimizer-update boundary")
            phase_clock_source="checkpoint"
            resume_grad_accum=int(ck.get("grad_accum",args.grad_accum))
        else:
            legacy_ga=int(args.legacy_resume_grad_accum)
            if legacy_ga < 1 or ck_step % legacy_ga:
                raise RuntimeError(f"cannot derive legacy optimizer clock: step={ck_step} legacy_grad_accum={legacy_ga}")
            optimizer_update_clock=ck_step // legacy_ga
            micro_in_update=0
            phase_clock_source="legacy_derived"
            resume_grad_accum=legacy_ga
        if isinstance(ck.get("data_state"),dict):
            t_stream0=time.perf_counter()
            stream.load_state_dict(ck["data_state"])
            stream_restore_s=time.perf_counter()-t_stream0
            ds=ck["data_state"]
            print(json.dumps({"event":"data_cursor_resumed","schema":ds.get("schema"),"rows_emitted":ds.get("rows_emitted",{}),"token_buffer":int(ds.get("token_buffer",torch.empty(0)).numel())}),flush=True)
        else:
            print(json.dumps({"event":"data_cursor_migration_reset","reason":"legacy checkpoint lacks exact stream cursor","checkpoint_step":ck_step,"seen_tokens":seen,"note":"one-time data-order reset; future v4 checkpoints resume exactly"}),flush=True)
        print(json.dumps({"event":"resumed","path":args.resume,"step":start_step-1,"seen_tokens":seen,"optimizer_update_clock":optimizer_update_clock,"phase":optimizer_update_clock%(STAGES+1),"phase_clock_source":phase_clock_source,"resume_grad_accum":resume_grad_accum,"requested_grad_accum":args.grad_accum,"target_alignment":TARGET_ALIGNMENT,"resume_target_alignment":resume_target_alignment,**({} if args.v1_telemetry else {"ts":time.time(),"ckpt_load_s":ckpt_load_s,"stream_restore_s":stream_restore_s,"resume_total_s":time.perf_counter()-t_resume0,"proc_uptime_s":time.time()-_PROC_T0})}),flush=True)
        del ck
        import gc; gc.collect(); torch.cuda.empty_cache()
    train_stage = None
    global_anchor = False
    active_opt = None
    active_params = None
    # ---- v2 full-stack host path. Nothing here alters batches, RNG use, loss, gradients,
    # optimizer updates, the phase clock, or checkpoint contents/schema.
    lazy = not args.strict_step_sync
    tel2 = not args.v1_telemetry
    ckpt_writer = _CkptWriter()
    ckpt_state = {"async_disabled": False, "retry": False, "saved_clock": optimizer_update_clock}
    _BACKGROUND.update({"stream": stream, "writer": ckpt_writer, "ckpt_state": ckpt_state})
    pending = deque()
    # Full pipeline rates count only train records whose CUDA completion event has
    # completed. Host enqueue timestamps and GPU event durations are not wall time.
    _pipeline_start = time.perf_counter()
    _pipeline_seen0 = seen
    _pipeline_targets = 0

    def _finalize_pipeline_telemetry(rec):
        nonlocal _pipeline_targets
        _elapsed = time.perf_counter() - _pipeline_start
        _pipeline_targets += int(rec.get("n_targets", args.batch * args.seq))
        rec.update({
            "data_pipeline": "astra-speedstack-completed-cursor-v1",
            "pipeline_tok_s": (int(rec["seen_tokens"]) - _pipeline_seen0) / _elapsed,
            "pipeline_target_s": _pipeline_targets / _elapsed,
            "wall_time_s": _elapsed,
            "data_prefetch_qsize": rec.get("prefetch_q", stream.last_qsize),
        })

    def _flush_train_log(block):
        # Deferred per-step record + NaN/Inf guard. Same scalars v1 checked, read from async
        # host copies once the step's end event has completed. Blocks only when asked to
        # (defer bound reached, or before any state capture).
        while pending:
            rec,host,ev0,ev1=pending[0]
            if block: ev1.synchronize()
            elif not ev1.query(): return
            pending.popleft()
            loss_v,aux_v,gn_v=host.tolist()
            bad="nonfinite loss" if not math.isfinite(loss_v) else ("nonfinite grad norm" if rec["optimizer_step"] and not math.isfinite(gn_v) else None)
            if bad:
                pending.clear()  # as in v1, nothing at or after the bad step is ever logged or saved
                raise RuntimeError(f"{bad} (step {rec['step']})")
            dt=ev0.elapsed_time(ev1)/1000.0
            step_times.append(dt)
            rec["loss"]=loss_v; rec["aux"]=aux_v; rec["grad_norm"]=gn_v if rec["optimizer_step"] else None
            rec["step_s"]=dt; rec["tok_s"]=args.batch*args.seq/dt
            _finalize_pipeline_telemetry(rec)
            _emit(rec)

    _BACKGROUND["flush"]=_flush_train_log

    def _checkpoint(step,seen_tokens,force_sync=False,reason="periodic"):
        # The v1 save block, hoisted so the periodic boundary and the signal/retry path share
        # it. Callers guarantee an optimizer-update boundary and a passed NaN/Inf guard.
        t_req=time.perf_counter()
        _report_ckpt(ckpt_writer.finish(block=True),ckpt_state)  # never more than one write in flight
        sd=Path(args.save_dir); sd.mkdir(parents=True,exist_ok=True)
        dst=sd/"latest.pt"; tmp=sd/"latest.pt.tmp"
        # Auto-prune only an incomplete stale temp file from this exact save dir.
        # Complete checkpoints are never deleted here; post-upload pruning is handled separately.
        if tmp.exists():
            try:
                age=time.time()-tmp.stat().st_mtime
                if age >= 300:
                    tmp.unlink()
                    _emit({"event":"checkpoint_autoprune","path":str(tmp),"reason":"stale_tmp","age_s":age})
            except FileNotFoundError:
                pass
        expected = dst.stat().st_size if dst.exists() else 12*1024**3
        free = shutil.disk_usage(sd).free
        need = expected + 1024**3
        if free < need:
            _emit({"event":"checkpoint_skipped_disk","step":step,"free_bytes":free,"need_bytes":need,"path":str(dst)})
            return False
        payload={"schema":("agillm.gb10.1pf.recovery.v5" if AH is None else ALLHEADS_CKPT_SCHEMA),"target_alignment":TARGET_ALIGNMENT,"step":step,"seen_tokens":seen_tokens,"model":model.state_dict(),"optimizer_bank":{"stages":[o.state_dict() for o in stage_opts],"head":head_opt.state_dict()},"optimizer_name":optimizer_name,"seed":args.seed,"optimizer_update_clock":optimizer_update_clock,"grad_accum":args.grad_accum,"micro_in_update":micro_in_update,"data_state":stream.consumed_state_dict(),**({} if AH is None else AH.ckpt_extra())}
        event={"event":"checkpoint","path":str(dst),"step":step,"seen_tokens":seen_tokens,"schema":("v5" if AH is None else "v6"),"target_alignment":TARGET_ALIGNMENT,"optimizer_update_clock":optimizer_update_clock,"grad_accum":args.grad_accum}
        use_async=not (args.sync_checkpoint or force_sync or ckpt_state["async_disabled"])
        fallback=None; snap=None; avail=None
        if use_async:
            # Unified memory: the host snapshot competes with the GPU and /dev/shm peers.
            avail=_mem_available_bytes(); need_mem=expected+int(args.async_ckpt_reserve_gb*1024**3)
            if avail is None or avail < need_mem:
                use_async=False; fallback="low_host_memory"
        if use_async:
            try:
                torch.cuda.synchronize()  # one sync: paged optimizer state is host-visible managed memory
                t_snap=time.perf_counter(); snap=_host_snapshot(payload,{}); snapshot_s=time.perf_counter()-t_snap
            except Exception as e:
                use_async=False; snap=None; fallback="snapshot_failed:"+type(e).__name__
        ckpt_state["saved_clock"]=optimizer_update_clock
        if use_async:
            del payload
            stall_s=time.perf_counter()-t_req
            ckpt_writer.start(snap,tmp,dst,event,not args.no_ckpt_fsync,{"async":True,"reason":reason,"stall_s":stall_s,"snapshot_s":snapshot_s} if tel2 else None,t_req)
            if tel2: _emit({"event":"checkpoint_async_started","ts":time.time(),"step":step,"optimizer_update_clock":optimizer_update_clock,"stall_s":stall_s,"snapshot_s":snapshot_s,"mem_available_bytes":avail})
        else:
            _write_checkpoint(payload,tmp,dst,False)
            if force_sync and not args.no_ckpt_fsync: _fsync_best_effort(dst)  # after the rename: keeps the tmp window at v1 length
            if tel2:
                save_s=time.perf_counter()-t_req
                event.update({"ts":time.time(),"async":False,"reason":reason,"stall_s":save_s,"save_s":save_s})
                if fallback: event["async_fallback"]=fallback
            _emit(event)
        return True

    def _boundary_service(step_done):
        # Update-boundary only. (a) a failed background write -> synchronous retry;
        # (b) lossless SIGTERM/SIGINT: join any in-flight write, save synchronously, stop.
        if lazy: _flush_train_log(True)  # NaN/Inf guard covers every step up to step_done
        t0=time.perf_counter()
        _report_ckpt(ckpt_writer.finish(block=True),ckpt_state)
        saved=None
        if ckpt_state["saved_clock"]!=optimizer_update_clock:
            saved=_checkpoint(step_done,seen,force_sync=True,reason="signal" if _STOP["n"] else "async_retry")
        ckpt_state["retry"]=False
        if not _STOP["n"]: return False
        _emit({"event":"signal_exit","ts":time.time(),"signal":_STOP["sig"],"signals_received":_STOP["n"],"step":step_done,"seen_tokens":seen,"optimizer_update_clock":optimizer_update_clock,"checkpoint_saved":saved,"already_saved":saved is None,"exit_s":time.perf_counter()-t0})
        stream.stop_prefetch()
        return True

    if args.gil_switch_interval_ms > 0: sys.setswitchinterval(args.gil_switch_interval_ms/1000.0)
    if tel2:
        _emit({"event":"fullstack_config","ts":time.time(),"version":"v2_fullstack_20260919","prefetch":not args.no_prefetch,"prefetch_depth":args.prefetch_depth,"prefetch_pin":not args.no_prefetch_pin,"async_checkpoint":not args.sync_checkpoint,"async_ckpt_reserve_gb":args.async_ckpt_reserve_gb,"ckpt_fsync":not args.no_ckpt_fsync,"signal_save":not args.no_signal_save,"lazy_step_sync":lazy,"finite_check_every":args.finite_check_every,"proc_uptime_s":time.time()-_PROC_T0})
    if not args.no_signal_save:
        # Installed only now: during startup (nothing to save yet) a signal keeps its default action.
        signal.signal(signal.SIGTERM,_on_signal); signal.signal(signal.SIGINT,_on_signal)
    if not args.no_prefetch:
        stream.abort_check=lambda: _STOP["n"]>0 and micro_in_update==0
        stream.start_prefetch(args.batch,args.seq,depth=args.prefetch_depth,pin=not args.no_prefetch_pin)
    t_prev=time.perf_counter()
    for step in range(start_step,args.steps+1):
        if micro_in_update == 0:
            # Persisted optimizer-update clock decouples stage curriculum from
            # raw microstep numbering, allowing safe grad-accum migrations.
            phase = optimizer_update_clock % (STAGES + 1)
            if args.force_head_only:
                global_anchor = True
                train_stage = None
            else:
                global_anchor = (phase == STAGES)  # receipt-compatible name; this is head-only
                train_stage = None if global_anchor else (STAGES - 1 - phase)
            model.set_trainable_stage(train_stage, head_anchor=global_anchor)
            if global_anchor:
                active_opt=head_opt; active_params=head_params
            else:
                active_opt=stage_opts[int(train_stage)]; active_params=stage_param_sets[int(train_stage)]
            active_opt.zero_grad(set_to_none=True)
        t_fetch=time.perf_counter()
        try:
            ids,labels,srcs=stream.batch(args.batch,args.seq)
        except PrefetchAborted:
            # Stop signal while starved for data at an update boundary; nothing was consumed.
            _boundary_service(step-1)
            return
        data_wait_s=time.perf_counter()-t_fetch
        if tel2 and step == start_step:
            _emit({"event":"stream_ready","ts":time.time(),"step":step,"stream_ff_s":data_wait_s,"prefetch":not args.no_prefetch,"proc_uptime_s":time.time()-_PROC_T0})
        if lazy:
            # CUDA events time the step on the GPU timeline without blocking the host.
            ev0=torch.cuda.Event(enable_timing=True); ev0.record()
        else:
            torch.cuda.synchronize()
        s0=time.perf_counter()
        _ah_obj="ar" if (AH is None or global_anchor) else AH.objective_for(optimizer_update_clock,micro_in_update)
        if _ah_obj == "ar":
            loss,_,aux,counts=model(ids,labels,train_stage=train_stage,head_anchor=global_anchor)
        else:
            loss,aux,counts=AH.forward(_ah_obj,model,ids,labels,train_stage,optimizer_update_clock,micro_in_update)
        if not lazy and not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
        (loss / args.grad_accum).backward()
        micro_in_update += 1
        do_update = (micro_in_update >= args.grad_accum) or (step == args.steps)
        grad_norm = None
        if do_update:
            tokens_after = seen + args.batch*args.seq
            if args.warmup_tokens > 0 and tokens_after < args.warmup_tokens:
                lr_now = args.lr * max(1e-4, tokens_after / args.warmup_tokens)
            else:
                prog = max(0.0,min(1.0,(tokens_after-args.warmup_tokens)/max(1,TOTAL_TRAIN_TOKENS-args.warmup_tokens)))
                lr_now = args.lr * (args.min_lr_mult + (1.0-args.min_lr_mult)*0.5*(1.0+math.cos(math.pi*prog)))
            for pg in active_opt.param_groups: pg["lr"] = lr_now
            grad_norm=torch.nn.utils.clip_grad_norm_(active_params,(1.0 if (AH is None or _ah_obj == "ar") else AH.nonar_max_norm(train_stage)))
            if AH is not None and _ah_obj == "ar" and not global_anchor: AH.note_ar_grad_norm(train_stage,grad_norm)
            if not lazy and not torch.isfinite(grad_norm): raise RuntimeError("nonfinite grad norm")
            active_opt.step()
            if AH is not None: AH.after_step(_ah_obj,lr_now,head_opt)
            if not global_anchor:
                model.invalidate_stage(train_stage)
            active_opt.zero_grad(set_to_none=True)
            optimizer_update_clock += 1
            micro_in_update = 0
            update_number = optimizer_update_clock
            if args.save_every_updates and update_number % args.save_every_updates == 0:
                if lazy:
                    # Deferred NaN/Inf guard: every earlier step, then this one, must be finite
                    # before any state is captured, so a non-finite state can never be saved.
                    _flush_train_log(True)
                    if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
                    if not torch.isfinite(grad_norm): raise RuntimeError("nonfinite grad norm")
                _checkpoint(step,seen + args.batch*args.seq)
        if lazy:
            # float64 on device is exact for the f32/bf16 scalars, so the logged values equal v1's float(x).
            gn_dev=grad_norm.detach().double() if grad_norm is not None else torch.zeros((),device=loss.device,dtype=torch.float64)
            scal=torch.stack([loss.detach().double(),aux.detach().double(),gn_dev]).to("cpu",non_blocking=True)
            ev1=torch.cuda.Event(enable_timing=True); ev1.record()
            dt=loss_v=aux_v=gn_v=None
        else:
            torch.cuda.synchronize(); dt=time.perf_counter()-s0
            step_times.append(dt)
            loss_v=float(loss.detach()); aux_v=float(aux.detach()); gn_v=None if grad_norm is None else float(grad_norm.detach())
        seen += args.batch*args.seq
        tel=model.sparse_telemetry()
        rec={"event":"train","step":step,"seen_tokens":seen,"loss":loss_v,"aux":aux_v,"grad_norm":gn_v,"optimizer_step":do_update,"grad_accum":args.grad_accum,"optimizer_update_clock":optimizer_update_clock,"micro_in_update":micro_in_update,"train_stage":train_stage,"global_anchor":global_anchor,"global_anchor_every":args.global_anchor_every,"lr":float(active_opt.param_groups[0]["lr"]),"optimizer":optimizer_name,"step_s":dt,"tok_s":None if dt is None else args.batch*args.seq/dt,"sources":srcs,"sparse":tel,"cuda_alloc":torch.cuda.memory_allocated(),"cuda_reserved":torch.cuda.memory_reserved(),"route_min":min(min(x) for x in counts),"route_max":max(max(x) for x in counts)}
        if AH is not None: rec.update(AH.telemetry(_ah_obj,args.batch*args.seq))
        if tel2:
            # step_wall_s: host wall time since the previous step's record, i.e. everything
            # (data wait, compute, save stall, logging). Summed over steps it is the run's wall time.
            t_now=time.perf_counter()
            rec.update({"ts":time.time(),"data_wait_s":data_wait_s,"step_wall_s":t_now-t_prev,"prefetch_q":stream.last_qsize})
            t_prev=t_now
        if lazy:
            pending.append((rec,scal,ev0,ev1))
            _flush_train_log(len(pending)>=max(1,args.finite_check_every))
        else:
            _finalize_pipeline_telemetry(rec)
            _emit(rec)
        _report_ckpt(ckpt_writer.finish(block=False),ckpt_state)
        if micro_in_update == 0 and (_STOP["n"] or ckpt_state["retry"]) and _boundary_service(step):
            return
    if lazy: _flush_train_log(True)
    _report_ckpt(ckpt_writer.finish(block=True),ckpt_state)
    if ckpt_state["retry"]: _boundary_service(args.steps)
    stream.stop_prefetch()
    result={"schema":"agillm.gb10.1pf.canary.v1","ok":True,"steps":args.steps,"seen_tokens":seen,"median_step_s":statistics.median(step_times),"median_tok_s":args.batch*args.seq/statistics.median(step_times),"parameters":params,"target_training_tokens":TOTAL_TRAIN_TOKENS,"verified_sparse_microkernel_tflops":VERIFIED_SPARSE_TFLOPS,"finished_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    Path(args.receipt).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description="AGILLM-GB10-1PF single-file hardware-locked trainer")
    sub=p.add_subparsers(dest="cmd",required=True)
    sub.add_parser("profile")
    d=sub.add_parser("dataset-probe"); d.add_argument("--dataset",default="fineweb-edu",choices=list(HF_SOURCES)+["mix"])
    t=sub.add_parser("train"); t.add_argument("--dataset",default="fineweb-edu",choices=list(HF_SOURCES)+["mix"]); t.add_argument("--batch",type=int,default=1); t.add_argument("--seq",type=int,default=2048); t.add_argument("--steps",type=int,default=1); t.add_argument("--grad-accum",type=int,default=8); t.add_argument("--legacy-resume-grad-accum",type=int,default=8); t.add_argument("--global-anchor-every",type=int,default=64); t.add_argument("--lr",type=float,default=2e-4); t.add_argument("--warmup-tokens",type=int,default=100000000); t.add_argument("--min-lr-mult",type=float,default=0.1); t.add_argument("--save-every-updates",type=int,default=2048); t.add_argument("--ce-chunk",type=int,default=4096); t.add_argument("--force-head-only",action="store_true"); t.add_argument("--save-dir",default="/workspace/agillm-gb10-1pf-checkpoints"); t.add_argument("--resume",default=""); t.add_argument("--seed",type=int,default=42); t.add_argument("--receipt",default="/workspace/agillm_gb10_1pf_canary.json")
    t.add_argument("--objectives",default="ar,sat,nat",help="'ar,sat,nat' (owner contract) or 'ar' (exact v1 reproduction)")
    t.add_argument("--allheads-hot-config",default="",help="hot JSON reloaded on mtime change; default <save-dir>/allheads_hot.json")
    t.add_argument("--dblock-ar-prob",type=float,default=0.60); t.add_argument("--dblock-sat-prob",type=float,default=0.25); t.add_argument("--dblock-nat-prob",type=float,default=0.15)
    t.add_argument("--sat-fixed-share",type=float,default=0.5); t.add_argument("--satvar-block-probs",default="1:0.5,2:0.5")
    t.add_argument("--satvar-lambda",type=float,default=1.0); t.add_argument("--satvar-merge-cost",type=float,default=-1.0,help="<0 -> 0.5*lambda (S default)")
    t.add_argument("--satvar-regret-weight",type=float,default=0.05); t.add_argument("--satvar-log-every",type=int,default=20); t.add_argument("--satvar-regret-max-blocks",type=int,default=4096,help="fused-CE path only: blocks sampled per SAT step for the regret targets")
    t.add_argument("--nat-mask-rate",default="uniform:0.1:1.0",help="uniform:lo:hi per-sequence sampled rate, or a fixed float")
    t.add_argument("--nat-span-mask-prob",type=float,default=0.35); t.add_argument("--nat-suffix-mask-prob",type=float,default=0.20); t.add_argument("--nat-region-partial-prob",type=float,default=0.5)
    t.add_argument("--nat-attn",default="causal",choices=["causal","bidir"],help="bidir = natmaskedhead_v2 kernel call; position-blind on this NoPE network")
    t.add_argument("--allheads-head-grad",default="sat",choices=["none","sat","sat+nat"]); t.add_argument("--allheads-aux-lr-mult",type=float,default=1.0)
    t.add_argument("--allheads-nonar-clip-mult",type=float,default=1.0,help="SAT/NAT stage gradients are clipped to mult x the stage's typical AR grad norm (log-EMA); 0 = v3_3 behaviour (clip 1.0)")
    # v2 full-stack switches. New behaviour is the default; each flag restores v1 for one feature.
    t.add_argument("--no-prefetch","--no-data-prefetch",action="store_true",help="v1: build batches synchronously on the main thread")
    t.add_argument("--prefetch-depth","--data-prefetch-depth",type=int,default=3,help="ready CPU batches held by the producer (bounded queue)")
    t.add_argument("--no-prefetch-pin",action="store_true",help="do not pin prefetched CPU batches")
    t.add_argument("--sync-checkpoint",action="store_true",help="v1: synchronous torch.save at the save boundary")
    t.add_argument("--async-ckpt-reserve-gb",type=float,default=8.0,help="async save needs MemAvailable >= checkpoint size + this, else it falls back to a synchronous save")
    t.add_argument("--no-ckpt-fsync",action="store_true",help="skip fsync on async and signal saves (periodic synchronous saves never fsync, as in v1)")
    t.add_argument("--no-signal-save",action="store_true",help="v1: default SIGTERM/SIGINT disposition (no save on signal)")
    t.add_argument("--strict-step-sync",action="store_true",help="v1: per-step cuda synchronize, blocking isfinite checks, immediate log line")
    t.add_argument("--finite-check-every",type=int,default=16,help="max steps a train record and its NaN/Inf check may be deferred (always forced before a checkpoint)")
    t.add_argument("--v1-telemetry",action="store_true",help="v1: omit the added telemetry keys/events")
    t.add_argument("--v1-behavior",action="store_true",help="all of: --no-prefetch --sync-checkpoint --no-signal-save --strict-step-sync --v1-telemetry")
    t.add_argument("--gil-switch-interval-ms",type=float,default=0.0,help="experiment knob for the GPU gate: sys.setswitchinterval; 0 leaves the interpreter default")
    t.add_argument("--stagefwd-v1-routing",action="store_true",help="restore v1 nonzero()/index_copy_ expert dispatch (6 host syncs per stage forward)")
    t.add_argument("--stagefwd-v1-active-silu",action="store_true",help="restore v1 F.silu(gate)*up + separate down-proj quant in the active stage")
    t.add_argument("--stagefwd-v1-repack",action="store_true",help="restore v1 destroy/re-create of native sparse contexts on weight-version change")
    # Promotion gate uses reference CE: fused CE currently omits per-row NLL and
    # activates the 4096-block SAT regret sampling branch. Preserve full SAT-var
    # regret training until fused CE can expose every row NLL without sampling.
    t.add_argument("--fused-ce",action="store_true",help="use fused rank-16 CE kernel; default keeps reference CE")
    a=p.parse_args()
    if a.cmd == "train":
        STAGEFWD["syncfree_routing"] = not a.stagefwd_v1_routing
        STAGEFWD["active_fused_silupack"] = not a.stagefwd_v1_active_silu
        STAGEFWD["inplace_repack"] = not a.stagefwd_v1_repack
        print(json.dumps({"event":"stagefwd_config","variant":"v2b_stagefwd",**STAGEFWD}),flush=True)
    # Compatibility guard for the recovery wrapper accidentally launched with the
    # temporary migration save cadence of 128. The already-running process has
    # parsed its CLI and is unaffected by this file edit; only a future restart
    # of the exact production save directory is upgraded to the established 512.
    if (a.cmd == "train" and a.save_every_updates == 128
            and a.save_dir == "/workspace/agillm-gb10-1pf-targetfix-active"):
        a.save_every_updates = 512
        print(json.dumps({"event":"checkpoint_cadence_compat_upgrade","from_updates":128,"to_updates":512,"reason":"post-migration production cadence"}),flush=True)
    if a.cmd=="profile": profile()
    elif a.cmd=="dataset-probe":
        dataset_probe(list(HF_SOURCES) if a.dataset=="mix" else [a.dataset]); sys.stdout.flush(); os._exit(0)
    else:
        if a.v1_behavior:
            a.no_prefetch = a.sync_checkpoint = a.no_signal_save = a.strict_step_sync = a.v1_telemetry = True
        try:
            train(a)
        finally:
            _shutdown_background()  # stop the producer; let an in-flight checkpoint write finish
        sys.stdout.flush(); os._exit(0)

if __name__=="__main__": main()
