from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import random
import statistics
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path

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
        return x + y.view_as(x), aux, counts


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

    def set_trainable_stage(self, stage_idx=None, head_anchor=False):
        # GB10-native local-update schedule. Exactly one Transformer stage owns
        # gradients on stage updates. A head anchor keeps all stages frozen and
        # updates only the tied token interface, avoiding a 2B-parameter global
        # gradient spike while still training embeddings/input/output/norm.
        for p in self.parameters():
            p.requires_grad_(False)
        if head_anchor:
            # Output-side tied embedding + final projection only. The Transformer
            # body is evaluated under no_grad during this phase.
            for mod in (self.norm, self.out_proj, self.factor_head, self.mix_head):
                for p in mod.parameters(): p.requires_grad_(True)
        elif stage_idx is not None:
            for p in self.stages[int(stage_idx)].parameters():
                p.requires_grad_(True)
            # Stage 0 already requires an end-to-end gradient path, so train the
            # input token interface there at essentially no extra graph depth.
            if int(stage_idx) == 0:
                for mod in (self.embed, self.in_proj):
                    for p in mod.parameters(): p.requires_grad_(True)

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
        for m in self.modules():
            if isinstance(m, SparseLinear): m.invalidate()
    def invalidate_stage(self, stage_idx):
        for m in self.stages[int(stage_idx)].modules():
            if isinstance(m, SparseLinear): m.invalidate()
    def sparse_telemetry(self):
        xs=[m for m in self.modules() if isinstance(m,SparseLinear)]
        return {"layers":len(xs),"kernel_calls":sum(m.kernel_calls for m in xs),"packs":sum(m.pack_count for m in xs)}


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

    def _open(self, name):
        repo,cfg,_=HF_SOURCES[name]
        kw=dict(split="train",streaming=True)
        ds=self.load_dataset(repo,cfg,**kw) if cfg else self.load_dataset(repo,**kw)
        # Do NOT call HF IterableDataset.shuffle here: its shuffle buffer is not
        # included in state_dict(). Our explicit row_buffers below are.
        self.datasets[name]=ds
        self.iters[name]=iter(ds)
        return self.iters[name]

    def _text(self,row):
        for k in ("text","content","code"):
            v=row.get(k) if isinstance(row,dict) else None
            if isinstance(v,str) and v.strip(): return v
        return ""

    def _raw_ids(self,name):
        while True:
            if name not in self.iters: self._open(name)
            row=next(self.iters[name])
            text=self._text(row)
            if not text: continue
            ids=self.tok.encode(text).ids
            if len(ids)>4095: ids=ids[:4095]
            if ids: return ids

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
                self.disabled.add(n); self.iters.pop(n,None); self.datasets.pop(n,None)
                available=[x for x in available if x!=n]
                if not available: raise RuntimeError("all HF sources exhausted")
            except Exception as e:
                print(json.dumps({"dataset_disable":n,"error":type(e).__name__,"detail":str(e)[:240]}),flush=True)
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
        for name,ds_state in state.get("dataset_state",{}).items():
            if name in self.disabled: continue
            self._open(name)
            ds=self.datasets[name]
            if not hasattr(ds,"load_state_dict"):
                raise RuntimeError(f"stream source {name} lacks load_state_dict()")
            ds.load_state_dict(ds_state)
            self.iters[name]=iter(ds)

    def batch(self,batch,seq):
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
    """Non-AR attention dispatch (the AR branch stays inline in Attention.forward, untouched)."""
    mode = _AH_ATTN.mode
    if mode == "sat":
        fn = _ah_flash_causal if q.is_cuda else (lambda a, b, c, w: ah_flash_semantics_dense(a, b, c, w))
        return ah_block_causal_twocall(q, k, v, window, _AH_ATTN.first2, fn)
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


def ah_fused_ce_fn(model):
    """The composed stack embeds rpv16_fused_ce (fused_ce patch set) and switches it on with model.fused_ce.
    The v1-based candidate has neither, so this returns None there and the reference CE below is used."""
    fn = globals().get("rpv16_fused_ce")
    return fn if (callable(fn) and bool(getattr(model, "fused_ce", False))) else None


def ah_ce_mean(model, hh, yy):
    """Mean RPV CE over the given rows -> (ce, per-row NLL or None).  The fused kernel keeps per-row NLL internal,
    so the fused path returns None and callers that need per-row values compute them on a sampled subset."""
    fn = ah_fused_ce_fn(model)
    if fn is not None:
        hint = float(getattr(model, "fused_ce_grad_scale_hint", getattr(model, "ce_grad_scale", 1.0)))
        ce = fn(hh, yy, model.factor_head.weight, model.mix_head.weight, ce_chunk=int(getattr(model, "ce_chunk", 4096)), grad_scale_hint=hint)
        return ce, None
    nll = ah_rpv_token_nll(model, hh, yy)
    return nll.mean(), nll


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
            # fused CE: per-row NLL is not exposed. Like S (which samples 256 blocks) take the regret targets from an
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
                "ce_path": "reference" if nll is not None else "fused", "regret_blocks": int(last_idx.numel()),
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
        self.info = {"objective": "nat", "n_targets": int(sel.numel()), "nat_ce": float(ce.detach()), "ce_path": "reference" if nll is not None else "fused", "nat": dict(stats, attn=self.nat_attn)}
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


def train(args):
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sources = list(HF_SOURCES) if args.dataset == "mix" else [args.dataset]
    stream=HFStream(sources,args.seed)
    print(json.dumps({"event":"dataset_ready","sources":sources,"tokenizer":str(TOKENIZER_JSON)}),flush=True)
    t0=time.time(); model=build_model(); model.ce_chunk=args.ce_chunk; torch.cuda.synchronize()
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
        ck=torch.load(args.resume,map_location="cpu",weights_only=False)
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
            stream.load_state_dict(ck["data_state"])
            ds=ck["data_state"]
            print(json.dumps({"event":"data_cursor_resumed","schema":ds.get("schema"),"rows_emitted":ds.get("rows_emitted",{}),"token_buffer":int(ds.get("token_buffer",torch.empty(0)).numel())}),flush=True)
        else:
            print(json.dumps({"event":"data_cursor_migration_reset","reason":"legacy checkpoint lacks exact stream cursor","checkpoint_step":ck_step,"seen_tokens":seen,"note":"one-time data-order reset; future v4 checkpoints resume exactly"}),flush=True)
        print(json.dumps({"event":"resumed","path":args.resume,"step":start_step-1,"seen_tokens":seen,"optimizer_update_clock":optimizer_update_clock,"phase":optimizer_update_clock%(STAGES+1),"phase_clock_source":phase_clock_source,"resume_grad_accum":resume_grad_accum,"requested_grad_accum":args.grad_accum,"target_alignment":TARGET_ALIGNMENT,"resume_target_alignment":resume_target_alignment}),flush=True)
        del ck
        import gc; gc.collect(); torch.cuda.empty_cache()
    train_stage = None
    global_anchor = False
    active_opt = None
    active_params = None
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
        ids,labels,srcs=stream.batch(args.batch,args.seq)
        torch.cuda.synchronize(); s0=time.perf_counter()
        _ah_obj="ar" if (AH is None or global_anchor) else AH.objective_for(optimizer_update_clock,micro_in_update)
        if _ah_obj == "ar":
            loss,_,aux,counts=model(ids,labels,train_stage=train_stage,head_anchor=global_anchor)
        else:
            loss,aux,counts=AH.forward(_ah_obj,model,ids,labels,train_stage,optimizer_update_clock,micro_in_update)
        if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
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
            if not torch.isfinite(grad_norm): raise RuntimeError("nonfinite grad norm")
            active_opt.step()
            if AH is not None: AH.after_step(_ah_obj,lr_now,head_opt)
            if not global_anchor:
                model.invalidate_stage(train_stage)
            active_opt.zero_grad(set_to_none=True)
            optimizer_update_clock += 1
            micro_in_update = 0
            update_number = optimizer_update_clock
            if args.save_every_updates and update_number % args.save_every_updates == 0:
                sd=Path(args.save_dir); sd.mkdir(parents=True,exist_ok=True)
                dst=sd/"latest.pt"; tmp=sd/"latest.pt.tmp"
                # Auto-prune only an incomplete stale temp file from this exact save dir.
                # Complete checkpoints are never deleted here; post-upload pruning is handled separately.
                if tmp.exists():
                    try:
                        age=time.time()-tmp.stat().st_mtime
                        if age >= 300:
                            tmp.unlink()
                            print(json.dumps({"event":"checkpoint_autoprune","path":str(tmp),"reason":"stale_tmp","age_s":age}),flush=True)
                    except FileNotFoundError:
                        pass
                expected = dst.stat().st_size if dst.exists() else 12*1024**3
                free = shutil.disk_usage(sd).free
                need = expected + 1024**3
                if free < need:
                    print(json.dumps({"event":"checkpoint_skipped_disk","step":step,"free_bytes":free,"need_bytes":need,"path":str(dst)}),flush=True)
                else:
                    torch.save({"schema":("agillm.gb10.1pf.recovery.v5" if AH is None else ALLHEADS_CKPT_SCHEMA),"target_alignment":TARGET_ALIGNMENT,"step":step,"seen_tokens":seen + args.batch*args.seq,"model":model.state_dict(),"optimizer_bank":{"stages":[o.state_dict() for o in stage_opts],"head":head_opt.state_dict()},"optimizer_name":optimizer_name,"seed":args.seed,"optimizer_update_clock":optimizer_update_clock,"grad_accum":args.grad_accum,"micro_in_update":micro_in_update,"data_state":stream.state_dict(),**({} if AH is None else AH.ckpt_extra())},tmp)
                    os.replace(tmp,dst)
                    print(json.dumps({"event":"checkpoint","path":str(dst),"step":step,"seen_tokens":seen + args.batch*args.seq,"schema":("v5" if AH is None else "v6"),"target_alignment":TARGET_ALIGNMENT,"optimizer_update_clock":optimizer_update_clock,"grad_accum":args.grad_accum}),flush=True)
        torch.cuda.synchronize(); dt=time.perf_counter()-s0
        seen += args.batch*args.seq; step_times.append(dt)
        tel=model.sparse_telemetry()
        rec={"event":"train","step":step,"seen_tokens":seen,"loss":float(loss.detach()),"aux":float(aux.detach()),"grad_norm":None if grad_norm is None else float(grad_norm.detach()),"optimizer_step":do_update,"grad_accum":args.grad_accum,"optimizer_update_clock":optimizer_update_clock,"micro_in_update":micro_in_update,"train_stage":train_stage,"global_anchor":global_anchor,"global_anchor_every":args.global_anchor_every,"lr":float(active_opt.param_groups[0]["lr"]),"optimizer":optimizer_name,"step_s":dt,"tok_s":args.batch*args.seq/dt,"sources":srcs,"sparse":tel,"cuda_alloc":torch.cuda.memory_allocated(),"cuda_reserved":torch.cuda.memory_reserved(),"route_min":min(min(x) for x in counts),"route_max":max(max(x) for x in counts)}
        if AH is not None: rec.update(AH.telemetry(_ah_obj,args.batch*args.seq))
        print(json.dumps(rec),flush=True)
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
    a=p.parse_args()
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
        train(a); sys.stdout.flush(); os._exit(0)

if __name__=="__main__": main()
