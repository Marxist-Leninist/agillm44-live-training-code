#!/usr/bin/env python3
"""bintf2: native-binary transformer, trainable when the model is bigger than the GPU.

The model family is bintf.py's: every linear layer and expert computes with +-1 weights times a
learned row scale, a mixture-of-experts feed-forward, RoPE attention, and DiffusionBlocks
(arXiv 2506.14202, masked-diffusion form): the layers are split into independent nets, each owning
one band of mask ratios. What changes is the vocabulary and how it trains.

  tokens          --vocab bytes keeps bintf.py's raw bytes. --vocab deepseek-v4.1 uses the tokenizer
                  of DeepSeek V4.1 Flash (129,280 tokens). The token embedding is then binary too:
                  one +-1 matrix per net, tied between input and output, scored one slice of the
                  vocabulary at a time so the logits never exist all at once.
  int8 latents    Each binary weight is stored as one int8: its sign is the weight, its magnitude is
                  how settled the sign is. The update (factored second moments, stochastic rounding,
                  the --opt int8 rule from bintf.py) runs inside the backward pass, one matrix at a
                  time, so a weight gradient is freed before the next matrix is touched.
  one net on GPU  Only the net being trained holds its int8 latents on the GPU; the others wait in RAM.
  normed inputs   Every binary matrix reads an RMS-normalised input (as BitNet's SubLN does): queries
                  and keys are normalised per head, and so are the attention output and the squared-ReLU
                  hidden layer before their projections. A +-1 row cannot shrink, so without this a
                  2048-wide row that lines up with its input drives squared activations past what
                  half precision can hold. Each of those inputs is also centred on its running mean
                  over tokens, so a row has no constant direction to line up with.
  dense targets   A causal clean stream predicts the next token at every position (the AR decoder).
                  A noisy stream of partly masked blocks, each attending to the clean tokens before
                  it and to itself, predicts the masked tokens (the semi / NAT / diffusion decoder).
                  bintf.py's rows carried one block each, so an AR row had a single target.
  streaming       Text comes from Hugging Face: one file, or the parquet shards of a dataset
                  directory read in a fixed shuffled order with a resumable cursor, each token once.
  step sizes      AdamW moves a parameter about one learning rate per step whatever its size, so the
                  row scales (about 0.02) and the router take a small share of --lr; at the full rate
                  a few rows of a 2048-wide matrix grow until one direction swamps the residual stream.
                  The output bias starts at each token's log frequency, so no layer has to carry it,
                  and the readout starts small. Nothing that belongs to a single token is stepped by
                  a rule that divides by its own gradient: the token table divides each row by its rms
                  or by one occurrence's, the output bias takes SGD steps, the output scale is fixed.
  ParScale        built in and on (arXiv 2505.10475; see ParAdapter). Every net can run its batch as P streams
                  that share all of its weights: streams 1 .. P - 1 each read their own learned prefix keys and
                  values in every attention layer, and the P last hidden states are merged by a small learned
                  weighting before the head, which then scores the vocabulary once. --parscale-duty of each
                  net's steps (default 1 in 8, at --parscale 4 streams) are taken that way and the rest are the
                  plain model (stream 0), so the add-on keeps pace with the backbone however long the run goes
                  on. It lives beside each checkpoint (ckpt/parscale{i}.pt, parscale.json): older revisions read
                  every checkpoint as before. --parscale-freeze-backbone trains the add-on alone (to add it to
                  any checkpoint, or to bring it up to date). --parscale 1 or --parscale-duty 0 is the plain
                  trainer, bit for bit. Revision 2 (v2): every stream's prefixes start from a real batch and
                  have a learned logit offset per head; a ParScale step takes its rows in sub-steps of at most
                  --parscale-max-pos positions (default --step-tokens), one optimiser step each.

Usage:
  python bintf2.py info --preset 4b --dblocks 3 --vocab deepseek-v4.1
  python bintf2.py train --preset small-moe --minutes 10 --run runs/small
  python bintf2.py train --preset 4b --dblocks 3 --vocab deepseek-v4.1 --resident one \
      --data hfds:HuggingFaceFW/fineweb-edu:data --target-tokens 500e9 --sched isqrt \
      --run /workspace/bintf4b_run --resume
  python bintf2.py status --run /workspace/bintf4b_run
  python bintf2.py eval --run /workspace/bintf4b_run --val-len 4096
  python bintf2.py export --run /workspace/bintf4b_run --out model_bits.pt     (import --file .. --run .. undoes it)
  python bintf2.py generate --run runs/small --prompt "Once upon a time"
  python bintf2.py train --run /workspace/bintf4b_run --resume --parscale-freeze-backbone --parscale-refresh-tokens 20e6
"""
import argparse
import contextlib
import fcntl
import glob
import hashlib
import json
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass

if len(sys.argv) > 1 and sys.argv[1] in ('status', 'info', 'export', 'import', 'prep'):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # these never compute: no CUDA context (0.3 GB) beside a running trainer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LN2 = math.log(2)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA = DEV.type == 'cuda'
HALF = torch.float16 if CUDA else torch.float32
_ONE = torch.ones((), device=DEV, dtype=HALF)
_NEG = -_ONE
ZF = None  # a list while evaluating: share of exactly-zero feed-forward activations per expert call
# Speed switches, read from the environment once; each is on unless set to 0, and off gives back the path it replaced.
#   BINTF_LEAN     the per-position norm and bound in one kernel pass each way: the same arithmetic, results equal
#                  to the tensor operations up to half-precision rounding (so a run drifts from an old-path run
#                  the way two rounding seeds drift apart, not bit for bit)
#   BINTF_CE_KEEP  the output head keeps its forward scores for backward while the GPU has room: the same numbers
#   BINTF_ROPE     rotary positions in one kernel pass each way: the same numbers
#   BINTF_MOE      the feed-forward experts of a layer in kernels over every expert's rows at once (see _MoE): the
#                  same numbers
#   BINTF_LEAN2    not on/off but a number of positions (rows x length, both streams): a training step of that many
#                  or more keeps 12 tensors of the stream's size per layer between its two passes, not 19, and
#                  computes the others again in its backward pass (see _AttnOut): the same numbers, 2% more time,
#                  a third less memory per position. Default 3000 (a step of 4,096 tokens does, one of 2,048 does
#                  not); 0 for every step, -1 for none.
_LEAN = os.environ.get('BINTF_LEAN', '1') != '0'
_CE_KEEP = os.environ.get('BINTF_CE_KEEP', '1') != '0'
_ROPE_K = os.environ.get('BINTF_ROPE', '1') != '0'
_MOE_K = os.environ.get('BINTF_MOE', '1') != '0'
_LEAN2 = int(os.environ.get('BINTF_LEAN2', '3000'))
_GCAP_NORM = os.environ.get('BINTF_GCAP_NORM', '1') != '0'  # fp32 accumulation without a full fp32 gradient copy
_PS_LEAN2 = os.environ.get('BINTF_PS_LEAN2', '1') != '0'  # ParScale steps on the lean2 path (_AttnOutP); 0: attend()
_RV_FIXC = os.environ.get('BINTF_RV_FIXC', '1') != '0'  # review FIX_C (D1): a ParScale step's rows centred stream by stream
_RV_FIXG = os.environ.get('BINTF_RV_FIXG', '1') != '0'  # review FIX_G (D2): _GCap's median from stream 0's positions
# tests (D6): 'P' or 'P:j': a ParScale (sub-)step of P streams or more, from its sub-step j on, runs out of GPU memory
_PS_FAKE_OOM = [int(x) for x in os.environ.get('BINTF_PS_FAKE_OOM', '').split(':') if x] or None
_PS_CAPTURE = None  # ps_init_data (D3): a list while it runs; attend() puts every layer's keys and values into it
PS_TRACE = os.environ.get('BINTF_PS_TRACE')  # a file: one line per training step (net, streams, hashes of what was drawn)
_CE_ROOM = 1.0e9  # bytes of GPU memory that must stay free beside the kept scores

# The vocabulary. Bytes by default; set_vocab() switches these to a tokenizer.
VNAME, VOCAB, MASK, BOS, EOS, VCHUNK, TOK = 'bytes', 257, 256, 0, 0, 0, None
TOKENIZERS = {
    # DeepSeek V4.1 Flash: 129,280 ids. <|fim_hole|> (128800) never occurs in text and serves as [MASK].
    # The placeholders from 128000 on never occur either: under Config.tiers the first few stand for the groups.
    'deepseek-v4.1': dict(repo='deepseek-ai/DeepSeek-V4.1-Flash', file='tokenizer.json', vocab=129280,
                          mask=128800, bos=0, eos=1, chunk=16160, spare=128000),
}


def set_vocab(name):
    global VNAME, VOCAB, MASK, BOS, EOS, VCHUNK
    if name != 'bytes':
        t = TOKENIZERS[name]
        VNAME, VOCAB, MASK, BOS, EOS, VCHUNK = name, t['vocab'], t['mask'], t['bos'], t['eos'], t['chunk']


def hf_token():
    t = os.environ.get('HF_TOKEN')
    if t:
        return t.strip()
    p = os.path.expanduser('~/.cache/huggingface/token')
    return open(p).read().strip() if os.path.exists(p) else None


def tokenizer():
    global TOK
    if TOK is None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        t = TOKENIZERS[VNAME]
        cache = os.path.join(os.path.expanduser('~'), '.cache', 'bintf', VNAME)
        TOK = Tokenizer.from_file(hf_hub_download(t['repo'], t['file'], local_dir=cache, token=hf_token()))
    return TOK


def encode(text):
    if VNAME == 'bytes':
        return list(text.encode('utf-8'))
    return [BOS] + tokenizer().encode(text, add_special_tokens=False).ids


def show(ids):
    text = bytes(ids).decode('utf-8', errors='replace') if VNAME == 'bytes' else tokenizer().decode(ids)
    return text.replace('\n', '\\n')


def unit():
    return 'bytes' if VNAME == 'bytes' else 'tokens'


def lossf(nats):
    """Bits per byte for bytes, nats per token for a tokenizer (the scale AGILLM's CE is quoted in)."""
    return nats / LN2 if VNAME == 'bytes' else nats


def lossu():
    return 'bits' if VNAME == 'bytes' else 'nats'


@dataclass
class Config:
    d: int = 192
    n_layer: int = 4
    n_head: int = 4
    ff: int = 768
    n_exp: int = 0      # 0 = dense feed-forward, otherwise experts per layer
    topk: int = 2
    ctx: int = 128
    act: str = 'relu2'  # relu2 | gelu
    dblocks: int = 1    # DiffusionBlocks nets; 1 = one end-to-end net
    vocab: str = 'bytes'
    pos: str = 'rope'   # rope | alibi | both: how attention knows where tokens are
    cap: float = 0.0    # > 0: no branch adds more than this rms per position to the residual stream, and the
                        # attention values are bounded per head (0 = unbounded: older checkpoints keep their function)
    centre2: bool = False  # every binary matrix reads an input centred again after its norm or bound (see _center)
    bias: bool = False  # full-precision offsets for queries, keys and feed-forward pre-activations (see _Bias)
    cov: bool = False   # every binary matrix learns from its output gradient minus that gradient's running mean over
                        # tokens (see Sign.centred), and the offsets are kept per kind of position (see _Bias2)
    tiers: str = ''     # token vocabularies. '' = one softmax over every word. '1,1,2,4' = the table is kept in
                        # order of frequency and its slices are grouped: the first group is scored for every row,
                        # a later group only for the rows whose word is in it (see _TiedCE)


PRESETS = {
    'tiny':      dict(d=192, n_layer=4, n_head=4, ff=768, ctx=128),
    'tiny-moe':  dict(d=192, n_layer=4, n_head=4, ff=384, n_exp=4, topk=2, ctx=128),
    'small':     dict(d=384, n_layer=6, n_head=6, ff=1536, ctx=256),
    'small-moe': dict(d=384, n_layer=6, n_head=6, ff=768, n_exp=8, topk=2, ctx=256),
    '4b':        dict(d=2048, n_layer=24, n_head=16, ff=2304, n_exp=16, topk=2, ctx=4096),
}


# ---------------------------------------------------------------- int8 sign latents

class SignOpt:
    """Settings shared by every sign-latent update; the trainer sets them once per step."""
    b2 = 0.99
    q = 0.08 / 127      # one int8 unit in the weight units bintf.py used (bound 0.08)
    init_std = 0.02
    a = 0.0             # step in int8 units: sign_lr * schedule / q
    inv_scale = 1.0     # 1 / loss scale
    table = 1.0         # the token table's step, as a multiple of a
    learn = False       # set during a training forward pass: the running means over tokens are updated
    enabled = True
    count = False       # count sign flips this step
    seen = 0
    gmax = 0.0          # the largest output-gradient entry met since the last log line (fp16 holds 65504)
    cov = False         # Config.cov of the model in this process: Sign.centred runs before every weight gradient
    cb = 0.99           # share of a running mean over tokens that a step keeps (_CB for a step of --ref-tokens ids)
    kick = 0            # int8 units a latent is put further on when it crosses to the other sign (--sign-kick)
    kick_table = 0      # the same for the token table (--table-kick)
    gcap = 0.0          # between layers, backward: no position's gradient longer than this many medians (--gcap)
    wcap = 0.0          # what a matrix learns from: no row of its output gradient longer than this many medians (--wcap)
    ema = True          # ParScale: False while no running mean over tokens may move (refresh mode, the fit probe)
    n0 = None           # ParScale step: the rows of stream 0 (the plain model; stream-major), the only rows a
                        # running mean over tokens reads (set by Net.forward, read again in backward by Sign.centred)
    frozen = False      # --parscale-freeze-backbone: backward takes no weight gradient and updates no backbone state
    gcap_mid = 0.0      # the bound of --gcap inside a layer: on what its expert branch hands its attention branch (--gcap-mid)
    gcap_qkv = 0.0      # and on what attention hands its q/k/v matrix (--gcap-qkv)


SO = SignOpt()
SO.bad_t = torch.zeros(1, dtype=torch.int32, device=DEV)    # non-finite gradient entries met this step
SO.flips_t = torch.zeros(1, dtype=torch.int32, device=DEV)  # sign flips this step, when counted
SO.head_bad_t = torch.zeros(1, dtype=torch.int32, device=DEV)  # non-finite entries in the token table's gradient
# since the last log line, summed over training steps: what the branches wrote before their bound (mean rms),
# the share of positions that were at the bound, and a count; and the last residual stream's mean and largest
# rms per position, and a count
SO.br_t = torch.zeros(3, device=DEV)
SO.xs_t = torch.zeros(3, device=DEV)
SO.xc_t = torch.zeros((), device=DEV)  # and the rms of that stream's running mean: the part every position shares
SO.gc_t = torch.zeros(2, device=DEV)  # --gcap: the share of positions held back, summed over its calls, and the calls
SO.wc_t = torch.zeros(2, device=DEV)  # --wcap: the same for the rows of the output gradients matrices learn from
SO.gm_t = torch.zeros(2, device=DEV)  # --gcap-mid: as gc_t
SO.gz_t = torch.zeros(2, device=DEV)  # --gcap-qkv: as gc_t

_KSRC = r'''
#include <cuda_fp16.h>
static __device__ __forceinline__ unsigned int pcg(unsigned int v) {
    unsigned int s = v * 747796405u + 2891336453u;
    unsigned int w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
// ---- one warp per row. row_stats, upd_sq and the norm and bound kernels need one sum per row. They ran one block
// of T threads per row: thread t added up the row's entries t, t + T, ..., and the T sums were then added pairwise
// at distances T / 2, T / 4, ..., 1 through shared memory, with a barrier at every level. Now one warp does a row.
// With L = min(T, 32) lanes in use and M = T / L, lane w keeps the sums of the threads w, w + L, ..., w + L (M - 1),
// adds those at the distances down to L itself, and the last levels go from lane to lane. Every addition is one
// that was made before, in the same order, so the sums are the same bit for bit; nothing is shared, nothing waits,
// and a row is read front to back with M reads under way at a time.
// WROW: the row, the lane and the row's first entry for this thread (eight rows to a block of 256 threads).
// WALK(LOAD, USE): one pass over the row's entries j that belong to this lane, m being the thread of the lane each
// belonged to; the M reads of a stretch (LOAD) come before what is done with them (USE). Stretches that lie
// inside the row are taken without a test per entry; the row's end, if it cuts one, comes last.
#define WROW(n_rows) const int lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); \
    if (row >= (n_rows)) return; \
    const long long base = (long long)row * n_in;
#define WALK(LOAD, USE) if (lane < L) { int j0 = lane; \
    for (; j0 + L * (M - 1) < n_in; j0 += L * M) { \
        _Pragma("unroll") for (int m = 0; m < M; ++m) { const int j = j0 + L * m; LOAD } \
        _Pragma("unroll") for (int m = 0; m < M; ++m) { const int j = j0 + L * m; (void)j; USE } } \
    _Pragma("unroll") for (int m = 0; m < M; ++m) { const int j = j0 + L * m; if (j < n_in) { LOAD USE } } }
// The M sums of a lane into the sum of the row, which lane 0 returns.
template <int M> static __device__ __forceinline__ float wsum(float* a, int lane, int L) {
    _Pragma("unroll") for (int h = M >> 1; h > 0; h >>= 1) {
        _Pragma("unroll") for (int m = 0; m < h; ++m) a[m] += a[m + h];
    }
    float x = a[0];
    _Pragma("unroll") for (int s = 16; s > 0; s >>= 1) {
        const float o = __shfl_down_sync(0xffffffffu, x, s);
        if (s < L && lane < s) x += o;
    }
    return x;
}
// One warp per row: the row's sum of g^2 and of sign(P) * g, as a block of 256 threads summed them. Non-finite
// entries are skipped and counted.
// With R: the row's second moment is brought up to date too, R = R * b2 + (rowsq * w * w * inv_in + eps) * k1 for
// the row's scale w, rounded as the tensor operations R.mul_(b2).add_(rowsq * w * w / n_in + eps, alpha=k1)
// round it (their division by a number is a multiplication by its reciprocal, their add_ a fused multiply-add).
extern "C" __global__ void row_stats(const __half* g, const signed char* P, float* rowsq, float* rowsg,
                                     int* bad, float* R, const float* sc, float b2, float k1, float inv_in,
                                     float eps, int n_out, int n_in) {
    WROW(n_out)
    const int L = 32, M = 8;
    float a[M], b[M];
    __half h[M];
    signed char q[M];
    int nb = 0;
    _Pragma("unroll") for (int m = 0; m < M; ++m) a[m] = b[m] = 0.f;
    WALK(h[m] = g[base + j]; q[m] = P[base + j];,
         const float v = __half2float(h[m]);
         if (!isfinite(v)) nb++; else { a[m] = __fmaf_rn(v, v, a[m]); b[m] += (q[m] >= 0) ? v : -v; })
    if (nb) atomicAdd(bad, nb);
    const float sq = wsum<M>(a, lane, L), sg = wsum<M>(b, lane, L);
    if (lane == 0) {
        rowsq[row] = sq;
        rowsg[row] = sg;
        if (R != 0) {
            const float w = sc[row];
            R[row] = __fmaf_rn(__fadd_rn(__fmul_rn(__fmul_rn(__fmul_rn(sq, w), w), inv_in), eps), k1, __fmul_rn(R[row], b2));
        }
    }
}
// One thread per column: the column's sum of (s_i * g_ij)^2, added up row after row, and with C the column's
// second moment brought up to date, C = C * b2 + (colsq * inv_out + eps) * k1 (rounded as row_stats rounds R).
// A thread reads UNC rows ahead of its sum. Reading one row at a time, each read waiting for the one before, a
// few dozen warps each sat idle for the length of a memory access 2,000 times over; the sum itself and its
// order are unchanged. (Measured per training step: 16 rows ahead 19.7 ms, 32 18.2, 64 15.0.)
#define UNR 16
#define UNC 64
extern "C" __global__ void col_stats(const __half* __restrict__ g, const float* __restrict__ s,
                                     float* __restrict__ colsq, float* __restrict__ C, float b2, float k1,
                                     float inv_out, float eps, int n_out, int n_in) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_in) return;
    const __half* q = g + j;
    float a = 0.f;
    int i = 0;
    for (; i + UNC <= n_out; i += UNC) {
        __half h[UNC];
        #pragma unroll
        for (int u = 0; u < UNC; ++u) h[u] = q[(long long)(i + u) * n_in];
        #pragma unroll
        for (int u = 0; u < UNC; ++u) {
            const float v = s[i + u] * __half2float(h[u]);
            if (isfinite(v)) a += v * v;
        }
    }
    for (; i < n_out; ++i) {
        const float v = s[i] * __half2float(q[(long long)i * n_in]);
        if (isfinite(v)) a += v * v;
    }
    colsq[j] = a;
    if (C != 0) C[j] = __fmaf_rn(__fadd_rn(__fmul_rn(a, inv_out), eps), k1, __fmul_rn(C[j], b2));
}
// What an update multiplies a gradient by. One thread per row and per column: rr = s / sqrt(max(R / mean(R) *
// inv_bc, lo)) for the row (inv_bc undoes the second moments' start from zero), cc = 1 / sqrt(max(C, lo)) for
// the column; a NaN stays one. The arithmetic of (R / R.mean()).div_(1 - b2 ** t).clamp_(min=lo).rsqrt_() * s
// and of C.clamp(min=lo).rsqrt_(), rounded where those round.
extern "C" __global__ void upd_rrcc(const float* __restrict__ R, const float* __restrict__ C,
                                    const float* __restrict__ mean, const float* __restrict__ s,
                                    float* __restrict__ rr, float* __restrict__ cc, float inv_bc, float lo,
                                    int n_out, int n_in) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_out) {
        float v = __fmul_rn(__fdiv_rn(R[i], mean[0]), inv_bc);
        if (v == v) v = fmaxf(v, lo);
        rr[i] = __fmul_rn(rsqrtf(v), s[i]);
    }
    if (i < n_in) {
        float v = C[i];
        if (v == v) v = fmaxf(v, lo);
        cc[i] = rsqrtf(v);
    }
}
// One warp per row: the row's sum of (g_ij * rr_i * cc_j)^2, for the update's rms, as a block of 256 threads
// summed it.
extern "C" __global__ void upd_sq(const __half* g, const float* rr, const float* cc, float* out, int n_out, int n_in) {
    WROW(n_out)
    const int L = 32, M = 8;
    const float r = rr[row];
    float a[M], c[M];
    __half h[M];
    _Pragma("unroll") for (int m = 0; m < M; ++m) a[m] = 0.f;
    WALK(h[m] = g[base + j]; c[m] = cc[j];,
         const float v = __half2float(h[m]) * r * c[m];
         if (isfinite(v)) a[m] = __fmaf_rn(v, v, a[m]);)
    const float sq = wsum<M>(a, lane, L);
    if (lane == 0) out[row] = sq;
}
// kick (0 = none): a latent whose new value is on the other side of the sign boundary (>= 0 against < 0) is put
// kick units further on before the clamp, so that it takes kick units of travel to come back.
// One block per row: P = clamp(floor(P - coef * g * rr_i * cc_j + U), -127, 127), U uniform in [0, 1). coef is
// a, or with tot (the sum of upd_sq's rows) a / max(sqrt(tot * inv_n), 1): Adafactor's update clipping, rounded
// as a / (tot / n).sqrt_().clamp_(min=1.0) rounds it (a number divided by a tensor is its reciprocal times a).
extern "C" __global__ void apply(signed char* P, const __half* g, const float* rr, const float* cc,
                                 const float* tot, float inv_n, float a, int* flips, unsigned int seed,
                                 int count, int n_in, int kick) {
    const int T = blockDim.x;
    const long long base = (long long)blockIdx.x * n_in;
    float coef = a;
    if (tot != 0) {
        float v = sqrtf(__fmul_rn(tot[0], inv_n));
        if (v == v) v = fmaxf(v, 1.f);
        coef = __fmul_rn(__fdiv_rn(1.f, v), a);
    }
    const float r = rr[blockIdx.x] * coef;
    int f = 0;
    for (int j = threadIdx.x; j < n_in; j += T) {
        const long long k = base + j;
        float d = r * cc[j] * __half2float(g[k]);
        if (!isfinite(d)) d = 0.f;
        const float u = (pcg((unsigned int)k + seed) >> 8) * (1.0f / 16777216.0f);
        const signed char old = P[k];
        float t = floorf((float)old - d + u);
        if (kick != 0 && ((t >= 0.f) != (old >= 0))) t += t >= 0.f ? (float)kick : -(float)kick;
        const signed char nw = (signed char)fminf(fmaxf(t, -127.f), 127.f);
        f += ((nw >= 0) != (old >= 0));
        P[k] = nw;
    }
    if (count && f) atomicAdd(flips, f);
}
// The same step, eight weights per thread, for a matrix whose rows are a multiple of 8 long at addresses that are:
// one read of eight latents, two of four gradients each, and one write, where the kernel above makes a read and a
// write per weight. Each weight gets the arithmetic it had, with the same random number.
extern "C" __global__ void apply8(unsigned long long* __restrict__ P, const unsigned long long* __restrict__ g,
                                  const float* __restrict__ rr, const float* __restrict__ cc, const float* tot,
                                  float inv_n, float a, int* flips, unsigned int seed, int count, long long n8,
                                  int n_in, int kick) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    float coef = a;
    if (tot != 0) {
        float v = sqrtf(__fmul_rn(tot[0], inv_n));
        if (v == v) v = fmaxf(v, 1.f);
        coef = __fmul_rn(__fdiv_rn(1.f, v), a);
    }
    const long long k0 = 8 * i;
    const int row = (int)(k0 / n_in), j0 = (int)(k0 - (long long)row * n_in);
    const float r = rr[row] * coef;
    const unsigned long long pw = P[i], gw[2] = {g[2 * i], g[2 * i + 1]};
    unsigned long long nwv = 0;
    int f = 0;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        const long long k = k0 + u;
        float d = r * cc[j0 + u] * __half2float(__ushort_as_half((unsigned short)(gw[u >> 2] >> (16 * (u & 3)))));
        if (!isfinite(d)) d = 0.f;
        const float uu = (pcg((unsigned int)k + seed) >> 8) * (1.0f / 16777216.0f);
        const signed char old = (signed char)(pw >> (8 * u));
        float t = floorf((float)old - d + uu);
        if (kick != 0 && ((t >= 0.f) != (old >= 0))) t += t >= 0.f ? (float)kick : -(float)kick;
        const signed char nw = (signed char)fminf(fmaxf(t, -127.f), 127.f);
        f += ((nw >= 0) != (old >= 0));
        nwv |= (unsigned long long)(unsigned char)nw << (8 * u);
    }
    P[i] = nwv;
    if (count && f) atomicAdd(flips, f);
}
// The bound and the norm per position, forward and backward, each in one pass over the tensor (the tensor-op
// versions read and write it eight times in full precision). One warp per row of n_in values throughout; T is the
// number of threads a row had as a block of its own (256, or the largest power of two the row holds), which fixes
// the order its sum is taken in (see "one warp per row" above). WKERNEL calls the body written for M = T / 32.
#define WKERNEL(F, ...) { const int L = T < 32 ? T : 32; switch (T >> 5) { \
    case 8: F<8>(__VA_ARGS__, L, lane, row, base); break; case 4: F<4>(__VA_ARGS__, L, lane, row, base); break; \
    case 2: F<2>(__VA_ARGS__, L, lane, row, base); break; default: F<1>(__VA_ARGS__, L, lane, row, base); } }
// r = c / rms where the row's rms is above c, else 1; y = x * r; ms = the row's mean square as it came in.
template <int M> static __device__ __forceinline__ void cap_fwd_w(
        const __half* x, __half* y, float* r, float* ms, int n_in, float c, int L, int lane, int row, long long base) {
    float a[M];
    __half h[M];
    _Pragma("unroll") for (int m = 0; m < M; ++m) a[m] = 0.f;
    WALK(h[m] = x[base + j];, const float v = __half2float(h[m]); a[m] = __fmaf_rn(v, v, a[m]);)
    const float mean = __shfl_sync(0xffffffffu, wsum<M>(a, lane, L), 0) / n_in;
    const float rr = mean > c * c ? c * rsqrtf(mean) : 1.f;
    if (lane == 0) { r[row] = rr; ms[row] = mean; }
    WALK(h[m] = x[base + j];, y[base + j] = __float2half(__half2float(h[m]) * rr);)
}
extern "C" __global__ void cap_fwd(const __half* x, __half* y, float* r, float* ms, int n_rows, int n_in, int T, float c) {
    WROW(n_rows)
    WKERNEL(cap_fwd_w, x, y, r, ms, n_in, c)
}
// Where the row was scaled (r < 1): gx = (g - y * mean(g * y) / c^2) * r with y = x * r; elsewhere gx = g.
template <int M> static __device__ __forceinline__ void cap_bwd_w(
        const __half* x, const __half* g, const float* r, __half* gx, int n_in, float c, int L, int lane, int row,
        long long base) {
    const float rr = r[row];
    float a[M];
    __half hg[M], hx[M];
    _Pragma("unroll") for (int m = 0; m < M; ++m) a[m] = 0.f;
    if (rr < 1.f) {
        WALK(hg[m] = g[base + j]; hx[m] = x[base + j];, a[m] = __fmaf_rn(__half2float(hg[m]), __half2float(hx[m]), a[m]);)
    }
    const float tot = __shfl_sync(0xffffffffu, wsum<M>(a, lane, L), 0);
    const float k = rr < 1.f ? tot * rr / n_in / (c * c) * rr : 0.f;  // mean(g * y) / c^2 * r
    WALK(hg[m] = g[base + j]; hx[m] = x[base + j];,
         gx[base + j] = __float2half((__half2float(hg[m]) - __half2float(hx[m]) * k) * rr);)
}
extern "C" __global__ void cap_bwd(const __half* x, const __half* g, const float* r, __half* gx, int n_rows, int n_in,
                                   int T, float c) {
    WROW(n_rows)
    WKERNEL(cap_bwd_w, x, g, r, gx, n_in, c)
}
// r = 1 / sqrt(mean square + 1e-6); y = x * r * w_j (w has n_in entries, or one when wn == 1).
template <int M> static __device__ __forceinline__ void norm_fwd_w(
        const __half* x, const float* w, __half* y, float* r, int n_in, int wn, int L, int lane, int row, long long base) {
    float a[M];
    __half h[M];
    _Pragma("unroll") for (int m = 0; m < M; ++m) a[m] = 0.f;
    WALK(h[m] = x[base + j];, const float v = __half2float(h[m]); a[m] = __fmaf_rn(v, v, a[m]);)
    const float rr = rsqrtf(__shfl_sync(0xffffffffu, wsum<M>(a, lane, L), 0) / n_in + 1e-6f);
    if (lane == 0) r[row] = rr;
    WALK(h[m] = x[base + j];, y[base + j] = __float2half(__half2float(h[m]) * rr * (wn == 1 ? w[0] : w[j]));)
}
extern "C" __global__ void norm_fwd(const __half* x, const float* w, __half* y, float* r, int n_rows, int n_in, int T,
                                    int wn) {
    WROW(n_rows)
    WKERNEL(norm_fwd_w, x, w, y, r, n_in, wn)
}
// gx = (g * w - xr * mean(g * w * xr)) * r with xr = x * r.
template <int M> static __device__ __forceinline__ void norm_bwd_w(
        const __half* x, const __half* g, const float* w, const float* r, __half* gx, int n_in, int wn, int L, int lane,
        int row, long long base) {
    const float rr = r[row];
    float a[M];
    __half hg[M], hx[M];
    _Pragma("unroll") for (int m = 0; m < M; ++m) a[m] = 0.f;
    WALK(hg[m] = g[base + j]; hx[m] = x[base + j];,
         a[m] = __fmaf_rn(__half2float(hg[m]) * (wn == 1 ? w[0] : w[j]), __half2float(hx[m]), a[m]);)
    const float k = __shfl_sync(0xffffffffu, wsum<M>(a, lane, L), 0) * rr / n_in * rr;  // mean(g * w * xr) * r
    WALK(hg[m] = g[base + j]; hx[m] = x[base + j];,
         gx[base + j] = __float2half((__half2float(hg[m]) * (wn == 1 ? w[0] : w[j]) - __half2float(hx[m]) * k) * rr);)
}
extern "C" __global__ void norm_bwd(const __half* x, const __half* g, const float* w, const float* r, __half* gx,
                                    int n_rows, int n_in, int T, int wn) {
    WROW(n_rows)
    WKERNEL(norm_bwd_w, x, g, w, r, gx, n_in, wn)
}
// One thread per column: the gain's gradient, gw_j = sum over rows of g_ij * x_ij * r_i.
extern "C" __global__ void norm_gw(const __half* x, const __half* g, const float* r, float* gw, int n_rows, int n_in) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_in) return;
    float a = 0.f;
    int i = 0;
    for (; i + UNR <= n_rows; i += UNR) {  // reads UNR rows ahead of the sum, as col_stats does
        __half hg[UNR], hx[UNR];
        #pragma unroll
        for (int u = 0; u < UNR; ++u) {
            const long long k = (long long)(i + u) * n_in + j;
            hg[u] = g[k];
            hx[u] = x[k];
        }
        #pragma unroll
        for (int u = 0; u < UNR; ++u) a += __half2float(hg[u]) * __half2float(hx[u]) * r[i + u];
    }
    for (; i < n_rows; ++i) {
        const long long k = (long long)i * n_in + j;
        a += __half2float(g[k]) * __half2float(x[k]) * r[i];
    }
    gw[j] = a;
}
// Rotary positions, one thread per pair of features. x, y (B, nh, L, hd) half; cs, sn (B, 1, L, hd / 2) float.
// y_even = x_even * cos - x_odd * sin, y_odd = x_even * sin + x_odd * cos. Every product and sum is rounded on its
// own (no fused multiply-add), as separate tensor operations round them, so the result is theirs bit for bit.
extern "C" __global__ void rope_fwd(const __half* x, const float* cs, const float* sn, __half* y,
                                    long long n_pairs, int nh, int L, int hp) {
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) return;
    const long long row = p / hp;  // (b * nh + h) * L + l
    const long long k = (row / L / nh * L + row % L) * hp + p % hp;
    const float c = cs[k], s = sn[k];
    const float x1 = __half2float(x[2 * p]), x2 = __half2float(x[2 * p + 1]);
    y[2 * p] = __float2half(__fadd_rn(__fmul_rn(x1, c), -__fmul_rn(x2, s)));
    y[2 * p + 1] = __float2half(__fadd_rn(__fmul_rn(x1, s), __fmul_rn(x2, c)));
}
// The same pairs backward: gx_even = g_even * cos + g_odd * sin, gx_odd = g_odd * cos - g_even * sin. Autograd adds
// the even and the odd half of this gradient to zeros, which turns a -0 into +0; ZFIX does the same.
#define ZFIX(v) __float2half(__fadd_rn(__half2float(__float2half(v)), 0.f))
extern "C" __global__ void rope_bwd(const __half* g, const float* cs, const float* sn, __half* gx,
                                    long long n_pairs, int nh, int L, int hp) {
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) return;
    const long long row = p / hp;
    const long long k = (row / L / nh * L + row % L) * hp + p % hp;
    const float c = cs[k], s = sn[k];
    const float ga = __half2float(g[2 * p]), gb = __half2float(g[2 * p + 1]);
    gx[2 * p] = ZFIX(__fadd_rn(__fmul_rn(ga, c), __fmul_rn(gb, s)));
    gx[2 * p + 1] = ZFIX(__fadd_rn(__fmul_rn(-ga, s), __fmul_rn(gb, c)));
}
// One thread per element: +-1 in half precision from the sign of an int8 latent.
extern "C" __global__ void sgn(const signed char* p, __half* out, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(p[i] >= 0 ? 1.f : -1.f);
}
// The same, eight latents per thread, for a matrix whose size is a multiple of 8 at addresses that are: 0x3C00 is
// 1.0 in half precision and its top bit makes it -1.0, so each latent's sign bit is copied to the top of its half.
extern "C" __global__ void sgn8(const unsigned long long* __restrict__ p, unsigned long long* __restrict__ out,
                                long long n8) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    const unsigned long long w = p[i], v = w >> 32, one = 0x3C003C003C003C00ULL;
    out[2 * i] = one | ((w & 0x80ULL) << 8) | ((w & 0x8000ULL) << 16) | ((w & 0x800000ULL) << 24)
                     | ((w & 0x80000000ULL) << 32);
    out[2 * i + 1] = one | ((v & 0x80ULL) << 8) | ((v & 0x8000ULL) << 16) | ((v & 0x800000ULL) << 24)
                         | ((v & 0x80000000ULL) << 32);
}
// One block per row: logsumexp over one slice of the vocabulary of z * s + b.
extern "C" __global__ void ce_lse(const __half* z, const float* s, const float* b, float* out, int n_cols) {
    extern __shared__ float sh[];
    const int T = blockDim.x;
    const long long base = (long long)blockIdx.x * n_cols;
    float m = -1e30f, a = 0.f;
    for (int v = threadIdx.x; v < n_cols; v += T) {
        const float x = __half2float(z[base + v]) * s[v] + b[v];
        if (x > m) { a = a * expf(m - x) + 1.f; m = x; } else { a += expf(x - m); }
    }
    sh[threadIdx.x] = m;
    sh[T + threadIdx.x] = a;
    __syncthreads();
    for (int st = T >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            const float m1 = sh[threadIdx.x], m2 = sh[threadIdx.x + st];
            const float a1 = sh[T + threadIdx.x], a2 = sh[T + threadIdx.x + st];
            const float M = fmaxf(m1, m2);
            sh[threadIdx.x] = M;
            sh[T + threadIdx.x] = (a1 > 0.f ? a1 * expf(m1 - M) : 0.f) + (a2 > 0.f ? a2 * expf(m2 - M) : 0.f);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sh[0] + logf(sh[T]);
}
// One thread per vocabulary entry of the slice: the softmax error of every row, written over z as
// (p - onehot) * k, plus the row-weighted gradients of that entry's output scale and bias.
extern "C" __global__ void ce_grad(__half* z, const float* s, const float* b, const float* lse, const float* w,
                                   const int* off, float* gs, float* gb, float k, int n_rows, int n_cols) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_cols) return;
    const float sv = s[v], bv = b[v];
    float as = 0.f, ab = 0.f;
    for (int n = 0; n < n_rows; ++n) {
        const long long i = (long long)n * n_cols + v;
        const float zz = __half2float(z[i]);
        float p = expf(zz * sv + bv - lse[n]);
        if (off[n] == v) p -= 1.f;
        ab += p * w[n];
        as += p * w[n] * zz;
        z[i] = __float2half(p * k);
    }
    gs[v] = as;
    gb[v] = ab;
}
'''


class _Fused:
    """The int8 update as four CUDA kernels (through CuPy): two passes of statistics, one for the
    update's rms and one that applies it. About 11 bytes of memory traffic per weight, against roughly
    90 when the same arithmetic goes through tensor operations, and nothing waits on the GPU."""

    def __init__(self):
        import cupy
        mod = cupy.RawModule(code=_KSRC)
        self.row, self.col, self.usq, self.app, self.sgn, self.k_lse, self.k_grad = (
            mod.get_function(n) for n in ('row_stats', 'col_stats', 'upd_sq', 'apply', 'sgn', 'ce_lse', 'ce_grad'))
        self.k_cf, self.k_cb, self.k_nf, self.k_nb, self.k_ng = (
            mod.get_function(n) for n in ('cap_fwd', 'cap_bwd', 'norm_fwd', 'norm_bwd', 'norm_gw'))
        self.k_rf, self.k_rb = mod.get_function('rope_fwd'), mod.get_function('rope_bwd')
        self.k_rr, self.k_s8, self.k_a8 = (mod.get_function(n) for n in ('upd_rrcc', 'sgn8', 'apply8'))
        self.calls = 0
        self._ones = {}

    @staticmethod
    def _rows(x):
        """x (rows, n_in) -> how a one-warp-per-row kernel is launched for it (eight rows to a block), and T: the
        number of threads a row had when it was a block of its own (256, or the largest power of two the row
        holds). The kernel adds a row up in the order those threads did."""
        t = 256
        while t > x.shape[1]:
            t >>= 1
        return ((x.shape[0] + 7) // 8,), (256,), np.int32(t)

    def cap_fwd(self, x, c):
        """x (rows, n_in) half, contiguous -> the bounded rows, each row's factor and its mean square coming in."""
        p, i32 = (lambda t: np.uintp(t.data_ptr())), np.int32
        y, r, ms = torch.empty_like(x), torch.empty(x.shape[0], device=DEV), torch.empty(x.shape[0], device=DEV)
        grid, block, T = self._rows(x)
        self.k_cf(grid, block, (p(x), p(y), p(r), p(ms), i32(x.shape[0]), i32(x.shape[1]), T, np.float32(c)))
        return y, r, ms

    def cap_bwd(self, x, g, r, c):
        p, i32 = (lambda t: np.uintp(t.data_ptr())), np.int32
        gx = torch.empty_like(g)
        grid, block, T = self._rows(x)
        self.k_cb(grid, block, (p(x), p(g), p(r), p(gx), i32(x.shape[0]), i32(x.shape[1]), T, np.float32(c)))
        return gx

    def norm_fwd(self, x, w):
        """x (rows, n_in) half, contiguous; w (n_in,) or (1,) float -> the normed rows times w, and each row's 1 / rms."""
        p, i32 = (lambda t: np.uintp(t.data_ptr())), np.int32
        y, r = torch.empty_like(x), torch.empty(x.shape[0], device=DEV)
        grid, block, T = self._rows(x)
        self.k_nf(grid, block, (p(x), p(w), p(y), p(r), i32(x.shape[0]), i32(x.shape[1]), T, i32(w.numel())))
        return y, r

    def norm_bwd(self, x, g, w, r, need_gw):
        p, i32 = (lambda t: np.uintp(t.data_ptr())), np.int32
        gx = torch.empty_like(g)
        grid, block, T = self._rows(x)
        self.k_nb(grid, block, (p(x), p(g), p(w), p(r), p(gx), i32(x.shape[0]), i32(x.shape[1]), T, i32(w.numel())))
        gw = None
        if need_gw:
            gw = torch.empty(x.shape[1], device=DEV)
            self.k_ng(((x.shape[1] + 255) // 256,), (256,), (p(x), p(g), p(r), p(gw), np.int32(x.shape[0]), np.int32(x.shape[1])))
        return gx, gw

    def rope(self, x, cos, sin, back):
        """x (B, nh, L, hd) half, contiguous; cos, sin (B, 1, L, hd / 2) float -> x rotated, or with back the
        gradient of the input for the gradient x of the output."""
        B, nh, L, hd = x.shape
        cos, sin = cos.contiguous(), sin.contiguous()
        assert x.is_contiguous() and cos.shape == (B, 1, L, hd // 2) and cos.dtype == torch.float32
        y = torch.empty_like(x)
        n = x.numel() // 2
        p = lambda t: np.uintp(t.data_ptr())
        (self.k_rb if back else self.k_rf)(((n + 255) // 256,), (256,), (
            p(x), p(cos), p(sin), p(y), np.int64(n), np.int32(nh), np.int32(L), np.int32(hd // 2)))
        return y

    def signs(self, P):
        out = torch.empty(P.shape, device=DEV, dtype=HALF)
        n, a, b = P.numel(), P.data_ptr(), out.data_ptr()
        if n % 8 == 0 and a % 8 == 0 and b % 8 == 0:  # eight latents per thread
            self.k_s8(((n // 8 + 255) // 256,), (256,), (np.uintp(a), np.uintp(b), np.int64(n // 8)))
        else:
            self.sgn(((n + 255) // 256,), (256,), (np.uintp(a), np.uintp(b), np.int64(n)))
        return out

    def lse(self, z, s, b):
        p = lambda t: np.uintp(t.data_ptr())
        out = torch.empty(z.shape[0], device=DEV)
        self.k_lse((z.shape[0],), (256,), (p(z), p(s), p(b), p(out), np.int32(z.shape[1])), shared_mem=2048)
        return out

    def ce_grad(self, z, s, b, lse, w, off, gs, gb, k):
        p = lambda t: np.uintp(t.data_ptr())
        self.k_grad(((z.shape[1] + 255) // 256,), (256,), (p(z), p(s), p(b), p(lse), p(w), p(off), p(gs), p(gb),
                                                           np.float32(k), np.int32(z.shape[0]), np.int32(z.shape[1])))

    def update(self, sign, g, s, P, e, floor=None):
        n_out, n_in = g.shape
        p, i32, f32 = (lambda t: np.uintp(0 if t is None else t.data_ptr())), np.int32, np.float32
        rowsq, rowsg = torch.empty(n_out, device=DEV), torch.empty(n_out, device=DEV)
        # the second moments, one per row (R) and one per column (C), are brought up to date by the two kernels
        # that take the statistics; the token table (floor) has none
        R, C = (sign.R[e], sign.C[e]) if SO.enabled and floor is None else (None, None)
        b2, k1, eps = f32(SO.b2), f32(1 - SO.b2), f32(1e-30)
        rows = ((n_out + 7) // 8,), (256,)  # one warp per row, eight rows to a block
        self.row(*rows, (p(g), p(P), p(rowsq), p(rowsg), p(SO.bad_t), p(R), p(s), b2, k1, f32(1) / f32(n_in), eps,
                         i32(n_out), i32(n_in)))
        if not SO.enabled:
            return rowsg
        if floor is not None:  # the token table (see Sign.update): each row by its own rms, or by floor if larger
            rr = (torch.sign(s) / (rowsq / n_in).sqrt_().clamp_(min=floor)).contiguous()
            if n_in not in self._ones:
                self._ones[n_in] = torch.ones(n_in, device=DEV)
            cc = self._ones[n_in]
            tot, inv_n, a = None, f32(0), f32(SO.a * SO.table)
        else:
            colsq = torch.empty(n_in, device=DEV)
            self.col(((n_in + 31) // 32,), (32,), (p(g), p(s), p(colsq), p(C), b2, k1, f32(1) / f32(n_out), eps,
                                                   i32(n_out), i32(n_in)))  # blocks of one warp: 10% faster than of eight
            sign.t[e] += 1
            mean, rr, cc = R.mean(), torch.empty(n_out, device=DEV), torch.empty(n_in, device=DEV)
            self.k_rr(((max(n_out, n_in) + 255) // 256,), (256,), (
                p(R), p(C), p(mean), p(s), p(rr), p(cc), f32(1) / f32(1 - SO.b2 ** sign.t[e]), eps, i32(n_out), i32(n_in)))
            usq = torch.empty(n_out, device=DEV)
            self.usq(*rows, (p(g), p(rr), p(cc), p(usq), i32(n_out), i32(n_in)))
            tot, inv_n, a = usq.sum(), f32(1) / f32(n_out * n_in), f32(SO.a)  # Adafactor update clipping, in the kernel
        self.calls += 1
        seed = np.uint32((self.calls * 2654435761) & 0xffffffff)
        kick = i32(SO.kick_table if floor is not None else SO.kick)
        if n_in % 8 == 0 and P.data_ptr() % 8 == 0 and g.data_ptr() % 8 == 0:  # eight weights per thread
            n8 = n_out * n_in // 8
            self.k_a8(((n8 + 255) // 256,), (256,), (p(P), p(g), p(rr), p(cc), p(tot), inv_n, a, p(SO.flips_t), seed,
                                                     i32(SO.count), np.int64(n8), i32(n_in), kick))
        else:
            self.app((n_out,), (256,), (p(P), p(g), p(rr), p(cc), p(tot), inv_n, a, p(SO.flips_t), seed, i32(SO.count),
                                        i32(n_in), kick))
        if SO.count:
            SO.seen += n_out * n_in
        return rowsg


FUSED = None
if CUDA and os.environ.get('BINTF_KERNELS', 'cupy') != 'torch':
    try:
        FUSED = _Fused()
    except Exception as e:
        print(f'CuPy kernels unavailable ({type(e).__name__}: {e}); the int8 update uses tensor ops', file=sys.stderr)


_KSEG = r'''
#include <cuda_fp16.h>
// One thread per column: for each segment of rows (start[e] .. start[e + 1]), the sum of g over the segment's
// ordinary rows and over its rows marked in mk (mk may be null: no row is marked), and how many rows of each.
extern "C" __global__ void seg_sums(const __half* g, const unsigned char* mk, const int* start, float* out,
                                    float* cnt, int n_seg, int n_cols) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_cols) return;
    const __half* q = g + j;
    for (int e = 0; e < n_seg; ++e) {
        const int r0 = start[e], r1 = start[e + 1];
        float a = 0.f, b = 0.f;
        int nb = 0, i = r0;
        for (; i + 16 <= r1; i += 16) {  // 16 rows are read ahead of the sums (see col_stats in _KSRC); same sums
            __half h[16];
            #pragma unroll
            for (int u = 0; u < 16; ++u) h[u] = q[(long long)(i + u) * n_cols];
            #pragma unroll
            for (int u = 0; u < 16; ++u) {
                const float v = __half2float(h[u]);
                if (mk != 0 && mk[i + u]) { b += v; nb++; } else a += v;
            }
        }
        for (; i < r1; ++i) {
            const float v = __half2float(q[(long long)i * n_cols]);
            if (mk != 0 && mk[i]) { b += v; nb++; } else a += v;
        }
        out[((long long)e * 2) * n_cols + j] = a;
        out[((long long)e * 2 + 1) * n_cols + j] = b;
        if (j == 0) { cnt[e * 2] = (float)(r1 - r0 - nb); cnt[e * 2 + 1] = (float)nb; }
    }
}
// One block per row: out = g - m[the row's segment][the row's kind], m (n_seg, 2, n_cols) in half precision.
extern "C" __global__ void seg_sub(const __half* g, const unsigned char* mk, const int* seg, const __half* m,
                                   __half* out, int n_cols) {
    const int i = blockIdx.x;
    const long long base = (long long)i * n_cols;
    const __half* mm = m + ((long long)(seg != 0 ? seg[i] : 0) * 2 + ((mk != 0 && mk[i]) ? 1 : 0)) * n_cols;
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x)
        out[base + j] = __float2half(__half2float(g[base + j]) - __half2float(mm[j]));
}
'''


class _Seg:
    """Sign.centred as two CUDA kernels: the sums over tokens, per matrix and per kind of position, in one pass
    over the gradient, and the subtraction in one more. As tensor operations the same needed a mask lookup
    that waits for the GPU, twice per matrix and 68 times per layer, and the step was 15% slower."""

    def __init__(self):
        import cupy
        mod = cupy.RawModule(code=_KSEG)
        self.k_sums, self.k_sub = mod.get_function('seg_sums'), mod.get_function('seg_sub')
        self._whole = {}

    def whole(self, n):
        """start for one segment that is all n rows."""
        if n not in self._whole:
            self._whole[n] = torch.tensor([0, n], dtype=torch.int32, device=DEV)
        return self._whole[n]

    def sums(self, g, mk, start, n_seg):
        """g (N, C) half, contiguous; mk (N,) bool or None; start (n_seg + 1,) int32. Returns the sums (n_seg, 2, C)
        and the row counts (n_seg, 2) of the ordinary and of the marked rows of each segment."""
        p = lambda t: np.uintp(0 if t is None else t.data_ptr())
        C = g.shape[1]
        out, cnt = torch.empty(n_seg, 2, C, device=DEV), torch.empty(n_seg, 2, device=DEV)
        self.k_sums(((C + 255) // 256,), (256,), (p(g), p(mk), p(start), p(out), p(cnt), np.int32(n_seg), np.int32(C)))
        return out, cnt

    def sub(self, g, mk, rows, m):
        """g minus m[rows[i], kind of row i] (m (n_seg, 2, C) half; rows int32 per row, or None for one segment)."""
        p = lambda t: np.uintp(0 if t is None else t.data_ptr())
        out = torch.empty_like(g)
        self.k_sub((g.shape[0],), (256,), (p(g), p(mk), p(rows), p(m), p(out), np.int32(g.shape[1])))
        return out


SEG = None
if FUSED is not None:
    try:
        SEG = _Seg()
    except Exception as e:
        print(f'CuPy kernels for centred gradients unavailable ({type(e).__name__}: {e}); tensor ops instead', file=sys.stderr)


_KMOE = r'''
#include <cuda_fp16.h>
// The feed-forward experts of a layer, the rows of every expert at once (see _MoE). The rows are the (token, expert)
// pairs grouped by expert; ex[i] is row i's expert and mk[i] its kind (1 where the token's input is [MASK]). Half
// precision is read and written four entries at a time. Every entry gets the arithmetic the tensor operations gave
// it, each step rounded where they round it, and a row is added up in the order torch.sum(-1) adds one on a GPU:
// the 32 lanes of a warp share the row, lane u adds every 32nd group of four entries into four sums, adds those
// one after the other, and the lanes are then added pairwise at distances 1, 2, 4, 8, 16. (torch does that for rows
// longer than 128, a multiple of 4, in tensors of 16 rows or more; _moe_whole checks all three.) This source is
// compiled without CuPy's flush-to-zero, as torch's own kernels are, so the tiniest numbers come out the same too.
#define H2F(w, i) __half2float(__ushort_as_half((unsigned short)((w) >> (16 * (i)))))
#define F2W(f, i) ((unsigned long long)__half_as_ushort(__float2half(f)) << (16 * (i)))
#define RELU(f) ((f) > 0.f ? (f) : ((f) == (f) ? 0.f : (f)))  /* a NaN stays one, as it does in torch.relu */
#define WROW const int lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); \
    if (row >= n_rows) return;
// the four sums of a lane into the row's sum, in every lane
#define ROWSUM(a) { a[0] = __fadd_rn(__fadd_rn(__fadd_rn(a[0], a[1]), a[2]), a[3]); \
    _Pragma("unroll") for (int off = 1; off < 32; off <<= 1) a[0] = __fadd_rn(a[0], __shfl_down_sync(0xffffffffu, a[0], off)); \
    a[0] = __shfl_sync(0xffffffffu, a[0], 0); }
// Forward, between an expert's two matmuls. HP comes in as x @ Wu.T and leaves as the pre-activations
// hp = raw * s_up + offset (half precision at each step); h = relu(hp)^2 goes to H when H is given (for the running
// means); r = 1 / sqrt(mean over the row of (h - mu)^2 + eps) to R; hn = (h - mu) * r, in half precision, to HN.
extern "C" __global__ void moe_fwd(unsigned long long* HP, const int* ex, const unsigned char* mk,
                                   const unsigned long long* s16, const unsigned long long* b16, const float* mu,
                                   unsigned long long* HN, float* R, float* H, int n_rows, int ff, float inv_ff,
                                   float eps) {
    WROW
    const int e = ex[row], g4 = ff >> 2;
    const long long ek = (long long)e * 2 + (mk != 0 && mk[row] ? 1 : 0);
    unsigned long long* hp = HP + (long long)row * g4;
    const unsigned long long* sv = s16 + (long long)e * g4;
    const unsigned long long* bv = b16 != 0 ? b16 + ek * g4 : 0;
    const float* m = mu + ek * ff;
    float a[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = lane; q < g4; q += 32) {
        const unsigned long long w = hp[q], s = sv[q], b = bv != 0 ? bv[q] : 0ULL;
        unsigned long long o = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float f = __half2float(__float2half(__fmul_rn(H2F(w, i), H2F(s, i))));
            if (bv != 0) f = __half2float(__float2half(__fadd_rn(f, H2F(b, i))));
            o |= F2W(f, i);
            const float t = RELU(f), h = __fmul_rn(t, t), c = __fsub_rn(h, m[4 * q + i]);
            if (H != 0) H[(long long)row * ff + 4 * q + i] = h;
            a[i] = __fadd_rn(a[i], __fmul_rn(c, c));
        }
        hp[q] = o;
    }
    ROWSUM(a)
    const float r = rsqrtf(__fadd_rn(__fmul_rn(a[0], inv_ff), eps));
    if (lane == 0) R[row] = r;
    unsigned long long* hn = HN + (long long)row * g4;
    for (int q = lane; q < g4; q += 32) {
        const unsigned long long w = hp[q];
        unsigned long long o = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float f = H2F(w, i), t = RELU(f);
            o |= F2W(__fmul_rn(__fsub_rn(__fmul_rn(t, t), m[4 * q + i]), r), i);
        }
        hn[q] = o;
    }
}
// hn once more, from the kept pre-activations and row factors: what the down matrices read, for their gradient.
// One thread per four entries.
extern "C" __global__ void moe_hn(const unsigned long long* HP, const int* ex, const unsigned char* mk, const float* mu,
                                  const float* R, unsigned long long* HN, int n_rows, int ff) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int g4 = ff >> 2;
    const long long row = i / g4;
    if (row >= n_rows) return;
    const float* m = mu + ((long long)ex[row] * 2 + (mk != 0 && mk[row] ? 1 : 0)) * ff + 4 * (i - row * g4);
    const float r = R[row];
    const unsigned long long w = HP[i];
    unsigned long long o = 0;
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        const float f = H2F(w, u), t = RELU(f);
        o |= F2W(__fmul_rn(__fsub_rn(__fmul_rn(t, t), m[u]), r), u);
    }
    HN[i] = o;
}
// Backward, between an expert's two transposed matmuls. GR is the gradient at hn as the matmul left it; with hn in
// full precision again, ghp = (gh - hn * mean over the row of gh * hn) * r * 2 relu(hp) goes to GH in half
// precision, and ghp times the up matrix's row scales to GS.
extern "C" __global__ void moe_bwd(const unsigned long long* GR, const unsigned long long* HP, const int* ex,
                                   const unsigned char* mk, const float* mu, const float* R,
                                   const unsigned long long* s16, unsigned long long* GH, unsigned long long* GS,
                                   int n_rows, int ff, float inv_ff) {
    WROW
    const int e = ex[row], g4 = ff >> 2;
    const float* m = mu + ((long long)e * 2 + (mk != 0 && mk[row] ? 1 : 0)) * ff;
    const unsigned long long* gr = GR + (long long)row * g4;
    const unsigned long long* hp = HP + (long long)row * g4;
    const unsigned long long* sv = s16 + (long long)e * g4;
    const float r = R[row];
    float a[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = lane; q < g4; q += 32) {
        const unsigned long long wg = gr[q], wh = hp[q];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float f = H2F(wh, i), t = RELU(f), hn = __fmul_rn(__fsub_rn(__fmul_rn(t, t), m[4 * q + i]), r);
            a[i] = __fadd_rn(a[i], __fmul_rn(H2F(wg, i), hn));
        }
    }
    ROWSUM(a)
    const float mean = __fmul_rn(a[0], inv_ff);
    unsigned long long* gh = GH + (long long)row * g4;
    unsigned long long* gs = GS + (long long)row * g4;
    for (int q = lane; q < g4; q += 32) {
        const unsigned long long wg = gr[q], wh = hp[q], s = sv[q];
        unsigned long long o = 0, os = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float f = H2F(wh, i), t = RELU(f), hn = __fmul_rn(__fsub_rn(__fmul_rn(t, t), m[4 * q + i]), r);
            const float g2 = __fmul_rn(__fsub_rn(H2F(wg, i), __fmul_rn(hn, mean)), r);
            const float gp = __half2float(__float2half(__fmul_rn(g2, __fmul_rn(t, 2.f))));
            o |= F2W(gp, i);
            os |= F2W(__fmul_rn(gp, H2F(s, i)), i);
        }
        gh[q] = o;
        gs[q] = os;
    }
}
// Per row, the sum of a * b for two half-precision rows, taken in full precision: (a.float() * b.float()).sum(-1).
extern "C" __global__ void rowdot(const unsigned long long* A, const unsigned long long* B, float* out, int n_rows,
                                  int n) {
    WROW
    const int g4 = n >> 2;
    const unsigned long long* pa = A + (long long)row * g4;
    const unsigned long long* pb = B + (long long)row * g4;
    float a[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = lane; q < g4; q += 32) {
        const unsigned long long wa = pa[q], wb = pb[q];
        #pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = __fadd_rn(a[i], __fmul_rn(H2F(wa, i), H2F(wb, i)));
    }
    ROWSUM(a)
    if (lane == 0) out[row] = a[0];
}
// out[t] = S[pos[2t]] + S[pos[2t + 1]] for rows of n half-precision entries: what index_add_ of a token's two rows
// into zeros gives (the first of them is added to zero, which turns a -0 into +0). One thread per four entries.
extern "C" __global__ void pairsum(const unsigned long long* S, const int* pos, unsigned long long* out, int n_tok,
                                   int n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int g4 = n >> 2;
    const long long t = i / g4;
    if (t >= n_tok) return;
    const long long q = i - t * g4;
    const unsigned long long a = S[(long long)pos[2 * t] * g4 + q], b = S[(long long)pos[2 * t + 1] * g4 + q];
    unsigned long long o = 0;
    #pragma unroll
    for (int u = 0; u < 4; ++u) o |= F2W(__fadd_rn(__fadd_rn(H2F(a, u), 0.f), H2F(b, u)), u);
    out[i] = o;
}
'''


class _Moe:
    """The kernels of _KMOE: _MoE on the rows of every expert of a layer at once. All tensors contiguous; half
    precision where the kernel says so, ex int32, mk bool or None."""

    def __init__(self):
        from cupy._core import core
        from cupy.cuda import compiler, function
        # The include paths a RawModule gets, but compiled here so that CuPy's -ftz=true is left out.
        out = compiler.compile_using_nvrtc(_KMOE, tuple(core.assemble_cupy_compiler_options(())), None, 'bintf_moe.cu')
        self.mod = function.Module()
        self.mod.load(out[0] if isinstance(out, tuple) else out)
        self.k_fwd, self.k_hn, self.k_bwd, self.k_dot, self.k_pair = (
            self.mod.get_function(n) for n in ('moe_fwd', 'moe_hn', 'moe_bwd', 'rowdot', 'pairsum'))

    @staticmethod
    def _p(t):
        return np.uintp(0 if t is None else t.data_ptr())

    def fwd(self, HP, ex, mk, s16, b16, mu, keep):
        """HP (rows, ff) = x @ Wu.T, overwritten with the pre-activations. Returns hn (rows, ff) half, each row's
        factor r, and h (rows, ff) in full precision when keep is set, else None."""
        p, n, ff = self._p, HP.shape[0], HP.shape[1]
        HN, R = torch.empty_like(HP), torch.empty(n, device=DEV)
        H = torch.empty(n, ff, device=DEV) if keep else None
        self.k_fwd(((n + 7) // 8,), (256,), (p(HP), p(ex), p(mk), p(s16), p(b16), p(mu), p(HN), p(R), p(H), np.int32(n),
                                             np.int32(ff), np.float32(1) / np.float32(ff), np.float32(1e-6)))
        return HN, R, H

    def hn(self, HP, ex, mk, mu, R):
        p, n, ff = self._p, HP.shape[0], HP.shape[1]
        HN = torch.empty_like(HP)
        self.k_hn(((n * (ff // 4) + 255) // 256,), (256,), (p(HP), p(ex), p(mk), p(mu), p(R), p(HN), np.int32(n), np.int32(ff)))
        return HN

    def bwd(self, GR, HP, ex, mk, mu, R, s16):
        """GR (rows, ff): the gradient at hn. Returns ghp and ghp times the up matrices' row scales, both half."""
        p, n, ff = self._p, HP.shape[0], HP.shape[1]
        GH, GS = torch.empty_like(HP), torch.empty_like(HP)
        self.k_bwd(((n + 7) // 8,), (256,), (p(GR), p(HP), p(ex), p(mk), p(mu), p(R), p(s16), p(GH), p(GS), np.int32(n),
                                             np.int32(ff), np.float32(1) / np.float32(ff)))
        return GH, GS

    def rowdot(self, A, B):
        p, n = self._p, A.shape[0]
        out = torch.empty(n, device=DEV)
        self.k_dot(((n + 7) // 8,), (256,), (p(A), p(B), p(out), np.int32(n), np.int32(A.shape[1])))
        return out

    def pairsum(self, S, pos, n_tok):
        p, d = self._p, S.shape[1]
        out = torch.empty(n_tok, d, dtype=S.dtype, device=DEV)
        self.k_pair(((n_tok * (d // 4) + 255) // 256,), (256,), (p(S), p(pos), p(out), np.int32(n_tok), np.int32(d)))
        return out


MOE = None
if FUSED is not None:
    try:
        MOE = _Moe()
    except Exception as e:
        print(f'CuPy kernels for whole layers of experts unavailable ({type(e).__name__}: {e}); tensor ops instead', file=sys.stderr)


def _pin(t):
    return t.pin_memory() if CUDA else t


_FOLD_N = 1 << 24


class _Fold:
    """The running mean of every latent over its net's last turns, kept in RAM beside the latents (Sign.avg, int8),
    for whoever reads the model: evaluation, export, eval, generate. Training never reads it.

    Why: a latent near zero changes sign from one step to the next (0.1-0.25% of all signs flip per step, and
    95% of a step is noise), and a reader at one instant gets those signs as they happen to stand. Measured at
    step 21,310 on the same validation rows: the signs of the mean of the last 2 checkpoints (240 steps apart)
    score 0.07 nats better than the last one alone, of the last 3 0.11 better, for every net. A smaller step
    buys the same and pays for it in learning; a mean costs memory.

    When a net leaves the GPU at the end of its turn its latents are copied to RAM anyway. A thread folds that
    copy in: avg <- avg + (latent - avg) * m / 256, with m = 256 / --avg-turns. The quotient is rounded down
    or up at random in its proportions (a dither of 0..255 before the shift), so a mean that sits a fraction of
    a unit from the latent still follows it on average."""

    def __init__(self):
        self.q, self.thread, self.dither, self.calls = queue.Queue(), None, None, 0

    def put(self, sign, m):
        if self.thread is None:
            g = torch.Generator().manual_seed(1234)
            self.dither = torch.randint(0, 256, (_FOLD_N + 4096,), generator=g, dtype=torch.int16)
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        sign.job = threading.Event()
        self.q.put((sign, m, sign.job))

    def join(self):
        self.q.join()

    def _run(self):
        while True:
            sign, m, done = self.q.get()
            try:
                self.fold(sign, m)
            except Exception as e:  # the mean is for readers: never the reason training stops
                print(f'  running mean of the latents: {type(e).__name__}: {e}', flush=True)
            finally:
                done.set()
                self.q.task_done()

    def fold(self, sign, m):
        cur = sign.cpu
        if sign.avg is None or sign.avg.shape != cur.shape:
            sign.avg = cur.clone()
            return
        a, c, n = sign.avg.view(-1), cur.view(-1), cur.numel()
        for i in range(0, n, _FOLD_N):
            j = min(n, i + _FOLD_N)
            self.calls += 1
            o = (self.calls * 2654435761) % 4096
            x = c[i:j].to(torch.int16)
            x.sub_(a[i:j]).mul_(m).add_(self.dither[o:o + j - i]).bitwise_right_shift_(8)
            a[i:j].add_(x.to(torch.int8))


FOLD = _Fold()


class Sign:
    """n binary matrices (n, d_out, d_in) held as int8 latents, each with one row and one column
    second-moment vector (Adafactor, no first moment)."""

    def __init__(self, n, d_out, d_in):
        self.shape = (n, d_out, d_in)
        self.P = None       # on the GPU while its net is active
        self.cpu = None     # the copy that waits in RAM
        self.R = torch.zeros(n, d_out, 1, device=DEV)
        self.C = torch.zeros(n, 1, d_in, device=DEV)
        self.t = [0] * n
        self._hot = None    # (e, int8 matrix on the GPU) while a non-resident matrix is in use
        self.G = self.gk = None  # running means over tokens of the output gradient and their weights (see centred)
        self.tiers = self.pos_of = None  # the token table under Config.tiers (see set_tiers)
        self.avg = None     # int8 in RAM: the running mean of the latents over the last turns (see _Fold), or None
        self.job = None     # an Event, set once that mean has been brought up to date from self.cpu

    def set_tiers(self, tiers):
        """Config.tiers for the token table (n slices of R rows): `tiers` is the number of slices in each group.
        The table is then kept in order of frequency: pos_of (V,) is the row of each id. Ids keep the tokenizer's
        numbering everywhere outside the table, the output scale and the output bias. The last rows of the first
        group are the later groups' entries: rows of spare ids that never occur in text."""
        n, R, _ = self.shape
        sizes = [int(v) for v in str(tiers).split(',')]
        if len(sizes) < 2 or min(sizes) < 1 or sum(sizes) != n:
            raise SystemExit(f'--tiers {tiers}: slices per group, two groups or more, {n} slices in all')
        ends = np.cumsum(sizes).tolist()
        self.tiers = tiers
        self.spans = [(e - s, e) for s, e in zip(sizes, ends)]  # the slices of each group
        self.entry = torch.arange(sizes[0] * R - len(sizes) + 1, sizes[0] * R, device=DEV)  # the entries' rows
        self.edge = torch.tensor([e * R for e in ends[:-1]], device=DEV)  # the first row of each later group

    def numel(self):
        return self.shape[0] * self.shape[1] * self.shape[2]

    def init_(self):
        parts = []
        for _ in range(self.shape[0]):
            p = torch.randn(self.shape[1:], device=DEV).mul_(SO.init_std / SO.q).add_(torch.rand(self.shape[1:], device=DEV))
            parts.append(p.floor_().clamp_(-127, 127).to(torch.int8).cpu())
        self.cpu = _pin(torch.stack(parts))
        self.P = None

    def load(self):
        if self.P is None:
            self.P = self.cpu.to(DEV)

    def store(self, release):
        if self.P is not None:
            if self.job is not None:  # the running mean is still reading the copy in RAM
                self.job.wait()
                self.job = None
            self.cpu.copy_(self.P)
            if release:
                self.P = None

    def latent(self, e):
        """Matrix e as int8 on the GPU: a view if this Sign is resident, else streamed up from RAM."""
        if self.P is not None:
            return self.P[e]
        if self._hot is None or self._hot[0] != e:
            self._hot = (e, self.cpu[e].to(DEV, non_blocking=True))
        return self._hot[1]

    def signs(self, e=0):
        P = self.latent(e)
        return FUSED.signs(P) if FUSED is not None else torch.where(P >= 0, _ONE, _NEG)

    def rows(self, idx):
        """Rows idx of the stack viewed as one (n * d_out, d_in) matrix, as int8 on the GPU."""
        if self.P is not None:
            return self.P.view(-1, self.shape[2])[idx]
        return self.cpu.view(-1, self.shape[2])[idx.cpu()].to(DEV)  # not resident: fetch these rows only

    @torch.no_grad()
    def centred(self, gy, e=0, mk=None, seg=None):
        """gy (N, d_out): the scaled loss gradient at the outputs of matrix e, one row per token. Returns gy
        minus the running mean of gy over tokens, and the sums over tokens it took (ordinary positions, [MASK]
        positions; in gy's units). While updates are on, this batch's means are folded into the running ones.
        Ordinary positions and [MASK] positions (mk) have a mean each, as the inputs have in _center.
        seg = (start, rows), both int32: gy holds the tokens of every matrix of this stack one after the other,
        matrix e's in rows start[e] .. start[e + 1], and rows names each row's matrix; all are done in one pass.

        Why: the weight gradient is the sum over tokens of gy(t) x(t)^T. Split gy and x into their means over
        tokens and the rest, and it is N mean(gy) mean(x)^T plus the part that follows the tokens. The first
        part points the same way at every step, so it is what the signs line up with first, and all it can do
        is turn the constant part of the input into an offset of the output, with sqrt(d) times the gain of
        anything else. On the sixth start that offset chased itself: units wanted offsets, the matrices that
        write to the residual stream moved its constant part to supply them, the running means of _center took
        the move away again 100 steps later, and so the constant never stopped moving (a third of its length
        per 158 steps of a net; 10-38% of the energy every matrix read was left-over mean, at cosine 0.9 with
        the direction of the move; the [MASK] positions' mean moved twice as fast as the others'; feed-forward
        pre-activations sat 7-14 standard deviations from zero). With the mean of gy taken out a matrix learns
        only how its output should follow its input, which is also the true gradient of a network whose inputs
        are centred by their mean over the data. The offsets a unit needs are stored ones (_Bias2), and those
        take the whole mean gradient."""
        n = self.shape[0] if seg is not None else 1
        if self.G is None:
            self.G = torch.zeros(self.shape[0], 2, self.shape[1], device=DEV)
            self.gk = torch.zeros(self.shape[0], 2, device=DEV)
        G, k = (self.G, self.gk) if seg is not None else (self.G[e:e + 1], self.gk[e:e + 1])
        fast = SEG is not None and gy.dtype == torch.float16  # two kernels: no mask lookup that waits for the GPU
        if fast:
            gy = gy.contiguous()
            start, rows = seg[:2] if seg is not None else (SEG.whole(gy.shape[0]), None)
            sums, cnt = SEG.sums(gy, mk, start, n)
        else:
            idx = seg[1].long() * 2 if seg is not None else torch.zeros(gy.shape[0], dtype=torch.long, device=gy.device)
            if mk is not None:
                idx = idx + mk.long()
            sums = torch.zeros(n * 2, gy.shape[1], device=gy.device).index_add_(0, idx, gy.float()).view(n, 2, -1)
            cnt = torch.zeros(n * 2, device=gy.device).index_add_(0, idx, torch.ones(idx.numel(), device=gy.device)).view(n, 2)
        if SO.enabled and not (_RV_FIXC and SO.n0 is not None):  # REVIEW FIX_C: no running-mean update on a ParScale step
            sums0, cnt0 = sums, cnt
            if SO.n0 is not None:  # ParScale step: the running mean is the plain stream's (stream 0's rows)
                tk = seg[2] if seg is not None else torch.arange(gy.shape[0], device=gy.device)
                s0 = tk < SO.n0
                i0 = seg[1].long() * 2 if seg is not None else torch.zeros(gy.shape[0], dtype=torch.long, device=gy.device)
                if mk is not None:
                    i0 = i0 + mk.long()
                i0 = i0[s0]
                sums0 = torch.zeros(n * 2, gy.shape[1], device=gy.device).index_add_(0, i0, gy[s0].float()).view(n, 2, -1)
                cnt0 = torch.zeros(n * 2, device=gy.device).index_add_(
                    0, i0, torch.ones(i0.numel(), device=gy.device)).view(n, 2)
            w = (cnt0 > 0).float().mul_(1 - SO.cb)  # a kind with no token in this batch keeps its mean
            d = (sums0 / cnt0.clamp(min=1.0)[..., None]).mul_(SO.inv_scale).sub_(G)  # in loss units, whatever the scale
            G.add_(d.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0).mul_(w[..., None]))  # what is not finite is left out
            k.add_((1 - k) * w)
        m = (G / (k.clamp(min=1e-8) * SO.inv_scale)[..., None]).clamp_(-3e4, 3e4).to(gy.dtype)
        if _RV_FIXC and SO.n0 is not None:  # REVIEW FIX_C: every stream's rows minus that stream's own batch mean
            tk_ = seg[2] if seg is not None else torch.arange(gy.shape[0], device=gy.device)
            key = torch.div(tk_, SO.n0, rounding_mode='floor').long() * n
            if seg is not None:
                key = key + seg[1].long()
            key = key * 2 + (mk.long() if mk is not None else 0)
            nk = int(key.max()) + 1
            s_ = torch.zeros(nk, gy.shape[1], device=gy.device).index_add_(0, key, gy.float())
            c_ = torch.zeros(nk, device=gy.device).index_add_(0, key, torch.ones(key.numel(), device=gy.device))
            out = (gy.float() - (s_ / c_.clamp(min=1.0)[:, None])[key]).to(gy.dtype)
        else:
            out = SEG.sub(gy, mk, rows, m) if fast else gy - m.view(n * 2, -1)[idx]
        if SO.wcap > 0:  # --wcap: no row (one position's say in this matrix's gradient) longer than wcap medians
            rn = torch.linalg.vector_norm(out.float(), dim=1)
            if SO.n0 is None:
                lim = rn.median() * SO.wcap
            else:  # ParScale step (D2b): the median of stream 0's rows, the plain model's, as a plain step takes it
                tk0 = seg[2] if seg is not None else torch.arange(gy.shape[0], device=gy.device)
                lim = rn[tk0 < SO.n0].median() * SO.wcap
            over = rn > lim
            SO.wc_t += torch.stack((over.float().mean(), rn.new_ones(())))
            if float(lim) > 0 and bool(over.any()):
                out = out * (lim / rn.clamp(min=1e-30)).clamp_(max=1.0).to(out.dtype)[:, None]
        return out, sums

    @torch.no_grad()
    def update(self, g, scale, e=0, floor=None):
        """g: (d_out, d_in) in half precision, the scaled loss gradient for the +-1 matrix before its
        row scale. Applies bintf.py's Int8Sign.step to this one matrix and returns, per row, the sum of
        sign * g, which is the gradient of the row scale.

        floor: given for the token table, most of whose rows belong to tokens that are not in the batch.
        Each row is then divided by its own rms, or by `floor` if that is larger; `floor` is the rms one
        occurrence of a token leaves in its row. A token that occurs moves its row a full step, and a
        row whose token did not occur moves only as far as the token was wrongly predicted. A second
        moment per row would divide such a row by its own tiny gradient instead, and every rare token's
        row would drift, a full step at a time, to one shared pattern: away from the mean hidden state."""
        s = scale.detach().float() * SO.inv_scale
        P = self.latent(e)
        if FUSED is not None:
            gs = FUSED.update(self, g, s, P, e, floor)
        else:
            g = g.float()
            gs = (g * torch.where(P >= 0, 1.0, -1.0)).sum(1)
            if SO.enabled and floor is not None:
                ok = torch.isfinite(g)
                SO.bad_t += (~ok).sum(dtype=torch.int32)
                g = torch.where(ok, g, torch.zeros_like(g))
                g.mul_(torch.sign(s)[:, None] / g.square().mean(1, keepdim=True).sqrt_().clamp_(min=floor))
                self._step(P, g, SO.a * SO.table, SO.kick_table)
            elif SO.enabled:
                g.mul_(s[:, None])
                if bool(torch.isfinite(g.sum())):
                    R, C = self.R[e], self.C[e]
                    R.mul_(SO.b2).add_(torch.linalg.vector_norm(g, dim=1, keepdim=True).square_()
                                       .div_(g.shape[1]).add_(1e-30), alpha=1 - SO.b2)
                    C.mul_(SO.b2).add_(torch.linalg.vector_norm(g, dim=0, keepdim=True).square_()
                                       .div_(g.shape[0]).add_(1e-30), alpha=1 - SO.b2)
                    self.t[e] += 1
                    g.mul_((R / R.mean()).div_(1 - SO.b2 ** self.t[e]).clamp_(min=1e-30).rsqrt_())
                    g.mul_(C.clamp(min=1e-30).rsqrt_())
                    rms = (torch.linalg.vector_norm(g) / math.sqrt(g.numel())).clamp_(min=1.0)  # update clipping
                    self._step(P, g, SO.a / float(rms), SO.kick)
                else:
                    SO.bad_t += 1
        if self.P is None:  # streamed matrix: write it back to RAM
            self.cpu[e].copy_(P)
            self._hot = None
        return gs

    @staticmethod
    def _step(P, u, step, kick=0):
        """P <- clamp(floor(P - step * u + U)), U uniform in [0, 1): stochastic rounding onto the int8 grid.
        kick: a latent that lands on the other side of the sign boundary is put that many units further on first
        (--sign-kick: it then takes as much travel again to come back)."""
        new = P.float().add_(u, alpha=-step).add_(torch.rand_like(u)).floor_()
        if kick:
            new.add_(torch.where(new >= 0, float(kick), -float(kick)).mul_((new >= 0) != (P >= 0)))
        new.clamp_(-127, 127)
        if SO.count:
            SO.flips_t += ((new >= 0) != (P >= 0)).sum(dtype=torch.int32)
            SO.seen += P.numel()
        P.copy_(new)


# ---------------------------------------------------------------- layers

def _act(hp, act):
    return F.relu(hp).square() if act == 'relu2' else F.gelu(hp)


def _dact(hp, act):
    if act == 'relu2':
        return F.relu(hp) * 2
    h = hp.float()
    return (0.5 * (1 + torch.erf(h * 0.7071067811865476))
            + h * torch.exp(-0.5 * h * h) * 0.3989422804014327).to(hp.dtype)


_GK = 0.5  # the largest entry of an output gradient as a weight-gradient matmul sees it


def _wscale(m):
    """For an output gradient gy whose largest entry is m: (f, c), with f a power of two to multiply gy by
    before the half-precision matmul gy.t() @ x that gives a binary matrix its gradient, and c = 1 / f to put
    back afterwards. f brings the largest entry to at most _GK, so the sum over every token of the batch stays
    inside fp16 whatever the loss scale is and however large one token's gradient is (a position that many
    others attend to collects the gradients of all of them, thousands of times the usual). A power of two
    only shifts exponents, so nothing is rounded. (1, 1) in the usual case, where nothing can overflow and
    nothing is lost. An inf or nan in gy stays one and is counted by the update as before."""
    if m > SO.gmax or m != m:
        SO.gmax = m if m == m else float('inf')
    if 0.015625 <= m <= 1.0:
        return 1.0, 1.0
    k = min(17, max(-15, math.ceil(math.log2(max(m, 1e-30) / _GK)))) if math.isfinite(m) else 17
    return 2.0 ** -k, 2.0 ** k


def _wupd(sign, gy, x, scale, e, f, c, mk=None, centred=False):
    """sign.update with the weight gradient gy.t() @ x taken at the scale f (see _wscale); returns the
    gradient of the row scales. Config.cov: gy is centred first (Sign.centred; mk marks the [MASK] positions)
    unless the caller has done that already (centred)."""
    if SO.frozen:  # --parscale-freeze-backbone: no weight gradient, no update, no row-scale gradient
        return None
    if SO.cov and not centred:
        gy = sign.centred(gy, e, mk)[0]
    if c == 1.0:
        return sign.update(gy.t() @ x, scale, e)
    return sign.update((gy * f).t() @ x, scale * c, e) * c


class _RouterLin(torch.autograd.Function):
    """MoE router scores x @ W.T. Its weight gradient is a sum over every token; in half precision that sum
    overflowed, and each overflow halved the loss scale of the whole model. Here it is taken on the
    gradient divided by four times its largest entry and returned in full precision."""

    @staticmethod
    def forward(ctx, x, W, *opt):
        ctx.again, ctx.nopt = (opt[0] if opt else None), len(opt)  # an _Again: x is not kept but computed again
        ctx.save_for_backward(*((W,) if ctx.again is not None else (x, W)))
        return F.linear(x, W.to(x.dtype))

    @staticmethod
    def backward(ctx, gy):
        if ctx.again is not None:
            (W,), x = ctx.saved_tensors, ctx.again.get()
            ctx.again = None
        else:
            x, W = ctx.saved_tensors
        gf = gy.float()
        m = torch.where(torch.isfinite(gf), gf.abs(), gf.new_zeros(())).max().clamp_(min=1e-30) * 4.0
        gw = ((gf / m).to(x.dtype).t() @ x).float() * m if ctx.needs_input_grad[1] else None  # None: refresh mode
        return (gy @ W.to(gy.dtype), gw) + (None,) * ctx.nopt


class _Bias(torch.autograd.Function):
    """x + b, one b per feature; b's gradient is summed over the tokens in full precision.

    Why offsets at all: a +-1 matrix has none, and its inputs are centred. A unit that needs a threshold (a
    squared ReLU does, a query or key that should lean one way does) can then only get it by lining its row
    up with whatever constant part its inputs still have, which is what the running means are slow to
    remove. So the constant part was regrown as fast as it was subtracted: centring the inputs a second time
    (Config.centre2) left MORE of it, 10-58% of the energy against 2-16%, and cost 0.17 nats by step 1,800.
    Rows lined up with a constant are also what let the branches grow without limit on the third start.
    A stored offset is exact, costs nothing, and takes that job off the signs."""

    @staticmethod
    def forward(ctx, x, b):
        return x + b.to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        return g, g.float().sum(tuple(range(g.dim() - 1)))


class _Bias2(torch.autograd.Function):
    """x (N, D) plus an offset on its first b.shape[1] features: b[0] at ordinary positions, b[1] where the
    input is [MASK] (mk; None when there are none). The running means that centre what a matrix reads are
    kept per kind of position, so the kind itself is not in what the matrix reads; an offset per kind is
    where a [MASK] position is told to behave unlike a position that holds a token. (Sharing one offset left
    that to the constant part of the stream: on the sixth start the mean of the [MASK] positions moved away
    from the mean of the others twice as fast as either moved.) b's gradient is summed over the tokens of
    each kind in full precision."""

    @staticmethod
    def forward(ctx, x, b, mk):
        n = b.shape[1]
        ctx.mk, ctx.n = mk, n
        bb = b.to(x.dtype)
        out = x.clone()
        out[:, :n] += bb[0] if mk is None else bb[mk.long()]
        return out

    @staticmethod
    def backward(ctx, g):
        n, mk = ctx.n, ctx.mk
        if SEG is not None and g.dtype == torch.float16:
            gc = g.contiguous()
            gb = SEG.sums(gc, mk, SEG.whole(gc.shape[0]), 1)[0][0, :, :n]
        else:
            idx = mk.long() if mk is not None else torch.zeros(g.shape[0], dtype=torch.long, device=g.device)
            gb = torch.zeros(2, n, device=g.device).index_add_(0, idx, g[:, :n].float())
        return g, gb, None


class _BinLin(torch.autograd.Function):
    """y = (x @ sign(P).T) * scale for x (N, d_in). Backward returns the gradients for x and scale and
    applies the int8 update for P in place; the weight gradient is never kept."""

    @staticmethod
    def forward(ctx, x, scale, sign, e, *opt):
        z = F.linear(x, sign.signs(e))
        ctx.sign, ctx.e = sign, e
        ctx.mk, ctx.nopt = (opt[0] if opt else None), len(opt)  # mk: the [MASK] positions, for Sign.centred
        ctx.save_for_backward(x, scale)
        return z * scale.to(z.dtype)

    @staticmethod
    def backward(ctx, gy):
        x, scale = ctx.saved_tensors
        gx = (gy * scale.to(gy.dtype)) @ ctx.sign.signs(ctx.e)
        f, c = _wscale(float(torch.linalg.vector_norm(gy, ord=float('inf'))))
        return (gx, _wupd(ctx.sign, gy, x, scale, ctx.e, f, c, ctx.mk), None, None) + (None,) * ctx.nopt


class _RMSNorm(torch.autograd.Function):
    """RMS norm computed in fp32, keeping only its input and one scalar per token for backward."""

    @staticmethod
    def forward(ctx, x, w):
        ctx.fused = None
        if _LEAN and FUSED is not None and x.dtype == torch.float16:  # one kernel pass (see _KSRC); the same arithmetic
            xc, wc = x.contiguous(), w.detach().float().reshape(-1).contiguous()
            y, r = FUSED.norm_fwd(xc.view(-1, x.shape[-1]), wc)
            ctx.fused = (xc, wc, r)
            return y.view(x.shape)
        xf = x.float()
        r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
        ctx.save_for_backward(x, w, r)
        return (xf * r * w).to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        if ctx.fused is not None:
            (xc, wc, r), ctx.fused = ctx.fused, ()  # () and not None: still the kernel path if asked again
            n = xc.shape[-1]
            gx, gw = FUSED.norm_bwd(xc.view(-1, n), gy.contiguous().view(-1, n), wc, r, ctx.needs_input_grad[1])
            return gx.view(xc.shape), gw
        x, w, r = ctx.saved_tensors
        xr, g = x.float() * r, gy.float()
        gw = (g * xr).sum(tuple(range(g.dim() - 1))) if ctx.needs_input_grad[1] else None
        g = g * w
        return ((g - xr * (g * xr).mean(-1, keepdim=True)) * r).to(gy.dtype), gw


class _RMSCap(torch.autograd.Function):
    """x scaled down to rms c where its rms is above c, and left alone elsewhere: a bound that does not turn
    a small input into a full-sized one. Attention spread evenly over the context returns nearly the same
    small vector at every position; normalised to rms 1, that vector is the largest thing every layer adds
    to the residual stream and the tokens become indistinguishable.

    The same bound sits on what a branch writes (Config.cap). A +-1 row that lines up with a direction its
    inputs share answers it sqrt(d) times as strongly as anything else, so a branch can grow what it writes
    without limit, and the norms downstream make only the proportions count: every branch gains by writing
    more than the others. Unbounded, the first positions of a row (few tokens to attend to, attended to by
    all the rest) reached four times the stream of the others in 6,000 steps, their gradients left half
    precision, and the running means that centre the stream were mostly those positions. rec: add what came
    in to SO.br_t, for the log."""

    @staticmethod
    def forward(ctx, x, *opt):
        c, rec = (opt[0] if opt else 1.0), (len(opt) > 1 and opt[1])
        again = opt[2] if len(opt) > 2 else None  # an _Again: x is not kept here but computed again for backward
        ctx.fused = None
        if _LEAN and FUSED is not None and x.dtype == torch.float16:  # one kernel pass (see _KSRC); the same arithmetic
            xc = x.contiguous()
            y, r, ms = FUSED.cap_fwd(xc.view(-1, x.shape[-1]), c)
            if rec and SO.learn:
                SO.br_t += torch.stack((ms.sqrt().mean(), (ms > c * c).float().mean(), ms.new_ones(())))
            ctx.c, ctx.rest, ctx.fused = c, (None,) * len(opt), (xc if again is None else again, r)
            return y.view(x.shape)
        xf = x.float()
        ms = xf.square().mean(-1, keepdim=True)
        over = ms > c * c
        if rec and SO.learn:
            SO.br_t += torch.stack((ms.sqrt().mean(), over.float().mean(), ms.new_ones(())))
        r = torch.where(over, c * torch.rsqrt(ms.clamp(min=1e-30)), ms.new_ones(()))
        ctx.c, ctx.over, ctx.rest = c, over, (None,) * len(opt)
        ctx.save_for_backward(x, r)
        return (xf * r).to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        if ctx.fused is not None:
            (xc, r), ctx.fused = ctx.fused, ()
            if isinstance(xc, _Again):
                xc = xc.get()
            n = xc.shape[-1]
            return (FUSED.cap_bwd(xc.view(-1, n), gy.contiguous().view(-1, n), r, ctx.c).view(xc.shape),) + ctx.rest
        x, r = ctx.saved_tensors
        xr, g = x.float() * r, gy.float()
        return (((g - xr * (g * xr).mean(-1, keepdim=True) * (ctx.over / (ctx.c * ctx.c))) * r).to(gy.dtype),) + ctx.rest


_UNIT = torch.ones((), device=DEV)  # gain of the parameter-free norms
_CB = 0.99  # share of a running mean over tokens that is kept at a step of --ref-tokens ids; a step uses SO.cb


def _ema(num, den, xs, mk):
    """Fold the mean over tokens of xs (N, D) into the running means num (2, D) / den (2,): row 0 for
    ordinary positions, row 1 for the positions whose input is [MASK] (mk; None when there are none).
    A mean that is not finite is left out: once inside a running mean it would stay there."""
    with torch.autocast(DEV.type, enabled=False):  # under autocast the matmul below would run, and overflow, in fp16
        xs = xs.float()
        if mk is None:
            mean = xs.mean(0)
            if bool(torch.isfinite(mean).all()):
                num[0].mul_(SO.cb).add_(mean, alpha=1 - SO.cb)
                den[0].mul_(SO.cb).add_(1 - SO.cb)
            return
        w = torch.stack((~mk, mk)).float()
        n = w.sum(1)
        mean = (w @ xs) / n.clamp(min=1)[:, None]
        step = (1 - SO.cb) * ((n > 0) & torch.isfinite(mean).all(1)).float()  # an absent kind keeps its mean
        num.add_((torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0) - num) * step[:, None])
        den.add_((1 - den) * step)


def _center(x, num, den, mk=None, give_mu=False):
    """x minus the running mean of x over tokens, one value per feature (num / den, both updated here
    while SO.learn is set). A +-1 row lines up with whatever its inputs have in common sooner than with
    anything else, and what it then writes is the same for every token; a few hundred steps of that and
    the common part is most of the residual stream. Inputs with no common part leave a row nothing of
    the kind to line up with, and a binary matrix fed centred inputs writes a centred output. The mean
    is a slow average over many batches, not this batch's: one row of text has a mean of its own (what
    it is about), and that has to stay. Positions whose input is [MASK] (mk) have a mean of their own:
    what they share is the [MASK] row itself, which the mean of all tokens does not remove.

    Config.centre2: the same again on what the matrix actually reads. A norm or a bound after the centring
    scales every position by its own factor, and what comes out has a common part once more. Measured on one
    net: without the bound on branches (third start, step 4,287) one constant vector was 10-15% of the energy
    the attention matrices read and 40-70% of what the attention output matrix read; with the bound (fourth
    start, step 1,883) 2-6% and 4-16%. A +-1 row answers that part 45 times as strongly as the rest, so about
    half of what the attention branches wrote was that one vector."""
    mu = (num / den.clamp(min=1e-8)[:, None]).to(x.dtype)  # as it stood before this batch
    if SO.learn and SO.ema:
        with torch.no_grad():
            if SO.n0 is None:
                _ema(num, den, x.detach().reshape(-1, x.shape[-1]).float(), mk)
            else:  # ParScale step: the running means are the plain stream's (its rows come first)
                _ema(num, den, x.detach().reshape(-1, x.shape[-1])[:SO.n0].float(), None if mk is None else mk[:SO.n0])
    y = x - (mu[0] if mk is None else mu[mk.long()].view(x.shape))
    return (y, mu) if give_mu else y  # give_mu: for a caller that takes the same means away again later


def _moe_whole(x, K, counts, ff, d, act, c2, b_up, mk_r):
    """Whether _MoE can take the rows of every expert of a layer at once (the kernels of _KMOE). They are written
    for the 4b run's case: squared ReLU, [MASK] positions in the batch, a mean and an offset per kind of position
    (or no offsets), two experts per token, no second centring. The widths must be ones whose rows torch adds up
    in the order the kernels use (longer than 128, a multiple of 4), and so must every expert's share of the rows
    (16 or more: for fewer, torch lays a row over more than one warp). While ZF is set (evaluation counts the zero
    activations) the experts go one at a time."""
    return (_MOE_K and MOE is not None and x.dtype == torch.float16 and act == 'relu2' and c2 is None and ZF is None
            and mk_r is not None and K == 2 and (b_up is None or b_up.dim() == 3)
            and ff % 4 == 0 and ff > 128 and d % 4 == 0 and d > 128
            and min((n for n in counts if n), default=0) >= 16)


def _ema_whole(num, den, H, mk_r, start, counts):
    """_ema for every expert of a layer: num (E, 2, D) / den (E, 2) are the running means, H (rows, D) the
    activations of the (token, expert) pairs grouped by expert (expert e's in rows start[e] .. start[e + 1]) and
    mk_r marks the rows whose token reads [MASK]. The sums over an expert's tokens stay one matmul per expert, as
    in _ema, so they are added in the same order; everything after them is done for all experts at once. An expert
    with no token, or a kind with none, keeps its mean."""
    with torch.autocast(DEV.type, enabled=False):
        W = torch.stack((~mk_r, mk_r)).float()  # (2, rows): which rows are of which kind
        sums, o0 = torch.zeros_like(num), 0
        for e, n in enumerate(counts):
            if n:
                torch.matmul(W[:, o0:o0 + n], H[o0:o0 + n], out=sums[e])
                o0 += n
        cm = torch.cat((mk_r.new_zeros(1, dtype=torch.long), mk_r.cumsum(0)))[start.long()]
        nm = (cm[1:] - cm[:-1]).float()  # every expert's [MASK] rows
        cnt = torch.stack((torch.tensor(counts, dtype=torch.float32, device=H.device) - nm, nm), 1)  # (E, 2)
        mean = sums / cnt.clamp(min=1)[..., None]
        step = (1 - SO.cb) * ((cnt > 0) & torch.isfinite(mean).all(-1)).float()
        num.add_((torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0) - num) * step[..., None])
        den.add_((1 - den) * step)


class _MoE(torch.autograd.Function):
    """All experts of one layer for x (N, d): out = sum over a token's experts of
    gate * (rms_norm(act(x @ Wu.T * s_up) - mean) @ Wd.T * s_dn), mean being the expert's running mean
    activation (ch / kh, see _center). Tokens are grouped by expert; each expert's
    two binary matrices are used and updated one after the other, and only the pre-activations and the
    expert outputs are kept for backward.

    There are two ways through the same arithmetic. One expert after the other in tensor operations: about 80
    small ones per expert on its few hundred rows, 128 experts a step, a fifth of the step. Or, where _moe_whole
    allows, the rows of every expert of the layer at once (MOE): the matmuls and the int8 updates stay one per
    expert and in the same order, and what lay between them is one kernel per layer each way. Those kernels give
    every entry the arithmetic it had and add rows up as torch does, so the two ways give the same bits."""

    @staticmethod
    def forward(ctx, x, top_p, top_i, s_up, s_dn, up, dn, act, ch, kh, mk, *opt):
        c2 = opt[0] if opt else None                 # (running means, counts) of the second centring, or None
        b_up = opt[1] if len(opt) > 1 else None      # (E, ff) offsets of the pre-activations, or None
        ctx.again = opt[2] if len(opt) > 2 else None  # an _Again: x is not kept but computed again for backward
        ya = opt[3] if len(opt) > 3 else None        # an _Again to fill: how the output can be computed again
        N, K = top_i.shape
        E = s_up.shape[0]
        flat = top_i.reshape(-1)
        order = torch.argsort(flat, stable=True)
        tok = torch.div(order, K, rounding_mode='floor')
        counts = torch.bincount(flat, minlength=E).tolist()
        gate = top_p.reshape(-1)[order].to(x.dtype)
        rows = flat[order]  # the expert of every (token, expert) pair, the pairs grouped by expert
        mk_r = None if mk is None else mk.index_select(0, tok)
        ctx.save_for_backward(*((top_p, s_up, s_dn) if ctx.again is not None else (x, top_p, s_up, s_dn)))
        ctx.nopt, ctx.has_b = len(opt), b_up is not None
        ctx.b_shape = tuple(b_up.shape) if b_up is not None else None
        ctx.whole = _moe_whole(x, K, counts, s_up.shape[1], s_dn.shape[1], act, c2, b_up, mk_r)
        if ctx.whole:
            start = torch.tensor(np.cumsum([0] + counts), dtype=torch.int32, device=x.device)
            xg = x.index_select(0, tok)  # what every pair's expert reads
            HP = torch.empty(N * K, s_up.shape[1], dtype=x.dtype, device=x.device)
            o0 = 0
            for e, n in enumerate(counts):
                if n:
                    torch.matmul(xg[o0:o0 + n], up.signs(e).t(), out=HP[o0:o0 + n])
                    o0 += n
            del xg
            mu = ch / kh.clamp(min=1e-8)[..., None]  # every expert's running means as they stood, (E, 2, ff)
            ex = rows.int()
            keep = SO.learn and SO.ema
            HN, R, H = MOE.fwd(HP, ex, mk_r, s_up.to(x.dtype), None if b_up is None else b_up.to(x.dtype), mu, keep)
            if keep and SO.n0 is None:
                _ema_whole(ch, kh, H, mk_r, start, counts)
            elif keep:  # ParScale step: the experts' means from the plain stream's pairs (still grouped by expert)
                s0 = tok < SO.n0
                c0 = torch.bincount(rows[s0], minlength=E).tolist()
                _ema_whole(ch, kh, H[s0], mk_r[s0], torch.tensor(np.cumsum([0] + c0), dtype=torch.int32, device=x.device),
                           c0)
            del H
            O = torch.empty(N * K, s_dn.shape[1], dtype=x.dtype, device=x.device)
            o0 = 0
            for e, n in enumerate(counts):
                if n:
                    torch.matmul(HN[o0:o0 + n], dn.signs(e).t(), out=O[o0:o0 + n])
                    o0 += n
            del HN
            O.mul_(s_dn.to(x.dtype).index_select(0, rows))
            pos = torch.empty(N * K, dtype=torch.int32, device=x.device)  # where each token's two pairs are
            pos[order] = torch.arange(N * K, dtype=torch.int32, device=x.device)
            ctx.misc = (order, tok, counts, gate, (HP, R, mu, pos), O, up, dn, act, ex, mk_r)
            if ya is not None:  # O, gate and pos are kept for the backward pass in any case
                ya.make = lambda: MOE.pairsum(O * gate[:, None], pos, N)
            return MOE.pairsum(O * gate[:, None], pos, N)
        bk = None
        if b_up is not None and b_up.dim() == 3:  # an offset per kind of position (Config.cov): 0 ordinary, 1 [MASK]
            bk = b_up.to(x.dtype).view(E * 2, -1)[rows * 2 if mk_r is None else rows * 2 + mk_r.long()]
        out = torch.zeros(N, s_dn.shape[1], device=x.device, dtype=x.dtype)
        hps, outs, o0 = [], [], 0
        for e, n in enumerate(counts):
            if n == 0:
                hps.append(None)
                outs.append(None)
                continue
            idx = tok[o0:o0 + n]
            hp = F.linear(x.index_select(0, idx), up.signs(e)) * s_up[e].to(x.dtype)
            mk_e = None if mk_r is None else mk_r[o0:o0 + n]
            if bk is not None:
                hp = hp + bk[o0:o0 + n]
            elif b_up is not None:
                hp = hp + b_up[e].to(x.dtype)
            h = _act(hp.float(), act)  # fp32: a squared activation can pass the half-precision range
            if ZF is not None:
                ZF.append((h == 0).float().mean())
            mu = ch[e] / kh[e].clamp(min=1e-8)[:, None]
            sel = None if SO.n0 is None else idx < SO.n0  # ParScale step: the means from the plain stream's tokens
            if SO.learn and SO.ema:
                _ema(ch[e], kh[e], h if sel is None else h[sel], mk_e if sel is None or mk_e is None else mk_e[sel])
            h = h - (mu[0] if mk_e is None else mu[mk_e.long()])
            r = torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6)
            hn, mu2 = h * r, None
            if c2 is not None:  # centred again, after the norm: what the down matrix reads has no common part
                mu2 = c2[0][e] / c2[1][e].clamp(min=1e-8)[:, None]
                if SO.learn and SO.ema:
                    _ema(c2[0][e], c2[1][e], hn if sel is None else hn[sel],
                         mk_e if sel is None or mk_e is None else mk_e[sel])
                hn = hn - (mu2[0] if mk_e is None else mu2[mk_e.long()])
            o = F.linear(hn.to(x.dtype), dn.signs(e)) * s_dn[e].to(x.dtype)
            out.index_add_(0, idx, o * gate[o0:o0 + n, None])
            hps.append((hp, r, mu, mk_e, mu2))
            outs.append(o)
            o0 += n
        ctx.misc = (order, tok, counts, gate, hps, outs, up, dn, act, rows.int(), mk_r)
        if ya is not None:  # one expert at a time: nothing to compute the sum from in one pass, so it is kept
            ya.t = out
        return out

    @staticmethod
    def backward(ctx, g):
        if ctx.again is not None:
            (top_p, s_up, s_dn), x = ctx.saved_tensors, ctx.again.get()
            ctx.again = None
        else:
            x, top_p, s_up, s_dn = ctx.saved_tensors
        order, tok, counts, gate, hps, outs, up, dn, act, rows, mk_r = ctx.misc
        ctx.misc = None  # the expert activations are freed when this pass ends, not a step later
        gs_up, gs_dn = torch.zeros_like(s_up), torch.zeros_like(s_dn)
        # one rescaling factor for every expert's down matrix (a gate is at most 1, so no entry of go is above
        # the largest of g) and, after the loop, one for every up matrix: two reads of the GPU per layer, not 32
        f_dn, c_dn = _wscale(float(torch.linalg.vector_norm(g, ord=float('inf'))))
        GE = g.index_select(0, tok)  # the output gradient of every (token, expert) pair, the pairs grouped by expert
        GO = GE * gate[:, None]
        seg = GOc = GHc = sums_up = None
        if SO.cov and not SO.frozen:  # every expert's rows are centred in one pass (Sign.centred), not expert by expert
            seg = (torch.tensor(np.cumsum([0] + counts), dtype=torch.int32, device=x.device), rows, tok)
            GOc = dn.centred(GO, 0, mk_r, seg)[0]
        if ctx.whole:
            (HP, R, mu, pos), O = hps, outs
            ggate = MOE.rowdot(GE, O)
            del GE
            S = GO * s_dn.to(GO.dtype).index_select(0, rows.long())  # what the down matrices' transposes read
            HN = None if SO.frozen else MOE.hn(HP, rows, mk_r, mu, R)  # what the down matrices read, for their gradient
            GR = torch.empty(order.numel(), up.shape[1], dtype=g.dtype, device=x.device)
            o0 = 0
            for e, n in enumerate(counts):
                if n == 0:
                    continue
                sl = slice(o0, o0 + n)
                o0 += n
                torch.matmul(S[sl], dn.signs(e), out=GR[sl])  # before this expert's down matrix is updated
                if not SO.frozen:
                    gs_dn[e] = _wupd(dn, GO[sl] if seg is None else GOc[sl], HN[sl], s_dn[e], e, f_dn, c_dn, mk_r[sl],
                                     seg is not None)
            del S, HN, GO, GOc
            GH, S = MOE.bwd(GR, HP, rows, mk_r, mu, R, s_up.to(g.dtype))  # every expert's ghp, and ghp * s_up
            del GR
            GX = torch.empty(order.numel(), x.shape[1], dtype=g.dtype, device=x.device)
            o0 = 0
            for e, n in enumerate(counts):
                if n:
                    torch.matmul(S[o0:o0 + n], up.signs(e), out=GX[o0:o0 + n])
                    o0 += n
            del S
            gx = MOE.pairsum(GX, pos, x.shape[0])
            del GX
            xg = None if SO.frozen else x.index_select(0, tok)
        else:
            gx = torch.zeros_like(x)
            ggate = torch.zeros(order.numel(), device=x.device)
            GH = torch.empty(order.numel(), up.shape[1], dtype=g.dtype, device=x.device)  # every expert's ghp
            xg, o0 = None, 0
            for e, n in enumerate(counts):
                if n == 0:
                    continue
                sl = slice(o0, o0 + n)
                o0 += n
                idx, (hp, r, mu, mk_e, mu2), o = tok[sl], hps[e], outs[e]
                ggate[sl] = (GE[sl].float() * o.float()).sum(-1)
                go = GO[sl]
                # down: o = (hn @ Wd.T) * s_dn[e], hn = rms_norm(act(hp) - mu)
                hn = (_act(hp.float(), act) - (mu[0] if mk_e is None else mu[mk_e.long()])) * r
                gh = ((go * s_dn[e].to(go.dtype)) @ dn.signs(e)).float()
                if not SO.frozen:
                    hr = hn if mu2 is None else hn - (mu2[0] if mk_e is None else mu2[mk_e.long()])  # what the matrix read
                    gs_dn[e] = _wupd(dn, go if seg is None else GOc[sl], hr.to(go.dtype), s_dn[e], e, f_dn, c_dn, mk_e,
                                     seg is not None)
                gh = (gh - hn * (gh * hn).mean(-1, keepdim=True)) * r  # back through the norm
                ghp = GH[sl]
                ghp.copy_(gh * _dact(hp.float(), act))
                # up: hp = (x[idx] @ Wu.T) * s_up[e]
                gx.index_add_(0, idx, (ghp * s_up[e].to(ghp.dtype)) @ up.signs(e))
        f_up, c_up = _wscale(float(torch.linalg.vector_norm(GH, ord=float('inf'))))
        if seg is not None:
            GHc, sums_up = up.centred(GH, 0, mk_r, seg)
        o0 = 0
        for e, n in enumerate(counts if not SO.frozen else ()):
            if n:
                sl = slice(o0, o0 + n)
                o0 += n
                gs_up[e] = _wupd(up, GH[sl] if seg is None else GHc[sl],
                                 x.index_select(0, tok[sl]) if xg is None else xg[sl], s_up[e], e, f_up, c_up,
                                 None if mk_r is None else mk_r[sl], seg is not None)
        gb_up = None
        if SO.frozen:  # --parscale-freeze-backbone: nothing here learns
            gs_up = gs_dn = None
        elif ctx.has_b and len(ctx.b_shape) == 3:  # an offset per kind: the sums over each expert's tokens of either kind
            if sums_up is None:
                kidx = rows.long() * 2 if mk_r is None else rows.long() * 2 + mk_r.long()
                sums_up = torch.zeros(ctx.b_shape[0] * 2, ctx.b_shape[2], device=x.device).index_add_(0, kidx, GH.float())
            gb_up = sums_up.view(ctx.b_shape)
        elif ctx.has_b:  # an offset's gradient is the pre-activation gradient summed over its expert's tokens
            gb_up, o0 = torch.zeros(ctx.b_shape, device=x.device), 0
            for e, n in enumerate(counts):
                if n:
                    gb_up[e] = GH[o0:o0 + n].sum(0, dtype=torch.float32)
                    o0 += n
        gp = torch.zeros_like(ggate)
        gp[order] = ggate
        return (gx, gp.view_as(top_p).to(top_p.dtype), None, gs_up, gs_dn, None, None, None, None, None, None,
                None, gb_up, None, None)[:11 + ctx.nopt]


_TCE_K = 16.0  # softmax errors are multiplied by this before the fp16 matmuls so small ones stay representable


def _ce_keep(ctx, h, n_vocab):
    """Whether _TiedCE keeps this step's scores for its backward pass (2 bytes per row per vocabulary entry,
    0.6 GB for the 4b preset): on the kernel path, when a backward pass will come, and while the GPU has room
    for them beside _CE_ROOM. Kept or computed again, backward sees the same numbers; keeping them saves one of
    the head's four passes over the vocabulary."""
    if not (_CE_KEEP and FUSED is not None and h.dtype == torch.float16 and ctx.needs_input_grad[0]):
        return False
    free, _ = torch.cuda.mem_get_info()
    spare = free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    return spare > 2 * h.shape[0] * n_vocab + _CE_ROOM


class _TiedCE(torch.autograd.Function):
    """Cross-entropy of h (N, d) against a tied binary embedding emb (n, R, d), n * R = vocabulary:
    logit_v = (h . sign(E_v)) * s_out[v] + b_out[v]. The vocabulary is scored one slice of R rows at
    a time, forward and again in backward, so the full logit matrix is never held; each slice of the
    embedding gets its int8 update as soon as its gradient exists. Returns the loss per row.

    Config.tiers (emb.tiers): the table is in order of frequency and its slices are grouped. The first group is a
    softmax of its own that every row is scored against; its last rows are entries, one for each later group. A
    row whose word is in a later group pays that group's entry in the first softmax plus the word's loss in a
    softmax over its group alone, and only such rows are scored against that group: p(word) = p(group) x
    p(word | group), so the loss is an exact log-likelihood as before. Why: the head was 170 ms of a 700 ms step,
    all of it matmul against 129,280 rows, while 16,157 words are 92.4% of this text and 31,000 of the words never
    occur in it. tgt holds table rows here, not ids (Net.nll converts)."""

    @staticmethod
    def forward(ctx, h, tgt, s_out, b_out, emb):
        n, R, _ = emb.shape
        if emb.tiers is None:
            groups = [(0, n, None, h, tgt)]
        else:  # one sort puts the rows of each later group together; their counts are the one wait for the GPU
            g = torch.bucketize(tgt, emb.edge, right=True)
            by, cnt = torch.argsort(g, stable=True), torch.bincount(g, minlength=len(emb.spans)).tolist()
            groups = [(0, emb.spans[0][1], None, h, torch.where(g > 0, emb.entry[(g - 1).clamp(min=0)], tgt))]
            o = cnt[0]
            for (c0, c1), m in zip(emb.spans[1:], cnt[1:]):
                if m:
                    groups.append((c0, c1, by[o:o + m], h[by[o:o + m]], tgt[by[o:o + m]]))
                o += m
        scored = sum((c1 - c0) * R * len(tk) for c0, c1, _, _, tk in groups) // max(1, h.shape[0])
        keep = _ce_keep(ctx, h, scored)  # each slice's scores, for backward (see _ce_keep)
        loss, parts = None, []
        for c0, c1, rows, hk, tk in groups:
            lse = torch.full((hk.shape[0],), float('-inf'), device=h.device)
            kept = [] if keep else None
            for c in range(c0, c1):
                sl = slice(c * R, (c + 1) * R)
                z = F.linear(hk, emb.signs(c))
                lse = torch.logaddexp(lse, FUSED.lse(z, s_out[sl], b_out[sl]) if FUSED is not None
                                      else torch.logsumexp(z.float().mul_(s_out[sl]).add_(b_out[sl]), 1))
                if kept is not None:
                    kept.append(z)
            zt = (hk.float() * torch.where(emb.rows(tk) >= 0, 1.0, -1.0)).sum(1)  # the target's own score
            nll = lse - (zt * s_out[tk] + b_out[tk])
            loss = nll if rows is None else loss.index_add(0, rows, nll)
            parts.append((c0, c1, rows, tk, lse, kept))
        ctx.emb, ctx.parts = emb, parts
        ctx.save_for_backward(h, s_out, b_out)
        return loss

    @staticmethod
    def backward(ctx, g):
        h, s_out, b_out = ctx.saved_tensors
        emb = ctx.emb
        R = emb.shape[1]
        parts, ctx.parts = ctx.parts, None
        gmax = g.abs().max().clamp_(min=1e-30)
        w = (g / gmax)[:, None]  # relative weight of each row, at most 1
        gh = torch.zeros(h.shape, device=h.device)
        gs, gb = torch.zeros_like(s_out), torch.zeros_like(b_out)
        if FUSED is not None:  # the same arithmetic, one kernel per slice and no fp32 copy of the logits
            w = w[:, 0].contiguous()
            hw = h.float() * w[:, None]  # each row of h times its weight
            floor = max(float(hw.square().mean().sqrt()) * _TCE_K, 1e-20)  # the rms one occurrence leaves in a row
            hw = hw.to(h.dtype)
            bad = SO.bad_t.clone()
            for c0, c1, rows, tk, lse, kept in parts:  # one softmax each: all rows, or the rows of one later group
                hk, wk, hwk, ghk = (h, w, hw, gh) if rows is None else \
                    (h[rows], w[rows], hw[rows], gh.new_zeros(len(rows), h.shape[1]))
                for j, c in enumerate(range(c0, c1)):
                    sl = slice(c * R, (c + 1) * R)
                    W = emb.signs(c)
                    if kept is None:
                        z = F.linear(hk, W)
                    else:  # the forward pass's scores of this slice: what the line above computes, bit for bit
                        z, kept[j] = kept[j], None
                    FUSED.ce_grad(z, s_out[sl], b_out[sl], lse, wk, (tk - c * R).int(), gs[sl], gb[sl], _TCE_K)
                    ghk.add_((z @ (W * s_out[sl].to(W.dtype)[:, None])).float())  # z now holds (p - onehot) * K
                    del W
                    if not SO.frozen:  # --parscale-freeze-backbone: the token table learns nothing
                        emb.update(z.t() @ hwk, s_out[sl] * (gmax / _TCE_K), c, floor)
                    del z
                if rows is not None:
                    gh.index_add_(0, rows, ghk)
            SO.head_bad_t += SO.bad_t - bad  # an overflow here does not depend on the loss scale
            SO.bad_t.copy_(bad)
            return (gh * (w[:, None] * (gmax / _TCE_K))).to(h.dtype), None, gs * gmax, gb * gmax, None
        floor = max(float((h.float() * w).square().mean().sqrt()) * _TCE_K, 1e-20)
        for c0, c1, rows, tk, lse, _ in parts:
            hk, wk, ghk = (h, w, gh) if rows is None else (h[rows], w[rows], gh.new_zeros(len(rows), h.shape[1]))
            for c in range(c0, c1):
                sl = slice(c * R, (c + 1) * R)
                W = emb.signs(c)
                z = F.linear(hk, W).float()
                p = (z * s_out[sl]).add_(b_out[sl]).sub_(lse[:, None]).exp_()
                off = tk - c * R
                p.scatter_add_(1, off.clamp(0, R - 1)[:, None], -((off >= 0) & (off < R)).float()[:, None])
                p.mul_(wk)  # (softmax - onehot) * row weight
                gb[sl] = p.sum(0)
                gs[sl] = (p * z).sum(0)
                del z
                dl = p.mul_(_TCE_K).to(h.dtype)
                del p
                ghk.add_(((dl * s_out[sl].to(dl.dtype)) @ W).float())
                del W
                if not SO.frozen:  # --parscale-freeze-backbone: the token table learns nothing
                    emb.update(dl.t() @ hk, s_out[sl] * (gmax / _TCE_K), c, floor)
                del dl
            if rows is not None:
                gh.index_add_(0, rows, ghk)
        return (gh * (gmax / _TCE_K)).to(h.dtype), None, gs * gmax, gb * gmax, None


_ENT = 10.0  # Config.tiers: the scale of a group's entry is _ENT times its parameter (see Net.s_ent)


def rope_tables(pos, hd):
    """pos (B, L) text positions -> cos, sin (B, 1, L, hd / 2)."""
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=pos.device).float() / hd))
    f = pos.float()[..., None] * inv
    return f.cos()[:, None], f.sin()[:, None]


def alibi(pos, mask, nh):
    """mask (B, 1, L, L), who may attend to whom -> an additive score bias (B, nh, L, L): head h pays
    slope_h for every token of distance, slopes 2^-0.5 .. 2^-8 for 16 heads (ALiBi, arXiv 2108.12409),
    and -inf where the mask forbids. The steep heads see only their neighbourhood from the first step."""
    dist = (pos[:, :, None] - pos[:, None, :]).abs().to(HALF)
    slopes = torch.pow(2.0, -8.0 * torch.arange(1, nh + 1, device=pos.device) / nh).to(HALF)
    return (dist[:, None] * -slopes[None, :, None, None]).masked_fill_(~mask, float('-inf'))


class _Rope(torch.autograd.Function):
    """rope() below in one kernel pass each way (rope_fwd, rope_bwd in _KSRC). Every product and sum is rounded
    on its own, as the tensor operations round them, so values and gradients are the same bit for bit."""

    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        return FUSED.rope(x, cos, sin, False)

    @staticmethod
    def backward(ctx, g):
        cos, sin = ctx.saved_tensors
        return FUSED.rope(g.to(torch.float16).contiguous(), cos, sin, True), None, None


def rope(x, cs):
    if cs is None:  # --pos alibi: position comes from the distance bias alone
        return x
    cos, sin = cs
    if _ROPE_K and FUSED is not None and x.dtype == torch.float16 and x.dim() == 4 and cos.dim() == 4:
        return _Rope.apply(x.contiguous(), cos, sin)
    x1, x2 = x[..., ::2].float(), x[..., 1::2].float()
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2).to(x.dtype)


def attend(q, k, v, mask, pre=None):
    """Attention on the plain path (the lean2 path is _AttnOut). pre: None, or a ParScale step's view of this layer
    (_PSLayer): q, k, v then hold P streams of B rows one after the other, and a stream from pre.s0 on attends to its
    own n prefix keys and values in front of its own keys: no rotation, no distance, visible to every query and
    outside every block of the noisy stream (open mask columns), each prefix column scored with the stream's learned
    logit offset for this layer and head (revision 2, D3). One call per stream, so the mask is never repeated P
    times; a stream before pre.s0 makes exactly the plain call. The CPU worker replaces this function while a
    KV-cached forward runs. While ps_init_data runs (_PS_CAPTURE a list) every call hands over its keys and values."""
    if _PS_CAPTURE is not None:
        _PS_CAPTURE.append((k.detach(), v.detach()))
    if pre is None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    B = q.shape[0] // pre.P
    out = []
    for p in range(pre.P):
        sl = slice(p * B, (p + 1) * B)
        if p < pre.s0:
            out.append(F.scaled_dot_product_attention(q[sl], k[sl], v[sl], attn_mask=mask))
        else:
            kx, vx, bx = _ps_kv(k[sl], v[sl], *pre.kv_all(), p - pre.s0, mask, pre.mask(mask))
            out.append(F.scaled_dot_product_attention(q[sl], kx, vx, attn_mask=bx))
    return torch.cat(out)


def _ps_bias(bx, po, n, dtype):
    """bx (B, 1 or nh, Lq, n + Lk): a mask with n open prefix columns in front (bool, or additive with 0 there);
    po (nh,) or (S, nh): the prefix logit offset per head of one stream, or of S streams (the worker's KV cache: one
    row per stream) -> the additive bias (max(B, S), nh, Lq, n + Lk) with po in the n prefix columns. dtype: the
    bias's when bx is bool (D3)."""
    add = bx if bx.dtype != torch.bool else \
        torch.zeros(bx.shape, dtype=dtype, device=bx.device).masked_fill_(~bx, float('-inf'))
    po = po.reshape(-1, po.shape[-1])
    Bn, nh, Lq = max(add.shape[0], po.shape[0]), po.shape[1], add.shape[2]
    return torch.cat((po.to(add.dtype)[:, :, None, None].expand(Bn, nh, Lq, n),
                      add[..., n:].expand(Bn, nh, Lq, add.shape[3] - n)), -1)


def _ps_kv(k, v, kp, vp, po, j, mask, bx):
    """Keys, values and mask of one ParScale stream for an attention call: its own (j < 0: no prefix), or prefix
    slot j's keys and values (nh, n, hd) in front of its own (B, nh, L, hd), with bx (the mask with open prefix
    columns in front) made into a bias that carries slot j's logit offsets po[j] (nh,) in those columns (D3)."""
    if j < 0:
        return k, v, mask
    B = k.shape[0]
    return (torch.cat((kp[j].to(k.dtype)[None].expand(B, -1, -1, -1), k), 2),
            torch.cat((vp[j].to(v.dtype)[None].expand(B, -1, -1, -1), v), 2),
            _ps_bias(bx, po[j], kp.shape[2], k.dtype))


class _Again:
    """A tensor that is not kept between a step's forward and backward pass: make() computes it again from tensors
    that are kept anyway. The same kernel on the same numbers gives the same bits, so the backward pass sees what
    it saw when the tensor was kept. readers: how many backward passes will ask (get); the tensor is held from
    the first one that asks to the last, and after that this object holds nothing."""

    def __init__(self, make=None, readers=1):
        self.make, self.left, self.t = make, readers, None

    def get(self):
        t = self.t if self.t is not None else self.make()
        self.left -= 1
        self.t, self.make = (None, None) if self.left <= 0 else (t, self.make)
        return t


class _NormLin(torch.autograd.Function):
    """_RMSNorm, then _BinLin on what it returns, as one function: y = (rms_norm(x) * w) @ sign(P).T * scale.
    As two functions the norm keeps x and _BinLin keeps the normed x; here only x is kept, and backward runs the
    norm's kernel on it again for the matrix's gradient. Kernel path only (see _lean2)."""

    @staticmethod
    def forward(ctx, x, w, scale, sign, e, *opt):
        xc, wc = x.contiguous(), w.detach().float().reshape(-1).contiguous()
        y, r = FUSED.norm_fwd(xc.view(-1, x.shape[-1]), wc)
        z = F.linear(y, sign.signs(e))
        ctx.sign, ctx.e, ctx.keep = sign, e, (xc, wc, r)
        ctx.mk, ctx.nopt = (opt[0] if opt else None), len(opt)  # mk: the [MASK] positions, for Sign.centred
        ctx.save_for_backward(scale)
        return z * scale.to(z.dtype)

    @staticmethod
    def backward(ctx, gy):
        (xc, wc, r), ctx.keep = ctx.keep, None
        scale, = ctx.saved_tensors
        n = xc.shape[-1]
        if SO.frozen:  # --parscale-freeze-backbone: the gradient passes on, the matrix learns nothing
            gx, gs = (gy * scale.to(gy.dtype)) @ ctx.sign.signs(ctx.e), None
        else:
            y = FUSED.norm_fwd(xc.view(-1, n), wc)[0]  # what the matrix read
            gx = (gy * scale.to(gy.dtype)) @ ctx.sign.signs(ctx.e)
            f, c = _wscale(float(torch.linalg.vector_norm(gy, ord=float('inf'))))
            gs = _wupd(ctx.sign, gy, y, scale, ctx.e, f, c, ctx.mk)
            del y
        gx, gw = FUSED.norm_bwd(xc.view(-1, n), gx.contiguous().view(-1, n), wc, r, ctx.needs_input_grad[1])
        return (gx.view(xc.shape), gw, gs, None, None) + (None,) * ctx.nopt


class _AttnOut(torch.autograd.Function):
    """A layer's attention, from q, k, v as the q/k/v matrix wrote them to what the output matrix writes, as one
    function: the per-head norms of q and k and the bound on v, the rotary positions, attention, the centring and
    the bound of what it returns, and the binary output matrix.

    Why one function: each of those keeps its input for its backward pass, and the next one keeps the same numbers
    once more as its own input. That is nine tensors of the stream's size per layer (q, k, v; q and k normed and
    rotated, v bounded, attention's output; that output centred; and bounded), and what a step keeps per position
    is what limits the tokens a step can hold on a 6 GB card. Here q, k, v and attention's output are kept, with
    one number per row for each norm and bound, and backward runs the same kernels on them again for the rest.
    Every value is the one the separate functions had, so a step comes out the same. (Attention's gradient for the
    queries differs in its last bits from one call to the next at this model's sizes, in either form; that is the
    library's kernel.)

    z (3, B, nh, L, hd): q, k, v. bias (B, nh, L, L): alibi(). num, den: the running means of what attention
    returns (_center). Returns (B * L, D)."""

    @staticmethod
    def forward(ctx, z, cos, sin, bias, scale, sign, num, den, mk):
        _, B, nh, L, hd = z.shape
        wc = _UNIT.detach().float().reshape(-1).contiguous()
        q, k, v = z[0].contiguous(), z[1].contiguous(), z[2].contiguous()
        qn, rq = FUSED.norm_fwd(q.view(-1, hd), wc)
        qr = FUSED.rope(qn.view(B, nh, L, hd), cos, sin, False)
        del qn
        kn, rk = FUSED.norm_fwd(k.view(-1, hd), wc)
        kr = FUSED.rope(kn.view(B, nh, L, hd), cos, sin, False)
        del kn
        vc, rv, _ = FUSED.cap_fwd(v.view(-1, hd), 1.0)
        out, lse, seed, off = torch.ops.aten._scaled_dot_product_efficient_attention(
            qr, kr, vc.view(B, nh, L, hd), bias, True, 0.0, False, scale=None)
        del qr, kr, vc
        yc, mu = _center(out.transpose(1, 2).reshape(B * L, nh * hd), num, den, mk, True)
        yb, rc, _ = FUSED.cap_fwd(yc, 1.0)
        del yc
        o = F.linear(yb, sign.signs(0))
        ctx.sign, ctx.mk = sign, mk
        ctx.keep = (q, k, v, rq, rk, rv, out, lse, seed, off, mu, rc, cos, sin, bias)
        ctx.save_for_backward(scale)
        return o * scale.to(o.dtype)

    @staticmethod
    def backward(ctx, gy):
        (q, k, v, rq, rk, rv, out, lse, seed, off, mu, rc, cos, sin, bias), ctx.keep = ctx.keep, None
        scale, = ctx.saved_tensors
        sign, mk = ctx.sign, ctx.mk
        B, nh, L, hd = q.shape
        wc = _UNIT.detach().float().reshape(-1).contiguous()
        # the output matrix, as _BinLin: what it read is the bounded, centred output of attention, made again
        yt = out.transpose(1, 2).reshape(B * L, nh * hd)
        yc = yt - (mu[0] if mk is None else mu[mk.long()].view(yt.shape))
        del yt
        if SO.frozen:  # --parscale-freeze-backbone
            g, gs = (gy * scale.to(gy.dtype)) @ sign.signs(0), None
        else:
            yb = FUSED.cap_fwd(yc, 1.0)[0]
            g = (gy * scale.to(gy.dtype)) @ sign.signs(0)
            f, c = _wscale(float(torch.linalg.vector_norm(gy, ord=float('inf'))))
            gs = _wupd(sign, gy, yb, scale, 0, f, c, mk)
            del yb
        g = FUSED.cap_bwd(yc, g.contiguous(), rc, 1.0)  # the bound; the centring passes its gradient on as it is
        del yc
        g = g.view(B, L, nh, hd).transpose(1, 2)
        # attention: what it read, made again
        qr = FUSED.rope(FUSED.norm_fwd(q.view(-1, hd), wc)[0].view(B, nh, L, hd), cos, sin, False)
        kr = FUSED.rope(FUSED.norm_fwd(k.view(-1, hd), wc)[0].view(B, nh, L, hd), cos, sin, False)
        vc = FUSED.cap_fwd(v.view(-1, hd), 1.0)[0].view(B, nh, L, hd)
        gq, gk, gv, _ = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
            g, qr, kr, vc, bias, out, lse, seed, off, 0.0, [True, True, True, False], False, scale=None)
        del qr, kr, vc, g, out
        gq = FUSED.rope(gq.to(torch.float16).contiguous(), cos, sin, True)
        gq = FUSED.norm_bwd(q.view(-1, hd), gq.view(-1, hd), wc, rq, False)[0]
        gk = FUSED.rope(gk.to(torch.float16).contiguous(), cos, sin, True)
        gk = FUSED.norm_bwd(k.view(-1, hd), gk.view(-1, hd), wc, rk, False)[0]
        gv = FUSED.cap_bwd(v.view(-1, hd), gv.contiguous().view(-1, hd), rv, 1.0)
        gz = torch.stack((gq.view(B, nh, L, hd), gk.view(B, nh, L, hd), gv.view(B, nh, L, hd)))
        return gz, None, None, None, gs, None, None, None, None


def _eff_fwd(q, k, v, bias):
    return torch.ops.aten._scaled_dot_product_efficient_attention(q, k, v, bias, True, 0.0, False, scale=None)


def _eff_bwd(g, q, k, v, bias, out, lse, seed, off, need_bias=False):
    """need_bias: the bias's gradient too (a prefixed stream: summed over its prefix columns it is the gradient of
    the stream's logit offsets, D3)."""
    return torch.ops.aten._scaled_dot_product_efficient_attention_backward(
        g, q, k, v, bias, out, lse, seed, off, 0.0, [True, True, True, bool(need_bias)], False, scale=None)


class _AttnOutP(torch.autograd.Function):
    """_AttnOut for a ParScale step (lean2 path). z (3, P * B, nh, L, hd) holds q, k, v of the P streams,
    stream-major. The norms, rotary positions and bounds, the centring and bound of what attention returns and the
    output matrix are _AttnOut's, on all P * B rows. Attention is _AttnOut's efficient-attention call made one
    stream at a time on that stream's B rows: a stream before s0 on its own keys with the unchanged bias
    (B, nh, L, L); a later stream on [its n prefix keys, its own keys] with the bias [po .. po, bias]
    (B, nh, L, n + L), made for that stream's call alone (_ps_kv), so a bias P times the size never exists. kp, vp
    (P - s0, nh, n, hd): the bounded prefixes in full precision; po (P - s0, nh): their logit offsets (revision 2,
    D3). The prefixes' gradients are the prefix columns of the kernel's key and value gradients summed over the
    rows, the offsets' the kernel's bias gradient summed over the prefix columns. (GPU only: not exercised on a CPU.)"""

    @staticmethod
    def forward(ctx, z, cos, sin, bias, bx, scale, sign, num, den, mk, kp, vp, po, s0):
        _, PB, nh, L, hd = z.shape
        P = kp.shape[0] + s0
        B = PB // P
        wc = _UNIT.detach().float().reshape(-1).contiguous()
        q, k, v = z[0].contiguous(), z[1].contiguous(), z[2].contiguous()
        qn, rq = FUSED.norm_fwd(q.view(-1, hd), wc)
        qr = FUSED.rope(qn.view(PB, nh, L, hd), cos, sin, False)
        del qn
        kn, rk = FUSED.norm_fwd(k.view(-1, hd), wc)
        kr = FUSED.rope(kn.view(PB, nh, L, hd), cos, sin, False)
        del kn
        vc, rv, _ = FUSED.cap_fwd(v.view(-1, hd), 1.0)
        vc = vc.view(PB, nh, L, hd)
        out, lses, seeds = qr.new_empty(PB, L, nh, hd).transpose(1, 2), [], []  # the layout the kernel returns
        for p in range(P):
            sl = slice(p * B, (p + 1) * B)
            kx, vx, bb = _ps_kv(kr[sl], vc[sl], kp, vp, po, p - s0, bias, bx)
            o, ls, sd, of = _eff_fwd(qr[sl], kx, vx, bb)
            out[sl] = o
            lses.append(ls)
            seeds.append((sd, of))
            del kx, vx, bb, o
        del qr, kr, vc
        yc, mu = _center(out.transpose(1, 2).reshape(PB * L, nh * hd), num, den, mk, True)
        yb, rc, _ = FUSED.cap_fwd(yc, 1.0)
        del yc
        o = F.linear(yb, sign.signs(0))
        ctx.sign, ctx.mk, ctx.s0, ctx.P = sign, mk, s0, P
        ctx.keep = (q, k, v, rq, rk, rv, out, lses, seeds, mu, rc, cos, sin, bias, bx, kp.detach(), vp.detach(),
                    po.detach())
        ctx.save_for_backward(scale)
        return o * scale.to(o.dtype)

    @staticmethod
    def backward(ctx, gy):
        (q, k, v, rq, rk, rv, out, lses, seeds, mu, rc, cos, sin, bias, bx, kp, vp, po), ctx.keep = ctx.keep, None
        scale, = ctx.saved_tensors
        sign, mk, s0, P = ctx.sign, ctx.mk, ctx.s0, ctx.P
        PB, nh, L, hd = q.shape
        B, n = PB // P, kp.shape[2]
        wc = _UNIT.detach().float().reshape(-1).contiguous()
        yt = out.transpose(1, 2).reshape(PB * L, nh * hd)
        yc = yt - (mu[0] if mk is None else mu[mk.long()].view(yt.shape))
        del yt
        if SO.frozen:  # --parscale-freeze-backbone
            g, gs = (gy * scale.to(gy.dtype)) @ sign.signs(0), None
        else:
            yb = FUSED.cap_fwd(yc, 1.0)[0]
            g = (gy * scale.to(gy.dtype)) @ sign.signs(0)
            f, c = _wscale(float(torch.linalg.vector_norm(gy, ord=float('inf'))))
            gs = _wupd(sign, gy, yb, scale, 0, f, c, mk)
            del yb
        g = FUSED.cap_bwd(yc, g.contiguous(), rc, 1.0)  # the bound; the centring passes its gradient on as it is
        del yc
        g = g.view(PB, L, nh, hd).transpose(1, 2)
        qr = FUSED.rope(FUSED.norm_fwd(q.view(-1, hd), wc)[0].view(PB, nh, L, hd), cos, sin, False)
        kr = FUSED.rope(FUSED.norm_fwd(k.view(-1, hd), wc)[0].view(PB, nh, L, hd), cos, sin, False)
        vc = FUSED.cap_fwd(v.view(-1, hd), 1.0)[0].view(PB, nh, L, hd)
        gq, gk, gv = torch.empty_like(qr), torch.empty_like(kr), torch.empty_like(vc)
        dkp, dvp, dpo = torch.zeros_like(kp), torch.zeros_like(vp), torch.zeros_like(po)
        for p in range(P):
            sl = slice(p * B, (p + 1) * B)
            kx, vx, bb = _ps_kv(kr[sl], vc[sl], kp, vp, po, p - s0, bias, bx)
            a, b_, c_, gb = _eff_bwd(g[sl], qr[sl], kx, vx, bb, out[sl], lses[p], *seeds[p], p >= s0)
            gq[sl] = a
            if p < s0:
                gk[sl], gv[sl] = b_, c_
            else:
                gk[sl], gv[sl] = b_[:, :, n:], c_[:, :, n:]
                dkp[p - s0] = b_[:, :, :n].float().sum(0)
                dvp[p - s0] = c_[:, :, :n].float().sum(0)
                dpo[p - s0] = gb[..., :n].float().sum((0, 2, 3))
            del kx, vx, bb, a, b_, c_, gb
        del qr, kr, vc, g, out
        gq = FUSED.rope(gq.to(q.dtype).contiguous(), cos, sin, True)
        gq = FUSED.norm_bwd(q.view(-1, hd), gq.view(-1, hd), wc, rq, False)[0]
        gk = FUSED.rope(gk.to(k.dtype).contiguous(), cos, sin, True)
        gk = FUSED.norm_bwd(k.view(-1, hd), gk.view(-1, hd), wc, rk, False)[0]
        gv = FUSED.cap_bwd(v.view(-1, hd), gv.contiguous().view(-1, hd), rv, 1.0)
        gz = torch.stack((gq.view(PB, nh, L, hd), gk.view(PB, nh, L, hd), gv.view(PB, nh, L, hd)))
        return gz, None, None, None, None, gs, None, None, None, None, dkp, dvp, dpo, None


def _lean2(x, cfg, cs, mask, pre=None):
    """Whether a layer takes _NormLin, _AttnOut and _Again for this batch: a training pass of BINTF_LEAN2 positions
    or more, on the kernels, in the configuration those were written for (the 4b run's: bounded branches, rotary
    positions and a distance bias, experts, no second centring) and a length attention takes without padding."""
    B, L, _ = x.shape
    if pre is not None and (not _PS_LEAN2 or pre.n % 16):  # ParScale step: _AttnOutP, for prefixes of 16s
        return False
    Bm = B if pre is None else B // pre.P  # a ParScale step's mask is one stream's, shared by its P streams
    return (0 <= _LEAN2 <= B * L and _LEAN and _ROPE_K and FUSED is not None and x.dtype == torch.float16
            and torch.is_grad_enabled() and bool(cfg.cap) and not cfg.centre2 and cfg.n_exp > 0 and cs is not None
            and mask.dtype == torch.float16 and tuple(mask.shape) == (Bm, cfg.n_head, L, L) and L % 16 == 0)


class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, E = cfg.d, max(1, cfg.n_exp)
        self.n1, self.n2 = nn.Parameter(torch.ones(d)), nn.Parameter(torch.ones(d))
        self.s_qkv, self.s_out = nn.Parameter(torch.full((3 * d,), 0.02)), nn.Parameter(torch.full((d,), 0.02))
        self.s_up, self.s_dn = nn.Parameter(torch.full((E, cfg.ff), 0.02)), nn.Parameter(torch.full((E, d), 0.02))
        if cfg.n_exp:
            self.router = nn.Linear(d, cfg.n_exp, bias=False)
        if cfg.bias:  # see _Bias. Values get none: theirs would be one vector at every position, removed after attention
            g = (2,) if cfg.cov else ()  # Config.cov: an offset for ordinary positions and one for [MASK] positions
            self.b_qk, self.b_up = nn.Parameter(torch.zeros(*g, 2 * d)), nn.Parameter(torch.zeros(E, *g, cfg.ff))
        self.qkv, self.out = Sign(1, 3 * d, d), Sign(1, d, d)
        self.up, self.dn = Sign(E, cfg.ff, d), Sign(E, d, cfg.ff)
        for name, shape in (('c1', (2, d)), ('cy', (2, d)), ('c2', (2, d)), ('ch', (E, 2, cfg.ff))):  # see _center
            self.register_buffer(name, torch.zeros(shape))
            self.register_buffer('k' + name[1:], torch.zeros(shape[:-1]))
            if cfg.centre2:  # the second centring's running means: d1/j1, dy/jy, d2/j2, dh/jh
                self.register_buffer('d' + name[1:], torch.zeros(shape))
                self.register_buffer('j' + name[1:], torch.zeros(shape[:-1]))

    def forward(self, x, cs, mask, cfg, mk=None, first=False, pre=None):
        B, L, D = x.shape
        nh = cfg.n_head
        # pre: a ParScale step's view of this layer (x holds P streams, stream-major; see ParAdapter), else None
        lean2 = _lean2(x, cfg, cs, mask, pre)  # keep less for the backward pass and compute it again there (see _AttnOut)
        # first: the net's input. Every [MASK] position holds the same vector there, so a mean of their own
        # would leave exactly zero for the norm to divide by; they share the mean of all positions instead.
        if lean2:
            z = _NormLin.apply(_center(x, self.c1, self.k1, None if first else mk), self.n1, self.s_qkv, self.qkv, 0,
                               None if first else mk)
        else:
            a = _RMSNorm.apply(_center(x, self.c1, self.k1, None if first else mk), self.n1)
            if cfg.centre2:
                a = _center(a, self.d1, self.j1, None if first else mk)
            a = a.reshape(B * L, D)
            z = _BinLin.apply(a, self.s_qkv, self.qkv, 0, None if first else mk)
        if cfg.bias:
            z = (_Bias2.apply(z, self.b_qk, mk) if cfg.cov
                 else _Bias.apply(z, torch.cat((self.b_qk, self.b_qk.new_zeros(D)))))
        if SO.gcap_qkv > 0 and SO.learn and z.requires_grad:  # backward: what attention hands this matrix is bounded
            z = _GCap.apply(z, SO.gcap_qkv, 2)
        if lean2 and pre is not None:  # ParScale step: _AttnOut stream by stream, each stream with its own prefix
            y = _AttnOutP.apply(z.view(B, L, 3, nh, D // nh).permute(2, 0, 3, 1, 4), cs[0], cs[1], mask, pre.mask(mask),
                                self.s_out, self.out, self.cy, self.ky, mk, *pre.kv_all(), pre.s0)
        elif lean2:
            y = _AttnOut.apply(z.view(B, L, 3, nh, D // nh).permute(2, 0, 3, 1, 4), cs[0], cs[1], mask, self.s_out,
                               self.out, self.cy, self.ky, mk)
        else:
            q, k, v = z.view(B, L, 3, nh, D // nh).permute(2, 0, 3, 1, 4)
            q, k = _RMSNorm.apply(q, _UNIT), _RMSNorm.apply(k, _UNIT)  # scores are bounded by sqrt(head_dim)
            if cfg.cap:  # a value is at most rms 1 in each head, so what attention returns is too, before any mean is taken
                v = _RMSCap.apply(v)
            y = attend(rope(q, cs), rope(k, cs), v, mask, pre)
            y = _RMSCap.apply(_center(y.transpose(1, 2).reshape(B * L, D), self.cy, self.ky, mk))
            if cfg.centre2:
                y = _center(y, self.dy, self.jy, mk)
            y = _BinLin.apply(y, self.s_out, self.out, 0, mk)
        x = x + (_RMSCap.apply(y, cfg.cap, True) if cfg.cap else y).view(B, L, D)
        if SO.gcap_mid > 0 and SO.learn and x.requires_grad:  # backward: what the expert branch hands back is bounded
            x = _GCap.apply(x, SO.gcap_mid, 1)
        xc = _center(x, self.c2, self.k2, mk)
        m = _RMSNorm.apply(xc, self.n2)
        if cfg.centre2:
            m = _center(m, self.d2, self.j2, mk)
        m = m.reshape(B * L, D)
        again = ()
        if lean2:  # the normed stream (read by the router and by the experts) and the experts' summed output
            wc = self.n2.detach().float().reshape(-1).contiguous()
            again = (_Again(lambda: FUSED.norm_fwd(xc.view(-1, D), wc)[0], 2), _Again())
        if cfg.n_exp:
            probs = _RouterLin.apply(m, self.router.weight, *again[:1]).float().softmax(-1)
            top_p, top_i = probs.topk(cfg.topk, dim=-1)
            top_p = top_p / top_p.sum(-1, keepdim=True)
            # Switch-style load balancing: share of first choices times mean router probability
            load = F.one_hot(top_i[:, 0], cfg.n_exp).float().mean(0)
            aux = cfg.n_exp * (load * probs.mean(0)).sum()
        else:
            top_p = torch.ones(B * L, 1, device=x.device)
            top_i = torch.zeros(B * L, 1, dtype=torch.long, device=x.device)
            aux = torch.zeros((), device=x.device)
        y = _MoE.apply(m, top_p, top_i, self.s_up, self.s_dn, self.up, self.dn, cfg.act, self.ch, self.kh, mk,
                       (self.dh, self.jh) if cfg.centre2 else None, self.b_up if cfg.bias else None, *again)
        return x + (_RMSCap.apply(y, cfg.cap, True, *again[1:]) if cfg.cap else y).view(B, L, D), aux


def _plen(g):
    """The length of every position's gradient (the last axis), in full precision. A row wider than the stream (q, k
    and v side by side) is squared a stream's width at a time, so that no full-precision copy of it is made."""
    D = g.shape[-1]
    if D <= 2048:
        if _GCAP_NORM and g.is_cuda and g.dtype == torch.float16:
            return torch.linalg.vector_norm(g, dim=-1, dtype=torch.float32)
        return torch.linalg.vector_norm(g.float(), dim=-1)
    n2 = g[..., :2048].float().square().sum(-1)
    for i in range(2048, D, 2048):
        n2 += g[..., i:i + 2048].float().square().sum(-1)
    return n2.sqrt_()


class _GCap(torch.autograd.Function):
    """The residual stream between two layers, unchanged on the way forward. On the way back no position's
    gradient is longer than c times the median position's: a longer one is scaled down to that length and keeps
    its direction. A position that many others attend to collects the gradients of all of them, a layer further
    down can do the same again with what it was handed, and after a few layers one position in a rare batch holds
    numbers that half precision cannot (65,504): see /workspace/b4/gcap/make_gcap.py for the measurements.
    Lengths are taken in full precision (the square of a half-precision entry may not fit one). An entry that has
    already overflowed is put at the largest number half precision has before its position is scaled back, so the
    step is taken instead of skipped.

    where: 0 between two layers (--gcap). 1 inside a layer, between its attention branch and its expert branch
    (--gcap-mid): the norm in front of the experts divides by the rms of the centred stream, which at a [MASK]
    position of the bottom layer is a quarter to a tenth of anyone else's, and a +-1 row that lines up with its
    input answers sqrt(d) times as strongly through an activation whose slope is that answer; what comes back from
    the bottom layer's expert branch was 77-405 medians long at its longest position in every batch looked at,
    with --gcap holding what went in to 16. 2 on what attention hands its q/k/v matrix, a row of 3 d numbers per
    position (--gcap-qkv): a token that the [MASK] positions of its block attend to collects their gradients.
    See /workspace/b4/gcap2/make_gcap2.py for the measurements."""

    @staticmethod
    def forward(ctx, x, c, where):
        ctx.c, ctx.where = c, where
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        n = _plen(g)
        if not bool(torch.isfinite(n).all()):
            g = torch.nan_to_num(g, nan=0.0, posinf=65504.0, neginf=-65504.0)
            n = _plen(g)
        # ParScale repeats rows stream-major; the cap threshold is defined by stream 0 only.
        base_n = n[:SO.n0 // n.shape[-1]] if _RV_FIXG and SO.n0 is not None else n
        lim = base_n.flatten().median() * ctx.c
        over = n > lim
        t = (SO.gc_t, SO.gm_t, SO.gz_t)[ctx.where]
        t += torch.stack((over.float().mean(), n.new_ones(())))
        if float(lim) > 0 and bool(over.any()):
            g = g * (lim / n.clamp(min=1e-30)).clamp_(max=1.0).to(g.dtype)[..., None]
        return g, None, None


_KINDS = ('gain', 'scale', 'router', 'bias')


def _kind(name, legacy=False):
    """Which optimiser group a full-precision parameter belongs to. legacy: as it was while the offsets were
    stepped with the gains."""
    last = name.rsplit('.', 1)[-1]
    return ('scale' if last.startswith('s_') else 'router' if '.router.' in name
            else 'bias' if last.startswith('b_') and not legacy else 'gain')


def _opt_state(net, opt, sd):
    """An AdamW state for this net's optimiser. One saved before the offsets had a group of their own has the
    same moments in another order: regroup them."""
    if len(sd['param_groups']) == len(opt.param_groups):
        return sd
    names = [n for n, _ in net.named_parameters() if n != 'b_out']
    order = lambda legacy: [n for k in _KINDS for n in names if _kind(n, legacy) == k]
    was = {n: i for i, n in enumerate(order(True))}
    state = {j: sd['state'][was[n]] for j, n in enumerate(order(False)) if was[n] in sd['state']}
    return {'state': state, 'param_groups': opt.state_dict()['param_groups']}


class Net(nn.Module):
    """One DiffusionBlocks net: its own embedding, layers and head."""

    def __init__(self, cfg, n_layer, emb=None):
        super().__init__()
        self.cfg = cfg
        self.emb = emb  # the tied binary token table, shared by every net (None for bytes)
        self.P, self.ps = 1, None  # ParScale: streams of the next forward, and the add-on (a ParAdapter, not a module)
        if emb is None:  # bytes: full-precision embedding and head, as bintf.py
            self.tok = nn.Embedding(VOCAB, cfg.d)
            nn.init.normal_(self.tok.weight, std=0.02)
            self.head = nn.Linear(cfg.d, VOCAB, bias=False)
            nn.init.normal_(self.head.weight, std=0.02)
        else:  # rows of the shared table are the input embeddings and the output classes
            self.gain = nn.Parameter(torch.full((cfg.d,), 0.5))
            # AdamW steps a parameter by its learning rate however small its gradient, so a scale or bias per
            # token drifts for every token that is not in the batch: the scale is fixed, the bias takes SGD steps
            self.register_buffer('s_out', torch.full((VOCAB,), 0.02))
            self.b_out = nn.Parameter(torch.zeros(VOCAB))
            # Config.tiers: an entry scores a whole group of words, which one fixed step of 0.02 per unit of
            # agreement fits badly (it cost 0.33 nats when tried on the flat model's hidden states, a constant
            # cost 0.012), so the entries' scale is a parameter, per net, and learns. It is held in units of _ENT:
            # AdamW steps every scale by lr x lr_scales whatever its gradient, and at that pace an entry that
            # starts at 0 needs 500 steps of its net to reach the scale every other row has
            self.s_ent = nn.Parameter(torch.full((len(emb.spans) - 1,), 0.02 / _ENT)) if emb.tiers else None
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(n_layer))
        # The readout starts small when the head is the binary table: at 1 its random scores cost 0.4 nats, and
        # the quickest way to lose them is a constant direction that drowns the token signal in every layer.
        self.norm = nn.Parameter(torch.full((cfg.d,), 1.0 if emb is None else 0.1))
        self.register_buffer('cf', torch.zeros(2, cfg.d))  # running means of the last residual stream: _center
        self.register_buffer('kf', torch.zeros(2))
        if cfg.centre2:
            self.register_buffer('df', torch.zeros(2, cfg.d))
            self.register_buffer('jf', torch.zeros(2))

    def signs(self):
        return [s for layer in self.layers for s in (layer.qkv, layer.out, layer.up, layer.dn)]

    def groups(self):
        """AdamW parameter groups by kind: row scales, router weights, offsets, and the rest (norm gains,
        the byte embedding and head). cmd_train gives each kind its own step."""
        kinds = {k: [] for k in _KINDS}
        for name, p in self.named_parameters():
            if name != 'b_out':  # the output bias takes plain SGD steps, in cmd_train
                kinds[_kind(name)].append(p)
        return [{'params': ps, 'kind': k} for k, ps in kinds.items() if ps]

    def forward(self, idx, pos, mask):
        """idx, pos: (B, L) ids and text positions; mask: (B, 1, L, L) who may attend to whom.
        Returns the normed hidden states (B, L, d) and the MoE balance loss."""
        if self.emb is None:
            x = self.tok(idx).to(HALF)
        else:  # the embedding rows learn through the output side; here they are read, not differentiated
            rows = self.emb.rows(idx if self.emb.pos_of is None else self.emb.pos_of[idx])
            x = torch.where(rows >= 0, _ONE, _NEG) * self.gain.to(HALF)
        cs = rope_tables(pos, self.cfg.d // self.cfg.n_head) if self.cfg.pos != 'alibi' else None
        if self.cfg.pos != 'rope':
            mask = alibi(pos, mask, self.cfg.n_head)
        mk = (idx == MASK).reshape(-1)  # positions that read [MASK]; None when there are none
        mk = mk if bool(mk.any()) else None
        P = self.P if self.ps is not None else 1
        B0, L0 = idx.shape
        SO.n0, pre = None, None
        if P > 1:  # ParScale: the batch P times over, stream-major; stream 0 is the plain model (see ParAdapter)
            x = x.repeat(P, 1, 1)
            cs = None if cs is None else (cs[0].repeat(P, 1, 1, 1), cs[1].repeat(P, 1, 1, 1))
            mk = None if mk is None else mk.repeat(P)
            SO.n0 = B0 * L0  # the running means read these rows only, here and in backward (Sign.centred)
            pre = self.ps.view(P, B0)
        aux = 0.0
        for i, layer in enumerate(self.layers):
            if i and SO.gcap > 0 and SO.learn and x.requires_grad:  # backward: what layer i hands down is bounded
                x = _GCap.apply(x, SO.gcap, 0)
            x, a = layer(x, cs, mask, self.cfg, mk, i == 0) if pre is None else                 layer(x, cs, mask, self.cfg, mk, i == 0, pre.at(i))
            aux = aux + a
        if SO.learn:  # for the log: how large the stream has become, on average and at its largest position
            with torch.no_grad():
                r = (x if pre is None else x[:B0]).detach().float().square().mean(-1).sqrt()  # ParScale: stream 0's
                SO.xs_t += torch.stack((r.mean(), r.max(), r.new_ones(())))
                SO.xc_t += (self.cf[0] / self.kf[0].clamp(min=1e-8)).square().mean().sqrt()
        h = _RMSNorm.apply(_center(x, self.cf, self.kf, mk), self.norm)
        h = _center(h, self.df, self.jf, mk) if self.cfg.centre2 else h
        if pre is not None:  # the P streams merged BEFORE the head, which then scores the vocabulary once
            if self.ps.want_h0:  # for the log: stream 0 alone, on the same rows
                self.ps.h0 = h[:B0].detach().clone()
            h = self.ps.merge(h.view(P, B0, L0, -1))
        return h, aux

    def out_scale(self):
        """The output scale of every row of the table: fixed, but for the entries of Config.tiers."""
        return self.s_out if self.s_ent is None else self.s_out.index_copy(0, self.emb.entry, self.s_ent.float() * _ENT)

    def nll(self, h, tgt, narrow=False):
        """Loss per row for hidden states h (N, d) and targets (N,). narrow: leave [MASK] out of the
        byte softmax, as bintf.py's evaluation does."""
        if self.emb is not None:
            if self.emb.tiers:
                tgt = self.emb.pos_of[tgt]
            return _TiedCE.apply(h, tgt, self.out_scale(), self.b_out, self.emb)
        logits = self.head(h).float()
        return F.cross_entropy(logits[:, :256] if narrow else logits, tgt, reduction='none')

    def logits(self, h):
        """Scores over the vocabulary without [MASK], for sampling: h (n, d) -> (n, V), in the order of the ids.
        Under Config.tiers they are log-probabilities: a word of a later group gets its group's entry in the first
        group's softmax plus its own share of its group."""
        if self.emb is None:
            return self.head(h).float()[:, :256]
        out = torch.cat([F.linear(h, self.emb.signs(c)).float() for c in range(self.emb.shape[0])], 1)
        out = out * self.out_scale() + self.b_out
        if self.emb.tiers:
            R, spans, entry = self.emb.shape[1], self.emb.spans, self.emb.entry
            head = out[:, :spans[0][1] * R]
            head -= torch.logsumexp(head, 1, keepdim=True)
            for k, (c0, c1) in enumerate(spans[1:]):
                part = out[:, c0 * R:c1 * R]
                part += (head[:, entry[k]] - torch.logsumexp(part, 1))[:, None]
            out[:, entry] = float('-inf')  # an entry stands for a group: it is not a word
            out = out.index_select(1, self.emb.pos_of)
        out[:, MASK] = float('-inf')
        return out


# ---------------------------------------------------------------- ParScale (arXiv 2505.10475), built in

PS_FORMAT = 'bintf2-parscale-2'  # revision 2: the add-on has prefix logit offsets (po) and fresh-slot flags
_PS_FORMATS = ('bintf2-parscale-1', PS_FORMAT)  # what this revision reads (1: no offsets, read as 0)
PS_P = 4           # streams of a ParScale training step, of evaluation and of the readers (generate, eval, the worker)
PS_DUTY = 0.125    # the share of each net's steps taken with PS_P streams
_PS_SALT = 0x5053  # the add-on's random numbers: a torch.Generator of its own, seeded from --seed and this


def _bound(t):
    """t scaled down to rms 1 along its last dimension where it is above that (_RMSCap with c = 1)."""
    return t * t.square().mean(-1, keepdim=True).add(1e-12).rsqrt().clamp(max=1.0)


class ParAdapter:
    """The ParScale add-on of one net (Chen et al., arXiv 2505.10475): P streams share every weight of the net.

    Stream 0 is the plain model: it has no prefix, so the plain steps of the duty cycle train exactly the stream a
    reader at P = 1 gets, and refresh mode can only add to what the model does. Streams 1 .. P - 1 (every stream
    with meta['prefix0']) each have n learned prefix keys and values in every attention layer (kp, vp: (slots,
    layers, nh, n, hd)): no rotation, no distance, visible to every query. The keys are bounded at rms 1 per head (a
    normed key has exactly 1) and so are the values where the model bounds its own (Config.cap). Prefixes are never
    tokens: nothing routes them and no running mean reads them.

    Revision 2 (D3): a slot's prefixes start from a real batch (ps_init_data, at its net's first ParScale step): in
    every layer the keys, at rms 1 per head, and the values of n positions of it, other positions for every slot.
    (Revision 1 started them at std 0.02: such keys score ~0 against keys of rms 1, so the n of them took 28-52% of
    every query's attention and gave back values of ~0, and the extra streams stayed near copies of each other.)
    Every slot also has a learned logit offset per layer and head on its prefix columns (po: (slots, layers, nh)),
    from -ln(n) - 2: the n prefix keys together start at about e^-2 of one key of the row. An add-on file of format 1
    has no po; it reads with po = 0, what it was trained with.

    The P last hidden states (after the net's last norm) are merged before the head: w = Softmax(MLP(concat
    h_1 .. h_P)), w <- w (1 - smooth) + smooth / P, so the vocabulary is scored once whatever P is. The MLP's last
    layer starts at zero and stream 0's logit at 3: at P = 4 stream 0 starts with ~0.8 of the merge, so switching
    ParScale on does not set the loss back, and the other streams learn their way in.

    Not a module of the net: net{i}.pt, its optimiser, state_dict() and Config stay the plain model's, so every
    revision reads every checkpoint as before. Kept in ckpt/parscale{i}.pt, the shared meta in parscale.json
    (ps_write); trained by an AdamW of its own (ps_update)."""

    KEYS = ('kp', 'vp', 'w1', 'b1', 'w2', 'b2', 'po')

    def __init__(self, cfg, n_layer, meta, gen):
        nh, hd, d = cfg.n_head, cfg.d // cfg.n_head, cfg.d
        Pm, n, hid = int(meta['P_max']), int(meta['n_prefix']), int(meta['hidden'])
        self.s0 = 0 if meta.get('prefix0') else 1
        S = Pm - self.s0
        self.kp = torch.randn(S, n_layer, nh, n, hd, generator=gen) * 0.02  # until ps_init_data fills them (D3)
        self.vp = torch.randn(S, n_layer, nh, n, hd, generator=gen) * 0.02
        self.w1 = (torch.rand(hid, Pm * d, generator=gen) * 2 - 1) * (Pm * d) ** -0.5  # nn.Linear's range
        self.b1 = torch.zeros(hid)
        self.w2, self.b2 = torch.zeros(Pm, hid), torch.zeros(Pm)
        if self.s0:
            self.b2[0] = 3.0
        self.po = torch.full((S, n_layer, nh), _ps_po0(n))  # D3: the prefix columns' logit offsets
        for t in self.params():
            t.requires_grad_()
        self.Pm, self.n, self.d, self.smooth, self.cap = Pm, n, d, float(meta['smooth']), bool(cfg.cap)
        self.live, self.dirty, self.src, self.steps, self.tok = False, True, None, 0, 0
        self.last_w, self.want_h0, self.h0 = None, False, None
        self.fresh = [True] * S  # D3: the slots whose prefixes are still the random start (ps_init_data fills them)

    def params(self):
        return [self.kp, self.vp, self.w1, self.b1, self.w2, self.b2, self.po]

    def kv(self, i, P):
        """Layer i's prefix keys and values of streams s0 .. P - 1: (P - s0, nh, n, hd), full precision, bounded;
        and their logit offsets (P - s0, nh)."""
        k, v = self.kp[:P - self.s0, i], self.vp[:P - self.s0, i]
        return _bound(k), (_bound(v) if self.cap else v), self.po[:P - self.s0, i]

    def view(self, P, B):
        return _PSView(self, P, B)

    def merge(self, h):
        """h (P, B, L, d): every stream's last hidden states -> their weighted sum (B, L, d)."""
        P, B, L, d = h.shape
        with torch.autocast(DEV.type, enabled=False):
            hf = h.float()
            w1 = self.w1.view(-1, self.Pm, d)[:, :P].reshape(-1, P * d)  # fewer streams than P_max: their columns
            z = F.linear(F.silu(F.linear(hf.permute(1, 2, 0, 3).reshape(B, L, P * d), w1, self.b1)),
                         self.w2[:P], self.b2[:P])
            w = z.softmax(-1) * (1 - self.smooth) + self.smooth / P  # each in [smooth / P, 1 - smooth + smooth / P]
            out = torch.einsum('blp,pbld->bld', w, hf)
        self.last_w = w.detach()
        return out.to(h.dtype)

    def state(self, opt=None):
        return {'w': {k: t.detach().cpu() for k, t in zip(self.KEYS, self.params())},
                'opt': None if opt is None else opt.state_dict(), 'steps': self.steps, 'tok': self.tok,
                'fresh': list(self.fresh)}

    def load(self, st):
        with torch.no_grad():
            for k, t in zip(self.KEYS, self.params()):
                w = st['w'].get(k)
                if w is None and k == 'po':  # format 1: no offsets, which is 0, what it was trained with
                    t.fill_(0.0 if int(st.get('steps') or 0) else _ps_po0(self.n))
                    continue
                if w is None or tuple(w.shape) != tuple(t.shape):
                    raise ValueError(f'ParScale {k}: {None if w is None else tuple(w.shape)} in the file, '
                                     f'{tuple(t.shape)} expected')
                t.copy_(w.float())
        self.steps, self.tok = int(st.get('steps') or 0), int(st.get('tok') or 0)
        fr = st.get('fresh')  # a file without it (format 1, an import): still random if it never took a step
        self.fresh = [bool(x) for x in fr] if fr is not None and len(fr) == len(self.fresh) else \
            [self.steps == 0] * len(self.fresh)

    def grow(self, Pm, gen):
        """More streams than the file has: new prefixes (random, as at the start, until ps_init_data) and the
        merge's new inputs and outputs at zero, so the streams there were compute what they did."""
        hid, d, add = self.w1.shape[0], self.d, Pm - self.Pm
        with torch.no_grad():
            sh = self.kp.shape[1:]
            kp = torch.cat((self.kp.detach().cpu(), torch.randn(add, *sh, generator=gen) * 0.02))
            vp = torch.cat((self.vp.detach().cpu(), torch.randn(add, *sh, generator=gen) * 0.02))
            w1 = torch.cat((self.w1.detach().cpu().view(hid, self.Pm, d), torch.zeros(hid, add, d)), 1).reshape(hid, -1)
            w2 = torch.cat((self.w2.detach().cpu(), torch.zeros(add, hid)))
            b2 = torch.cat((self.b2.detach().cpu(), torch.zeros(add)))
            po = torch.cat((self.po.detach().cpu(), torch.full((add, *self.po.shape[1:]), _ps_po0(self.n))))
        self.kp, self.vp, self.w1, self.b1, self.w2, self.b2, self.po = kp, vp, w1, self.b1.detach().cpu(), w2, b2, po
        for t in self.params():
            t.requires_grad_()
        self.Pm, self.dirty, self.src = Pm, True, None
        self.fresh = self.fresh + [True] * add

    def to(self, dev, opt=None):
        for t in self.params():
            if t.device != dev:
                t.data = t.data.to(dev)
                if t.grad is not None:
                    t.grad = t.grad.to(dev)
        for st in (opt.state.values() if opt is not None else ()):
            for k, v in st.items():
                if torch.is_tensor(v) and k != 'step' and v.device != dev:
                    st[k] = v.to(dev)

    def zero_grad(self):
        for t in self.params():
            t.grad = None


class _PSView:
    """One ParScale forward of a net: P streams of B rows each. at(i) is layer i's view."""

    def __init__(self, ad, P, B):
        self.ad, self.P, self.B, self.s0, self.n = ad, P, B, ad.s0, ad.n
        self._m = None

    def at(self, i):
        return _PSLayer(self, i)

    def mask(self, mask):
        """mask (B, 1 or nh, L, L), bool or additive -> the same with n open prefix columns in front; made once per
        forward (every layer gets the same mask)."""
        if self._m is None or self._m[0] is not mask:
            col = (torch.ones if mask.dtype == torch.bool else torch.zeros)(
                *mask.shape[:-1], self.n, dtype=mask.dtype, device=mask.device)
            self._m = (mask, torch.cat((col, mask), -1))
        return self._m[1]


class _PSLayer:
    """Layer i of a ParScale forward (Layer.forward, attend, _AttnOutP, the worker's KV cache)."""

    def __init__(self, v, i):
        self.v, self.i, self.P, self.s0, self.n, self._kv = v, i, v.P, v.s0, v.n, None

    def kv_all(self):
        """This layer's bounded prefix keys and values of streams s0 .. P - 1, (P - s0, nh, n, hd) each, and their
        logit offsets per head (P - s0, nh) (revision 2, D3)."""
        if self._kv is None:
            self._kv = self.v.ad.kv(self.i, self.P)
        return self._kv

    def mask(self, mask):
        return self.v.mask(mask)


def duty_hit(n, f):
    """Whether a net's step n (its own count) runs with ParScale streams: exactly the share f of each net's steps,
    spread evenly (at f = 1/8 its steps 7, 15, 23, ..). No random number is drawn for it, so the rows and the noisy
    plans of every step are those of a plain run."""
    return f > 0 and math.floor((n + 1) * f + 1e-9) > math.floor(n * f + 1e-9)


def ps_meta_new(cfg, a):
    return {'format': PS_FORMAT, 'P_max': max(PS_P, a.parscale), 'P_trained': 1, 'n_prefix': a.parscale_prefix,
            'hidden': a.parscale_hidden, 'smooth': a.parscale_smooth, 'prefix0': bool(a.parscale_prefix0), 'd': cfg.d,
            'n_head': cfg.n_head, 'layers_per_net': cfg.n_layer // cfg.dblocks, 'dblocks': cfg.dblocks, 'p_fit': {}}


def _ps_shape_ok(meta, cfg):
    return meta.get('format') in _PS_FORMATS and (meta.get('d'), meta.get('n_head'), meta.get('layers_per_net'),
                                                 meta.get('dblocks')) == (cfg.d, cfg.n_head, cfg.n_layer // cfg.dblocks,
                                                                          cfg.dblocks)


def _ps_files(d, n=None):
    """(meta, the per-net files) of the add-on in directory d, or None."""
    p = os.path.join(d, 'parscale.json')
    try:
        meta = json.load(open(p))
    except (OSError, ValueError):
        return None
    files = [os.path.join(d, f'parscale{i}.pt') for i in range(n or int(meta.get('dblocks') or 0))]
    return (meta, files) if files and all(os.path.exists(f) for f in files) else None


def load_ps(model, ck, run, a, gen):
    """Every net's add-on (net.ps). From the checkpoint's side files; else from run/parscale_last, the newest this
    revision wrote (a later checkpoint may come from a revision that leaves them out); else a fresh one, written at
    the next save. Returns the shared meta and the per-net optimiser states (None: start them afresh)."""
    cfg = model.cfg
    for d, src in ((ck, 'checkpoint'), (os.path.join(run, 'parscale_last') if run else None, 'parscale_last')):
        got = _ps_files(d, len(model.nets)) if d else None
        if got is None:
            continue
        meta, files = got
        if not _ps_shape_ok(meta, cfg):
            print(f'  parscale: {d} holds an add-on for another shape: not used', flush=True)
            continue
        Pm = max(int(meta['P_max']), a.parscale)
        states = [torch.load(f, map_location='cpu', weights_only=False) for f in files]
        for net, f, st in zip(model.nets, files, states):
            net.ps = ParAdapter(cfg, len(net.layers), meta, gen)
            net.ps.load(st)
            if Pm > net.ps.Pm:
                net.ps.grow(Pm, gen)
            else:
                net.ps.src, net.ps.dirty = f, False
        grown = Pm > int(meta['P_max'])
        meta['P_max'], meta['source'] = Pm, src
        meta['format'] = PS_FORMAT  # read as format 1 or 2, written as 2: it may now hold what 1 does not know
        return meta, (None if grown else [st.get('opt') for st in states])
    meta = ps_meta_new(cfg, a)
    for net in model.nets:
        net.ps = ParAdapter(cfg, len(net.layers), meta, gen)
    meta['source'] = 'new'
    return meta, None


def ps_optimizer(ad, a):
    return torch.optim.AdamW([{'params': ad.params()[:2], 'kind': 'ps_prefix'},
                              {'params': ad.params()[2:6], 'kind': 'ps_merge'},
                              {'params': ad.params()[6:], 'kind': 'ps_offset'}],  # D3: the prefix logit offsets
                             lr=a.parscale_lr, betas=(0.9, 0.95), weight_decay=0.0)


def ps_update(ad, opt, grads, S, a, bs):
    """One AdamW step of the add-on on this step's gradients (taken at loss scale S): clipped on their own, with
    the add-on's own warm-up and then a constant step times sqrt(ids / ref) (bs, as every step of the run): no
    point of any schedule that a longer run could move."""
    for g in grads:
        g.mul_(1.0 / S)
    torch.nn.utils.clip_grad_norm_(ad.params(), a.parscale_clip)
    for gr in opt.param_groups:
        gr['lr'] = a.parscale_lr * min(1.0, (ad.steps + 1) / max(1, a.parscale_warmup)) * bs
    opt.step()
    ad.steps += 1
    ad.dirty = True


def _ps_po0(n):
    """D3: where a slot's prefix logit offsets start: its n prefix keys together weigh about e^-2 of one key."""
    return -math.log(max(1, int(n))) - 2.0


@torch.no_grad()
def ps_init_data(net, rows, gen):
    """D3: the prefixes of every stream slot of net's add-on that is still the random start (ad.fresh), from rows
    (B, l), a real training batch on the device: one forward of the plain model over them (nothing learns, no
    running mean moves, no random number of the run is drawn: the add-on's own generator picks the positions).
    Slot s then holds, in every layer, the keys (normed and rotated, as attention reads them, at rms 1 per head)
    and the values (bounded) of n positions of the batch: the same positions in every layer, so it starts as n real
    tokens in front of the row, and positions no other slot has while the batch has enough of them, so the streams
    differ from their first step. Its logit offsets start at -ln(n) - 2 (_ps_po0)."""
    global _PS_CAPTURE
    ad = net.ps
    slots = [s for s, f in enumerate(ad.fresh) if f]
    if not slots:
        return
    was, learn, n0 = net.P, SO.learn, SO.n0
    net.P, SO.learn, _PS_CAPTURE = 1, False, []
    try:
        idx, pos, mask, _, _ = build_streams(rows, None, 0, None)
        with torch.autocast('cuda', dtype=torch.float16, enabled=CUDA):
            net(idx, pos, mask)
        kv = _PS_CAPTURE
    finally:
        _PS_CAPTURE = None
        net.P, SO.learn, SO.n0 = was, learn, n0
    nl, n = ad.kp.shape[1], ad.n
    if len(kv) != nl:
        raise RuntimeError(f'ParScale prefix start: {len(kv)} attention calls for {nl} layers')
    Bk, nh, L, hd = kv[0][0].shape
    N, need = Bk * L, len(slots) * n
    pick = torch.cat([torch.randperm(N, generator=gen) for _ in range(-(-need // N))])[:need]
    for i, (k, v) in enumerate(kv):
        kf = k.float().transpose(1, 2).reshape(N, nh, hd)
        kf = kf * kf.square().mean(-1, keepdim=True).add(1e-12).rsqrt()  # rms 1 per head: a normed key's own scale
        vf = v.float().transpose(1, 2).reshape(N, nh, hd)
        for j, s in enumerate(slots):
            sel = pick[j * n:(j + 1) * n].to(kf.device)
            ad.kp[s, i] = kf[sel].transpose(0, 1).to(ad.kp.device, ad.kp.dtype)
            ad.vp[s, i] = vf[sel].transpose(0, 1).to(ad.vp.device, ad.vp.dtype)
    for s in slots:
        ad.po[s] = _ps_po0(n)
        ad.fresh[s] = False
    ad.dirty = True
    print(f'  parscale: prefixes of stream{"s" if len(slots) > 1 else ""} {",".join(str(s + ad.s0) for s in slots)} '
          f'started from {n} positions each of a batch of {Bk} x {L} (keys at rms 1, offsets {_ps_po0(n):.2f})',
          flush=True)


def ps_maxpos(a, cfg):
    """D5: the most positions, every stream counted, that one sub-step of a ParScale step holds (-1: no limit)."""
    mp = int(getattr(a, 'parscale_max_pos', 0) or 0)
    return mp if mp else (a.step_tokens or 32 * cfg.ctx)


def ps_plan(k, l, P, mp):
    """D5: (streams, rows per sub-step) of a ParScale step of k rows of l positions at up to P streams whose
    sub-steps hold at most mp positions, every stream counted (mp < 0: no limit, the whole step in one pass): as
    many streams as one row can have (min(P, mp // l)), and as many rows as fit with them. (1, k) when fewer than
    2 streams fit: the step is a plain one."""
    if mp < 0:
        return P, k
    P = min(P, mp // l)
    if P < 2:
        return 1, k
    return P, max(1, min(k, mp // (P * l)))


def ps_fit_key(a, cfg, meta):
    """D6: the key under which parscale.json keeps how many streams fit a step (ps_probe, or the halving after a
    ParScale step ran out of memory): every setting that changes what a step holds on the device."""
    dev = torch.cuda.get_device_name(0) if CUDA else 'cpu'
    return (f'{dev}|{a.step_tokens}|{a.lengths}|{a.noisy}|{a.parscale}|lean {int(_LEAN)}|lean2 {_LEAN2}|'
            f'ps_lean2 {int(_PS_LEAN2)}|resident {a.resident}|gpu_sublayers {a.gpu_sublayers}|'
            f'prefix {meta.get("n_prefix")}|max_pos {ps_maxpos(a, cfg)}')


def ps_refusal(a, cfg, fit):
    """D5: why the ParScale steps this run asks for cannot run, or None: fewer than 2 streams fit the device (fit),
    or --parscale-max-pos holds fewer than 2 streams of a row of some training length. Those steps are taken as
    plain steps, and every step's log line and status.json say so ('PARSCALE NOT RUNNING ...')."""
    lengths = [int(x) for x in str(a.lengths).split(',')] if a.lengths else [cfg.ctx]
    P, mp = min(a.parscale, fit), ps_maxpos(a, cfg)
    if P < 2:
        return f'PARSCALE NOT RUNNING: {fit} stream{"s fit" if fit != 1 else " fits"} a step on this device'
    bad = [l for l in lengths if ps_plan(1, l, P, mp)[0] < 2]
    if not bad:
        return None
    where = '' if len(bad) == len(lengths) else f' on rows of {",".join(str(l) for l in bad)}'
    return f'PARSCALE NOT RUNNING{where}: --parscale-max-pos {mp} holds fewer than 2 streams of a row of {max(bad)}'


def ps_source(ck, run, cfg):
    """D9: where a reader (generate, eval, export) takes the add-on from: the checkpoint's side files, else
    run/parscale_last (the newest a ParScale revision wrote; a checkpoint saved by a revision without ParScale has
    no side files, and the trainer reads the same fallback), else None."""
    for d in (ck, os.path.join(run, 'parscale_last') if run else None):
        got = _ps_files(d, cfg.dblocks) if d else None
        if got is not None and _ps_shape_ok(got[0], cfg):
            if d != ck:
                print(f'ParScale add-on from {d}: the checkpoint has none', flush=True)
            return d
    return None


def ps_eval_record(st):
    """D7: the run's last evaluation of every net (state.json 'eval'), plain (P1; P1_means: read from the running
    means of the latents) and with its trained streams merged (P<k>), in nats per token: what a reader can check
    before it serves P = k."""
    nets = {}
    for nb, e in sorted((st.get('eval') or {}).items()):
        r = {}
        if 'ar' in e:
            r['P1'] = {k: e[k] for k in ('ar', 'ar_tail', 'step') if k in e}
        if isinstance(e.get('avg'), dict) and 'ar' in e['avg']:
            r['P1_means'] = {k: e['avg'][k] for k in ('ar', 'ar_tail', 'step') if k in e['avg']}
        ps = e.get('ps')
        if isinstance(ps, dict) and 'ar' in ps:
            r[f'P{ps.get("P")}'] = {k: ps[k] for k in ('P', 'ar', 'ar_tail', 'step') if k in ps}
        if r:
            nets[nb] = r
    return {'unit': 'nats per token: next-token loss on the run\'s validation rows', 'nets': nets}


_RUN_LOCK = None  # D8: the open run/trainer.lock of this process (its flock is let go when the process ends)


def _lock_run(run):
    """D8: one trainer or refresh at a time on a run directory: an exclusive fcntl.flock on run/trainer.lock, held
    until this process ends, however it ends. (A trainer of a revision without the lock is still caught by
    ps_freeze's look at status.json.)"""
    global _RUN_LOCK
    f = open(os.path.join(run, 'trainer.lock'), 'a+')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.seek(0)
        who = f.read().strip()
        f.close()
        raise SystemExit(f'{run}/trainer.lock is held by another process (last taken by {who or "an unknown process"}); '
                         f'stop it first')
    f.seek(0)
    f.truncate()
    f.write(f'pid {os.getpid()}: {" ".join(sys.argv[:2])}\n')
    f.flush()
    _RUN_LOCK = f


def ps_freeze(model, run):
    """--parscale-freeze-backbone: only the add-on learns. No backbone parameter takes a gradient, no weight
    gradient of a binary matrix or of the token table is taken (SO.frozen), no sign latent, second moment, running
    mean or optimiser state moves; and the run's own trainer must not be running."""
    try:
        pid = int(json.load(open(os.path.join(run, 'status.json')))['pid'])
        cmd = open(f'/proc/{pid}/cmdline', 'rb').read().decode('utf-8', 'replace') if pid != os.getpid() else ''
    except (OSError, ValueError, KeyError, TypeError):
        pid, cmd = None, ''
    if 'bintf' in cmd and 'train' in cmd:
        raise SystemExit(f'--parscale-freeze-backbone: a trainer (pid {pid}) is running on {run}; stop it first')
    for net in model.nets:
        for p in net.parameters():
            p.requires_grad_(False)
    SO.frozen, SO.enabled, SO.ema = True, False, False


def ps_probe(model, b, val, a, cfg):
    """The most streams (--parscale, halved down to 2) whose ParScale step fits this GPU: one forward and backward
    on validation rows of the longest training length with a fixed noisy plan, below 92% of the card. Nothing is
    learned (no sign, second moment or running mean moves), no random number of the run is drawn, the gradients
    are thrown away. 1 if not even 2 streams fit."""
    if not CUDA or a.parscale < 2:
        return a.parscale
    lengths = [int(x) for x in str(a.lengths).split(',')] if a.lengths else [cfg.ctx]
    l = min(max(lengths), val.shape[1])
    k = max(1, (a.step_tokens or 32 * cfg.ctx) // max(lengths))
    rows = val[torch.arange(k) % len(val), :l].to(DEV)
    net, total = model.nets[b], torch.cuda.get_device_properties(0).total_memory
    keep = (SO.enabled, SO.ema, SO.learn, SO.count, SO.gmax)
    mp = ps_maxpos(a, cfg)  # D5: what has to fit is one sub-step of a ParScale step
    P, fit = ps_plan(k, l, a.parscale, mp)[0], 1
    while P >= 2:
        ok = False
        SO.enabled, SO.ema, SO.learn, SO.count = False, False, True, False
        net.P = P
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.autocast('cuda', dtype=torch.float16):
                ar, nz, aux, share = forward_loss(net, rows[:ps_plan(k, l, P, mp)[1]], model.band(b),
                                                  int(l * a.noisy) // 2 * 2,
                                                  np.random.default_rng(0))
                loss = ar + a.w_noisy * share * nz + a.aux * aux
            loss.backward()
            torch.cuda.synchronize()
            ok = torch.cuda.max_memory_allocated() < 0.92 * total
        except torch.cuda.OutOfMemoryError:
            pass
        net.P, SO.n0 = 1, None
        ar = nz = aux = loss = None
        net.zero_grad(set_to_none=True)
        net.ps.zero_grad()
        torch.cuda.empty_cache()
        if ok:
            fit = P
            break
        P //= 2
    SO.enabled, SO.ema, SO.learn, SO.count, SO.gmax = keep
    for t in (SO.bad_t, SO.head_bad_t, SO.br_t, SO.xs_t, SO.xc_t, SO.gc_t, SO.wc_t, SO.flips_t):
        t.zero_()
    torch.cuda.reset_peak_memory_stats()
    return fit


@torch.no_grad()
def ps_stream0(net, clean, l, ar):
    """On a log step: stream 0 alone (net.ps.h0) on the same clean rows, minus the merged loss ar."""
    h0 = net.ps.h0
    with torch.autocast('cuda', dtype=torch.float16, enabled=CUDA):
        nll = net.nll(h0[:, :l - 1].reshape(-1, h0.shape[-1]), clean[:, 1:].reshape(-1))
    return float(nll.float().mean()) - float(ar)


def ps_evaluate(model, b, val, bs, P):
    """evaluate() with P streams merged, or None if the GPU has no room; leaves together()'s NLLS as it was."""
    global ZF
    net, ev = model.nets[b], None
    was, keep = net.P, NLLS.pop(b, None)
    try:
        ev = evaluate(model, b, val, bs, P)
    except torch.cuda.OutOfMemoryError:
        ZF = None
    net.P = was
    NLLS.pop(b, None)
    if keep is not None:
        NLLS[b] = keep
    if ev is None and CUDA:  # outside the handler, so the failed pass has let go of its tensors
        torch.cuda.empty_cache()
    return ev


def ps_line(model, a, on, frozen):
    m, ad = model.ps_meta, model.nets[0].ps
    how = (f'{a.parscale} streams on {a.parscale_duty:g} of each net\'s steps' if on and not frozen else
           f'refresh mode, {a.parscale} streams: the add-on alone learns' if on else
           'off (--parscale 1 or --parscale-duty 0): the plain trainer')
    return (f'parscale: {how}; add-on from {m.get("source")}: {ad.steps:,} steps, trained at up to {m["P_trained"]} '
            f'streams, {m["n_prefix"]} prefix tokens, room for {m["P_max"]} streams'
            + (', stream 0 plain' if ad.s0 else ', a prefix on every stream')
            + (f'; a ParScale step in sub-steps of at most {ps_maxpos(a, model.cfg):,} positions, every stream '
               f'counted' if on and ps_maxpos(a, model.cfg) > 0 else ''))


def ps_log(ad, meta, a, fit, acc, n, gain):
    ar, nz = (acc / n).tolist() if n else (float('nan'), float('nan'))
    w = ad.last_w.float().mean((0, 1)).tolist() if ad.last_w is not None else None
    return {'P': min(a.parscale, fit), 'duty': a.parscale_duty, 'steps': ad.steps, 'P_trained': meta['P_trained'],
            'n': n, 'ar': ar, 'masked': nz, 'gain': gain, 'w': w}


def ps_log_line(b, r):
    return (f'  parscale net {b} | {r["n"]} sub-steps x{r["P"]} since its last line: next {lossf(r["ar"]):.3f} '
            f'masked {lossf(r["masked"]):.3f} {lossu()}'
            + (f' | gain over stream 0 alone {lossf(r["gain"]):+.3f}' if r['gain'] is not None else '')
            + (' | merge ' + ' '.join(f'{x:.2f}' for x in r['w']) if r['w'] else '')
            + f' | add-on steps {r["steps"]:,}')


def ps_report(model, val, a, exclusive):
    """Refresh mode's end: every net on the validation rows, plain and with the trained streams merged."""
    P = min(a.parscale, model.ps_meta['P_trained'])
    for i in range(len(model.nets)):
        model.activate(i, exclusive, a.gpu_sublayers)
        ev = evaluate(model, i, val, a.eval_bs)
        NLLS.pop(i, None)
        evp = ps_evaluate(model, i, val, a.eval_bs, P) if P > 1 else None
        print(f'  net {i}: plain next {lossf(ev["ar"]):.3f} {lossu()}'
              + (f', ParScale x{P} {lossf(evp["ar"]):.3f}' if evp is not None else ''), flush=True)


def ps_trace(path, gstep, b, n, P, clean, rs, rng, data, sub=None):
    """BINTF_PS_TRACE: what a step drew, hashed (rows, the random state before and after, the data cursor); and
    (D5) its rows and row length, and for a ParScale step its sub-steps and each one's sqrt(ids / ref)."""
    h = lambda o: hashlib.sha1(o if isinstance(o, bytes) else
                               json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()[:16]
    rec = {'step': gstep, 'net': b, 'net_step': n, 'P': P, 'clean': h(clean.cpu().numpy().tobytes()), 'rng_pre': h(rs),
           'rng_post': h(rng.bit_generator.state), 'data': h(data.state()),
           'data_rng': h(data.rng.bit_generator.state) if hasattr(data, 'rng') else None,
           'k': int(clean.shape[0]), 'l': int(clean.shape[1]), 'subs': sub['n'] if sub else 1,
           'bs': sub.get('bs') if sub else None}
    with open(path, 'a') as f:
        f.write(json.dumps(rec) + '\n')


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _ps_meta_file(meta):
    return {k: v for k, v in meta.items() if k != 'source'}


def ps_write(model, d):
    """The add-on into checkpoint directory d (save_ckpt, before state.json): parscale{i}.pt per net (weights,
    AdamW state, counters) and parscale.json. A net whose add-on has not changed since it was read or written gets
    a hard link to that file: carried forward byte for byte, at no cost."""
    meta = getattr(model, 'ps_meta', None)
    if meta is None:
        return
    for i, net in enumerate(model.nets):
        dst = os.path.join(d, f'parscale{i}.pt')
        if not net.ps.dirty and net.ps.src and os.path.exists(net.ps.src):
            _link_or_copy(net.ps.src, dst)
        else:
            torch.save(net.ps.state(model.popts[i]), dst)
    with open(os.path.join(d, 'parscale.json'), 'w') as f:
        json.dump(_ps_meta_file(meta), f)


def ps_saved(run, model, cur):
    """After a save: the add-on's files are now those in cur, and run/parscale_last holds hard links of them (no
    older revision touches it, so the add-on survives a checkpoint written by one)."""
    if getattr(model, 'ps_meta', None) is None:
        return
    for i, net in enumerate(model.nets):
        net.ps.src, net.ps.dirty = os.path.join(cur, f'parscale{i}.pt'), False
    last, tmp = os.path.join(run, 'parscale_last'), os.path.join(run, 'parscale_last.new')
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for f in [f'parscale{i}.pt' for i in range(len(model.nets))] + ['parscale.json']:
        _link_or_copy(os.path.join(cur, f), os.path.join(tmp, f))
    shutil.rmtree(last, ignore_errors=True)
    os.rename(tmp, last)


def ps_save_inplace(run, model, ck):
    """Refresh mode's save: the side files of checkpoint ck replaced one at a time (each written beside it and
    renamed over it), then run/parscale_last. Nothing else in ck is touched."""
    for i, net in enumerate(model.nets):
        dst, tmp = os.path.join(ck, f'parscale{i}.pt'), os.path.join(ck, f'parscale{i}.pt.tmp')
        if os.path.exists(tmp):
            os.remove(tmp)
        if not net.ps.dirty and net.ps.src and os.path.exists(net.ps.src):
            if os.path.exists(dst) and os.path.samefile(net.ps.src, dst):
                continue
            _link_or_copy(net.ps.src, tmp)
        else:
            torch.save(net.ps.state(model.popts[i]), tmp)
        os.replace(tmp, dst)
    write_json(os.path.join(ck, 'parscale.json'), _ps_meta_file(model.ps_meta))
    ps_saved(run, model, ck)


def ps_export(ck, st=None):
    """The add-on in directory ck for export: the meta and every net's weights in half precision (no optimiser
    state). st (the checkpoint's state.json): the meta's 'eval' then holds the run's last evaluation of every net,
    plain and with its trained streams merged (D7, ps_eval_record)."""
    got = _ps_files(ck) if ck else None
    if got is None:
        return None
    meta, files = got
    nets = []
    for f in files:
        sd = torch.load(f, map_location='cpu', weights_only=False)
        nets.append({'w': {k: v.half() for k, v in sd['w'].items()}, 'steps': sd.get('steps', 0), 'tok': sd.get('tok', 0)})
    meta = _ps_meta_file(meta)
    if st is not None:
        meta['eval'] = ps_eval_record(st)
    return {'meta': meta, 'nets': nets}


def ps_import(ps, d):
    for i, sd in enumerate(ps['nets']):
        torch.save({'w': {k: v.float() for k, v in sd['w'].items()}, 'opt': None, 'steps': sd.get('steps', 0),
                    'tok': sd.get('tok', 0)}, os.path.join(d, f'parscale{i}.pt'))
    with open(os.path.join(d, 'parscale.json'), 'w') as f:
        json.dump({k: v for k, v in ps['meta'].items() if k != 'eval'}, f)  # (D7's record belongs to the export)


def attach_parscale(model, src, P=PS_P):
    """The reading side (generate, eval, the CPU worker): every net gets its add-on and runs P streams, at most as
    many as were trained (an untrained stream's prefixes are random). src: a checkpoint directory, or the
    'parscale' entry of an export. Returns the streams in use: 1 (the plain model) when there is no add-on."""
    if src is None:
        return 1
    if isinstance(src, str):
        got = _ps_files(src, len(model.nets))
        if got is None:
            return 1
        meta, files = got
        nets = [torch.load(f, map_location='cpu', weights_only=False) for f in files]
    else:
        meta, nets = src['meta'], src['nets']
    if not _ps_shape_ok(meta, model.cfg) or len(nets) != len(model.nets):
        print('ParScale add-on of another shape: not used', flush=True)
        return 1
    use = max(1, min(int(P), int(meta['P_max']), int(meta.get('P_trained') or 1)))
    gen = torch.Generator().manual_seed(0)
    for net, st in zip(model.nets, nets):
        ad = ParAdapter(model.cfg, len(net.layers), meta, gen)
        ad.load(st)
        for t in ad.params():
            t.requires_grad_(False)
        ad.to(DEV)
        net.ps, net.P = ad, use
    model.ps_meta = dict(meta)
    return use


class Model:
    """cfg.dblocks independent nets. Net b owns mask ratios in band(b); net 0 owns the highest."""

    def __init__(self, cfg):
        if cfg.n_layer % cfg.dblocks:
            raise SystemExit(f'--dblocks {cfg.dblocks} must divide n_layer {cfg.n_layer}')
        self.cfg = cfg
        SO.cov = cfg.cov
        self.emb = Sign(VOCAB // VCHUNK, VCHUNK, cfg.d) if cfg.vocab != 'bytes' else None
        if cfg.tiers and self.emb is not None:
            self.emb.set_tiers(cfg.tiers)  # pos_of comes from the data on a fresh start, else from the checkpoint
        self.nets = [Net(cfg, cfg.n_layer // cfg.dblocks, self.emb).to(DEV) for _ in range(cfg.dblocks)]

    def band(self, b):
        n = len(self.nets)
        return (n - b - 1) / n, (n - b) / n

    def net_for(self, ratio):
        return min(len(self.nets) - 1, int((1 - ratio) * len(self.nets)))

    def all_signs(self):
        return ([self.emb] if self.emb is not None else []) + [s for net in self.nets for s in net.signs()]

    def init_signs(self):
        for s in self.all_signs():
            s.init_()

    def activate(self, b, exclusive, sub=-1):
        """Put net b's latents on the GPU. exclusive: the other nets go back to RAM. sub: how many of
        net b's sublayers (attention, feed-forward, attention, ..) stay resident; the rest are streamed
        from RAM one matrix at a time (-1 = all resident)."""
        if self.emb is not None:
            self.emb.load()
        if exclusive:
            for i, net in enumerate(self.nets):
                if i != b:
                    for s in net.signs():
                        s.store(release=True)
        for i, s in enumerate(self.nets[b].signs()):
            if sub < 0 or i // 2 < sub:
                s.load()
            else:
                s.store(release=True)
        popts = getattr(self, 'popts', None)
        for i, net in enumerate(self.nets):  # ParScale: the add-on of the net in use is on the GPU with it
            if CUDA and net.ps is not None and net.ps.live:
                net.ps.to(DEV if i == b or not exclusive else torch.device('cpu'), popts[i] if popts else None)

    def fold(self, left, m):
        """The end of a turn: the net that has just left the GPU goes into its running means (_Fold), and once
        per round of the nets the shared token table does. Nothing here waits for the thread."""
        for s in self.nets[left].signs():
            if s.P is not None:  # every net stays on the GPU (--resident all): a copy for the mean
                s.store(release=False)
            FOLD.put(s, m)
        if self.emb is not None and left == len(self.nets) - 1:
            self.emb.store(release=False)
            FOLD.put(self.emb, m)

    def has_avg(self):
        return all(s.avg is not None for s in self.all_signs())

    @contextlib.contextmanager
    def reading(self, b):
        """Inside: net b and the token table are read from the running means of their latents (_Fold)."""
        FOLD.join()
        held = []
        try:
            # Entry can fail partway through the transfers, before yield. Restore
            # every original training tensor even when an averaged copy runs out of memory.
            for s in self.nets[b].signs() + ([self.emb] if self.emb is not None else []):
                if s.avg is not None and s.P is not None:
                    held.append((s, s.P))
                    s.P = s.avg.to(DEV)
            yield
        finally:
            for s, P in reversed(held):
                s.P = P

    def counts(self):
        fp = sum(p.numel() for net in self.nets for p in net.parameters())
        binary = sum(s.numel() for s in self.all_signs())
        return fp + binary, binary


def describe(cfg, total, binary):
    active = total - (cfg.n_layer * (cfg.n_exp - cfg.topk) * 2 * cfg.d * cfg.ff if cfg.n_exp else 0)
    emb = VOCAB * cfg.d if cfg.vocab != 'bytes' else 0
    return (f'{total / 1e6:,.1f}M params ({binary / 1e6:,.1f}M binary signs x row scales as int8 latents'
            + (f', of which {emb / 1e6:,.1f}M the shared token table' if emb else '')
            + f'; {(total - binary) / 1e6:,.2f}M full precision), {(active - emb) / 1e6:,.1f}M active per token'
            + (f', {cfg.n_exp} experts top-{cfg.topk}' if cfg.n_exp else '')
            + (f', DiffusionBlocks x{cfg.dblocks} ({cfg.n_layer // cfg.dblocks} layers and '
               f'{(total - emb) / cfg.dblocks / 1e6:,.0f}M params per net)' if cfg.dblocks > 1 else '')
            + f', {cfg.act}, ctx {cfg.ctx}, vocab {cfg.vocab} ({VOCAB:,})'
            + (f', branches bounded at rms {cfg.cap:g}' if cfg.cap else '')
            + (', matrix inputs centred twice' if cfg.centre2 else '')
            + (', offsets for queries, keys and feed-forward units' if cfg.bias else '')
            + (', matrices learn from centred gradients' if cfg.cov else '')
            + (f', output head in groups of {cfg.tiers} slices by word frequency' if cfg.tiers else ''))


# ---------------------------------------------------------------- the two streams

def plan_noisy(l, m, band, rng):
    """Blocks of 2, 4, 8.. tokens that add up to m noisy tokens for one row of l tokens. For each
    token: its text position, the start of its block, its block number, and whether it is masked."""
    pos, start, bid = np.empty(m, np.int64), np.empty(m, np.int64), np.empty(m, np.int64)
    msk = np.zeros(m, bool)
    o, b = 0, 0
    while m - o >= 2:
        n = 1 << int(rng.integers(1, int(math.log2(min(m - o, l))) + 1))
        s = int(rng.integers(0, l - n + 1))
        mk = rng.random(n) < band[0] + (band[1] - band[0]) * rng.random()
        if not mk.any():
            mk[int(rng.integers(0, n))] = True
        pos[o:o + n], start[o:o + n], bid[o:o + n], msk[o:o + n] = np.arange(s, s + n), s, b, mk
        o, b = o + n, b + 1
    return pos, start, bid, msk


def build_streams(clean, band, m, rng):
    """clean (B, l) ids on the device. Returns idx and pos (B, l + m), the attention mask
    (B, 1, l + m, l + m), and the noisy stream's true ids and loss mask (B, m)."""
    B, l = clean.shape
    ar = torch.arange(l, device=DEV)
    tril = torch.ones(l, l, dtype=torch.bool, device=DEV).tril_()
    if not m:
        return clean, ar.expand(B, l), tril[None, None].expand(B, 1, l, l), None, None
    plans = [plan_noisy(l, m, band, rng) for _ in range(B)]
    pos, start, bid, msk = (torch.from_numpy(np.stack([p[i] for p in plans])).to(DEV) for i in range(4))
    tgt = clean.gather(1, pos)
    mask = torch.zeros(B, l + m, l + m, dtype=torch.bool, device=DEV)
    mask[:, :l, :l] = tril                                   # clean: causal
    mask[:, l:, :l] = ar[None, None, :] < start[:, :, None]  # noisy: the clean tokens before its block
    mask[:, l:, l:] = bid[:, :, None] == bid[:, None, :]     # noisy: its own block, both directions
    idx = torch.cat([clean, torch.where(msk, torch.full_like(tgt, MASK), tgt)], 1)
    return idx, torch.cat([ar.expand(B, l), pos], 1), mask[:, None], tgt, msk


def forward_loss(net, clean, band, m, rng):
    """Mean next-token loss of the clean stream, mean masked-token loss of the noisy stream, MoE balance,
    and how many masked targets there are for each next-token target."""
    idx, pos, mask, tgt, msk = build_streams(clean, band, m, rng)
    h, aux = net(idx, pos, mask)
    B, l = clean.shape
    hh, tt = h[:, :l - 1].reshape(B * (l - 1), -1), clean[:, 1:].reshape(-1)
    if m:
        hh, tt = torch.cat([hh, h[:, l:][msk]]), torch.cat([tt, tgt[msk]])
    nll = net.nll(hh, tt)
    n = B * (l - 1)
    return nll[:n].mean(), (nll[n:].mean() if m else torch.zeros((), device=DEV)), aux, (len(nll) - n) / n


# ---------------------------------------------------------------- evaluation and decoding

@torch.no_grad()
def evaluate(model, b, val, bs, P=1):
    """val (R, ctx) ids. Next-token loss from net b's clean stream over every position and over the
    last 32 (nearly full context), and, for the net that owns mask ratio 1, 16 tokens at once.
    Losses are in nats per token. P: ParScale streams (1: the plain model, stream 0)."""
    global ZF
    net, T = model.nets[b], val.shape[1]
    was = net.P
    net.P = P if net.ps is not None else 1
    ZF = []
    each = []
    tot = tail = nat = 0.0
    for i in range(0, len(val), bs):
        rows = val[i:i + bs].to(DEV)
        idx, pos, mask, _, _ = build_streams(rows, None, 0, None)
        with torch.autocast('cuda', dtype=torch.float16, enabled=CUDA):
            h = net(idx, pos, mask)[0][:, :-1]
            nll = net.nll(h.reshape(-1, h.shape[-1]), rows[:, 1:].reshape(-1), narrow=True).view(len(rows), T - 1)
        tot, tail = tot + nll.sum().item(), tail + nll[:, -32:].sum().item()
        each.append(nll.float().cpu())
        if b == model.net_for(1.0):
            l = T - 16
            idx = torch.cat([rows[:, :l], torch.full_like(rows[:, l:], MASK)], 1)
            m2 = torch.ones(T, T, dtype=torch.bool, device=DEV).tril_()
            m2[l:, l:] = True
            with torch.autocast('cuda', dtype=torch.float16, enabled=CUDA):
                h = net(idx, pos, m2[None, None].expand(len(rows), 1, T, T))[0][:, l:]
                nat += net.nll(h.reshape(-1, h.shape[-1]), rows[:, l:].reshape(-1), narrow=True).sum().item()
    R = len(val)
    out = {'ar': tot / (R * (T - 1)), 'ar_tail': tail / (R * 32)}
    if b == model.net_for(1.0):
        out['nat16'] = nat / (R * 16)
    if ZF:
        out['ff_zero'] = float(torch.stack(ZF).mean())
    ZF = None
    NLLS[b] = torch.cat(each)  # every validation token's loss, for together()
    net.P = was
    return out


NLLS = {}  # net -> the loss of every validation token at that net's evaluation in this process


def together(run, b, D, last_eval, tag=''):
    """The loss of the D nets answering together: a token's probability is the mean of what the nets give it
    (measured at step 4,289 of the eighth start: 0.11 nats below the mean of the three alone). Net b has just been
    scored; the others are taken as they were at their own last evaluation (the nets are scored in turn), from the
    loss per token every evaluation leaves in the run directory. None until every net has one from this run."""
    # tag 'avg': the evaluation read from the running means of the latents (last_eval[net]['avg'], files of their own)
    at = (lambda i: last_eval[str(i)].get(tag, {}).get('step')) if tag else (lambda i: last_eval[str(i)]['step'])
    np.savez(os.path.join(run, f'eval_nll_{tag}{b}.npz'), step=at(b), nll=NLLS.pop(b).numpy())
    each = []
    for i in range(D):
        p = os.path.join(run, f'eval_nll_{tag}{i}.npz')
        if str(i) not in last_eval or not os.path.exists(p):
            return None
        z = np.load(p)
        if at(i) is None or int(z['step']) != at(i):
            return None
        each.append(torch.from_numpy(z['nll']).double())
    if D < 2 or len({e.shape for e in each}) != 1:
        return None
    return float((math.log(D) - torch.logsumexp(-torch.stack(each), 0)).mean())


def eval_line(ev):
    return (f'next-{unit()[:-1]} {lossf(ev["ar"]):.3f} {lossu()} (last 32 positions {lossf(ev["ar_tail"]):.3f})'
            + (f', 16 at once {lossf(ev["nat16"]):.3f}' if 'nat16' in ev else '')
            + (f', feed-forward activations {ev["ff_zero"] * 100:.0f}% zero' if 'ff_zero' in ev else ''))


@torch.no_grad()
def evaluate_long(model, b, val, near, tail):
    """Does net b still read its context past the training length? val (R, T) ids, T above the longest
    training row. The last `tail` ids of each row are scored twice: at the end of the whole row (up to T
    ids of context, 'far'), and at the end of the row's last `near` ids alone (as much context as in
    training, 'near'). A net that reads further back than it was trained to scores far no worse than near."""
    net, T = model.nets[b], val.shape[1]
    tot = {'far': 0.0, 'near': 0.0}
    for i in range(len(val)):
        for key, rows in (('far', val[i:i + 1]), ('near', val[i:i + 1, T - near:])):
            rows = rows.to(DEV)
            idx, pos, mask, _, _ = build_streams(rows, None, 0, None)
            with torch.autocast('cuda', dtype=torch.float16, enabled=CUDA):
                h = net(idx, pos, mask)[0][:, -tail - 1:-1]
                tot[key] += net.nll(h.reshape(-1, h.shape[-1]), rows[:, -tail:].reshape(-1), narrow=True).sum().item()
    n = len(val) * tail
    return {'far': tot['far'] / n, 'near': tot['near'] / n, 'far_len': T, 'near_len': near, 'tail': tail}


@torch.no_grad()
def probe_ar(model, b, val, bs):
    """Net b's mean next-token loss on fixed rows: nothing but the model differs between two calls."""
    net, T = model.nets[b], val.shape[1]
    tot = 0.0
    for i in range(0, len(val), bs):
        rows = val[i:i + bs].to(DEV)
        idx, pos, mask, _, _ = build_streams(rows, None, 0, None)
        with torch.autocast('cuda', dtype=torch.float16, enabled=CUDA):
            h = net(idx, pos, mask)[0][:, :-1]
            tot += net.nll(h.reshape(-1, h.shape[-1]), rows[:, 1:].reshape(-1), narrow=True).sum().item()
    return tot / (len(val) * (T - 1))


def probe_log(run, step, b, when, ar):
    with open(os.path.join(run, 'probe.jsonl'), 'a') as f:
        f.write(json.dumps({'step': step, 'net': b, 'when': when, 'ar': ar}) + '\n')


def long_line(ev):
    return (f'the last {ev["tail"]:,} {unit()} of {ev["far_len"]:,}-long rows {lossf(ev["far"]):.3f} {lossu()}; '
            f'the same {unit()} with the {ev["near_len"]:,} of context it trains on {lossf(ev["near"]):.3f}')


@torch.no_grad()
def run_seq(model, b, clean, noisy=None):
    """One sequence: clean ids, then optionally one noisy block that starts right after them.
    Returns scores for the last clean position, or for every position of the block."""
    l, n = len(clean), len(noisy) if noisy else 0
    idx = torch.tensor([clean + (noisy or [])], device=DEV)
    mask = torch.ones(l + n, l + n, dtype=torch.bool, device=DEV).tril_()
    if n:
        mask[l:, l:] = True
    net = model.nets[b]
    with torch.autocast('cuda', dtype=torch.float16, enabled=CUDA):
        h, _ = net(idx, torch.arange(l + n, device=DEV)[None], mask[None, None])
        return net.logits(h[0, l:] if n else h[0, -1:])


@torch.no_grad()
def run_all(model, clean, a):
    """Scores for the token after `clean` from every net together: the log of the mean of the nets' probabilities.
    Each net is an 8-layer model of its own for left-to-right text; the mean was 0.11-0.13 nats better than one
    net on validation rows (steps 4,289 to 8,300). With --resident one every token swaps each net in and out."""
    lps = []
    for b in range(len(model.nets)):
        model.activate(b, a.resident == 'one', a.gpu_sublayers)
        lps.append(run_seq(model, b, clean).log_softmax(-1))
    return torch.logsumexp(torch.stack(lps), 0) - math.log(len(lps))


def _sample(logits, a):
    logits = logits / max(a.temp, 1e-5)
    if a.topk_sample:
        kth = logits.topk(min(a.topk_sample, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float('-inf'))
    probs = logits.softmax(-1)
    pick = probs.argmax(-1) if a.temp == 0 else torch.multinomial(probs, 1)[:, 0]
    return pick, probs.gather(1, pick[:, None])[:, 0]


@torch.no_grad()
def fill(model, prefix, n, steps, a):
    """Append n tokens to prefix as one noisy block, unmasking the most confident over `steps` passes;
    each pass goes to the net that owns the current mask ratio."""
    T = model.cfg.ctx
    prefix = prefix[max(0, len(prefix) - (T - n)):]
    block, todo, passes = [MASK] * n, list(range(n)), 0
    for s in range(steps):
        b = model.net_for(len(todo) / n)
        model.activate(b, a.resident == 'one', a.gpu_sublayers)
        pick, conf = _sample(run_seq(model, b, prefix, block)[todo], a)
        passes += 1
        k = len(todo) if s == steps - 1 else math.ceil(len(todo) / (steps - s))
        keep = set(conf.argsort(descending=True)[:k].tolist())
        for j in keep:
            block[todo[j]] = int(pick[j])
        todo = [t for j, t in enumerate(todo) if j not in keep]
        if not todo:
            break
    return block, passes


def decode(model, mode, prompt, n, a):
    T = model.cfg.ctx
    out, passes, t = [], 0, time.time()
    if mode == 'ar':
        b = model.net_for(1.0)
        model.activate(b, a.resident == 'one', a.gpu_sublayers)
        for _ in range(n):
            pick, _ = _sample(run_seq(model, b, (prompt + out)[-T:]), a)
            out, passes = out + [int(pick[0])], passes + 1
    elif mode == 'ar3':  # every net scores the next token (see run_all)
        for _ in range(n):
            pick, _ = _sample(run_all(model, (prompt + out)[-T:], a), a)
            out, passes = out + [int(pick[0])], passes + len(model.nets)
    elif mode == 'semi':
        while len(out) < n:
            blk, p = fill(model, prompt + out, min(a.block, n - len(out)), a.semi_steps, a)
            out, passes = out + blk, passes + p
    elif mode == 'nat':
        out, passes = fill(model, prompt, n, 1, a)
    elif mode == 'diffusion':
        out, passes = fill(model, prompt, n, a.steps, a)
    return out, passes, time.time() - t


@torch.no_grad()
def ar_score(model, prompt, cont, a):
    """How surprising `cont` is to the AR decoder read left to right (lower = more fluent to it)."""
    b = model.net_for(1.0)
    model.activate(b, a.resident == 'one', a.gpu_sublayers)
    idx = torch.tensor([(prompt + cont)[-model.cfg.ctx:]], device=DEV)
    _, pos, mask, _, _ = build_streams(idx, None, 0, None)
    with torch.autocast('cuda', dtype=torch.float16, enabled=CUDA):
        nll = model.nets[b].nll(model.nets[b](idx, pos, mask)[0][0, :-1], idx[0, 1:], narrow=True)
    return lossf(nll[-len(cont):].mean().item())


def sample_all(model, prompts, a):
    D = len(model.nets)
    modes = (['ar'] + (['ar3'] if D > 1 else []) + ['semi', 'nat', 'diffusion']) if a.modes == 'all' else a.modes.split(',')
    labels = {'ar': 'ar', 'ar3': f'ar, {D} nets', 'semi': f'semi {a.block}x{a.semi_steps}', 'nat': 'nat',
              'diffusion': f'diffusion {a.steps}'}
    for text in prompts:
        prompt = encode(text)
        n = min(a.n, model.cfg.ctx - 1)
        print(f'\nprompt: "{text}"  ({n} new {unit()})')
        print(f'  {"mode":<14} {"passes":>6} {"secs":>6} {lossu():>6}  output')
        for mode in modes:
            torch.manual_seed(a.seed)
            out, passes, secs = decode(model, mode, prompt, n, a)
            print(f'  {labels[mode]:<14} {passes:>6} {secs:>6.2f} {ar_score(model, prompt, out, a):>6.2f}  {show(out)}',
                  flush=True)


# ---------------------------------------------------------------- data

HF_TRAIN = 'hf:roneneldan/TinyStories/TinyStoriesV2-GPT4-train.txt'
HF_VAL = 'hf:roneneldan/TinyStories/TinyStoriesV2-GPT4-valid.txt'


def id_dtype():
    return np.uint8 if VNAME == 'bytes' else np.int32


def load_text(spec, mb):
    """A local path, or hf:owner/dataset/path/to/file.txt (first `mb` MB, cached in ~/.cache/bintf)."""
    if not spec.startswith('hf:'):
        with open(spec, 'rb') as f:
            return f.read()
    owner, name, path = spec[3:].split('/', 2)
    limit = int(mb * 1e6)
    cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'bintf')
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{owner}_{name}_{path.replace("/", "_")}_{mb:g}MB')
    if not os.path.exists(cache):
        url = f'https://huggingface.co/datasets/{owner}/{name}/resolve/main/{path}'
        print(f'fetching up to {mb:g} MB of {url}', flush=True)
        req = urllib.request.Request(url, headers={'Range': f'bytes=0-{limit - 1}'})
        with urllib.request.urlopen(req, timeout=60) as r, open(cache + '.part', 'wb') as f:
            got = 0
            while got < limit:
                chunk = r.read(min(1 << 20, limit - got))
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
        os.replace(cache + '.part', cache)
    with open(cache, 'rb') as f:
        raw = f.read()
    raw = raw[:raw.rfind(b'\n') + 1] or raw  # drop a line cut off by the byte limit
    return raw.replace(b'<|endoftext|>', b'\n')


def load_ids(spec, mb):
    """The ids of a single text file: its bytes, or its tokens (cached next to the text)."""
    raw = load_text(spec, mb)
    if VNAME == 'bytes':
        return np.frombuffer(raw, dtype=np.uint8)
    cache = os.path.join(os.path.expanduser('~'), '.cache', 'bintf',
                         f'{spec.replace("/", "_").replace(":", "_")}_{mb:g}MB.{VNAME}.npy')
    if os.path.exists(cache):
        return np.load(cache)
    chunks, cur, size = [], [], 0
    for line in raw.decode('utf-8', 'replace').split('\n'):
        cur.append(line)
        size += len(line) + 1
        if size > 8000:
            chunks.append('\n'.join(cur) + '\n')
            cur, size = [], 0
    chunks.append('\n'.join(cur))
    ids = np.concatenate([np.asarray(e.ids, dtype=np.int32)
                          for e in tokenizer().encode_batch(chunks, add_special_tokens=False)])
    np.save(cache, ids)
    return ids


def unigram_bias(ids, cap=1 << 28):
    """The log frequency of every id over the first `cap` of ids (add-half smoothing), and the
    entropy of that distribution: the loss of a model that knows only how common each token is."""
    c = np.zeros(VOCAB, np.int64)
    n = min(len(ids), cap)
    for i in range(0, n, 1 << 24):
        c += np.bincount(ids[i:min(i + (1 << 24), n)], minlength=VOCAB)[:VOCAB]
    p = (c + 0.5) / (c.sum() + 0.5 * VOCAB)
    return torch.from_numpy(np.log(p)).float(), float(-(p * np.log(p)).sum())


def table_order(score, tiers):
    """Config.tiers: the row of every id when the table is kept in order of frequency. score (V,): anything that
    grows with a word's frequency (words that tie keep the order of their ids). The spare ids that stand for the
    later groups take the last rows of the first group; every other id follows by frequency."""
    sizes = [int(v) for v in str(tiers).split(',')]
    k, spare = len(sizes) - 1, TOKENIZERS[VNAME]['spare']
    sc = np.asarray(score, dtype=np.float64).copy()
    sc[spare:spare + k] = -np.inf
    by = np.argsort(-sc, kind='stable')[:VOCAB - k]
    head = sizes[0] * VCHUNK - k
    order = np.concatenate([by[:head], np.arange(spare, spare + k), by[head:]])  # order[row] = id
    pos = np.empty(VOCAB, dtype=np.int64)
    pos[order] = np.arange(VOCAB)
    return torch.from_numpy(pos)


class MathReplay:
    """Deterministic, checkpoint-bound low-duty replay of a pretokenized corpus."""
    def __init__(self, ids_path, meta_path, every, seed, previous=None):
        self.path = os.path.abspath(ids_path)
        self.meta_path = os.path.abspath(meta_path)
        self.every = int(every)
        self.seed = int(seed)
        if self.every < 1:
            raise ValueError('--math-replay-every must be >= 1 when replay is enabled')
        meta = json.load(open(self.meta_path))
        if meta.get('schema') != 'bintf.math-replay.v1':
            raise ValueError('math replay metadata schema mismatch')
        if meta.get('tokenizer') != VNAME:
            raise ValueError(f"math replay tokenizer {meta.get('tokenizer')} != {VNAME}")
        h = hashlib.sha256()
        with open(self.path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != meta.get('ids_sha256'):
            raise ValueError('math replay ids SHA256 mismatch')
        self.ids = np.memmap(self.path, dtype=np.int32, mode='r')
        if len(self.ids) != int(meta.get('tokens', -1)):
            raise ValueError('math replay token count mismatch')
        self.info = {
            'schema': 'bintf.math-replay.state.v1',
            'ids_sha256': actual,
            'source_sha256': str(meta.get('source_sha256') or ''),
            'source_commit': str(meta.get('source_commit') or ''),
            'tokenizer': VNAME,
            'tokens': int(len(self.ids)),
            'every': self.every,
        }
        if previous is not None:
            for key in ('ids_sha256', 'source_sha256', 'tokenizer', 'tokens', 'every'):
                if previous.get(key) != self.info.get(key):
                    raise ValueError(f'math replay checkpoint binding mismatch for {key}')
        print('[math-replay] enabled ' + json.dumps(self.info, sort_keys=True), flush=True)

    def state(self):
        return dict(self.info)

    def hit(self, global_step):
        return int(global_step) % self.every == 0

    def rows(self, k, l, global_step):
        k, l = int(k), int(l)
        if l < 1 or k < 1 or len(self.ids) <= l:
            raise ValueError('math replay corpus is too small for requested rows')
        # Independent from the trainer RNG and deterministic from committed global step.
        z = (self.seed * 0x9E3779B185EBCA87 + int(global_step) * 0xD1B54A32D192ED03) & ((1 << 64) - 1)
        g = np.random.default_rng(z)
        starts = g.integers(0, len(self.ids) - l, size=k, endpoint=False)
        return np.stack([np.asarray(self.ids[int(a):int(a)+l], dtype=np.int32) for a in starts])


def open_math_replay(a, checkpoint_state):
    prev = checkpoint_state.get('math_replay')
    enabled = bool(a.math_replay_ids) or bool(a.math_replay_meta) or int(a.math_replay_every or 0) > 0
    if not enabled:
        if prev:
            raise ValueError('checkpoint requires math replay but launch disabled it')
        return None
    if not a.math_replay_ids or not a.math_replay_meta:
        raise ValueError('--math-replay-ids and --math-replay-meta are both required')
    return MathReplay(a.math_replay_ids, a.math_replay_meta, a.math_replay_every, a.seed, prev)


class RandomText:
    """An array of ids in RAM, sampled at random offsets (bintf.py's sampler)."""

    def __init__(self, ids, seed):
        self.data = ids
        self.rng = np.random.default_rng(seed)

    def rows(self, k, l):
        starts = self.rng.integers(0, len(self.data) - l, k)
        return np.stack([self.data[s:s + l] for s in starts])

    def sample(self):
        return self.data

    def state(self):
        return {}

    def where(self):
        return f'{len(self.data) / 1e6:,.1f}M {unit()} in RAM'


def cmd_prep(a):
    """Download one parquet shard and write its text column as a flat pool of ids: bytes with a 0x00
    after each document, or tokens as int32 with BOS before and EOS after each document. The
    trainer runs this as a subprocess so tokenising never stalls a training step."""
    set_vocab(a.vocab)
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    t = time.time()
    path = hf_hub_download(a.repo, a.file, repo_type='dataset', local_dir=os.path.join(os.path.dirname(a.out), 'dl'),
                           token=hf_token())
    pf = pq.ParquetFile(path)
    docs = 0
    with open(a.out + '.part', 'wb', buffering=1 << 24) as f:
        for rg in range(pf.num_row_groups):
            col = pf.read_row_group(rg, columns=['text']).column(0)
            docs += len(col)
            if a.vocab == 'bytes':
                for chunk in col.chunks:
                    bufs = chunk.buffers()
                    wide = str(chunk.type).startswith('large')
                    off = np.frombuffer(bufs[1], dtype=np.int64 if wide else np.int32, count=len(chunk) + 1,
                                        offset=chunk.offset * (8 if wide else 4)).tolist()
                    data = memoryview(bufs[2])
                    for i in range(len(chunk)):
                        f.write(data[off[i]:off[i + 1]])
                        f.write(b'\x00')
            else:
                texts = col.to_pylist()
                for i in range(0, len(texts), 2000):
                    for e in tokenizer().encode_batch([x or '' for x in texts[i:i + 2000]], add_special_tokens=False):
                        ids = np.empty(len(e.ids) + 2, np.int32)
                        ids[0], ids[1:-1], ids[-1] = BOS, e.ids, EOS
                        f.write(ids.tobytes())
    os.replace(a.out + '.part', a.out)
    os.remove(path)
    n = os.path.getsize(a.out) // np.dtype(id_dtype()).itemsize
    print(f'[data] {a.file}: {docs:,} documents, {n / 1e6:,.1f}M {unit()} in {time.time() - t:.0f}s', flush=True)


class ShardStream:
    """The parquet shards under one directory of a Hugging Face dataset, in a fixed shuffled order.
    A shard becomes a pool of ids (see cmd_prep), cut into windows of `win` ids that are handed out
    once each, shuffled. Rows shorter than a window take pieces of a window, so every token is used
    once."""

    def __init__(self, spec, work, seed, win, state=None, mix=256):
        parts = spec.split(':')
        self.repo, self.sub = parts[1], parts[2] if len(parts) > 2 else ''
        self.work, self.seed, self.win = work, seed, win
        os.makedirs(work, exist_ok=True)
        self.files = self._files()
        self.order = np.random.default_rng(seed).permutation(len(self.files)).tolist()
        st = state or {}
        self.si, self.wi = st.get('si', 0), st.get('wi', 0)
        self.bank = [list(x) for x in st.get('bank', [])]  # open windows: [window, ids already used]
        self.calls, self.mix = st.get('calls', 0), mix
        self.thread = None
        self._open()

    def _files(self):
        cache = os.path.join(self.work, 'files.json')
        if os.path.exists(cache):
            return json.load(open(cache))
        from huggingface_hub import HfApi
        files = sorted(f for f in HfApi().list_repo_files(self.repo, repo_type='dataset', token=hf_token())
                       if f.startswith(self.sub) and f.endswith('.parquet'))
        if not files:
            raise SystemExit(f'no parquet files under {self.repo}/{self.sub}')
        json.dump(files, open(cache, 'w'))
        return files

    def _pool(self, si):
        return os.path.join(self.work, f'pool_{si:06d}.bin')

    def _fetch(self, si):
        out, attempt = self._pool(si), 0
        cmd = [sys.executable, os.path.abspath(__file__), 'prep', '--repo', self.repo, '--file',
               self.files[self.order[si % len(self.files)]], '--out', out, '--vocab', VNAME]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='true', RAYON_NUM_THREADS='12')
        while not os.path.exists(out):
            if subprocess.run(cmd, env=env).returncode:
                attempt += 1
                print(f'[data] shard {si} failed; retry {attempt}', flush=True)
                time.sleep(min(600, 20 * attempt))

    def _open(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        self._fetch(self.si)
        self.pool = np.memmap(self._pool(self.si), dtype=id_dtype(), mode='r')
        self.n_win = len(self.pool) // self.win
        self.perm = np.random.default_rng([self.seed, self.si]).permutation(self.n_win)
        for p in glob.glob(os.path.join(self.work, 'pool_*.bin')):
            if int(os.path.basename(p)[5:11]) < self.si:
                os.remove(p)

    def sample(self):
        """Ids to count token frequencies over: the shard that is open."""
        return self.pool

    def take_val(self, n):
        """The first n windows of the first shard, set aside for evaluation."""
        self.wi = max(self.wi, n)
        return np.stack([self.pool[w * self.win:(w + 1) * self.win] for w in self.perm[:n].tolist()])

    def rows(self, k, l):
        """k rows of l ids. A window is handed out in pieces of whatever lengths are asked for, each id
        once; the pieces come from `mix` open windows chosen at random, so the next piece of a text
        arrives hundreds of steps after the one before it, not in the next batch."""
        if self.thread is None and self.wi > self.n_win // 2 and not os.path.exists(self._pool(self.si + 1)):
            self.thread = threading.Thread(target=self._fetch, args=(self.si + 1,), daemon=True)
            self.thread.start()
        fits = [e for e in self.bank if self.win - e[1] >= l]
        fresh = min(max(k, self.mix) - len(fits), self.n_win - self.wi)
        if len(fits) + fresh < k:  # this shard cannot fill the batch any more: on to the next one
            self.si, self.wi, self.bank, self.calls = self.si + 1, 0, [], 0
            self._open()
            fits, fresh = [], min(max(k, self.mix), self.n_win)
        for _ in range(max(0, fresh)):
            e = [int(self.perm[self.wi]), 0]
            self.wi += 1
            self.bank.append(e)
            fits.append(e)
        pick = np.random.default_rng([self.seed, self.si, self.calls]).choice(len(fits), k, replace=False)
        self.calls += 1
        out = np.empty((k, l), id_dtype())
        for i, j in enumerate(pick.tolist()):
            e = fits[j]
            s = e[0] * self.win + e[1]
            out[i] = self.pool[s:s + l]
            e[1] += l
        self.bank = [e for e in self.bank if e[1] < self.win]
        return out

    def state(self):
        return {'si': self.si, 'wi': self.wi, 'bank': self.bank, 'calls': self.calls}

    def where(self):
        return (f'shard {self.si + 1} of {len(self.files)} ({self.files[self.order[self.si % len(self.files)]]}), '
                f'window {self.wi:,}/{self.n_win:,}')


# ---------------------------------------------------------------- checkpoints

def find_ckpt(run):
    for name in ('ckpt', 'ckpt.new', 'ckpt.old'):
        p = os.path.join(run, name)
        if os.path.exists(os.path.join(p, 'state.json')):
            return p
    return None


def save_ckpt(run, model, opts, st):
    """Everything needed to resume: int8 latents (raw, 1 byte per weight), full-precision parameters,
    optimiser state, counters and the data cursor. state.json is written last and marks it complete."""
    new, cur, old = (os.path.join(run, n) for n in ('ckpt.new', 'ckpt', 'ckpt.old'))
    shutil.rmtree(new, ignore_errors=True)
    os.makedirs(new)
    FOLD.join()  # the running means are written as they stand after the last turn that ended
    groups = [(f'net{i}', net.signs(), {'fp': net.state_dict(), 'opt': opts[i].state_dict() if opts else None})
              for i, net in enumerate(model.nets)]
    if model.emb is not None:
        # Config.tiers: the table's files are embf.*, so that nothing written before takes its rows to be in id order
        groups.append(('embf', [model.emb], {'pos_of': model.emb.pos_of.cpu()}) if model.emb.tiers
                      else ('emb', [model.emb], {}))
    for name, signs, extra in groups:
        with open(os.path.join(new, name + '.i8'), 'wb') as f:
            for s in signs:
                s.store(release=False)
                s.cpu.numpy().tofile(f)
        if all(s.avg is not None for s in signs):  # the running means (_Fold), in the layout of the latents
            with open(os.path.join(new, name + '.avg.i8'), 'wb') as f:
                for s in signs:
                    s.avg.numpy().tofile(f)
        torch.save({**extra, 'R': [s.R.cpu() for s in signs], 'C': [s.C.cpu() for s in signs],
                    't': [s.t for s in signs],
                    'G': [None if s.G is None else (s.G.cpu(), s.gk.cpu()) for s in signs]},
                   os.path.join(new, name + '.pt'))
    ps_write(model, new)  # the ParScale add-on (side files) is in before state.json marks the checkpoint complete
    with open(os.path.join(new, 'state.json'), 'w') as f:
        json.dump(st, f)
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(cur):
        os.rename(cur, old)
    os.rename(new, cur)
    ps_saved(run, model, cur)


def keep_good(run):
    """A second name for the checkpoint just written (hard links, no copy). ckpt.good is the last one saved
    while the loss scale was whole and no step had been skipped since the save before: the one to go back to
    if the run degrades later, and the one the weekly backup uploads."""
    src, new, good = (os.path.join(run, n) for n in ('ckpt', 'ckpt.good.new', 'ckpt.good'))
    shutil.rmtree(new, ignore_errors=True)
    shutil.copytree(src, new, copy_function=os.link)
    shutil.rmtree(good, ignore_errors=True)
    os.rename(new, good)


def load_ckpt(model, opts, ck, avg=None):
    """avg: what to do with the running means of the latents (_Fold) if the checkpoint has them. None: nothing.
    'keep': into Sign.avg, to be carried on (the trainer). 'read': in place of the latents, so that whatever is
    done with the model reads the means (export, eval, generate). Returns how many groups had them."""
    found = 0
    groups = [(f'net{i}', net.signs()) for i, net in enumerate(model.nets)]
    if model.emb is not None:
        groups.append(('embf' if model.emb.tiers else 'emb', [model.emb]))
    for i, (name, signs) in enumerate(groups):
        d = torch.load(os.path.join(ck, name + '.pt'), map_location=DEV)
        if name == 'embf':  # Config.tiers: the row of every id
            model.emb.pos_of = d['pos_of'].to(DEV)
        if 'fp' in d:
            miss = model.nets[i].load_state_dict(d['fp'], strict=False)  # a buffer added since stays at its start
            if miss.missing_keys or miss.unexpected_keys:
                print(f'  {name}: not in the checkpoint {miss.missing_keys[:4]}.., not in the model '
                      f'{miss.unexpected_keys[:4]}..', flush=True)
            if opts and d['opt']:
                opts[i].load_state_dict(_opt_state(model.nets[i], opts[i], d['opt']))
        pa = os.path.join(ck, name + '.avg.i8')
        fa = open(pa, 'rb') if avg and os.path.exists(pa) else None
        found += fa is not None
        with open(os.path.join(ck, name + '.i8'), 'rb') as f:
            for s, R, C, t, G in zip(signs, d['R'], d['C'], d['t'], d.get('G') or [None] * len(signs)):
                s.cpu = _pin(torch.from_numpy(np.fromfile(f, dtype=np.int8, count=s.numel()).reshape(s.shape)))
                s.P = None
                s.R.copy_(R)
                s.C.copy_(C)
                s.t = list(t)
                s.G, s.gk = (None, None) if G is None else (G[0].to(DEV), G[1].to(DEV))
                if fa is not None:
                    mean = torch.from_numpy(np.fromfile(fa, dtype=np.int8, count=s.numel()).reshape(s.shape))
                    if avg == 'read':
                        s.cpu = _pin(mean)
                    else:
                        s.avg = mean
        if fa is not None:
            fa.close()
    return found


def write_json(path, obj):
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f)
    os.replace(path + '.tmp', path)


# ---------------------------------------------------------------- commands

def make_cfg(a):
    cfg = Config(**PRESETS[a.preset])
    cfg.act, cfg.dblocks, cfg.vocab, cfg.pos, cfg.cap = a.act, a.dblocks, a.vocab, a.pos, a.cap
    cfg.centre2, cfg.bias, cfg.cov = bool(a.centre2), bool(a.bias), bool(a.cov)
    cfg.tiers = a.tiers or ''
    if a.experts is not None:
        cfg.n_exp = a.experts
    if a.topk is not None:
        cfg.topk = a.topk
    return cfg


def cmd_info(a):
    cfg = make_cfg(a)
    set_vocab(cfg.vocab)
    L, E, D = cfg.n_layer, max(1, cfg.n_exp), cfg.dblocks
    tied = cfg.vocab != 'bytes'
    emb = VOCAB * cfg.d if tied else 0
    binary = L * (4 * cfg.d * cfg.d + 2 * E * cfg.d * cfg.ff) + emb
    fp = D * (cfg.d + (cfg.d + VOCAB if tied else 2 * VOCAB * cfg.d)) \
        + L * (2 * cfg.d + 4 * cfg.d + cfg.n_exp * cfg.d + E * (cfg.ff + cfg.d))
    print(f'preset {a.preset}: ' + describe(cfg, binary + fp, binary))
    print(f'  model file: {(binary / 8 + fp * 2) / 1e9:,.2f} GB (1 bit per weight packed, the rest fp16)')
    print(f'  training state: {binary / 1e9:,.2f} GB of int8 latents (1 byte per weight) + {fp * 12 / 1e6:,.0f} MB')
    print(f'  on the GPU at once: {((binary - emb) / D + emb) / 1e9:,.2f} GB of latents (one net of {D}'
          + (' + the token table' if tied else '') + '); fewer with --gpu-sublayers')


STOP = False
HOT = ('lr', 'sign_lr', 'sign_lr_nets', 'lr_scales', 'lr_router', 'lr_offsets', 'lr_bias', 'table_lr', 'w_noisy', 'aux', 'rotate', 'save_min', 'eval_every', 'log_every',
       't0', 'lr_floor', 'noisy', 'step_tokens', 'ref_tokens', 'lengths', 'val_long_every', 'eval_slow', 'probe',
       'avg_turns', 'sign_kick', 'sign_kick_nets', 'table_kick', 'avg_every', 'lr_nets', 'gcap', 'gcap_nets', 'wcap',
       'wcap_nets', 'gcap_mid', 'gcap_qkv', 'parscale_duty', 'parscale_lr', 'parscale_warmup', 'parscale_clip')


def _stop(*_):
    global STOP
    STOP = True


def _hot_updates(a, payload):
    """Validate a complete hot update before changing any live argument."""
    if not isinstance(payload, dict):
        raise ValueError('hot.json must contain a JSON object')
    updates = {}
    csv_keys = {'sign_lr_nets', 'lr_nets', 'sign_kick_nets', 'gcap_nets', 'wcap_nets'}
    integer_keys = {'rotate', 'eval_every', 'log_every', 'step_tokens', 'ref_tokens',
                    'val_long_every', 'eval_slow', 'probe', 'avg_turns', 'avg_every', 'parscale_warmup'}
    string_keys = csv_keys | {'lengths'}
    nonnegative = {'step_tokens', 'ref_tokens', 't0', 'noisy', 'save_min', 'probe',
                   'avg_turns', 'avg_every', 'eval_slow', 'val_long_every', 'parscale_warmup'}
    for key, value in payload.items():
        if key not in HOT:
            continue  # Preserve compatibility with notes and future-version keys.
        current = getattr(a, key)
        if key in integer_keys:
            if type(value) is not int:
                raise ValueError(f'{key} must be an integer')
        elif key in string_keys:
            if not isinstance(value, str):
                raise ValueError(f'{key} must be a string')
        else:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{key} must be a finite number')
        if key in {'log_every', 'eval_every', 'rotate'} and value < 1:
            raise ValueError(f'{key} must be at least 1')
        if key in nonnegative and value < 0:
            raise ValueError(f'{key} must be nonnegative')
        if key == 'parscale_duty' and not 0 <= value <= 1:
            raise ValueError('parscale_duty must be between 0 and 1')
        if key == 'lengths' and value:
            if any(int(part) < 2 for part in value.split(',')):
                raise ValueError('lengths must contain integers of at least 2')
        if key in csv_keys and value:
            if any(not math.isfinite(float(part)) for part in value.split(',')):
                raise ValueError(f'{key} must contain finite numbers')
        if current != value:
            updates[key] = value
    return updates


def hot_reload(a, seen_mtime):
    """Apply changed settings atomically after validating the complete update."""
    p = os.path.join(a.run, 'hot.json')
    try:
        m = os.path.getmtime(p)
        if m != seen_mtime:
            with open(p) as f:
                updates = _hot_updates(a, json.load(f))
            for key, value in updates.items():
                print(f'  hot.json: {key} {getattr(a, key)} -> {value}', flush=True)
            for key, value in updates.items():
                setattr(a, key, value)
        return m
    except (OSError, ValueError, TypeError, OverflowError) as e:
        if not isinstance(e, FileNotFoundError):
            print(f'  hot.json ignored: {e}', flush=True)
        return seen_mtime


def eval_period(a, gstep):
    """Steps between evaluations: --eval-every for the first 100 of them, then --eval-slow times as many."""
    return a.eval_every * (a.eval_slow if a.eval_slow > 1 and gstep >= 100 * a.eval_every else 1)


def sched(a, s, planned):
    warm = min(1.0, (s + 1) / a.warmup)
    if a.sched == 'cosine':
        return warm * (0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, s / max(1, planned)))))
    return warm * max(a.lr_floor, min(1.0, math.sqrt(a.t0 / (s + 1))))


def _resume_refuse(run, reason):
    """Fail closed only for an explicitly required resume; never overwrite a peer pause."""
    os.makedirs(run, exist_ok=True)
    message = '[resume] REFUSED: ' + reason
    with open(os.path.join(run, 'HALTED'), 'w') as f:
        f.write(message + '\n')
    try:
        with open(os.path.join(run, 'STOP'), 'x') as f:
            f.write(message + '\n')
    except FileExistsError:
        pass
    print(message, flush=True)
    raise SystemExit(6)


def _check_required_resume(a, ck):
    if getattr(a, 'require_resume_checkpoint', False) and (not a.resume or ck is None):
        _resume_refuse(a.run, 'required checkpoint is missing; refusing a new step-zero run')


def _capture_training_runtime(rng, data, turn_steps, good, streak):
    # Drain averaging before reading its rounding cursor; save_ckpt also drains before writing weights.
    FOLD.join()
    return {'version': 1, 'numpy_rng': rng.bit_generator.state,
            'torch_cpu_rng': torch.get_rng_state().tolist(),
            'torch_cuda_rng': [s.tolist() for s in torch.cuda.get_rng_state_all()] if CUDA else [],
            'random_text_rng': data.rng.bit_generator.state if isinstance(data, RandomText) else None,
            'fused_calls': FUSED.calls if FUSED is not None else None, 'fold_calls': FOLD.calls,
            'turn_steps': turn_steps, 'good_steps': good, 'overflow_streak': streak}


def _restore_training_runtime(saved, rng, data):
    if not saved:
        return 0, 0, 0
    if saved.get('version') != 1:
        raise ValueError('unsupported training runtime version')
    fields = ('turn_steps', 'good_steps', 'overflow_streak', 'fold_calls')
    for key in fields:
        if type(saved.get(key)) is not int or saved[key] < 0:
            raise ValueError('invalid runtime counter: ' + key)
    fused = saved.get('fused_calls')
    if fused is not None and (type(fused) is not int or fused < 0):
        raise ValueError('invalid fused rounding counter')
    if (fused is None) != (FUSED is None):
        raise ValueError('checkpoint rounding backend differs from this process')
    # Validate with temporary generators before touching the live RNG streams.
    trial = np.random.default_rng()
    trial.bit_generator.state = saved['numpy_rng']
    cpu = torch.tensor(saved['torch_cpu_rng'], dtype=torch.uint8, device='cpu')
    torch.Generator(device='cpu').set_state(cpu)
    cuda = [torch.tensor(x, dtype=torch.uint8, device='cpu') for x in saved['torch_cuda_rng']]
    if len(cuda) != (torch.cuda.device_count() if CUDA else 0):
        raise ValueError('checkpoint CUDA generator count differs from this process')
    for i, x in enumerate(cuda):
        torch.Generator(device='cuda:' + str(i)).set_state(x)
    drng = saved.get('random_text_rng')
    if isinstance(data, RandomText):
        if drng is None:
            raise ValueError('checkpoint is missing RandomText sampler state')
        trial.bit_generator.state = drng
    FOLD.join()
    rng.bit_generator.state = saved['numpy_rng']
    torch.set_rng_state(cpu)
    if CUDA:
        torch.cuda.set_rng_state_all(cuda)
    if isinstance(data, RandomText):
        data.rng.bit_generator.state = drng
    if FUSED is not None:
        FUSED.calls = fused
    FOLD.calls = saved['fold_calls']
    return saved['turn_steps'], saved['good_steps'], saved['overflow_streak']


def cmd_train(a):
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.run, exist_ok=True)
    _lock_run(a.run)  # D8: one trainer or refresh at a time on a run directory
    ck = find_ckpt(a.run) if a.resume else None
    _check_required_resume(a, ck)
    st = json.load(open(os.path.join(ck, 'state.json'))) if ck else {}
    cfg = Config(**st['cfg']) if ck else make_cfg(a)
    set_vocab(cfg.vocab)
    D = cfg.dblocks
    exclusive = a.resident == 'one'
    model = Model(cfg)
    params = [[p for n, p in net.named_parameters() if n != 'b_out'] for net in model.nets]  # what AdamW steps
    opts = [torch.optim.AdamW(net.groups(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0) for net in model.nets]
    if ck:
        load_ckpt(model, opts, ck, 'keep')  # with the running means, if it has them (dropped below if --avg-turns is 0)
    else:
        model.init_signs()
    # ParScale, built in and on (see ParAdapter): every net's add-on from the checkpoint's side files, else fresh
    frozen = bool(a.parscale_freeze_backbone)
    if frozen and not ck:
        raise SystemExit(f'--parscale-freeze-backbone trains the ParScale add-on of a checkpoint: there is none in {a.run}')
    ps_gen = torch.Generator().manual_seed(a.seed * 1000003 + _PS_SALT)  # its own random numbers, never the run's
    model.ps_meta, ps_opt = load_ps(model, ck, a.run, a, ps_gen)
    model.popts = popts = [ps_optimizer(net.ps, a) for net in model.nets]
    for o, sd in zip(popts, ps_opt or []):
        if sd:
            try:
                o.load_state_dict(sd)
            except (ValueError, KeyError, RuntimeError) as e:
                print(f'  parscale: optimiser state not loaded ({e}); it starts again', flush=True)
    PS_ON = a.parscale > 1 and (a.parscale_duty > 0 or frozen)
    for net in model.nets:
        net.ps.live = PS_ON
    if frozen:
        ps_freeze(model, a.run)
    ps_fit, ps_redo, ps_sub = a.parscale, None, None  # ps_fit: the most streams a step holds here (see ps_probe)
    acc_ps, acc_ps_n, ps_gain = torch.zeros(D, 2, device=DEV), [0] * D, [None] * D  # the merged losses, for the log
    print(ps_line(model, a, PS_ON, frozen), flush=True)
    total, binary = model.counts()
    print(f'model: {describe(cfg, total, binary)}')
    print(f'device: {torch.cuda.get_device_name(0) if CUDA else "cpu"}')

    if a.data.startswith('hfds:'):
        data = ShardStream(a.data, os.path.join(a.run, 'data'), a.seed, a.window, st.get('data'), a.mix_windows)
        vpath = os.path.join(a.run, 'val.bin')
        if not os.path.exists(vpath) or 'data' not in st:  # a fresh stream: step over the validation windows
            v = data.take_val(a.val_windows)
            if not os.path.exists(vpath):
                v.tofile(vpath)
        vraw = np.fromfile(vpath, dtype=id_dtype())
    else:
        ids = load_ids(a.data, a.data_mb)
        if a.val_data:
            vraw = load_ids(a.val_data, a.data_mb)
        else:
            split = int(len(ids) * 0.95)
            ids, vraw = ids[:split], ids[split:]
        data = RandomText(ids, a.seed)
    math_replay = open_math_replay(a, st)
    vlen = a.val_len or cfg.ctx
    vrows = min(a.val_rows, len(vraw) // vlen)
    vstep = max(1, len(vraw) // vlen // vrows)
    val = torch.from_numpy(np.stack([vraw[i * vstep * vlen:(i * vstep + 1) * vlen] for i in range(vrows)])).long()
    print(f'data: {a.data} -- {data.where()}; validation {val.numel():,} {unit()}')
    vlong = None
    if a.val_long:  # a few rows longer than any training row: does it read past the training length
        lnear = max(int(x) for x in str(a.lengths).split(',')) if a.lengths else cfg.ctx
        lfar = min(a.val_long, cfg.ctx, a.window if a.data.startswith('hfds:') else a.val_long)  # inside one window
        ln = min(a.val_long_rows, len(vraw) // lfar)
        if lfar > lnear and ln:
            lk = max(1, len(vraw) // lfar // ln)
            vlong = torch.from_numpy(np.stack([vraw[i * lk * lfar:(i * lk + 1) * lfar] for i in range(ln)])).long()
            print(f'past the training length: {ln} rows of {lfar:,} {unit()}, every {a.val_long_every} evaluations')
    if not ck and model.emb is not None and (a.bias_init == 'unigram' or cfg.tiers):
        bias, h1 = unigram_bias(data.sample())
        if cfg.tiers:  # the table in order of frequency; an entry starts at the log of its group's share of the text
            emb = model.emb
            emb.pos_of = table_order(bias.numpy(), cfg.tiers).to(DEV)
            tb = torch.empty_like(bias)
            tb[emb.pos_of.cpu()] = bias
            for k, (c0, c1) in enumerate(emb.spans[1:]):
                tb[int(emb.entry[k])] = torch.logsumexp(tb[c0 * VCHUNK:c1 * VCHUNK], 0)
            bias = tb
        if a.bias_init != 'unigram':
            bias = torch.zeros_like(bias)
        for net in model.nets:
            net.b_out.data.copy_(bias)
        print(f'output bias starts at each token\'s log frequency; frequency alone scores {h1:.3f} nats')

    steps, seen = st.get('steps', [0] * D), st.get('seen', 0)
    # the ids each net has been trained on (a checkpoint from before they were counted: its share of all ids, by steps)
    tok = st.get('tok') or [seen * s // max(1, sum(steps)) for s in steps]
    S, good, skipped = st.get('scale', 65536.0 if CUDA else 1.0), 0, st.get('skipped', 0)
    b = st.get('active', 0)
    model.activate(b, exclusive, a.gpu_sublayers)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    gstep = sum(steps)
    g0, planned, t0 = gstep, a.max_steps, time.time()
    last_save = tl = t0
    # hot.json is read before the first step as well: what it sets holds from the first step after a restart
    seen_l, last_eval, hot_m, rate_avg = seen, st.get('eval', {}), hot_reload(a, None), None
    acc, acc_n = torch.zeros(D, 2, device=DEV), [0] * D  # per net: the two losses summed since its last log line
    streak = 0  # steps skipped in a row
    tok_cap = 1 << 62  # the most ids a step may hold: lowered when a step runs out of GPU memory (below)
    again = False  # the step before was dropped for lack of GPU memory: this is the same step once more
    skip_mark = skipped  # skipped steps at the last periodic checkpoint
    seen0 = seen  # --parscale-freeze-backbone: its budget counts from here
    ps_key = ps_fit_key(a, cfg, model.ps_meta) if PS_ON else None  # D6: every setting that changes what a step holds
    if PS_ON and CUDA:  # the most streams a step of this run holds on this GPU, measured once per setting (ps_probe)
        ps_fit = model.ps_meta.setdefault('p_fit', {}).get(ps_key) or ps_probe(model, b, val, a, cfg)
        model.ps_meta['p_fit'][ps_key] = ps_fit
        print(f'parscale: {ps_fit} streams fit a step on this GPU', flush=True)
    elif PS_ON and (model.ps_meta.get('p_fit') or {}).get(ps_key):  # D6: a halving kept from an earlier start
        ps_fit = model.ps_meta['p_fit'][ps_key]
        print(f'parscale: {ps_fit} streams fit a step here (kept from an earlier start)', flush=True)
    why = ps_refusal(a, cfg, ps_fit) if PS_ON else None
    if why:  # D5: said now, and in every step's log line and in status.json while it holds
        print(f'{why}; those steps are taken as plain steps', flush=True)
        if frozen and why.startswith('PARSCALE NOT RUNNING:'):
            raise SystemExit(f'--parscale-freeze-backbone: {why}')
    u = unit()
    try:
        turn_steps, good, streak = _restore_training_runtime(st.get('runtime'), rng, data)
    except (ValueError, TypeError, KeyError, RuntimeError) as e:
        _resume_refuse(a.run, 'invalid runtime state: ' + str(e))
    print('[resume] runtime RNG/rounding/rotation state restored' if st.get('runtime') else
          '[resume] legacy/fresh checkpoint without runtime state; initializing once (not exact replay)', flush=True)
    print(f'gradient norm native fp32 accumulation: {_GCAP_NORM} (BINTF_GCAP_NORM)', flush=True)
    print(f'training from step {gstep:,} ({seen / 1e9:.4f}B {u} seen)'
          + (f', target {a.target_tokens / 1e9:g}B {u}' if a.target_tokens else ''), flush=True)

    def state():
        return {'cfg': asdict(cfg), 'steps': steps, 'tok': tok, 'seen': seen, 'scale': S, 'skipped': skipped, 'active': b,
                'data': data.state(), 'eval': last_eval, 'time': time.time(), 'args': vars(a),
                'math_replay': math_replay.state() if math_replay is not None else None,
                'runtime': _capture_training_runtime(rng, data, turn_steps, good, streak)}

    # D5: a ParScale step in sub-steps (ps_sub) is finished before the loop looks at anything else
    while ps_sub is not None or (not STOP and gstep < planned and not (a.target_tokens and seen >= a.target_tokens)
                                 and not (frozen and seen - seen0 >= a.parscale_refresh_tokens)):
        if a.minutes and time.time() - t0 > a.minutes * 60 and a.sched != 'cosine' and ps_sub is None:
            break
        if D > 1 and turn_steps >= a.rotate and not again and ps_sub is None:
            turn_steps = 0
            pr = val[:a.probe] if a.probe and not frozen else None
            if pr is not None:  # the net that hands over, on fixed rows, as its turn ends
                probe_log(a.run, gstep, b, 'end', probe_ar(model, b, pr, a.eval_bs))
            left, b = b, (b + 1) % D
            model.activate(b, exclusive, a.gpu_sublayers)
            if frozen:  # --parscale-freeze-backbone: the latents do not move, and neither do their running means
                pass
            elif a.avg_turns and a.gpu_sublayers < 0:  # the net that has left goes into its running means (_Fold)
                # on every avg_every-th of its turns; 0: one more for every 60,000 steps of the run, from 120,000 on
                every = a.avg_every if a.avg_every > 0 else max(1, min(32, gstep // 60000))
                if (steps[left] // max(1, a.rotate)) % every == 0:
                    model.fold(left, max(2, min(128, round(256 / a.avg_turns))))
            elif not a.avg_turns and any(s.avg is not None for s in model.all_signs()):  # switched off: no stale means
                FOLD.join()
                for s in model.all_signs():
                    s.avg = None
            if pr is not None:  # and the net that takes over, before its first step
                probe_log(a.run, gstep, b, 'start', probe_ar(model, b, pr, a.eval_bs))
        again = False
        net, opt = model.nets[b], opts[b]
        lengths = [int(x) for x in str(a.lengths).split(',')] if a.lengths else [cfg.ctx]
        rs = rng.bit_generator.state if PS_ON or PS_TRACE else None  # a ParScale step out of memory restarts here
        if ps_sub is not None:  # D5: the next sub-step of this ParScale step: its next rows; the draws go on from here
            l, Pst = ps_sub['l'], ps_sub['P']
            k = min(ps_sub['k'] - ps_sub['r0'], ps_sub['rows'])
            clean = ps_sub['clean'][ps_sub['r0']:ps_sub['r0'] + k]
        else:
            l = lengths[int(rng.integers(len(lengths)))]
            if ps_redo is not None:  # the same rows and random draws once more, with fewer streams
                (clean, k), ps_redo = ps_redo, None
            else:
                k = max(1, min(a.step_tokens or 32 * cfg.ctx, tok_cap) // l)
                math_hit = math_replay is not None and math_replay.hit(gstep)
                raw_clean = math_replay.rows(k, l, gstep) if math_hit else data.rows(k, l)
                clean = torch.from_numpy(raw_clean).to(DEV).long()
                if math_hit:
                    print(f'[math-replay] step={gstep} rows={k} length={l}', flush=True)
            # ParScale: P streams on a fixed share of each net's steps (no random number is drawn for it), else plain
            Pst = min(a.parscale, ps_fit) if PS_ON and (frozen or duty_hit(steps[b], a.parscale_duty)) else 1
            if Pst > 1:  # D5: as many streams as a row can have in --parscale-max-pos positions, in sub-steps of rows
                Pst, prow = ps_plan(k, l, Pst, ps_maxpos(a, cfg))  # that hold no more than that, every stream counted
                if Pst > 1:
                    ps_sub = {'clean': clean, 'k': k, 'l': l, 'r0': 0, 'rows': prow, 'P': Pst, 'rs': rs, 'n': 0}
                    clean, k = clean[:prow], min(k, prow)
            Pst = Pst if Pst > 1 else 1
        # --ref-tokens: the stated steps are for a step of that many ids. The int8 step and AdamW's are taken on a
        # gradient divided by its own size, and one step's update is mostly noise (5% of its energy is signal at
        # 2,048 ids, measured), so n / ref batches in one step give sqrt(n / ref) times the signal for the same
        # noise: sqrt(n / ref) times the step is then the same signal and the same noise per token. The schedule
        # counts ref ids as one step, so it too is the same per token.
        ref = a.ref_tokens
        bs = math.sqrt(k * l / ref) if ref else 1.0
        SO.cb = _CB ** (k * l / ref) if ref else _CB  # the running means over tokens: the same memory in tokens
        mult = sched(a, tok[b] / ref if ref else steps[b], planned / D) * bs
        lrx = {'gain': 1.0, 'scale': a.lr_scales, 'router': a.lr_router, 'bias': a.lr_offsets}
        lrs = [float(v) for v in str(a.lr_nets).split(',')] if a.lr_nets else []
        own_lr = lrs[b] if b < len(lrs) and lrs[b] > 0 else 1.0  # this net's multiple of --lr
        for g in opt.param_groups:
            g['lr'] = a.lr * own_lr * mult * lrx[g['kind']]
        per_net = [float(v) for v in str(a.sign_lr_nets).split(',')] if a.sign_lr_nets else []
        own = per_net[b] if b < len(per_net) and per_net[b] > 0 else 1.0  # this net's multiple of --sign-lr
        SO.a = a.sign_lr * own * mult / SO.q
        SO.inv_scale, SO.table = 1.0 / S, a.table_lr / own  # the table is shared: its step does not follow the net
        kicks = [float(v) for v in str(a.sign_kick_nets).split(',')] if a.sign_kick_nets else []
        kap = kicks[b] if b < len(kicks) else a.sign_kick  # in steps, so the units follow the step wherever it goes
        SO.kick = int(round(kap * SO.a)) if kap > 0 else 0
        SO.kick_table = int(round(a.table_kick * SO.a * SO.table)) if a.table_kick > 0 else 0
        gcs = [float(v) for v in str(a.gcap_nets).split(',')] if a.gcap_nets else []
        SO.gcap = gcs[b] if b < len(gcs) else a.gcap
        wcs = [float(v) for v in str(a.wcap_nets).split(',')] if a.wcap_nets else []
        SO.wcap = wcs[b] if b < len(wcs) else a.wcap
        SO.gcap_mid, SO.gcap_qkv = a.gcap_mid, a.gcap_qkv
        SO.bad_t.zero_()
        SO.count = (gstep + 1) % a.log_every == 0
        if SO.count and (ps_sub is None or ps_sub['r0'] == 0):  # (D5: counted over every sub-step of a step)
            SO.flips_t.zero_()
            SO.seen = 0
        if a.prof:
            torch.cuda.synchronize()
            tp = time.time()
        if Pst > 1 and any(net.ps.fresh):  # D3: the add-on's first ParScale step: its prefixes from this step's rows
            ps_init_data(net, ps_sub['clean'] if ps_sub is not None else clean, ps_gen)
        net.P, net.ps.want_h0 = Pst, Pst > 1 and ps_gain[b] is None  # stream 0 alone: once per log line of a net
        SO.learn = True
        full = False
        try:
            if _PS_FAKE_OOM and Pst >= _PS_FAKE_OOM[0] and ps_sub is not None and ps_sub['n'] >= (_PS_FAKE_OOM[1:] or [0])[0]:
                raise torch.cuda.OutOfMemoryError('BINTF_PS_FAKE_OOM')  # tests (D6): the out-of-memory path on a CPU
            with torch.autocast('cuda', dtype=torch.float16, enabled=CUDA):
                ar, nz, aux, share = forward_loss(net, clean, model.band(b), int(l * a.noisy) // 2 * 2, rng)
                # per target: a masked token weighs w_noisy times a next-token one. (As two plain means the few
                # masked tokens weighed 12 times as much each and supplied nine tenths of the gradient noise.)
                loss = ar + a.w_noisy * share * nz + a.aux * aux
            SO.learn = False
            if a.prof:
                torch.cuda.synchronize()
                tf = time.time()
            net.zero_grad(set_to_none=True)
            fine = bool(torch.isfinite(loss))  # False: the forward pass overflowed, so there is nothing to back-propagate
            if fine:
                (loss * S).backward()
        except torch.cuda.OutOfMemoryError:
            full = True
        net.P = 1
        if full and Pst > 1:  # a ParScale step out of GPU memory: the same rows and draws again with half the streams
            SO.learn, SO.n0 = False, None
            ar = nz = aux = loss = None
            net.zero_grad(set_to_none=True)
            net.ps.zero_grad()
            torch.cuda.empty_cache()
            ps_fit, again = Pst // 2, True
            if ps_key is not None:  # D6: kept in parscale.json, so the next start does not try more streams again
                model.ps_meta.setdefault('p_fit', {})[ps_key] = ps_fit
            rng.bit_generator.state = rs  # this (sub-)step's draws are made again
            if ps_sub['r0'] == 0:  # nothing of the step taken yet: all of it again (its row length is drawn again too)
                ps_redo, ps_sub = (ps_sub['clean'], ps_sub['k']), None
            else:  # D5: the sub-steps taken stand; the rest of the rows go on with fewer streams, or as a plain step
                rest = ps_sub['k'] - ps_sub['r0']
                P2, prow = ps_plan(rest, l, min(a.parscale, ps_fit), ps_maxpos(a, cfg)) if ps_fit > 1 else (1, rest)
                ps_sub['P'], ps_sub['rows'] = (P2, prow) if P2 > 1 else (1, rest)
            print(f'  no GPU memory for {Pst} ParScale streams at step {gstep:,}: the step is taken again with '
                  f'{ps_fit if ps_fit > 1 else "one stream (plain)"}, and so are the steps after it', flush=True)
            continue
        if full:  # outside the handler, so the failed pass has let go of its tensors
            # The step is dropped: its rows are not counted as seen, and the matrices its backward pass had already
            # reached keep their update (each update is whole). Steps hold half as many ids from here on; one row
            # that does not fit is not something a smaller step can cure, so that still ends the run.
            SO.learn = False
            ar = nz = aux = loss = None
            net.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if k == 1:
                raise SystemExit(f'no GPU memory for one row of {l:,} {u} at step {gstep:,}')
            tok_cap, again, ps_sub = max(l, k * l // 2), True, None  # (D5: and the rest of a ParScale step)
            print(f'  no GPU memory for {k} rows of {l:,} at step {gstep:,}: the step is dropped and steps hold at most '
                  f'{tok_cap:,} {u} until this process ends (hot.json step_tokens above that waits for a restart)',
                  flush=True)
            continue
        if a.prof:
            torch.cuda.synchronize()
            print(f'    prof: rows {k} x {l}: forward {tf - tp:.3f}s backward {time.time() - tf:.3f}s, '
                  f'peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB', flush=True)
        SO.n0 = None
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        pgs = [t.grad for t in net.ps.params() if t.grad is not None] if Pst > 1 else []  # the add-on's
        allg = grads + pgs
        if fine and int(SO.bad_t) == 0 and (not allg or bool(torch.isfinite(sum(g.sum() for g in allg)))):
            if not frozen:
                for g in grads:
                    g.mul_(1.0 / S)
                torch.nn.utils.clip_grad_norm_(params[b], 1.0)
                opt.step()
                if model.emb is not None:  # SGD on the output bias: a token that is not in the batch barely moves
                    net.b_out.data.add_(net.b_out.grad, alpha=-a.lr_bias * mult * bs)  # plain SGD: n / ref in all
            good += 1
            streak = 0
            if Pst > 1:  # the add-on's own AdamW step; the merged losses are logged apart from the plain ones
                ps_update(net.ps, popts[b], pgs, S, a, bs)
                net.ps.tok += k * l
                model.ps_meta['P_trained'] = max(model.ps_meta['P_trained'], Pst)
                acc_ps[b] += torch.stack((ar.detach().float(), nz.detach().float()))
                acc_ps_n[b] += 1
                if net.ps.h0 is not None:  # a log step: what stream 0 alone scores on the same rows
                    ps_gain[b] = ps_stream0(net, clean, l, ar)
            else:
                acc[b] += torch.stack((ar.detach().float(), nz.detach().float()))
                acc_n[b] += 1
            if CUDA and good % 200 == 0:  # (2000 let a few overflows pin the scale low for hours)
                S = min(S * 2.0, 65536.0)
        elif fine:  # fp16 overflow somewhere in this backward pass: back the loss scale off
            S, good, skipped, streak = max(1.0, S / 2.0), 0, skipped + 1, streak + 1
        else:  # the forward pass overflowed: the loss scale has no part in that, and no sign was updated
            skipped, streak = skipped + 1, streak + 1
        net.ps.zero_grad()
        net.ps.h0 = None
        ps_last = None
        if ps_sub is not None:  # D5: one optimiser step per sub-step; the step's counters, trace and log after its last
            ps_sub['r0'] += k
            ps_sub['n'] += 1
            ps_sub.setdefault('bs', []).append(round(bs, 6))
            if ps_sub['r0'] < ps_sub['k']:
                ps_last = 'more'
            else:
                ps_last, clean, k, rs, ps_sub = ps_sub, ps_sub['clean'], ps_sub['k'], ps_sub['rs'], None
        if PS_TRACE and ps_last != 'more':
            ps_trace(PS_TRACE, gstep, b, steps[b], Pst, clean, rs, rng, data, ps_last)
        # The scale halves at an overflow and doubles after 200 clean steps; the largest entry of a rare batch is
        # thousands of times an ordinary one's, so a few halvings in a row do happen. 2,048 times down they no longer
        # fit half precision at either end and every further step damages the model.
        if CUDA and S <= 65536.0 / 2048:
            print(f'loss scale down to {S:g} at step {gstep:,} ({skipped} steps skipped): stopping without a checkpoint. '
                  f'{a.run}/ckpt.good is the last checkpoint saved at full scale; ckpt and ckpt.old may be later and worse.',
                  flush=True)
            raise SystemExit(5)
        if streak >= 100:  # something is non-finite for good; a checkpoint now would keep it
            print(f'100 steps skipped in a row at step {gstep:,} (forward loss finite: {fine}); stopping without a '
                  f'checkpoint so the last good one is what restarts', flush=True)
            raise SystemExit(4)
        if ps_last == 'more':  # D5: the next sub-step of this ParScale step
            again = True
            continue
        steps[b] += 1
        tok[b] += k * l
        gstep += 1
        turn_steps += 1
        seen += k * l
        if a.minutes and a.sched == 'cosine':  # plan the cosine over the time budget, as bintf.py does
            if gstep - g0 == 20:
                t20 = time.time()
            if gstep - g0 == 60:
                per = (time.time() - t20) / 40
                planned = min(a.max_steps, max(gstep + 1, gstep + int((a.minutes * 60 - (time.time() - t0)) / per)))
                print(f'{per:.3f} s/step -> {planned - g0} steps in {a.minutes:g} min', flush=True)
        if gstep % a.log_every == 0:
            now = time.time()
            rate = (seen - seen_l) / (now - tl)
            rate_avg = rate if rate_avg is None else 0.95 * rate_avg + 0.05 * rate  # smooths over swaps, evals, saves
            tl, seen_l = now, seen
            flips = float(SO.flips_t) / max(1, SO.seen) * 100 if SO.count else float('nan')
            mem = torch.cuda.max_memory_allocated() / 1e9 if CUDA else 0.0
            # this net's means over its own steps since its last line
            ar_m, nz_m = (acc[b] / acc_n[b]).tolist() if acc_n[b] else (float('nan'), float('nan'))
            head_bad = int(SO.head_bad_t)
            acc[b].zero_()
            SO.head_bad_t.zero_()
            acc_n[b] = 0
            gm, SO.gmax = min(SO.gmax, 1e30), 0.0
            xs = (SO.xs_t[:2] / SO.xs_t[2].clamp(min=1)).tolist()  # the last residual stream: mean and largest rms
            xc = float(SO.xc_t / SO.xs_t[2].clamp(min=1))  # and the rms of the part every position of it shares
            br = (SO.br_t[:2] / SO.br_t[2].clamp(min=1)).tolist()  # branches: rms before the bound, share at it
            SO.xs_t.zero_()
            SO.xc_t.zero_()
            SO.br_t.zero_()
            gc = float(SO.gc_t[0] / SO.gc_t[1].clamp(min=1))  # --gcap: share of positions whose gradient was held back
            SO.gc_t.zero_()
            wc = float(SO.wc_t[0] / SO.wc_t[1].clamp(min=1))  # --wcap: share of rows a matrix's gradient had scaled back
            SO.wc_t.zero_()
            gmid = float(SO.gm_t[0] / SO.gm_t[1].clamp(min=1))  # --gcap-mid, --gcap-qkv: the shares held back inside the layers
            gqkv = float(SO.gz_t[0] / SO.gz_t[1].clamp(min=1))
            SO.gm_t.zero_()
            SO.gz_t.zero_()
            rec = {'step': gstep, 'net': b, 'net_step': steps[b], 'ids': k * l, 'seen': seen, 'ar': ar_m, 'masked': nz_m,
                   'lr_mult': mult, 'flips_pct': flips, 'rate': rate, 'scale': S, 'skipped': skipped, 'gpu_gb': mem,
                   'head_bad': head_bad, 'gmax': gm, 'x_rms': xs[0], 'x_max': xs[1], 'x_common': xc, 'br_rms': br[0],
                   'br_capped': br[1], 'time': now, 'kick': SO.kick, 'kick_table': SO.kick_table,
                   'gcap': SO.gcap, 'gcap_share': gc, 'wcap': SO.wcap, 'wcap_share': wc,
                   'gcap_mid': SO.gcap_mid, 'gcap_mid_share': gmid, 'gcap_qkv': SO.gcap_qkv, 'gcap_qkv_share': gqkv}
            if PS_ON:  # ParScale: this net's merged losses since its last line, and how its streams are weighted
                rec['ps'] = ps_log(net.ps, model.ps_meta, a, ps_fit, acc_ps[b], acc_ps_n[b], ps_gain[b])
                why = ps_refusal(a, cfg, ps_fit)
                if why:
                    rec['parscale_not_running'] = why
                acc_ps[b].zero_()
                acc_ps_n[b], ps_gain[b] = 0, None
            print(f'step {gstep:>8,} net {b} | {seen / 1e9:9.4f}B {u} | next {lossf(ar_m):.3f} '
                  f'masked {lossf(nz_m):.3f} {lossu()} | lr x{mult:.3f} flips {flips:.3f}% | '
                  f'{rate:,.0f} {u}/s | scale {S:g} grad {gm:.2g} | stream {xs[0]:.1f}/{xs[1]:.0f} common {xc:.1f}'
                  + (f' branch {br[0]:.2f} ({br[1] * 100:.0f}% bounded)' if cfg.cap else '') + f' | gpu {mem:.2f} GB'
                  + (f' | {head_bad} token-table overflows' if head_bad else '')
                  + (f' | {rec["parscale_not_running"]}' if 'parscale_not_running' in rec else ''), flush=True)
            if PS_ON and rec['ps']['n']:  # a line when this net took ParScale steps since its last one
                print(ps_log_line(b, rec['ps']), flush=True)
            with open(os.path.join(a.run, 'metrics.jsonl' if not frozen else 'parscale_refresh.jsonl'), 'a') as f:
                f.write(json.dumps(rec) + '\n')
            eta = (a.target_tokens - seen) / max(rate_avg, 1e-9) / 86400 if a.target_tokens else None
            write_json(os.path.join(a.run, 'status.json' if not frozen else 'parscale_refresh_status.json'), {
                **rec, 'rate_avg': rate_avg, 'steps': steps, 'target': a.target_tokens, 'eta_days': eta,
                'eval': last_eval, 'unit': u,
                'loss_unit': lossu(), 'bits': VNAME == 'bytes', 'data': data.where(),
                'model': describe(cfg, total, binary), 'started': t0, 'last_save': last_save, 'pid': os.getpid()})
            hot_m = hot_reload(a, hot_m)
        E = eval_period(a, gstep)
        if (gstep % E == 0 or gstep == planned) and not frozen:
            eb = (gstep // E) % D  # the nets take turns at being scored, whichever one is training
            if eb != b:
                model.activate(eb, exclusive, a.gpu_sublayers)
            ev = evaluate(model, eb, val, a.eval_bs)
            last_eval[str(eb)] = {**ev, 'step': gstep, 'seen': seen,
                                  **{kk: vv for kk, vv in last_eval.get(str(eb), {}).items() if kk == 'long'}}
            print(f'  eval net {eb} @ {gstep:,}: {eval_line(ev)}', flush=True)
            joint = together(a.run, eb, D, last_eval)
            if joint is not None:
                last_eval[str(eb)]['together'] = joint
                print(f'  the {D} nets together (the others as they were last scored): next-{unit()[:-1]} '
                      f'{lossf(joint):.3f} {lossu()}', flush=True)
            means = bool(a.avg_turns) and a.gpu_sublayers < 0 and model.has_avg()
            if means:  # the same rows, read from the running means of the latents (_Fold): what a reader gets
                ev2 = None
                try:
                    if CUDA:
                        torch.cuda.empty_cache()  # the means of one net and of the table go up beside the latents: 1.6 GB
                    with model.reading(eb):
                        ev2 = evaluate(model, eb, val, a.eval_bs)
                except torch.cuda.OutOfMemoryError:
                    globals()['ZF'] = None  # evaluate() did not reach its end
                    NLLS.pop(eb, None)
                if ev2 is None:  # outside the handler, so the failed pass has let go of its tensors
                    torch.cuda.empty_cache()
                    means = False
                    print('  read from the running means of the latents: skipped, no GPU memory', flush=True)
            if means:
                last_eval[str(eb)]['avg'] = {'ar': ev2['ar'], 'ar_tail': ev2['ar_tail'], 'step': gstep}
                j2 = together(a.run, eb, D, last_eval, 'avg')
                if j2 is not None:
                    last_eval[str(eb)]['avg']['together'] = j2
                print(f'  read from the running means of the latents: next-{unit()[:-1]} {lossf(ev2["ar"]):.3f} {lossu()} '
                      f'(last 32 positions {lossf(ev2["ar_tail"]):.3f})'
                      + (f'; the {D} nets together {lossf(j2):.3f}' if j2 is not None else ''), flush=True)
            if PS_ON and model.ps_meta['P_trained'] > 1:  # the same rows with the trained streams merged
                Pe = min(a.parscale, ps_fit, model.ps_meta['P_trained'])
                evp = ps_evaluate(model, eb, val, a.eval_bs, Pe) if Pe > 1 else None
                if evp is not None:
                    last_eval[str(eb)]['ps'] = {'P': Pe, 'ar': evp['ar'], 'ar_tail': evp['ar_tail'], 'step': gstep}
                    print(f'  ParScale x{Pe}, net {eb}: {eval_line(evp)} (plain {lossf(ev["ar"]):.3f})', flush=True)
            if vlong is not None and (gstep // E) % max(1, a.val_long_every) == 0:
                evl = None
                try:
                    evl = evaluate_long(model, eb, vlong, lnear, min(512, lnear // 2))
                except torch.cuda.OutOfMemoryError:
                    pass
                if evl is None:  # outside the handler, so the failed pass has let go of its tensors
                    torch.cuda.empty_cache()
                    print(f'  past the training length: skipped, no GPU memory for rows of {vlong.shape[1]:,}', flush=True)
                else:
                    last_eval[str(eb)]['long'] = {**evl, 'step': gstep}
                    print(f'  past the training length, net {eb}: {long_line(evl)}', flush=True)
                if CUDA:  # the log's GPU memory figure stays the training peak
                    torch.cuda.reset_peak_memory_stats()
            if a.eval_sample:  # a short AR continuation from this net, so the log shows what it writes
                with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if CUDA else []):
                    torch.manual_seed(gstep)
                    out, pr = [], encode(a.eval_prompt)
                    with (model.reading(eb) if means else contextlib.nullcontext()):  # from the means, if they are kept
                        for _ in range(a.eval_sample):
                            out.append(int(_sample(run_seq(model, eb, (pr + out)[-cfg.ctx:]), a)[0][0]))
                    last_eval[str(eb)]['sample'] = a.eval_prompt + show(out)
                    print(f'  sample net {eb}: {a.eval_prompt}|{show(out)}', flush=True)
            with open(os.path.join(a.run, 'evals.jsonl'), 'a') as f:
                f.write(json.dumps({'net': eb, **last_eval[str(eb)]}) + '\n')
            if eb != b:
                model.activate(b, exclusive, a.gpu_sublayers)
            if CUDA:  # the long rows grew the allocator's pool to 5.6 GB of the card's 6; training needs 4.2
                torch.cuda.empty_cache()
        if frozen and a.save_min and time.time() - last_save > a.save_min * 60 and streak == 0:
            ps_save_inplace(a.run, model, ck)
            last_save = time.time()
            print(f'  ParScale add-on written at step {gstep:,} (refresh mode: nothing else is)', flush=True)
        elif a.save_min and time.time() - last_save > a.save_min * 60 and streak == 0:
            t = time.time()
            save_ckpt(a.run, model, opts, state())
            # good: not more than two halvings down and at most 8 steps skipped since the checkpoint before (one
            # skipped step is the loss scale at work, not a run that is degrading)
            healthy = (not CUDA or S >= 16384.0) and skipped - skip_mark <= 8
            if healthy:
                keep_good(a.run)
            skip_mark = skipped
            last_save = time.time()
            print(f'  checkpoint at step {gstep:,} in {last_save - t:.1f}s'
                  + ('' if healthy else ' (not marked good: the loss scale is more than two halvings down or more than 8 '
                                        'steps were skipped since the checkpoint before)'), flush=True)

    t = time.time()
    if frozen:  # refresh mode ends here: the add-on's side files, and nothing else
        ps_save_inplace(a.run, model, ck)
        print(f'parscale refresh: {seen - seen0:,} {u} in {gstep - g0:,} steps; the add-on is in {ck} and '
              f'{a.run}/parscale_last; the backbone, its optimiser state, state.json and the data cursor are as they were',
              flush=True)
        ps_report(model, val, a, exclusive)
        return
    save_ckpt(a.run, model, opts, state())
    if a.target_tokens and seen >= a.target_tokens:
        print(f'target reached: {seen / 1e9:.4f}B {u}', flush=True)
    mem = f', checkpoint written in {time.time() - t:.1f}s'
    mem += f', peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB' if CUDA else ''
    print(f'stopped at step {gstep:,}: {gstep - g0:,} steps in {(time.time() - t0) / 60:.1f} min{mem}, '
          f'{skipped} skipped for fp16 overflow; checkpoint saved in {a.run}/ckpt', flush=True)
    if a.samples:
        for i in range(D):
            model.activate(i, exclusive, a.gpu_sublayers)
            print(f'final net {i}: {eval_line(evaluate(model, i, val, a.eval_bs))}', flush=True)
        if not exclusive:
            for i in range(D):
                model.activate(i, False)
        sample_all(model, a.prompt or ['Once upon a time', 'The little dog'], a)


def cmd_generate(a):
    torch.set_num_threads(a.threads)
    ck = find_ckpt(a.run)
    if not ck:
        raise SystemExit(f'no checkpoint in {a.run}')
    st = json.load(open(os.path.join(ck, 'state.json')))
    cfg = Config(**st['cfg'])
    set_vocab(cfg.vocab)
    model = Model(cfg)
    means = load_ckpt(model, None, ck, None if a.latest else 'read')
    total, binary = model.counts()
    print(f'{ck}: {describe(cfg, total, binary)}; steps {st["steps"]}, {st["seen"] / 1e9:.4f}B {unit()} seen'
          + ('; read from the running means of the latents' if means else ''))
    Pg = attach_parscale(model, ps_source(ck, a.run, cfg), a.parscale)  # D9: else run/parscale_last
    print(f'ParScale: {Pg} streams merged' if Pg > 1 else 'ParScale: one stream (the plain model)')
    if a.resident != 'one':
        for i in range(len(model.nets)):
            model.activate(i, False)
    sample_all(model, a.prompt or ['Once upon a time', 'The little dog'], a)


def cmd_export(a):
    """The model alone in one file: each binary weight as one bit (its sign), the full-precision parameters
    and running means as they are, and the counters and data cursor, so that import can make a checkpoint
    training continues from. One eighth the size of a checkpoint: what is left out is how settled each
    sign was and the optimisers' moments."""
    ck = find_ckpt(a.run)
    if not ck:
        raise SystemExit(f'no checkpoint in {a.run}')
    st = json.load(open(os.path.join(ck, 'state.json')))
    tiers = bool(st['cfg'].get('tiers'))  # then the table is 'embf' and the format has another name: see save_ckpt
    names = [f'net{i}' for i in range(st['cfg']['dblocks'])] \
        + ([('embf' if tiers else 'emb')] if st['cfg']['vocab'] != 'bytes' else [])
    out = {'format': 'bintf2-bits-2' if tiers else 'bintf2-bits-1', 'fp': {}, 'bits': {}, 'count': {},
           **{k: st.get(k) for k in ('cfg', 'steps', 'seen', 'scale', 'data', 'args')}}
    for name in names:
        d = torch.load(os.path.join(ck, name + '.pt'), map_location='cpu')
        if 'fp' in d:
            out['fp'][name] = d['fp']
        if name == 'embf':  # Config.tiers: the row of every id
            out['pos_of'] = d['pos_of']
        pa = os.path.join(ck, name + '.avg.i8')  # the running means of the latents (_Fold), if the run keeps them
        means = os.path.exists(pa) and not a.latest
        out['latents'] = out.get('latents', []) + ['mean' if means else 'latest']
        P = np.fromfile(pa if means else os.path.join(ck, name + '.i8'), dtype=np.int8)
        out['count'][name] = len(P)
        out['bits'][name] = torch.from_numpy(np.packbits(P >= 0))
    if os.path.exists(os.path.join(a.run, 'val.bin')):  # the held-out windows travel with the model
        out['val'] = torch.from_numpy(np.fromfile(os.path.join(a.run, 'val.bin'), dtype=np.uint8))
    # the ParScale add-on in half precision, with the run's last evaluations at P = 1 and P = k (D7); from
    # run/parscale_last if the checkpoint has none (D9). A reader that does not know it never looks
    ps = ps_export(ps_source(ck, a.run, Config(**st['cfg'])), st)
    if ps is not None:
        out['parscale'] = ps
        print(f'{a.out}: with the ParScale add-on (trained at up to {ps["meta"]["P_trained"]} streams, '
              f'{ps["meta"]["n_prefix"]} prefix tokens)')
    torch.save(out, a.out + '.tmp')
    os.replace(a.out + '.tmp', a.out)
    print(f'{a.out}: signs of ' + ('the running means of the latents' if set(out['latents']) == {'mean'} else
                                   'the latest latents' if set(out['latents']) == {'latest'} else str(out['latents'])))
    print(f'{a.out}: {os.path.getsize(a.out) / 1e9:.2f} GB, {sum(out["count"].values()) / 1e9:.2f}B binary weights, '
          f'step {sum(st["steps"]):,}, {st["seen"] / 1e9:.4f}B {"bytes" if st["cfg"]["vocab"] == "bytes" else "tokens"} seen')


def cmd_import(a):
    """Make a checkpoint in --run from a file written by export; train --resume then continues from it.
    The signs come back exactly. Every latent starts again at the usual initial magnitude, and the
    optimisers start with empty moments."""
    m = torch.load(a.file, map_location='cpu')
    if m.get('format') not in ('bintf2-bits-1', 'bintf2-bits-2'):
        raise SystemExit(f'{a.file} was not written by export')
    if find_ckpt(a.run):
        raise SystemExit(f'{a.run} already has a checkpoint')
    cfg = Config(**m['cfg'])
    set_vocab(cfg.vocab)
    model = Model(cfg)
    new = os.path.join(a.run, 'ckpt.new')
    shutil.rmtree(new, ignore_errors=True)
    os.makedirs(new)
    mag = np.int8(round(SO.init_std / SO.q))
    groups = [(f'net{i}', net.signs()) for i, net in enumerate(model.nets)]
    if model.emb is not None:
        groups.append(('embf' if model.emb.tiers else 'emb', [model.emb]))
    for name, signs in groups:
        if sum(s.numel() for s in signs) != m['count'][name]:
            raise SystemExit(f'{name}: the file holds {m["count"][name]:,} weights, this configuration has '
                             f'{sum(s.numel() for s in signs):,}')
        bits = np.unpackbits(m['bits'][name].numpy())[:m['count'][name]]
        np.where(bits.view(np.bool_), mag, -mag).tofile(os.path.join(new, name + '.i8'))  # int8 throughout
        extra = {'fp': m['fp'][name], 'opt': None} if name in m['fp'] else {}
        if name == 'embf':
            extra = {**extra, 'pos_of': m['pos_of']}
        torch.save({**extra, 'R': [torch.zeros_like(s.R).cpu() for s in signs],
                    'C': [torch.zeros_like(s.C).cpu() for s in signs], 't': [s.t for s in signs]},
                   os.path.join(new, name + '.pt'))
    if m.get('parscale') is not None:  # the ParScale add-on, as it was exported (half precision, no optimiser state)
        ps_import(m['parscale'], new)
    with open(os.path.join(new, 'state.json'), 'w') as f:
        json.dump({'cfg': m['cfg'], 'steps': m['steps'], 'seen': m['seen'], 'scale': m.get('scale') or 65536.0,
                   'skipped': 0, 'active': 0, 'data': m.get('data') or {}, 'eval': {}, 'time': time.time(),
                   'args': m.get('args') or {}}, f)
    if m.get('val') is not None and not os.path.exists(os.path.join(a.run, 'val.bin')):
        m['val'].numpy().tofile(os.path.join(a.run, 'val.bin'))
    os.rename(new, os.path.join(a.run, 'ckpt'))
    print(f'{a.run}/ckpt written from {a.file}: step {sum(m["steps"]):,}; latents start at +-{mag}')


def cmd_eval(a):
    """Score every net of the checkpoint in --run on its validation windows, in rows of --val-len ids:
    longer rows than it was trained on show how far it reads past its training length."""
    ck = find_ckpt(a.run)
    if not ck:
        raise SystemExit(f'no checkpoint in {a.run}')
    st = json.load(open(os.path.join(ck, 'state.json')))
    cfg = Config(**st['cfg'])
    set_vocab(cfg.vocab)
    model = Model(cfg)
    if load_ckpt(model, None, ck, None if a.latest else 'read'):
        print('read from the running means of the latents (--latest: the latest latents)')
    Pe = attach_parscale(model, ps_source(ck, a.run, cfg), a.parscale)  # D9: else run/parscale_last
    vraw = np.fromfile(os.path.join(a.run, 'val.bin'), dtype=id_dtype())
    vlen = min(a.val_len or cfg.ctx, st['args'].get('window', 4096))  # a row stays inside one window of text
    vrows = min(a.val_rows, len(vraw) // vlen)
    vstep = max(1, len(vraw) // vlen // vrows)  # the same rows the trainer evaluates on
    val = torch.from_numpy(np.stack([vraw[i * vstep * vlen:(i * vstep + 1) * vlen] for i in range(vrows)])).long()
    near = max(int(x) for x in str(st['args'].get('lengths') or cfg.ctx).split(','))  # the longest training row
    for b in range(cfg.dblocks):
        model.activate(b, True, a.gpu_sublayers)
        print(f'net {b} at step {st["steps"][b]:,}, {len(val)} rows of {vlen:,} {unit()}: '
              f'{eval_line(evaluate(model, b, val, a.eval_bs))}', flush=True)
        if Pe > 1:
            print(f'  ParScale x{Pe}: {eval_line(evaluate(model, b, val, a.eval_bs, Pe))}', flush=True)
        if vlen > near:
            print(f'  {long_line(evaluate_long(model, b, val[:8], near, min(512, near // 2)))}', flush=True)


def cmd_status(a):
    p = os.path.join(a.run, 'status.json')
    if not os.path.exists(p):
        raise SystemExit(f'no status.json in {a.run} yet')
    s = json.load(open(p))
    u, k = s['unit'], (1 / LN2 if s['bits'] else 1.0)
    if os.path.exists(os.path.join(a.run, 'HALTED')):
        print('HALTED: ' + open(os.path.join(a.run, 'HALTED')).read().strip())
    alive = os.path.exists(f'/proc/{s["pid"]}')
    print(s['model'])
    print(f'trainer pid {s["pid"]} {"running" if alive else "NOT RUNNING"}; last log line {time.time() - s["time"]:,.0f}s ago; '
          f'up {(s["time"] - s["started"]) / 3600:.1f} h this launch')
    print(f'step {s["step"]:,} (per net {s["steps"]}); {s["seen"] / 1e9:.4f}B {u} seen'
          + (f' of {s["target"] / 1e9:g}B ({s["seen"] / s["target"] * 100:.4f}%)' if s.get('target') else ''))
    print(f'{s.get("rate_avg", s["rate"]):,.0f} {u}/s (last interval {s["rate"]:,.0f})'
          + (f'; {s["eta_days"]:,.0f} days ({s["eta_days"] / 365.25:.1f} years) to target at this rate'
             if s.get('eta_days') else ''))
    print(f'last interval: net {s["net"]} next {s["ar"] * k:.3f} masked {s["masked"] * k:.3f} {s["loss_unit"]}; '
          f'lr x{s["lr_mult"]:.3f}; flips {s["flips_pct"]:.3f}%/step; loss scale {s["scale"]:g}; '
          f'{s["skipped"]} skipped steps; gpu {s["gpu_gb"]:.2f} GB'
          + (f'; largest gradient entry {s["gmax"]:.3g} (half precision holds 65,504)' if 'gmax' in s else ''))
    if 'x_rms' in s:
        print(f'residual stream at the last layer: rms {s["x_rms"]:.1f}, largest position {s["x_max"]:.0f}'
              + (f', the part every position shares {s["x_common"]:.1f}' if 'x_common' in s else '')
              + (f'; branches write rms {s["br_rms"]:.2f} before their bound, {s["br_capped"] * 100:.0f}% of positions '
                 f'are at it' if s.get('br_rms') else ''))
    if s['scale'] < 65536 and not s['bits']:
        print('ATTENTION: the loss scale has been backed off, so gradients overflowed half precision. Look at the '
              'largest gradient entry and the stream figures above before letting it run on.')
    for b, ev in sorted(s.get('eval', {}).items()):
        print(f'eval net {b} @ step {ev["step"]:,}: next {ev["ar"] * k:.3f} (last 32: {ev["ar_tail"] * k:.3f})'
              + (f', 16 at once {ev["nat16"] * k:.3f}' if 'nat16' in ev else '') + f' {s["loss_unit"]}')
        if ev.get('avg'):
            print(f'  read from the running means of the latents: next {ev["avg"]["ar"] * k:.3f}'
                  + (f', the nets together {ev["avg"]["together"] * k:.3f}' if 'together' in ev['avg'] else ''))
        if ev.get('sample'):
            print(f'  sample: {ev["sample"][:300]}')
        if ev.get('long'):
            g = ev['long']
            print(f'  past the training length @ step {g["step"]:,}: the last {g["tail"]:,} {u} of {g["far_len"]:,}-long rows '
                  f'{g["far"] * k:.3f} {s["loss_unit"]}; the same {u} with {g["near_len"]:,} of context {g["near"] * k:.3f}')
    print(f'data: {s["data"]}; last checkpoint {(time.time() - s["last_save"]) / 60:.0f} min ago')


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def model_args(sp):
        sp.add_argument('--preset', default='tiny', choices=PRESETS)
        sp.add_argument('--vocab', default='bytes', choices=['bytes'] + list(TOKENIZERS))
        sp.add_argument('--act', default='relu2', choices=['relu2', 'gelu'])
        sp.add_argument('--experts', type=int, help='override the preset: experts per layer (0 = dense)')
        sp.add_argument('--topk', type=int, help='override the preset: experts per token')
        sp.add_argument('--dblocks', type=int, default=1, help='DiffusionBlocks nets (1 = one end-to-end net)')
        sp.add_argument('--pos', default='rope', choices=['rope', 'alibi', 'both'],
                        help='rope: rotary positions; alibi: a per-head penalty on distance; both')
        sp.add_argument('--cap', type=float, default=1.0,
                        help='largest rms per position a branch may add to the residual stream (0 = no bound)')
        sp.add_argument('--centre2', type=int, default=0,
                        help='1: centre what every binary matrix reads again after its norm or bound (0 = once, before)')
        sp.add_argument('--bias', type=int, default=1,
                        help='1: full-precision offsets for queries, keys and feed-forward pre-activations')
        sp.add_argument('--cov', type=int, default=0,
                        help='1: every binary matrix learns from its output gradient minus the running mean of that '
                             'gradient over tokens, and the offsets are kept per kind of position (ordinary, [MASK])')
        sp.add_argument('--tiers', default='',
                        help='token vocabularies: slices of the table per group, e.g. 1,1,2,4. The table is kept in '
                             'order of word frequency; the first group is scored for every row, a later group only '
                             'for the rows whose word is in it (empty = one softmax over every word)')

    def decode_args(sp):
        sp.add_argument('--prompt', action='append', help='repeatable')
        sp.add_argument('--modes', default='all',
                        help='all, or a comma list of ar,ar3,semi,nat,diffusion (ar3: every net scores each token)')
        sp.add_argument('--n', type=int, default=64, help='tokens to generate')
        sp.add_argument('--block', type=int, default=16, help='semi-autoregressive block size')
        sp.add_argument('--semi-steps', type=int, default=1, help='parallel passes per block')
        sp.add_argument('--steps', type=int, default=16, help='diffusion passes')
        sp.add_argument('--temp', type=float, default=0.8)
        sp.add_argument('--topk-sample', type=int, default=0)
        sp.add_argument('--seed', type=int, default=0)
        sp.add_argument('--threads', type=int, default=4)
        sp.add_argument('--resident', default='all', choices=['all', 'one'],
                        help='one = only the net in use keeps its latents on the GPU')
        sp.add_argument('--gpu-sublayers', type=int, default=-1,
                        help='how many sublayers of the net in use keep their latents on the GPU; the rest are '
                             'streamed from RAM one matrix at a time, down to a single expert (-1 = all)')

    sp = sub.add_parser('info', help='size and memory for a preset')
    model_args(sp)

    sp = sub.add_parser('train', help='train (or resume), checkpointing into --run')
    model_args(sp)
    decode_args(sp)
    sp.add_argument('--run', required=True, help='directory for checkpoints, logs and the data cache')
    sp.add_argument('--resume', action='store_true', help='continue from the checkpoint in --run if there is one')
    sp.add_argument('--require-resume-checkpoint', action='store_true',
                    help='refuse and pause instead of starting from zero when the resume checkpoint is missing')
    sp.add_argument('--data', default=HF_TRAIN,
                    help='local file, hf:owner/dataset/file.txt, or hfds:owner/dataset:dir of parquet shards')
    sp.add_argument('--val-data', default=HF_VAL, help='held-out text for a single-file --data; empty = last 5%%')
    sp.add_argument('--data-mb', type=float, default=300, help='single file: read at most this many MB')
    sp.add_argument('--math-replay-ids', default='', help='pretokenized int32 replay corpus (optional)')
    sp.add_argument('--math-replay-meta', default='', help='SHA-bound JSON metadata for --math-replay-ids')
    sp.add_argument('--math-replay-every', type=int, default=0,
                    help='use one deterministic replay batch every N global steps (0 = off)')
    sp.add_argument('--window', type=int, default=4096, help='shard stream: ids per window')
    sp.add_argument('--val-windows', type=int, default=256, help='shard stream: windows set aside for evaluation')
    sp.add_argument('--mix-windows', type=int, default=256,
                    help='shard stream: open windows that each batch draws its rows from at random')
    sp.add_argument('--val-rows', type=int, default=64, help='rows per evaluation')
    sp.add_argument('--val-len', type=int, default=0, help='ids per evaluation row (0 = ctx)')
    sp.add_argument('--val-long', type=int, default=0,
                    help='also score a few rows of this many ids, longer than any training row (0 = off)')
    sp.add_argument('--val-long-rows', type=int, default=8)
    sp.add_argument('--val-long-every', type=int, default=4, help='every this many evaluations')
    sp.add_argument('--eval-bs', type=int, default=8)
    sp.add_argument('--eval-sample', type=int, default=0, help='ids to generate (AR) after each evaluation')
    sp.add_argument('--eval-prompt', default='The')
    sp.add_argument('--step-tokens', type=int, default=0, help='clean ids per step (0 = 32 x ctx)')
    sp.add_argument('--ref-tokens', type=int, default=0,
                    help='the ids a step holds that --sign-lr, --lr, --lr-bias, --t0 and --warmup are stated for. A step '
                         'of n ids then takes sqrt(n / ref) times the int8 and AdamW steps (the output bias, plain SGD, '
                         'n / ref times) and the schedule counts ref ids as one step: the same training per token '
                         'whatever a step holds. 0: the steps are as stated')
    sp.add_argument('--lengths', default='', help='comma list of row lengths to mix (default: ctx)')
    sp.add_argument('--noisy', type=float, default=0.125, help='noisy-stream tokens per clean token')
    sp.add_argument('--w-noisy', type=float, default=1.0,
                    help='weight of one masked token against one next-token target (1 = every target counts the same)')
    sp.add_argument('--aux', type=float, default=0.01, help='MoE load-balancing loss weight')
    sp.add_argument('--lr', type=float, default=2e-3, help='AdamW on the full-precision parameters')
    sp.add_argument('--sign-lr', type=float, default=5e-4, help='step size on the int8 sign latents')
    sp.add_argument('--sign-lr-nets', default='', help='comma list: a multiple of --sign-lr for each DiffusionBlocks net, '
                    'to compare step sizes on nets that train side by side (default: all 1); the shared token '
                    'table keeps --sign-lr x --table-lr')
    sp.add_argument('--lr-scales', type=float, default=0.02, help='row scales: their step as a share of --lr')
    sp.add_argument('--lr-router', type=float, default=0.02, help='MoE router: its step as a share of --lr')
    sp.add_argument('--lr-offsets', type=float, default=1.0,
                    help='offsets of queries, keys and feed-forward units: their step as a multiple of --lr (a row '
                         'lining up with a constant moves a unit\'s offset about 0.005 a step; --lr alone is 0.002)')
    sp.add_argument('--lr-bias', type=float, default=1.0, help='token vocabularies: SGD step on the output bias')
    sp.add_argument('--table-lr', type=float, default=1.0, help='token table: its step as a multiple of --sign-lr')
    sp.add_argument('--sign-kick', type=float, default=0.0,
                    help='hysteresis of the signs: a latent that crosses to the other sign is put this many int8 steps '
                         '(the step as it stands, rounded to units) further on, so it takes as much travel again to '
                         'come back; a latent near zero then stops changing sign from step to step (0 = off)')
    sp.add_argument('--sign-kick-nets', default='', help='comma list: --sign-kick for each DiffusionBlocks net, to '
                    'compare them on nets that train side by side')
    sp.add_argument('--table-kick', type=float, default=0.0, help='--sign-kick for the shared token table')
    sp.add_argument('--bias-init', default='unigram', choices=['unigram', 'zero'],
                    help='token vocabularies: start the output bias at each token\'s log frequency, or at zero')
    sp.add_argument('--warmup', type=int, default=100)
    sp.add_argument('--sched', default='cosine', choices=['cosine', 'isqrt'],
                    help='cosine over the planned steps, or 1/sqrt(step) after --t0 steps per net')
    sp.add_argument('--t0', type=float, default=10000)
    sp.add_argument('--lr-floor', type=float, default=0.01)
    sp.add_argument('--minutes', type=float, default=0, help='time budget; 0 = run --max-steps / --target-tokens')
    sp.add_argument('--max-steps', type=int, default=10 ** 12)
    sp.add_argument('--target-tokens', type=float, default=0, help='stop after this many ids')
    sp.add_argument('--rotate', type=int, default=1, help='steps on one net before moving to the next')
    sp.add_argument('--avg-turns', type=int, default=0,
                    help='keep a running mean of every latent over about this many of its net\'s turns (2..128), in RAM '
                         'and in the checkpoint, for evaluation, export, eval and generate to read: the signs of the '
                         'mean do not carry the flips of the last steps. 0: no mean is kept. Training never reads it')
    sp.add_argument('--avg-every', type=int, default=0,
                    help='--avg-turns: fold a net into its running means on every this-many-th of its turns, so the '
                         'means reach that many times as far back (0: 1 until step 120,000, then one more per 60,000 '
                         'steps, at most 32)')
    sp.add_argument('--gcap', type=float, default=0.0,
                    help='backward pass: between two layers no position hands down a gradient longer than this many '
                         'times the median position\'s (a position many others attend to collects all their '
                         'gradients, and in rare batches that outgrows half precision); 0 = no bound')
    sp.add_argument('--gcap-nets', default='', help='comma list: --gcap for each DiffusionBlocks net')
    sp.add_argument('--gcap-mid', type=float, default=0.0,
                    help='the bound of --gcap inside every layer too: on what the expert branch and its norm hand back '
                         'to the attention branch (the bottom layer makes its longest gradients there); 0 = no bound')
    sp.add_argument('--gcap-qkv', type=float, default=0.0,
                    help='and on what attention hands its q/k/v matrix, where a token that others attend to collects '
                         'their gradients; 0 = no bound')
    sp.add_argument('--wcap', type=float, default=0.0,
                    help='what a binary matrix learns from: no row of its output gradient (one position\'s say) is '
                         'longer than this many times the median row; what is passed on down the network is not '
                         'touched; 0 = no bound')
    sp.add_argument('--wcap-nets', default='', help='comma list: --wcap for each DiffusionBlocks net')
    sp.add_argument('--lr-nets', default='', help='comma list: a multiple of --lr for each DiffusionBlocks net, to '
                    'compare AdamW steps on nets that train side by side (default: all 1)')
    sp.add_argument('--log-every', type=int, default=50)
    sp.add_argument('--eval-every', type=int, default=500)
    sp.add_argument('--probe', type=int, default=0,
                    help='diagnostic: at every hand-over score both nets on this many validation rows (probe.jsonl)')
    sp.add_argument('--eval-slow', type=int, default=1,
                    help='after the first 100 evaluations, this many times as many steps between them')
    sp.add_argument('--save-min', type=float, default=30, help='minutes between checkpoints (0 = only at exit)')
    sp.add_argument('--samples', action='store_true', help='evaluate every net and sample all decoders at the end')
    sp.add_argument('--prof', action='store_true', help='print forward/backward seconds and peak memory every step')
    sp.add_argument('--parscale', type=int, default=PS_P,
                    help='ParScale (built in, see ParAdapter): streams of a ParScale step, and what evaluation merges '
                         '(1 = off: the plain trainer, bit for bit). More than the add-on has grows it')
    sp.add_argument('--parscale-duty', type=float, default=PS_DUTY,
                    help='the share of each net\'s steps taken with --parscale streams, spread evenly and drawn from no '
                         'random number (the rows and noisy plans are those of a plain run); the others are plain '
                         'steps of stream 0. 1 = every step, 0 = none (the plain trainer)')
    sp.add_argument('--parscale-freeze-backbone', action='store_true',
                    help='refresh mode: only the ParScale add-on learns, every step with --parscale streams, for '
                         '--parscale-refresh-tokens ids; nothing of the backbone moves and only the side files are '
                         'written (no state.json, no data cursor). Adds ParScale to any checkpoint, or brings it up '
                         'to date after a long stretch of plain steps. The run\'s trainer must not be running')
    sp.add_argument('--parscale-refresh-tokens', type=float, default=20e6, help='refresh mode: ids to train on')
    sp.add_argument('--parscale-lr', type=float, default=1e-3, help='AdamW on the add-on (prefixes and merge)')
    sp.add_argument('--parscale-warmup', type=int, default=100, help='the add-on\'s own warm-up, in its own steps')
    sp.add_argument('--parscale-clip', type=float, default=1.0, help='gradient norm clip of the add-on')
    sp.add_argument('--parscale-prefix', type=int, default=48,
                    help='a new add-on: prefix tokens per stream and layer (16s keep the GPU\'s lean path)')
    sp.add_argument('--parscale-hidden', type=int, default=512, help='a new add-on: width of the merge MLP')
    sp.add_argument('--parscale-smooth', type=float, default=0.1, help='a new add-on: label smoothing of the merge')
    sp.add_argument('--parscale-max-pos', type=int, default=0,
                    help='a ParScale step takes its rows in sub-steps of at most this many positions, every stream '
                         'counted, the same rows and draws as a plain step, one optimiser step per sub-step at '
                         'sqrt(its ids / ref); streams per step = min(--parscale, this / row length), and fewer than '
                         '2 is refused loudly (PARSCALE NOT RUNNING). 0 = --step-tokens (a sub-step holds what a '
                         'plain step does); -1 = no limit (the whole step at once, as revision 1)')
    sp.add_argument('--parscale-prefix0', type=int, default=0,
                    help='a new add-on: 1 gives stream 0 a prefix too (the paper\'s layout); 0 keeps stream 0 the '
                         'plain model, which the plain steps then train and P = 1 serves')

    sp = sub.add_parser('generate', help='sample all decoding modes from the checkpoint in --run')
    sp.add_argument('--run', required=True)
    sp.add_argument('--parscale', type=int, default=PS_P,
                    help='ParScale streams (at most as many as were trained; 1 = the plain model)')
    sp.add_argument('--latest', action='store_true',
                    help='read the latest latents even if the checkpoint has their running means (train --avg-turns)')
    decode_args(sp)

    sp = sub.add_parser('status', help='a few lines on a run: progress, speed, losses, last evaluation')
    sp.add_argument('--run', required=True)

    sp = sub.add_parser('export', help='write the model in --run as one file, one bit per binary weight')
    sp.add_argument('--run', required=True)
    sp.add_argument('--out', required=True)
    sp.add_argument('--latest', action='store_true',
                    help='read the latest latents even if the checkpoint has their running means (train --avg-turns)')

    sp = sub.add_parser('import', help='make a checkpoint in --run from a file written by export')
    sp.add_argument('--file', required=True)
    sp.add_argument('--run', required=True)

    sp = sub.add_parser('eval', help='score the checkpoint in --run on its validation windows at any row length')
    sp.add_argument('--run', required=True)
    sp.add_argument('--parscale', type=int, default=PS_P,
                    help='also score with this many ParScale streams merged (at most as many as were trained)')
    sp.add_argument('--val-len', type=int, default=0, help='ids per row (0 = ctx; at most one window)')
    sp.add_argument('--val-rows', type=int, default=64)
    sp.add_argument('--eval-bs', type=int, default=2)
    sp.add_argument('--gpu-sublayers', type=int, default=-1)
    sp.add_argument('--latest', action='store_true',
                    help='read the latest latents even if the checkpoint has their running means (train --avg-turns)')

    sp = sub.add_parser('prep', help='(used by train) turn one parquet shard into a pool of ids')
    sp.add_argument('--repo', required=True)
    sp.add_argument('--file', required=True)
    sp.add_argument('--out', required=True)
    sp.add_argument('--vocab', default='bytes')

    a = p.parse_args()
    {'info': cmd_info, 'train': cmd_train, 'generate': cmd_generate, 'status': cmd_status, 'eval': cmd_eval,
     'export': cmd_export, 'import': cmd_import, 'prep': cmd_prep}[a.cmd](a)


if __name__ == '__main__':
    main()
