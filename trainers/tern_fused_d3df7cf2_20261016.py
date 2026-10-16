"""Fused Triton kernels for terntf's int8 ternary-latent update (the bintf2 _Fused approach, in Triton).

The PyTorch TernLinear.update makes ~15 full-matrix fp32 passes (~150 B of memory traffic per weight) and two
host syncs per matrix (isfinite(g.sum()) and float(rms)). Here the same arithmetic is three tiled passes:
  stats  : one read of gW (fp32) + P (int8): row sum g^2, column sum g^2, row-scale gradient sum gW*tern(P),
           non-finite count          (g = gW * scale[r] * inv_scale, never materialised)
  usq    : one read of gW: sum over the matrix of u^2, u = g * rr[r] * cc[c] (Adafactor update clipping)
  apply  : one read of gW + P, write P: P + u*(-a/rms) + U[0,1) -> floor -> hysteresis kick -> clamp -> int8,
           flip count; everything gated on the device-side bad flag (the step skip), no host round trip.
The tiny R/C/t bookkeeping runs on (d_out,)/(d_in,) vectors on the device. ~15 B/weight, zero host syncs.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _stats(G, P, S, ROWSQ, COLSQ, GSC, BAD, n_out, n_in, inv_scale, dead,
           BM: tl.constexpr, BN: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    r = pm * BM + tl.arange(0, BM)
    c = pn * BN + tl.arange(0, BN)
    mr, mc = r < n_out, c < n_in
    m = mr[:, None] & mc[None, :]
    off = r.to(tl.int64)[:, None] * n_in + c[None, :]
    gw = tl.load(G + off, mask=m, other=0.0)
    p = tl.load(P + off, mask=m, other=0).to(tl.int32)
    s = tl.load(S + r, mask=mr, other=0.0) * inv_scale
    w = tl.where(p > dead, 1.0, tl.where(p < -dead, -1.0, 0.0))
    g = gw * s[:, None]
    ok = (g == g) & (tl.abs(g) < float('inf'))
    nbad = tl.sum(tl.sum(tl.where(m & (~ok), 1, 0), 1), 0)
    g = tl.where(ok, g, 0.0)
    gsq = g * g
    tl.atomic_add(ROWSQ + r, tl.sum(gsq, 1), mask=mr)
    tl.atomic_add(COLSQ + c, tl.sum(gsq, 0), mask=mc)
    tl.atomic_add(GSC + r, tl.sum(gw * w, 1), mask=mr)
    if nbad > 0:
        tl.atomic_add(BAD, nbad)


@triton.jit
def _usq(G, S, RR, CC, TOT, n_out, n_in, inv_scale, BM: tl.constexpr, BN: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    r = pm * BM + tl.arange(0, BM)
    c = pn * BN + tl.arange(0, BN)
    mr, mc = r < n_out, c < n_in
    m = mr[:, None] & mc[None, :]
    off = r.to(tl.int64)[:, None] * n_in + c[None, :]
    gw = tl.load(G + off, mask=m, other=0.0)
    s = tl.load(S + r, mask=mr, other=0.0) * inv_scale
    rr = tl.load(RR + r, mask=mr, other=0.0)
    cc = tl.load(CC + c, mask=mc, other=0.0)
    u = (gw * s[:, None]) * rr[:, None] * cc[None, :]
    tl.atomic_add(TOT, tl.sum(tl.sum(u * u, 1), 0))


@triton.jit(do_not_specialize=["seed"])
def _apply(G, P, S, RR, CC, TOT, BAD, FLIPS, NOISE, n_out, n_in, inv_scale, inv_n, a, dead, kick, seed,
           COUNT: tl.constexpr, USE_NOISE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    if tl.load(BAD) != 0:  # the matrix had a non-finite gradient: skip it (bintf2 / terntf semantics)
        return
    pm, pn = tl.program_id(0), tl.program_id(1)
    r = pm * BM + tl.arange(0, BM)
    c = pn * BN + tl.arange(0, BN)
    mr, mc = r < n_out, c < n_in
    m = mr[:, None] & mc[None, :]
    off = r.to(tl.int64)[:, None] * n_in + c[None, :]
    gw = tl.load(G + off, mask=m, other=0.0)
    p8 = tl.load(P + off, mask=m, other=0)
    s = tl.load(S + r, mask=mr, other=0.0) * inv_scale
    rr = tl.load(RR + r, mask=mr, other=0.0)
    cc = tl.load(CC + c, mask=mc, other=0.0)
    rms = tl.maximum(tl.sqrt(tl.load(TOT) * inv_n), 1.0)
    u = (gw * s[:, None]) * rr[:, None] * cc[None, :]
    if USE_NOISE:
        rnd = tl.load(NOISE + off, mask=m, other=0.0)
    else:
        rnd = tl.rand(seed, off)
    pf = p8.to(tl.float32)
    new = tl.floor((pf + u * (-a / rms)) + rnd)
    s0 = tl.where(pf > dead, 1, tl.where(pf < -dead, -1, 0))
    s1 = tl.where(new > dead, 1, tl.where(new < -dead, -1, 0))
    moved = s1 != s0
    new = tl.where(moved, new + tl.where(new >= 0, kick, -kick), new)
    new = tl.minimum(tl.maximum(new, -127.0), 127.0)
    if COUNT:
        s2 = tl.where(new > dead, 1, tl.where(new < -dead, -1, 0))
        tl.atomic_add(FLIPS, tl.sum(tl.sum(tl.where(m & (s2 != s0), 1, 0), 1), 0).to(tl.int64))
    tl.store(P + off, new.to(tl.int8), mask=m)


@triton.jit
def _tern(P, W, n, dead, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mk = i < n
    p = tl.load(P + i, mask=mk, other=0).to(tl.int32)
    w = tl.where(p > dead, 1.0, tl.where(p < -dead, -1.0, 0.0))
    tl.store(W + i, w.to(tl.float16), mask=mk)


class Fused:
    """Host side. Device-resident counters: bad_t (int32, non-finite matrices this step), flips_t (int64).
    update() never synchronises with the host; the trainer reads bad_t/flips_t once per step."""
    BM, BN = 32, 128

    def __init__(self, device='cuda'):
        self.dev = device
        self.bad_t = torch.zeros((), dtype=torch.int32, device=device)
        self.flips_t = torch.zeros((), dtype=torch.int64, device=device)
        self.calls = 0

    def reset(self):
        self.bad_t.zero_(); self.flips_t.zero_()

    def update(self, lin, gW, P, scale, TO, noise=None):
        """In-place equivalent of TernLinear.update(gW * scale[:, None], P) plus gscale = (gW * tern(P)).sum(1).
        gW: (out, in) fp32 raw weight gradient (before row scale). Returns gscale (out,) fp32."""
        n_out, n_in = gW.shape
        gW = gW.contiguous()
        f32 = dict(device=gW.device, dtype=torch.float32)
        rowsq, colsq, gsc = torch.zeros(n_out, **f32), torch.zeros(n_in, **f32), torch.zeros(n_out, **f32)
        bad = torch.zeros((), dtype=torch.int32, device=gW.device)
        tot = torch.zeros((), **f32)
        s = scale.detach().float().contiguous()
        grid = (triton.cdiv(n_out, self.BM), triton.cdiv(n_in, self.BN))
        inv = float(TO.inv_scale)
        _stats[grid](gW, P, s, rowsq, colsq, gsc, bad, n_out, n_in, inv, int(TO.dead), BM=self.BM, BN=self.BN)
        ok = bad == 0
        b2 = TO.b2
        # second moments + step counter, all gated on the device flag (original returns before touching them)
        Rn = lin.R * b2 + (rowsq[:, None] / n_in + 1e-30) * (1 - b2)
        Cn = lin.C * b2 + (colsq[None, :] / n_out + 1e-30) * (1 - b2)
        lin.R.copy_(torch.where(ok, Rn, lin.R)); lin.C.copy_(torch.where(ok, Cn, lin.C))
        lin.t.add_(ok.long())
        corr = 1 - torch.pow(torch.tensor(b2, **f32), lin.t.float())
        rr = ((lin.R / lin.R.mean()) / corr).clamp_(min=1e-30).rsqrt_().reshape(-1).contiguous()
        cc = (lin.C / corr).clamp_(min=1e-30).rsqrt_().reshape(-1).contiguous()
        _usq[grid](gW, s, rr, cc, tot, n_out, n_in, inv, BM=self.BM, BN=self.BN)
        self.calls += 1
        seed = (self.calls * 2654435761) & 0x7fffffff
        use_noise = noise is not None
        _apply[grid](gW, P, s, rr, cc, tot, bad, self.flips_t, noise if use_noise else gW, n_out, n_in, inv,
                     1.0 / (n_out * n_in), float(TO.a), int(TO.dead), int(TO.kick), seed,
                     COUNT=bool(TO.count), USE_NOISE=use_noise, BM=self.BM, BN=self.BN)
        self.bad_t.add_(bad.clamp(max=1))
        if TO.count:
            TO.seen += n_out * n_in
        return gsc

    def tern(self, P, dtype=torch.float16):
        W = torch.empty(P.shape, dtype=torch.float16, device=P.device)
        n = P.numel()
        _tern[(triton.cdiv(n, 4096),)](P, W, n, int(_DEAD[0]), BLOCK=4096)
        return W


_DEAD = [24]
