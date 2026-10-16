#!/usr/bin/env python3
"""terntf: native-ternary transformer (-1 / 0 / +1 weights, ~1.58 bits), derived from bintf2.py.

What it keeps from bintf2:
  int8 latents    Each weight is ONE int8 latent P in [-127, 127]. There is no full-precision master copy and no
                  Adam moments for it. The update (factored second moments per row and column, Adafactor update
                  clipping, stochastic rounding onto the int8 grid) runs inside the backward pass, one matrix at a
                  time, so a weight gradient is freed before the next matrix is touched.
  row scales      each output row has one fp32 scale, trained by AdamW with the other (tiny) fp parameters.
  normed inputs   every ternary matrix reads an RMS-normalised input (SubLN), squared-ReLU FFN.
What is new:
  ternary read    w = sign(P) if |P| > --dead else 0. The latent's magnitude is how settled a weight is (bintf2's
                  idea); the band |P| <= dead is the zero state, so a weight walks -1 <-> 0 <-> +1 and never jumps.
                  The dead zone is the counter: a weight must move `dead` units (many steps of evidence) to switch.
  hysteresis      --kick: a latent that crosses a state boundary is pushed `kick` units further (bintf2's
                  --sign-kick), so states do not chatter.
  ternary export  5 trits per byte (3^5 = 243 <= 256): 1.6 bits per weight on disk / in inference memory.

Usage:
  python terntf.py info  --preset 64b
  python terntf.py train --preset smoke --data file:/path/to.txt --steps 30 --run runs/smoke
  python terntf.py train --preset 64b --stream 1 --data hfds:HuggingFaceFW/fineweb-edu:data --run /workspace/tern64b_run --resume
  python terntf.py export --run runs/smoke --out model_trits.pt
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

PRESETS = {  # d model, layers, heads, ffn width, vocab, context
    'smoke': dict(d=640, L=12, H=10, ff=2560, vocab=256, ctx=256),          # ~59M ternary + tiny fp
    'tiny':  dict(d=384, L=6, H=6, ff=1536, vocab=256, ctx=256),            # ~10.6M, a fast sanity check
    '600m':  dict(d=1536, L=20, H=12, ff=6144, vocab=256, ctx=512),          # ~566M, fits a 12GB 3060 resident
    '1b':    dict(d=2048, L=20, H=16, ff=8192, vocab=129280, ctx=2048),
    '64b':   dict(d=8192, L=80, H=64, ff=32768, vocab=129280, ctx=2048),    # ~64.4B ternary + ternary table
}


# ---------------------------------------------------------------- int8 ternary latents
class TernOpt:
    """Settings shared by every latent update; the trainer sets them once per step (bintf2's SignOpt)."""
    b2 = 0.99
    a = 0.0          # step in int8 units
    dead = 24        # |P| <= dead reads as 0
    kick = 4
    inv_scale = 1.0
    count = False
    flips = 0
    seen = 0
    bad = 0


TO = TernOpt()

# TERNTF_FUSED=1 (default on CUDA): the latent update runs as three fused Triton passes (tern_fused.py), no host syncs.
# TERNTF_FUSED=0 restores the original PyTorch path exactly.
FUSED = None
if DEV == 'cuda' and os.environ.get('TERNTF_FUSED', '1') == '1':
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from tern_fused import Fused as _FusedCls
        FUSED = _FusedCls(DEV)
    except Exception as e:  # triton missing / broken: fall back to the torch path
        print(f'TERNTF_FUSED unavailable ({e!r}); using the PyTorch update', flush=True)


def tern(P):
    """int8 latent -> ternary weight in {-1, 0, +1} (as the compute dtype)."""
    return (torch.sign(P) * (P.abs() > TO.dead)).to(torch.float32 if DEV == 'cpu' else torch.float16)


def state_of(P):
    return torch.sign(P) * (P.abs() > TO.dead)


class _TernMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, lin):
        w = tern(lin.lat())
        ctx.lin = lin
        ctx.save_for_backward(x, scale)
        y = (x.to(w.dtype) @ w.t()).float() * scale
        # bintf2 46deeb0b FP16 clamp: keep a finite overflowing product in fp32 and clamp to the half range
        return y.clamp(-65504, 65504) if DEV != 'cpu' else y

    @staticmethod
    def backward(ctx, gy):
        x, scale = ctx.saved_tensors
        lin = ctx.lin
        P = lin.lat()
        w = tern(P)
        gys = (gy * scale).to(w.dtype)
        gx = (gys @ w).to(x.dtype)
        # weight gradient of the ternary matrix before its row scale; consumed here, never stored
        gW = gy.reshape(-1, gy.shape[-1]).t().float() @ x.reshape(-1, x.shape[-1]).float()  # (out, in)
        if FUSED is not None:
            del w
            gscale = FUSED.update(lin, gW, P, scale, TO)  # stats + R/C + rms + apply, gated on-device
            if lin.mm is not None:
                lin.mm[:] = P.cpu().numpy()
            return gx, gscale, None
        gscale = (gW * w.float()).sum(1)
        del w
        lin.update(gW * scale.detach()[:, None], P)
        return gx, gscale, None


class TernLinear(nn.Module):
    """A (d_out, d_in) ternary matrix held as int8 latents + one fp32 scale per row.
    Optimiser state per weight: 0 bytes (factored moments R: d_out floats, C: d_in floats)."""
    def __init__(self, d_in, d_out, stream=None):
        super().__init__()
        self.shape = (d_out, d_in)
        self.mm = None
        if stream is not None:  # --stream: the latents live in a file on disk (np.memmap); only the matrix in use
            path, off = stream    # is ever on the GPU, read for forward, read again + written back in backward
            self.mm = np.memmap(path, dtype=np.int8, mode='r+', offset=off, shape=self.shape)
            self.register_buffer('P', torch.zeros(0, dtype=torch.int8), persistent=False)
        else:
            init = torch.randint(-1, 2, (d_out, d_in), dtype=torch.int8) * (TO.dead + 8)  # 1/3 each state, settled
            self.register_buffer('P', init)
        self.scale = nn.Parameter(torch.full((d_out,), 1.0 / math.sqrt(d_in * 2 / 3)))
        self.register_buffer('R', torch.zeros(d_out, 1))
        self.register_buffer('C', torch.zeros(1, d_in))
        self.register_buffer('t', torch.zeros((), dtype=torch.long))

    def forward(self, x):
        return _TernMM.apply(x, self.scale, self)

    def lat(self):
        if self.mm is None:
            return self.P
        return torch.from_numpy(np.asarray(self.mm)).to(self.scale.device, non_blocking=True)

    @torch.no_grad()
    def update(self, g, P=None):
        P = self.P if P is None else P
        g = g * TO.inv_scale
        if not bool(torch.isfinite(g.sum())):
            TO.bad += 1  # bintf2: a non-finite weight gradient skips the whole step (overflow streak)
            return
        b2 = TO.b2
        self.R.mul_(b2).add_(g.square().mean(1, keepdim=True) + 1e-30, alpha=1 - b2)
        self.C.mul_(b2).add_(g.square().mean(0, keepdim=True) + 1e-30, alpha=1 - b2)
        self.t += 1
        corr = 1 - b2 ** int(self.t)
        u = g * ((self.R / self.R.mean()) / corr).clamp_(min=1e-30).rsqrt_() * (self.C / corr).clamp_(min=1e-30).rsqrt_()
        rms = float((u.norm() / math.sqrt(u.numel())).clamp(min=1.0))  # Adafactor update clipping
        new = P.float().add_(u, alpha=-TO.a / rms).add_(torch.rand_like(u)).floor_()
        if TO.kick:  # hysteresis: crossing a state boundary costs kick more units to come back
            s0, s1 = state_of(P.float()), state_of(new)
            moved = s1 != s0
            new.add_(torch.where(new >= 0, float(TO.kick), -float(TO.kick)) * moved)
        new.clamp_(-127, 127)
        if TO.count:
            TO.flips += int((state_of(new) != state_of(P.float())).sum())
            TO.seen += P.numel()
        P.copy_(new.to(torch.int8))
        if self.mm is not None:  # streamed: write the matrix back to its file
            self.mm[:] = P.cpu().numpy()


class RMS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * self.g


def rope(x, base=10000.0):
    T, hd = x.shape[-2], x.shape[-1]
    f = base ** (-torch.arange(0, hd, 2, device=x.device).float() / hd)
    ang = torch.arange(T, device=x.device).float()[:, None] * f[None]
    c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Store:
    """One int8 file holding every streamed matrix back to back (--stream). A new file starts every latent at
    -/0/+ settled, a third each, written in chunks so that a 64B model never has to exist in RAM."""
    def __init__(self, path, total):
        self.path, self.off = path, 0
        if not os.path.exists(path) or os.path.getsize(path) != total:
            g = np.random.default_rng(0)
            with open(path, 'wb') as f:
                for i in range(0, total, 1 << 28):
                    n = min(1 << 28, total - i)
                    f.write(((g.integers(-1, 2, n, dtype=np.int8)) * np.int8(TO.dead + 8)).tobytes())

    def take(self, n):
        o, self.off = self.off, self.off + n
        return (self.path, o)


def lin(d_in, d_out, st):
    return TernLinear(d_in, d_out, None if st is None else st.take(d_in * d_out))


class Block(nn.Module):
    def __init__(self, c, st=None):
        super().__init__()
        d, ff = c['d'], c['ff']
        self.H = c['H']
        self.n1, self.n2, self.nh = RMS(d), RMS(d), RMS(ff)
        self.qkv, self.o = lin(d, 3 * d, st), lin(d, d, st)
        self.no = RMS(d)
        self.up, self.down = lin(d, ff, st), lin(ff, d, st)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.H, D // self.H).permute(2, 0, 3, 1, 4)
        q, k = rope(F.normalize(q, dim=-1) * math.sqrt(D // self.H)), rope(F.normalize(k, dim=-1))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + self.o(self.no(a))
        h = F.relu(self.up(self.n2(x))).square()
        return x + self.down(self.nh(h))


class TernTF(nn.Module):
    def __init__(self, c, stream_path=None):
        super().__init__()
        self.c = c
        t, _, _ = counts(c)
        st = Store(stream_path, t) if stream_path else None
        self.ckpt = st is not None or DEV == 'cuda'  # streamed or GPU: recompute each block in backward (one block's activations at a time)
        self.emb = nn.Embedding(c['vocab'], c['d'])  # fp for small vocab; ternary table for the big presets
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(c, st) for _ in range(c['L']))
        self.nf = RMS(c['d'])

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            if self.ckpt and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False)
            else:
                x = b(x)
        return self.nf(x) @ self.emb.weight.t()  # tied readout


# ---------------------------------------------------------------- memory math
def counts(c):
    d, L, ff, V = c['d'], c['L'], c['ff'], c['vocab']
    tern_ = L * (4 * d * d + 2 * d * ff)
    table = V * d
    fp = L * (4 * d + ff + 4 * d + d)  # row scales + norm gains
    return tern_, table, fp


def info(a):
    c = PRESETS[a.preset]
    t, tab, fp = counts(c)
    big = c['vocab'] > 256
    tw = t + (tab if big else 0)
    fpn = fp + (0 if big else tab)
    G = 1e9
    print(f"preset {a.preset}: {c}")
    print(f"  {t / 1e6:,.1f}M ternary matrix weights, {tab / 1e6:,.1f}M token table "
          f"({'ternary' if big else 'fp32'}), {fpn / 1e6:,.2f}M fp params")
    print(f"  training state: {tw / G:,.2f} GB int8 latents (1 B/weight; NO fp master, NO Adam per weight)"
          f" + {fpn * 16 / G:,.3f} GB fp params with AdamW (16 B each)")
    print(f"  per layer on the GPU when streaming: {(4 * c['d'] ** 2 + 2 * c['d'] * c['ff']) / G:,.3f} GB latents"
          f" + transient fp16 grad of the largest matrix {c['d'] * c['ff'] * 2 / G:,.3f} GB")
    print(f"  inference: {tw * 1.6 / 8 / G:,.2f} GB packed 5 trits/byte (2-bit packing: {tw * 2 / 8 / G:,.2f} GB)")
    print(f"  BitNet-style fp master + Adam for comparison: {tw * 16 / G:,.0f} GB")


# ---------------------------------------------------------------- data
def batches(spec, B, T, seed=0):
    if spec.startswith('file:'):
        raw = np.frombuffer(open(spec[5:], 'rb').read(), dtype=np.uint8)
        rng = np.random.default_rng(seed)
        while True:
            i = rng.integers(0, len(raw) - T - 1, B)
            x = np.stack([raw[j:j + T + 1] for j in i]).astype(np.int64)
            yield torch.from_numpy(x).to(DEV)
    if spec.startswith('parquet:'):  # parquet:<dir or glob>: the text column of every shard, as UTF-8 bytes
        import glob, pyarrow.parquet as pq
        files = sorted(glob.glob(spec[8:] if any(ch in spec for ch in '*?') else os.path.join(spec[8:], '*.parquet')))
        rng = np.random.default_rng(seed)
        buf = np.zeros(0, np.uint8)
        while True:
            for fn in rng.permutation(files):
                pf = pq.ParquetFile(fn)
                for g in rng.permutation(pf.num_row_groups):
                    txt = '\n\n'.join(pf.read_row_group(int(g), columns=['text']).column(0).to_pylist())
                    buf = np.frombuffer(txt.encode('utf-8', 'replace'), np.uint8)
                    for _ in range(max(1, len(buf) // (B * T))):
                        i = rng.integers(0, len(buf) - T - 1, B)
                        yield torch.from_numpy(np.stack([buf[j:j + T + 1] for j in i]).astype(np.int64)).to(DEV)
    raise SystemExit(f'data {spec}: only file: is wired here; hfds: streaming is bintf2.py\'s loader (port it on GPU)')


# ---------------------------------------------------------------- train
def save(model, opt, step, run):
    os.makedirs(run, exist_ok=True)
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'cfg': model.c}, os.path.join(run, 'ckpt.pt'))
    json.dump({'step': step}, open(os.path.join(run, 'state.json'), 'w'))  # written last: marks it complete
    stamp = os.path.join(run, 'hf_last_upload')
    if HF_REPO and os.path.exists(stamp) and time.time() - os.path.getmtime(stamp) < HF_EVERY_H * 3600:
        return  # frugal with HF: local save only
    if HF_REPO:
        import subprocess
        open(stamp, 'w').write(str(step))
        subprocess.Popen([sys.executable, os.path.abspath(__file__), 'upload', '--run', run, '--repo', HF_REPO],
                         stdout=open(os.path.join(run, 'upload.log'), 'a'), stderr=subprocess.STDOUT)


HF_REPO = None
HF_EVERY_H = 6.0


def _lowprio():
    try:
        os.nice(19)
        import subprocess
        subprocess.run(['ionice', '-c3', '-p', str(os.getpid())], check=False)
    except Exception as e:
        print('lowprio:', e, flush=True)


def upload(a):
    """Frugal public backup, ONE copy only: the ternary export (5 trits/byte), its layout meta, ckpt.pt (fp state),
    state.json, config and code at latest/. Every other file in the repo is deleted in the same commit and the git
    history is squashed, so HF keeps just the newest. Runs nice 19 / ionice idle. HF_TOKEN from the environment."""
    _lowprio()
    import shutil
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
    api = HfApi()
    api.create_repo(a.repo, private=False, exist_ok=True)
    stream = os.path.exists(os.path.join(a.run, 'latents.i8'))
    out = os.path.join(a.run, 'latest_trits.bin' if stream else 'latest_trits.pt')
    export(argparse.Namespace(run=a.run, out=out, dead=TO.dead, preset=getattr(a, 'preset', '64b')))
    files = [(out, 'latest/model_trits.bin' if stream else 'latest/model_trits.pt'), (os.path.abspath(__file__), 'latest/terntf.py')]
    if stream:
        files.append((out + '.json', 'latest/model_trits.json'))
    ck, st = os.path.join(a.run, 'ckpt.pt'), os.path.join(a.run, 'state.json')
    tmpck = os.path.join(a.run, 'upload_ckpt.pt')
    if os.path.exists(ck) and os.path.exists(st) and os.path.getmtime(st) >= os.path.getmtime(ck):
        shutil.copyfile(ck, tmpck)  # a consistent copy: state.json is written after ckpt.pt
        files += [(tmpck, 'latest/ckpt.pt'), (st, 'latest/state.json')]
    keep = {n for _, n in files}
    ops = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=f) for f, n in files]
    ops += [CommitOperationDelete(path_in_repo=f) for f in api.list_repo_files(a.repo)
            if f not in keep and f != '.gitattributes']
    api.create_commit(a.repo, operations=ops, commit_message='latest backup (single copy)')
    try:
        api.super_squash_history(a.repo, commit_message='latest backup (single copy, history squashed)')
    except Exception as e:
        print('squash failed:', e, flush=True)
    for f in (out, out + '.json', tmpck):
        if os.path.exists(f):
            os.remove(f)
    open(os.path.join(a.run, 'hf_last_upload'), 'w').write(open(st).read() if os.path.exists(st) else 'no-ckpt')
    print('uploaded', a.repo, sorted(keep), flush=True)


def train(a):
    global HF_REPO, HF_EVERY_H
    HF_REPO = a.hf_repo or None
    HF_EVERY_H = a.hf_every_h
    TO.dead, TO.kick = a.dead, a.kick
    c = dict(PRESETS[a.preset])
    if a.ctx:
        c['ctx'] = a.ctx
    if a.vocab:
        c['vocab'] = a.vocab
    elif a.data.startswith(('file:', 'parquet:')) and c['vocab'] > 256:
        print(f"byte data: vocab {c['vocab']} -> 256 (UTF-8 bytes; the big table would not fit the GPU)", flush=True)
        c['vocab'] = 256
    torch.manual_seed(a.seed)
    os.makedirs(a.run, exist_ok=True)
    model = TernTF(c, os.path.join(a.run, 'latents.i8') if a.stream else None).to(DEV)  # MoE / DiffusionBlocks / tiers of the 8B are not ported yet: dense decoder
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    step = 0
    ck = os.path.join(a.run, 'ckpt.pt')
    if a.resume and os.path.exists(ck):
        s = torch.load(ck, map_location=DEV)
        model.load_state_dict(s['model']); opt.load_state_dict(s['opt'])
        step = json.load(open(os.path.join(a.run, 'state.json')))['step']
        print(f'resumed at step {step}')
    t, tab, fp = counts(c)
    print(f'{t / 1e6:,.1f}M ternary weights (int8 latents) + {sum(p.numel() for p in model.parameters()) / 1e6:,.2f}M fp params on {DEV}')
    data = batches(a.data, a.batch, c['ctx'], a.seed)
    t0 = last_save = time.time()
    S = 1.0 if DEV == 'cpu' else 1024.0
    good = streak = skipped = 0
    for it in range(a.steps):
        mult = min(1.0, (step + 1) / max(1, a.warmup))
        TO.a = a.sign_lr * mult
        for gopt in opt.param_groups:
            gopt['lr'] = a.lr * mult
        TO.count = (step % a.log_every == 0)
        TO.flips = TO.seen = 0
        x = next(data)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        TO.bad = 0
        if FUSED is not None:
            FUSED.reset()
        fine = bool(torch.isfinite(loss))
        if fine:
            TO.inv_scale = 1.0 / S
            (loss * S).backward()  # ternary latents are updated inside this
            if FUSED is not None:  # one host read per step for the whole model
                TO.bad = int(FUSED.bad_t)
                if TO.count:
                    TO.flips = int(FUSED.flips_t)
            gs = [p.grad for p in model.parameters() if p.grad is not None]
            fine = TO.bad == 0 and bool(torch.isfinite(sum(g.sum() for g in gs)))
            if fine:
                for g in gs:
                    g.mul_(1.0 / S)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        # bintf2 46deeb0b overflow-streak handling: back the loss scale off, count steps skipped in a row,
        # grow it again after 200 good steps; stop rather than spin when the streak runs away
        if fine:
            good, streak = good + 1, 0
            if good % 200 == 0:
                S = min(S * 2.0, 65536.0)
        else:
            S, good, skipped, streak = max(1.0, S / 2.0), 0, skipped + 1, streak + 1
            if streak >= a.max_streak:
                save(model, opt, step, a.run)
                raise SystemExit(f'{streak} overflowing steps in a row at step {step}: saved and stopped')
        step += 1
        if a.save_min and time.time() - last_save > a.save_min * 60:
            save(model, opt, step, a.run); last_save = time.time()
        if TO.count or it == a.steps - 1:
            zeros = sum(int((m.lat().abs() <= TO.dead).sum()) for m in model.modules() if isinstance(m, TernLinear)) if not a.stream else 0
            print(f'step {step} loss {loss.item():.4f} bits/byte {loss.item() / math.log(2):.3f} | flips '
                  f'{100 * TO.flips / max(1, TO.seen):.3f}% | zero share {100 * zeros / t:.1f}% | scale {S:g} skipped {skipped} | {(time.time() - t0) / (it + 1):.2f}s/step | gpu {torch.cuda.max_memory_allocated() / 1e9 if DEV == "cuda" else 0:.2f}GB',
                  flush=True)
    save(model, opt, step, a.run)
    print(f'saved {a.run}')


def pack_trits(s):
    """{-1,0,1} int tensor -> uint8, 5 trits per byte."""
    v = (s.flatten().to(torch.int64) + 1).numpy()
    v = np.concatenate([v, np.zeros((-len(v)) % 5, np.int64)]).reshape(-1, 5)
    return torch.from_numpy((v * np.array([1, 3, 9, 27, 81])).sum(1).astype(np.uint8))


def export_stream(a):
    """Streamed run: pack latents.i8 straight to trits, chunk by chunk (never the whole model in RAM, no int8 copy on
    disk). trit = sign(P) if |P| > dead else 0, stored +1 in base 3, 5 trits per byte, little end first (same code as
    pack_trits). Matrices are back to back in the Store order: per block qkv (3d,d), o (d,d), up (ff,d), down (d,ff),
    row-major; the layout goes to <out>.json. Page cache of what was read is dropped so the trainer keeps its cache."""
    src = os.path.join(a.run, 'latents.i8')
    total = os.path.getsize(src)
    ckp = os.path.join(a.run, 'ckpt.pt')
    cfg = torch.load(ckp, map_location='cpu', weights_only=False)['cfg'] if os.path.exists(ckp) else dict(PRESETS[getattr(a, 'preset', '64b')])
    d, ff, layout, off = cfg['d'], cfg['ff'], [], 0
    for L in range(cfg['L']):
        for n, sh in (('qkv', (3 * d, d)), ('o', (d, d)), ('up', (ff, d)), ('down', (d, ff))):
            layout.append({'name': f'blocks.{L}.{n}.P', 'shape': sh, 'offset': off}); off += sh[0] * sh[1]
    if off != total:
        raise SystemExit(f'latents.i8 is {total} bytes but cfg says {off} weights')
    W = np.array([1, 3, 9, 27, 81], np.uint8)
    CH = 5 << 24  # 80 MB of latents per chunk, a multiple of 5 so chunks pack independently
    zeros = 0
    fd = os.open(src, os.O_RDONLY)
    with open(a.out + '.part', 'wb') as f:
        for i in range(0, total, CH):
            n = min(CH, total - i)
            x = np.frombuffer(os.pread(fd, n, i), np.int8)
            t = (np.sign(x) * (np.abs(x.astype(np.int16)) > a.dead) + 1).astype(np.uint8)
            zeros += int((t == 1).sum())
            if n % 5:
                t = np.concatenate([t, np.ones((-n) % 5, np.uint8)])
            f.write((t.reshape(-1, 5) * W).sum(1, dtype=np.uint8).tobytes())
            try:
                os.posix_fadvise(fd, i, n, os.POSIX_FADV_DONTNEED)
            except Exception:
                pass
            if (i // CH) % 64 == 0:
                print(f'pack {100 * i / total:.1f}%', flush=True)
    os.close(fd)
    os.replace(a.out + '.part', a.out)
    meta = {'format': 'terntf-trits-stream-1', 'cfg': cfg, 'dead': a.dead, 'n_weights': total, 'trits_per_byte': 5,
            'encoding': 'byte = sum(t_k * 3**k, k=0..4), t = trit + 1', 'zero_share': zeros / total, 'layout': layout,
            'fp_state': 'latest/ckpt.pt (model + opt state_dict, fp; latents excluded)'}
    json.dump(meta, open(a.out + '.json', 'w'))
    print(f'{a.out}: {os.path.getsize(a.out) / 1e9:,.2f} GB, zero share {100 * zeros / total:.1f}%', flush=True)


def export(a):
    if os.path.exists(os.path.join(a.run, 'latents.i8')):
        return export_stream(a)
    s = torch.load(os.path.join(a.run, 'ckpt.pt'), map_location='cpu')
    TO.dead = a.dead
    out = {'format': 'terntf-trits-1', 'cfg': s['cfg'], 'dead': a.dead, 'trits': {}, 'shape': {}, 'fp': {}}
    for k, v in s['model'].items():
        if k.endswith('.P'):
            out['trits'][k] = pack_trits(state_of(v.float()).to(torch.int8)); out['shape'][k] = tuple(v.shape)
        elif not k.endswith(('.R', '.C', '.t')):
            out['fp'][k] = v.half()
    torch.save(out, a.out)
    print(f'{a.out}: {os.path.getsize(a.out) / 1e6:,.1f} MB')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sp = sub.add_parser('info'); sp.add_argument('--preset', default='64b', choices=PRESETS)
    sp = sub.add_parser('train')
    sp.add_argument('--preset', default='smoke', choices=PRESETS)
    sp.add_argument('--data', required=True)
    sp.add_argument('--run', required=True)
    sp.add_argument('--resume', action='store_true')
    sp.add_argument('--steps', type=int, default=100)
    sp.add_argument('--batch', type=int, default=8)
    sp.add_argument('--ctx', type=int, default=0)
    sp.add_argument('--lr', type=float, default=2e-3, help='AdamW on the full-precision parameters (scales, norms, fp table)')
    sp.add_argument('--sign-lr', type=float, default=4.0, help='step size on the int8 ternary latents, in int8 units')
    sp.add_argument('--dead', type=int, default=24, help='|latent| <= dead reads as 0 (the zero state)')
    sp.add_argument('--kick', type=int, default=4, help='hysteresis: units added when a latent changes state')
    sp.add_argument('--warmup', type=int, default=10)
    sp.add_argument('--log-every', type=int, default=5)
    sp.add_argument('--vocab', type=int, default=0, help='override the preset vocab (byte data defaults to 256)')
    sp.add_argument('--hf-every-h', type=float, default=6.0, help='push to HF at most this often (local saves keep --save-min)')
    sp.add_argument('--stream', type=int, default=0, help='(GPU port) keep only the layer in use on the GPU; the rest in RAM/disk')
    sp.add_argument('--seed', type=int, default=0)
    sp.add_argument('--hf-repo', default='', help='after every save, push a ternary export to this PUBLIC repo (latest/ only)')
    sp.add_argument('--max-streak', type=int, default=50)
    sp.add_argument('--save-min', type=float, default=30, help='checkpoint every this many minutes (0 = end only)')
    sp = sub.add_parser('export'); sp.add_argument('--run', required=True); sp.add_argument('--out', required=True)
    sp.add_argument('--dead', type=int, default=24)
    sp = sub.add_parser('upload'); sp.add_argument('--run', required=True); sp.add_argument('--repo', required=True)
    a = ap.parse_args()
    {'info': info, 'train': train, 'export': export, 'upload': upload}[a.cmd](a)


if __name__ == '__main__':
    main()
