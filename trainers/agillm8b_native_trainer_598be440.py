#!/usr/bin/env python3
"""Native packed MoE transformer with successive, independently differentiated sublayers.

Only the selected attention OR MoE sublayer executes during an optimizer step.
Its detached output becomes the next sublayer's input on the next step. There
is no frozen-prefix recomputation and no cross-sublayer backward graph. Each
sublayer uses the shared tied token head for a local AR/NAT/SAT objective.
All matrix weights, including routers and the tied head, are authoritative
packed sign bits. Floating tensors are transient activations/gradients, not
weight masters. This is an experimental local-learning algorithm; it is not
claimed equivalent to end-to-end training or a fused binary GEMM kernel.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

VERSION = '1.1.0-native-local'
FORMAT = 'agillm-native-local-packed-v2'
LIMIT = 20_000_000_000
STOP = False


def emit(**values):
    print(json.dumps(values, sort_keys=True, allow_nan=False), flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def unpack(bits, dtype):
    shifts = torch.arange(8, dtype=torch.uint8, device=bits.device)
    v = ((bits[..., None] >> shifts) & 1).reshape(bits.shape[0], -1)
    return v.to(dtype).mul_(2).sub_(1)


def pack(positive):
    x = positive.to(torch.uint8).reshape(positive.shape[0], -1, 8)
    shifts = torch.arange(8, dtype=torch.uint8, device=x.device)
    return (x << shifts).sum(-1).to(torch.uint8)


class Bank:
    def __init__(self, name, rows, cols, seed, tile=2048):
        if rows <= 0 or cols <= 0 or cols % 8:
            raise ValueError('Invalid packed geometry')
        self.name, self.rows, self.cols, self.tile = name, rows, cols, tile
        g = torch.Generator(device='cpu').manual_seed(seed)
        self.bits = torch.randint(0, 256, (rows, cols // 8), dtype=torch.uint8, generator=g)
        self.scale = cols ** -0.5
        self.revision = 0
        self.pending = None
        self.changed = 0
        self.seed = seed
        self.probability = 0.0
        self.learning = False

    def to(self, device):
        if self.pending is not None:
            raise RuntimeError('Cannot move an uncommitted proposal')
        self.bits = self.bits.to(device)
        return self

    def linear(self, x):
        return Packed.apply(x, self)

    def audit(self):
        if self.bits.dtype != torch.uint8 or tuple(self.bits.shape) != (self.rows, self.cols // 8):
            raise RuntimeError('Persistent matrix state is not bit packed')
        tensors = {k for k, v in vars(self).items() if isinstance(v, torch.Tensor)}
        if tensors != ({'bits', 'pending'} if self.pending is not None else {'bits'}):
            raise RuntimeError('Unexpected persistent tensor bank')
        if self.pending is not None and (self.pending.dtype != torch.uint8 or self.pending.shape != self.bits.shape):
            raise RuntimeError('Proposal is not packed')


class Packed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bank):
        if x.shape[-1] != bank.cols or x.device != bank.bits.device:
            raise ValueError('Input and bank geometry/device differ')
        flat = x.reshape(-1, bank.cols).contiguous()
        out = torch.empty((len(flat), bank.rows), dtype=x.dtype, device=x.device)
        for lo in range(0, bank.rows, bank.tile):
            hi = min(lo + bank.tile, bank.rows)
            w = unpack(bank.bits[lo:hi], x.dtype)
            out[:, lo:hi] = (flat @ w.T) * bank.scale
        ctx.bank, ctx.revision, ctx.shape = bank, bank.revision, x.shape
        ctx.save_for_backward(flat)
        return out.reshape(*x.shape[:-1], bank.rows)

    @staticmethod
    def backward(ctx, grad):
        bank = ctx.bank
        if ctx.revision != bank.revision:
            raise RuntimeError('Packed weights changed inside backward')
        (x,) = ctx.saved_tensors
        gy = grad.reshape(-1, bank.rows).contiguous()
        if not torch.isfinite(gy).all() or not torch.isfinite(x).all():
            raise FloatingPointError('Nonfinite activation/derivative; no update committed')
        dx = torch.zeros_like(x)
        proposal = torch.zeros_like(bank.bits) if bank.learning else None
        count = torch.zeros((), dtype=torch.int64, device=x.device)
        rng = torch.Generator(device=x.device).manual_seed(bank.seed)
        for lo in range(0, bank.rows, bank.tile):
            hi = min(lo + bank.tile, bank.rows)
            weight = unpack(bank.bits[lo:hi], x.dtype)
            dx.add_((gy[:, lo:hi] @ weight) * bank.scale)
            if proposal is not None:
                gradient = (gy[:, lo:hi].float().T @ x.float()) * bank.scale
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError('Nonfinite tile gradient; no update committed')
                # Independent row normalization stops frequent vocabulary rows
                # from determining every other row's discrete update scale.
                den = gradient.abs().mean(-1, keepdim=True).clamp_min(1e-20)
                chance = (gradient.abs() / den * bank.probability).clamp_max_(0.05)
                flips = (gradient * weight.float() > 0) & (torch.rand(
                    gradient.shape, generator=rng, device=x.device) < chance)
                proposal[lo:hi] = pack(flips)
                count += flips.sum()
        if proposal is not None:
            if bank.pending is not None:
                raise RuntimeError('A matrix was used twice before its local commit')
            bank.pending, bank.changed = proposal, int(count.item())
        return dx.reshape(ctx.shape), None


@dataclass(frozen=True)
class Config:
    width: int = 2048
    layers: int = 24
    groups: int = 3
    experts: int = 32
    top_k: int = 2
    hidden: int = 2304
    heads: int = 16
    vocab: int = 129280
    mask_id: int = 128800
    seed: int = 1701
    tile: int = 2048

    def validate(self):
        if (self.width % 8 or self.hidden % 8 or self.layers % self.groups or
            self.width % self.heads or (self.width // self.heads) % 2 or
            not 0 < self.top_k <= self.experts or not 0 <= self.mask_id < self.vocab):
            raise ValueError('Invalid transformer geometry')

    def count(self):
        return self.vocab * self.width + self.layers * (4 * self.width**2 +
            self.width * self.experts + 2 * self.experts * self.width * self.hidden)


class Model:
    def __init__(self, cfg: Config):
        cfg.validate()
        self.cfg = cfg
        self.banks: dict[str, Bank] = {}
        self.units: list[list[str]] = []
        self.events: list[int] = []
        self.head_gains = [0.1] * cfg.groups  # Three scalar calibrations, not floating matrix masters.
        def new(name, rows, cols):
            bank = Bank(name, rows, cols, cfg.seed + len(self.banks) * 1009, cfg.tile)
            self.banks[name] = bank
            return name
        self.table = self.banks[new('shared.table', cfg.vocab, cfg.width)]
        for layer in range(cfg.layers):
            a = f'l{layer}.attn'
            self.units.append([new(a+'.qkv', 3*cfg.width, cfg.width), new(a+'.out', cfg.width, cfg.width)])
            m = f'l{layer}.moe'
            names = [new(m+'.router', cfg.experts, cfg.width)]
            for expert in range(cfg.experts):
                names.extend([new(m+f'.e{expert}.up', cfg.hidden, cfg.width),
                              new(m+f'.e{expert}.down', cfg.width, cfg.hidden)])
            self.units.append(names)
        self.active = None
        if sum(b.rows*b.cols for b in self.banks.values()) != cfg.count():
            raise RuntimeError('Initialized parameter count mismatch')

    def activate(self, unit, device):
        if self.active is not None and self.active != unit:
            for name in self.units[self.active]:
                self.banks[name].to('cpu')
        self.table.to(device)
        for name in self.units[unit]:
            self.banks[name].to(device)
        self.active = unit
        return [self.table] + [self.banks[n] for n in self.units[unit]]

    def embedding(self, ids, dtype):
        flat = ids.reshape(-1)
        return unpack(self.table.bits[flat], dtype).reshape(*ids.shape, self.cfg.width)

    @staticmethod
    def rms(x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5).to(x.dtype)

    def unit(self, x, unit, causal):
        if unit != self.active:
            raise RuntimeError('Inactive sublayer execution rejected')
        self.events.append(unit)
        cfg, names = self.cfg, self.units[unit]
        B, T, D = x.shape
        gain = (2 * cfg.layers / cfg.groups) ** -0.5
        u = self.rms(x)
        if unit % 2 == 0:
            q, k, v = self.banks[names[0]].linear(u).chunk(3, -1)
            H, K = cfg.heads, D // cfg.heads
            q, k, v = [z.reshape(B, T, H, K).transpose(1, 2) for z in (q, k, v)]
            # Rotary position information, recomputed as bounded transient state.
            pos = torch.arange(T, device=x.device, dtype=torch.float32)
            freq = torch.exp(-math.log(10000) * torch.arange(0, K, 2, device=x.device).float() / K)
            angle = pos[:, None] * freq
            cs, sn = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
            def rotate(z):
                a, b = z[..., 0::2], z[..., 1::2]
                return torch.stack((a*cs-b*sn, a*sn+b*cs), -1).flatten(-2)
            q, k = rotate(q), rotate(k)
            scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(K)
            if causal:
                mask = torch.ones((T, T), device=x.device, dtype=torch.bool).triu(1)
                scores = scores.masked_fill(mask, float('-inf'))
            a = (scores.softmax(-1).to(x.dtype) @ v).transpose(1, 2).reshape(B, T, D)
            return x + gain*self.banks[names[1]].linear(a), x.new_zeros(())
        u = u.reshape(-1, D)
        logits = self.banks[names[0]].linear(u).float()
        chosen = torch.argsort(logits, descending=True, stable=True, dim=-1)[:, :cfg.top_k]
        gate = logits.gather(1, chosen).softmax(-1).to(u.dtype)
        out = torch.zeros_like(u)
        for e in range(cfg.experts):
            token, slot = torch.where(chosen == e)
            if token.numel() == 0:
                continue
            up, down = self.banks[names[1+2*e]], self.banks[names[2+2*e]]
            y = down.linear(F.relu(up.linear(u[token])).square())
            out = out.index_add(0, token, y*gate[token, slot, None])
        usage = F.one_hot(chosen, cfg.experts).float().mean((0, 1))
        balance = cfg.experts * (logits.softmax(-1).mean(0) * usage).sum()
        return x + gain*out.reshape(B, T, D), balance

    def audit(self):
        for bank in self.banks.values():
            bank.audit()
        return dict(parameters=self.cfg.count(), packed_weight_bytes=sum(b.bits.numel() for b in self.banks.values()),
                    floating_weight_master_bytes=0, int8_shadow_bytes=0,
                    per_weight_optimizer_bytes=0, floating_auxiliary_scalars=len(self.head_gains), packed_matrices=len(self.banks),
                    sublayers=len(self.units), independent_groups=self.cfg.groups,
                    backend='packed_authoritative_state_with_transient_decode_and_pytorch_gemm')

    def digest(self):
        return {name: hashlib.sha256(b.bits.cpu().numpy().tobytes()).hexdigest() for name, b in self.banks.items()}


def objective(tokens, batch_index, cfg):
    g = torch.Generator(device='cpu').manual_seed(cfg.seed+batch_index*13)
    mode = ('ar', 'sat2', 'sat3', 'sat_variable', 'nat')[batch_index % 5]
    if mode == 'ar':
        return tokens[:, :-1].clone(), tokens[:, 1:].clone(), torch.ones_like(tokens[:, 1:], dtype=torch.bool), True, mode, 0., 1
    truth, ids = tokens[:, :-1].clone(), tokens[:, :-1].clone()
    mask = torch.zeros_like(ids, dtype=torch.bool)
    if mode == 'nat':
        group = batch_index % cfg.groups
        rate = (cfg.groups-group-0.5) / cfg.groups
        mask = torch.rand(ids.shape, generator=g) < rate
        mask[:, -1] = True
        stride = 0
    else:
        stride = {'sat2': 2, 'sat3': 3}.get(mode)
        if stride is None:
            stride = int(torch.randint(1, 4, (), generator=g))
        mask[:, -stride:] = True
    ids[mask] = cfg.mask_id
    return ids, truth, mask, False, mode, float(mask.float().mean()), stride


class Corpus:
    def __init__(self, path, cfg, sequence, batch, limit_tokens=None):
        self.path = Path(path)
        self.cfg, self.sequence, self.batch = cfg, sequence, batch
        if not self.path.is_file() or self.path.stat().st_size % 4:
            raise ValueError('Completed aligned int32 token pool required')
        self.n = self.path.stat().st_size // 4
        if limit_tokens is not None and self.n != limit_tokens:
            raise ValueError('Corpus size changed since checkpoint')
        self.data = np.memmap(self.path, mode='r', dtype='<i4')
        self.validation_end = min(262144, self.n // 20)
        if self.n - self.validation_end <= batch*(sequence+1):
            raise ValueError('Not enough disjoint training and validation tokens')
        h = hashlib.sha256()
        with self.path.open('rb') as f:
            h.update(f.read(min(1<<20, self.path.stat().st_size)))
            if self.path.stat().st_size > 1<<20:
                f.seek(-min(1<<20, self.path.stat().st_size), 2)
                h.update(f.read())
        self.binding = dict(path=str(self.path), tokens=self.n, prefix_suffix_sha256=h.hexdigest(),
                            tokenizer='deepseek-v4.1', source='FineWeb-Edu completed token pool',
                            validation_end=self.validation_end)

    def batch_tokens(self, batch_index, validation=False):
        g = np.random.default_rng(self.cfg.seed + batch_index*100003)
        low = 0 if validation else self.validation_end
        high = self.validation_end if validation else self.n
        starts = g.integers(low, high-self.sequence-1, size=self.batch)
        a = np.stack([self.data[int(i):int(i)+self.sequence+1] for i in starts]).astype(np.int64)
        if a.min() < 0 or a.max() >= self.cfg.vocab or (a == self.cfg.mask_id).any():
            raise ValueError('Corpus vocabulary mismatch or reserved mask in target')
        return torch.from_numpy(a)


def pipeline_start(model, tokens, batch_index, device, dtype):
    ids, truth, mask, causal, mode, noise, stride = objective(tokens, batch_index, model.cfg)
    group = batch_index % model.cfg.groups
    unit = group * (len(model.units)//model.cfg.groups)
    model.activate(unit, device)
    with torch.no_grad():
        h = model.embedding(ids.to(device), dtype)
        freq = torch.exp(-math.log(10000)*torch.arange(model.cfg.width, device=device).float()/model.cfg.width)
        h = h + (0.1*torch.sin(noise*17*freq)).to(dtype)
    return dict(h=h, truth=truth.to(device), mask=mask.to(device), causal=causal,
                mode=mode, noise=noise, stride=stride, group=group, offset=0)


def local_step(model, pipeline, step, device, probability, head_probability):
    depth = len(model.units)//model.cfg.groups
    unit = pipeline['group']*depth + pipeline['offset']
    active = model.activate(unit, device)
    for index, bank in enumerate(active):
        bank.learning = True
        bank.probability = head_probability if bank is model.table else probability
        bank.seed = model.cfg.seed + step*1000003 + (unit+1)*1009 + index*97
    model.events.clear()
    x = pipeline['h'].detach().to(device).requires_grad_(True)
    try:
        h, balance = model.unit(x, unit, pipeline['causal'])
        mask = pipeline['mask'].to(device)
        truth = pipeline['truth'].to(device)
        gain = torch.tensor(model.head_gains[pipeline['group']], device=device, requires_grad=True)
        logits = model.table.linear(model.rms(h)[mask]).float() * gain
        ce = F.cross_entropy(logits.float(), truth[mask])
        loss = ce + 0.01*balance
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite local loss; no update committed')
        loss.backward()
        if model.events != [unit]:
            raise RuntimeError('Independent sublayer execution invariant failed')
        if gain.grad is None or not torch.isfinite(gain.grad):
            raise FloatingPointError('Nonfinite head calibration gradient')
        next_gain = max(0.025, min(8.0, float(gain.detach()) - 0.001*float(gain.grad)))
        changed = sum(b.changed for b in active if b.pending is not None)
        updates = [b for b in active if b.pending is not None]
        for bank in updates:
            bank.audit()
        # All proposals were validated before any weight is changed.
        for bank in updates:
            bank.bits.bitwise_xor_(bank.pending)
            bank.revision += 1
        model.head_gains[pipeline['group']] = next_gain
        result = dict(step=step+1, loss=float(ce.detach()), balance=float(balance.detach()), head_gain=next_gain,
                      changed_bits=changed, updated_matrices=len(updates), active_sublayer=unit,
                      group=pipeline['group'], mode=pipeline['mode'], stride=pipeline['stride'],
                      objective_tokens=int(mask.sum()), forward_units=list(model.events),
                      frozen_prefix_calls=0)
        pipeline['h'] = h.detach()
        pipeline['offset'] += 1
        return result
    finally:
        for bank in active:
            bank.pending = None
            bank.changed = 0
            bank.learning = False


def checkpoint(path, model, state, pipeline):
    if any(b.pending is not None for b in model.banks.values()):
        raise RuntimeError('Checkpoint requires completed local update')
    audit = model.audit()
    pl = None if pipeline is None else {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in pipeline.items()}
    payload = dict(format=FORMAT, version=VERSION, config=asdict(model.cfg),
                   weights={k:b.bits.detach().cpu() for k,b in model.banks.items()},
                   state=state, pipeline=pl, audit=audit, head_gains=list(model.head_gains))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Immutable checkpoint already exists')
    fd, temp = tempfile.mkstemp(prefix='.native-checkpoint-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            torch.save(payload, f)
            f.flush(); os.fsync(f.fileno())
        size = os.path.getsize(temp)
        if size >= LIMIT:
            raise RuntimeError('Complete resume checkpoint exceeds 20GB')
        os.rename(temp, path)
        return dict(path=str(path), bytes=size, sha256=sha256(path), step=state['step'])
    finally:
        if os.path.exists(temp): os.unlink(temp)


def restore(path, device='cpu'):
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if obj.get('format') != FORMAT:
        raise ValueError('Not a native-local complete resume checkpoint')
    cfg = Config(**obj['config'])
    model = Model(cfg)
    if set(model.banks) != set(obj['weights']):
        raise ValueError('Checkpoint bank set mismatch')
    for name, value in obj['weights'].items():
        bank = model.banks[name]
        if value.dtype != torch.uint8 or value.shape != bank.bits.shape:
            raise ValueError('Unpacked, latent or shape-incompatible checkpoint rejected')
        bank.bits = value
    gains = obj['head_gains']
    if len(gains) != cfg.groups or any(not math.isfinite(v) or not 0.025 <= v <= 8 for v in gains):
        raise ValueError('Invalid scalar head calibration state')
    model.head_gains = list(gains)
    model.audit()
    pl = obj['pipeline']
    if pl:
        pl = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in pl.items()}
    return model, obj['state'], pl


def test(device):
    cfg = Config(width=32, layers=6, groups=3, experts=4, top_k=2, hidden=48,
                 heads=4, vocab=128, mask_id=127, tile=32)
    dtype = torch.float16 if device.startswith('cuda') else torch.float32
    m = Model(cfg)
    gen = torch.Generator().manual_seed(42)
    truth = torch.randint(0, 126, (2, 9), generator=gen)
    covered, strides, steps = set(), set(), 0
    p = None
    for batch_index in range(15):
        p = pipeline_start(m, truth, batch_index, device, dtype)
        for sub in range(4):
            before = m.digest()
            r = local_step(m, p, steps, device, .025, .01)
            after = m.digest()
            allowed = set(m.units[r['active_sublayer']]) | {'shared.table'}
            changes = {k for k in before if before[k] != after[k]}
            assert changes and changes <= allowed, (changes, allowed)
            assert r['forward_units'] == [r['active_sublayer']]
            covered.add(r['mode'])
            if r['mode'] == 'sat_variable': strides.add(r['stride'])
            steps += 1
    assert covered == {'ar','sat2','sat3','sat_variable','nat'}
    for bi in range(3, 203, 5):
        ids, targets, masked, causal, mode, noise, stride = objective(truth, bi, cfg)
        strides.add(stride)
        assert torch.equal(ids[~masked], targets[~masked])
        assert (ids[masked] == cfg.mask_id).all()
    assert strides == {1, 2, 3}, strides
    b = Bank('gradient_check', 16, 32, 811, tile=8)
    b.to(device)
    x = torch.randn(2, 32, device=device, dtype=dtype, requires_grad=True)
    gy = torch.randn(2, 16, device=device, dtype=dtype)
    b.linear(x).backward(gy)
    expected = (gy @ unpack(b.bits, dtype)) * b.scale
    assert torch.allclose(x.grad, expected, atol=0.004 if dtype==torch.float16 else 1e-6, rtol=0.004 if dtype==torch.float16 else 1e-5)
    with tempfile.TemporaryDirectory(prefix='packed-local-canary-') as d:
        p = pipeline_start(m, truth, 15, device, dtype)
        r = local_step(m, p, steps, device, .025, .01)
        steps += 1
        ck = Path(d)/'resume.pt'
        receipt = checkpoint(ck, m, {'step':steps}, p)
        n, st, pp = restore(ck, device)
        r1 = local_step(m, p, steps, device, .025, .01)
        r2 = local_step(n, pp, steps, device, .025, .01)
        assert m.digest() == n.digest(), 'Packed next-update resume mismatch'
        assert m.head_gains == n.head_gains, 'Scalar calibration resume mismatch'
        assert r1 == r2, (r1, r2)
        assert torch.equal(p['h'], pp['h']), 'Boundary activation resume mismatch'
        # Failure must not modify any packed bank.
        original = m.digest()
        p['h'].fill_(float('nan'))
        try:
            local_step(m, p, steps+1, device, .025, .01)
            raise AssertionError('Expected rejection')
        except FloatingPointError:
            pass
        assert m.digest() == original
    emit(event='canary_pass', device=device, completed_local_steps=steps+2,
         exact_next_step_resume=True, inactive_banks_unchanged=True,
         all_objectives=sorted(covered), sampled_sat_variable_strides=sorted(strides),
         nonfinite_rejected_before_commit=True, audit=m.audit(),
         full_geometry_parameters=Config().count(), full_geometry_packed_bytes=Config().count()//8)


def main():
    global STOP
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['test','train','inspect'])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--run', type=Path)
    ap.add_argument('--data', type=Path)
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--sequence', type=int, default=128)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--probability', type=float, default=0.0001)
    ap.add_argument('--head-probability', type=float, default=0.00001)
    ap.add_argument('--save-seconds', type=int, default=1800)
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.command == 'test':
        test(args.device); return
    if args.command == 'inspect':
        obj = torch.load(args.resume, map_location='cpu', weights_only=True)
        emit(format=obj['format'], state=obj['state'], audit=obj['audit']); return
    if args.run is None or args.data is None:
        ap.error('--run and --data are required')
    if args.sequence < 4 or args.batch < 1 or args.sequence > 4096:
        ap.error('Invalid sequence/batch')
    if not (0 < args.probability <= .05 and 0 < args.head_probability <= .05):
        ap.error('Flip probabilities must be in (0,.05]')
    args.run.mkdir(parents=True,exist_ok=True)
    lock=(args.run/'trainer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.run/'trainer.pid').write_text(str(os.getpid()))
    def stop(signum, frame):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    device=args.device
    dtype=torch.float16 if device.startswith('cuda') else torch.float32
    if args.resume:
        model, state, pipeline=restore(args.resume,device)
        if state['sequence'] != args.sequence or state['batch'] != args.batch:
            raise ValueError('Resume batch geometry differs')
        if state['probability']!=args.probability or state['head_probability']!=args.head_probability:
            raise ValueError('Resume discrete optimizer settings differ')
    else:
        model=Model(Config())
        state=dict(step=0, batch_index=0, fresh_tokens=0, objective_tokens=0,
                   sequence=args.sequence,batch=args.batch,probability=args.probability,
                   head_probability=args.head_probability)
        pipeline=None
    corpus=Corpus(args.data,model.cfg,args.sequence,args.batch)
    if 'corpus' in state and state['corpus'] != corpus.binding:
        raise ValueError('Resume corpus binding differs')
    state['corpus']=corpus.binding
    audit=model.audit()
    (args.run/'architecture.json').write_text(json.dumps({'config':asdict(model.cfg),'audit':audit},indent=2))
    emit(event='native_initialized',pid=os.getpid(),audit=audit,corpus=corpus.binding,
         resume_step=state['step'],learning='one sublayer per update, detached next-boundary pipeline')
    last_save=time.monotonic()
    last_saved_step=state['step'] if args.resume else -1
    while not STOP and (not args.steps or state['step']<args.steps):
        if pipeline is None or pipeline['offset']>=len(model.units)//model.cfg.groups:
            tokens=corpus.batch_tokens(state['batch_index'])
            pipeline=pipeline_start(model,tokens,state['batch_index'],device,dtype)
            state['batch_index']+=1
            state['fresh_tokens']+=tokens.numel()
        start=time.monotonic()
        if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        result=local_step(model,pipeline,state['step'],device,args.probability,args.head_probability)
        if device.startswith('cuda'): torch.cuda.synchronize()
        dt=time.monotonic()-start
        state['step']=result['step']
        state['objective_tokens']+=result['objective_tokens']
        result.update(event='native_step',seconds=dt,objective_tokens_per_second=result['objective_tokens']/dt,
                      fresh_tokens_seen=state['fresh_tokens'],objective_tokens_seen=state['objective_tokens'],
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0)
        emit(**result)
        tmp=args.run/'status.json.tmp'
        tmp.write_text(json.dumps(result,sort_keys=True)); os.replace(tmp,args.run/'status.json')
        if state['step'] in (1,48) or time.monotonic()-last_save>=args.save_seconds:
            receipt=checkpoint(args.run/f'checkpoint-{state["step"]:09d}.pt',model,state,pipeline)
            (args.run/'latest.json').write_text(json.dumps(receipt,indent=2))
            emit(event='native_checkpoint',**receipt)
            last_save=time.monotonic();last_saved_step=state['step']
    if state['step']!=last_saved_step:
        receipt=checkpoint(args.run/f'checkpoint-{state["step"]:09d}.pt',model,state,pipeline)
        (args.run/'latest.json').write_text(json.dumps(receipt,indent=2))
        emit(event='native_checkpoint',**receipt)
    emit(event='native_stopped',step=state['step'],signal_requested=STOP)


if __name__=='__main__':
    main()
