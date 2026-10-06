#!/usr/bin/env python3
"""rpv16_infer.py - self-contained single-file inference for AGILLM RPV16 (~2B, 16-stage MoE, tied full-vocab head).

No imports from trainer files. Model code is a copy of the CPU int8 serving/eval engine
(fable cpu_eval/bundle/rpv16_cpu_server.py + convert_checkpoint.py quantisation + diag_local.py held-out),
which is what the quality-watch-v2 held-out evaluator uses. Weights are pulled straight from a v6 recovery
checkpoint via torch.load(mmap=True); optimizer state is never materialised, and the checkpoint mapping is
released as soon as the weights are converted (so a later atomic checkpoint rename does not keep the old inode alive).

Modes:
  --prompt TEXT [--prompt TEXT ...]   generate (greedy by default; --temperature/--top-p/--top-k to sample)
  --heldout                           24x512 fixed held-out CE (per-window, all-24 mean, prose-only mean)
Precision: int8 (default on CPU; matches the existing evaluator), bf16, fp32. --device cuda needs --allow-gpu.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, math, os, resource, sys, threading, time
from pathlib import Path

DEFAULT_CKPT = '/workspace/agillm-gb10-1pf-targetfix-active/RPV16-GB10-1PF-v6-current-resumable.pt'
DEFAULT_TOKENIZER = '/workspace/rpv16_sm120/tokenizer_bundle.json'
DEFAULT_HELDOUT = '/workspace/fable_trainers_20261003/cpu_eval/bundle/heldout_token_ids.json'
DEFAULT_CODE_WINDOWS = '14,15,16,17,18'   # code/whitespace windows of the fixed held-out set

VOCAB = 129_280; D = 1_280; STAGES = 16; EXPERTS = 6; FFN = 5_120
Q_HEADS = 20; KV_HEADS = 5; HEAD_DIM = 64
WINDOWS = (256,256,256,256,512,512,512,512,1024,1024,1024,1024,-1,-1,-1,-1)
STAGE_QUANT = ('.attn.q.weight', '.attn.k.weight', '.attn.v.weight', '.attn.o.weight', '.experts.')


def emit(d):
    print(json.dumps(d, allow_nan=False), flush=True)


# ----------------------------------------------------------------------------- resource monitoring
def _meminfo_avail():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return -1


def _cgroup_avail():
    try:
        mx = open('/sys/fs/cgroup/memory.max').read().strip()
        if mx == 'max':
            return None
        anon = shmem = 0
        for line in open('/sys/fs/cgroup/memory.stat'):
            k, v = line.split()
            if k == 'anon': anon = int(v)
            elif k == 'shmem': shmem = int(v)
        return int(mx) - anon - shmem
    except OSError:
        return None


def mem_available():
    vals = [v for v in (_meminfo_avail(), _cgroup_avail()) if v is not None and v >= 0]
    return min(vals) if vals else -1


def disk_free(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def proc_rss():
    out = {}
    for line in open('/proc/self/status'):
        if line.startswith(('VmHWM', 'VmRSS', 'RssAnon', 'RssFile')):
            k, v = line.split(':'); out[k] = int(v.split()[0]) * 1024
    return out


class Monitor(threading.Thread):
    """Samples MemAvailable / disk; sets .abort if floors are crossed while the checkpoint is mapped."""
    def __init__(self, disk_path, mem_floor, disk_floor):
        super().__init__(daemon=True)
        self.disk_path, self.mem_floor, self.disk_floor = disk_path, mem_floor, disk_floor
        self.min_mem = mem_available(); self.min_disk = disk_free(disk_path)
        self.peak_anon = 0; self.ckpt_open = False; self.abort = None; self._halt = threading.Event()
    def run(self):
        while not self._halt.wait(1.0):
            m, d = mem_available(), disk_free(self.disk_path)
            self.min_mem = min(self.min_mem, m); self.min_disk = min(self.min_disk, d)
            self.peak_anon = max(self.peak_anon, proc_rss().get('RssAnon', 0))
            if m < self.mem_floor:
                self.abort = f'MemAvailable {m/2**30:.1f} GiB < floor {self.mem_floor/2**30:.1f} GiB'
            if self.ckpt_open and d < self.disk_floor:
                self.abort = f'disk free {d/2**30:.1f} GiB < floor {self.disk_floor/2**30:.1f} GiB while checkpoint mapped'
            if self.abort:
                emit({'event': 'abort', 'reason': self.abort}); os._exit(3)
    def stop(self):
        self._halt.set()
    def summary(self):
        r = proc_rss()
        return {'min_mem_available_gib': round(self.min_mem / 2**30, 2), 'min_disk_free_gib': round(self.min_disk / 2**30, 2),
                'peak_rss_hwm_gib': round(r.get('VmHWM', 0) / 2**30, 2), 'peak_rss_anon_gib': round(max(self.peak_anon, r.get('RssAnon', 0)) / 2**30, 2),
                'rss_now_gib': round(r.get('VmRSS', 0) / 2**30, 2)}


# ----------------------------------------------------------------------------- tokenizer
def load_tokenizer(path):
    from tokenizers import Tokenizer
    obj = json.loads(Path(path).read_text())
    if isinstance(obj, dict) and isinstance(obj.get('tokenizer_bundle'), dict):
        raw = obj['tokenizer_bundle'].get('tokenizer.json')
        if isinstance(raw, str):
            return Tokenizer.from_str(raw)
    if isinstance(obj, dict) and isinstance(obj.get('tokenizer.json'), str):
        return Tokenizer.from_str(obj['tokenizer.json'])
    return Tokenizer.from_str(Path(path).read_text())


# ----------------------------------------------------------------------------- weights / linear layers
def unpack_sparse_mask(packed, shape):
    """Trainer's paired 4-of-8 topology (adjacent-pairs-low-nibble-first) -> bool mask. Optional (--sparse-mask)."""
    import torch
    groups = math.prod(shape) // 8
    bits = torch.empty(groups, dtype=torch.uint8)
    bits[0::2] = packed.bitwise_and(15)
    bits[1::2] = packed[: groups // 2].bitwise_right_shift(4)
    pairs = bits[:, None].bitwise_and(torch.tensor([1, 2, 4, 8], dtype=torch.uint8)).ne(0)
    return pairs.repeat_interleave(2, dim=1).reshape(shape)


class Lin:
    """y = x @ W^T. int8: torch dynamic-quantised Linear with per-tensor symmetric scale = max|W|/127
    (identical to convert_checkpoint.quantize_weight). bf16/fp32: plain F.linear."""
    def __init__(self, w, precision, device):
        import torch
        x = w.detach().to('cpu', torch.float32, copy=True).contiguous()   # always copy: never alias the checkpoint mmap
        self.precision = precision
        if precision == 'int8':
            from torch.ao.nn.quantized.dynamic import Linear as DQ
            scale = max(float(x.abs().max()) / 127.0, 1e-8)
            q = torch.quantize_per_tensor(x, scale=scale, zero_point=0, dtype=torch.qint8)
            self.layer = DQ(x.shape[1], x.shape[0], bias_=False, dtype=torch.qint8)
            self.layer.set_weight_bias(q, None); self.layer.eval()
            del q
        else:
            dt = torch.bfloat16 if precision == 'bf16' else torch.float32
            self.w = x.to(device=device, dtype=dt).contiguous()
        del x
    def __call__(self, h):
        import torch.nn.functional as F
        if self.precision == 'int8':
            return self.layer(h.float())
        return F.linear(h.to(self.w.dtype), self.w).float()


def rms_norm(x, weight):
    import torch
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-5)
    return y * weight


class Stage:
    def __init__(self, get, window, precision, device, perms):
        self.window = int(window); self.perms = perms
        self.n1 = get('n1.weight', 'f'); self.n2 = get('n2.weight', 'f'); self.router = get('router.weight', 'f')
        self.q = get('attn.q.weight', 'lin'); self.k = get('attn.k.weight', 'lin')
        self.v = get('attn.v.weight', 'lin'); self.o = get('attn.o.weight', 'lin')
        self.experts = [(get(f'experts.{i}.gate.weight', 'lin'), get(f'experts.{i}.up.weight', 'lin'),
                         get(f'experts.{i}.down.weight', 'lin')) for i in range(EXPERTS)]

    def _route(self, flat):
        import torch.nn.functional as F
        logits = F.linear(flat, self.router)
        P = self.perms
        if flat.shape[0] % EXPERTS == 0 and flat.shape[0] >= EXPERTS:
            groups = flat.shape[0] // EXPERTS
            gl = logits.view(EXPERTS, groups, EXPERTS).permute(1, 0, 2).contiguous()
            pick = P.view(1, P.shape[0], EXPERTS, 1).expand(groups, -1, -1, 1)
            scores = gl.unsqueeze(1).expand(-1, P.shape[0], -1, -1).gather(3, pick).squeeze(3).sum(2)
            return P.index_select(0, scores.argmax(1)).transpose(0, 1).reshape(-1)
        return logits.argmax(-1)

    def _ffn(self, h):
        import torch, torch.nn.functional as F
        flat = h.reshape(-1, D).float()
        route = self._route(flat)
        y = torch.empty_like(flat)
        for i, (gate, up, down) in enumerate(self.experts):
            idx = (route == i).nonzero(as_tuple=False).flatten()
            if idx.numel():
                z = flat.index_select(0, idx)
                y.index_copy_(0, idx, down(F.silu(gate(z)) * up(z)))
        return y.view_as(h)

    def prefill(self, x):
        import torch, torch.nn.functional as F
        h = rms_norm(x, self.n1); s = h.shape[0]
        q = self.q(h).view(s, Q_HEADS, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)
        k0 = self.k(h).view(s, KV_HEADS, HEAD_DIM).permute(1, 0, 2).contiguous()
        v0 = self.v(h).view(s, KV_HEADS, HEAD_DIM).permute(1, 0, 2).contiguous()
        kr = k0.repeat_interleave(Q_HEADS // KV_HEADS, dim=0).unsqueeze(0)
        vr = v0.repeat_interleave(Q_HEADS // KV_HEADS, dim=0).unsqueeze(0)
        pos = torch.arange(s, device=h.device); dist = pos[:, None] - pos[None, :]
        mask = dist >= 0
        if self.window >= 0:
            mask &= dist <= self.window
        a = F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask.view(1, 1, s, s), dropout_p=0.0, is_causal=False)
        a = a.squeeze(0).permute(1, 0, 2).reshape(s, D)
        x = x.float() + self.o(a)
        x = x + self._ffn(rms_norm(x, self.n2))
        return x, (k0, v0)

    def step(self, x, cache):
        import torch
        h = rms_norm(x, self.n1)
        q = self.q(h).view(1, Q_HEADS, HEAD_DIM).permute(1, 0, 2)
        k0 = torch.cat((cache[0], self.k(h).view(1, KV_HEADS, HEAD_DIM).permute(1, 0, 2)), dim=1)
        v0 = torch.cat((cache[1], self.v(h).view(1, KV_HEADS, HEAD_DIM).permute(1, 0, 2)), dim=1)
        if self.window >= 0 and k0.shape[1] > self.window + 1:
            k0 = k0[:, -(self.window + 1):]; v0 = v0[:, -(self.window + 1):]
        kr = k0.repeat_interleave(Q_HEADS // KV_HEADS, dim=0)
        vr = v0.repeat_interleave(Q_HEADS // KV_HEADS, dim=0)
        probs = (torch.matmul(q, kr.transpose(-1, -2)) / math.sqrt(HEAD_DIM)).softmax(-1)
        a = torch.matmul(probs, vr).permute(1, 0, 2).reshape(1, D)
        x = x.float() + self.o(a)
        x = x + self._ffn(rms_norm(x, self.n2))
        return x, (k0, v0)


class RPV16:
    def __init__(self, ckpt_path, precision='int8', device='cpu', sparse_mask=False, monitor=None):
        import torch
        t0 = time.time()
        self.device = device; self.precision = precision
        src = Path(ckpt_path).resolve()
        fd = os.open(src, os.O_RDONLY)          # pin the inode for the duration of the load
        ident = os.fstat(fd)
        if monitor: monitor.ckpt_open = True
        ck = torch.load(f'/proc/self/fd/{fd}', map_location='cpu', mmap=True, weights_only=False)
        state = ck['model'] if 'model' in ck else ck
        self.meta = {'source': str(src), 'inode': ident.st_ino, 'bytes': ident.st_size,
                     'step': int(ck.get('step', -1)) if isinstance(ck, dict) else -1,
                     'seen_tokens': int(ck.get('seen_tokens', -1)) if isinstance(ck, dict) else -1,
                     'schema': ck.get('schema') if isinstance(ck, dict) else None,
                     'precision': precision, 'device': device}
        masks = {}
        if sparse_mask:
            blob = ck.get('sparse_topology') or {}
            for name, e in (blob.get('entries') or {}).items():
                if e.get('packed') is not None:
                    masks[name + '.weight'] = (e['packed'], tuple(e['shape']))
            self.meta['sparse_masks_applied'] = len(masks)
        used = set()
        fdev = torch.device(device)

        def take(name, kind):
            used.add(name)
            t = state[name]
            if name in masks:
                packed, shape = masks[name]
                t = t.float() * unpack_sparse_mask(packed, shape).to(torch.float32)
            if kind == 'lin':
                return Lin(t, precision, fdev)
            return t.detach().to(device=fdev, dtype=torch.float32).clone()

        # Embedding: the int8 CPU export stores it as bf16 (also used for the tied head); keep that for exact parity.
        emb = state['embed.weight']; used.add('embed.weight')
        emb_dt = torch.bfloat16 if precision in ('int8', 'bf16') else torch.float32
        self.embed = emb.detach().to(device=fdev, dtype=emb_dt).clone()
        self.tied_w = self.embed.float().contiguous() if precision != 'bf16' else self.embed
        self.in_proj = take('in_proj.weight', 'lin'); self.out_proj = take('out_proj.weight', 'lin')
        self.norm = take('norm.weight', 'f')
        if 'tied_bias' not in state:
            raise SystemExit('checkpoint has no tied_bias: product-vocabulary (pre-tied) heads are not supported by this script')
        self.tied_bias = take('tied_bias', 'f').reshape(-1)
        self.tied_scale = float(state['tied_logit_scale'].float().reshape(())); used.add('tied_logit_scale')
        perms = torch.tensor([[(a * j + b) % EXPERTS for j in range(EXPERTS)] for a in (1, EXPERTS - 1) for b in range(EXPERTS)],
                             dtype=torch.long, device=fdev)
        self.stages = []
        for si in range(STAGES):
            pre = f'stages.{si}.'
            self.stages.append(Stage(lambda n, k, pre=pre: take(pre + n, k), WINDOWS[si], precision, fdev, perms))
            if monitor and monitor.abort: raise SystemExit(monitor.abort)
        unused = sorted(k for k, v in state.items() if torch.is_tensor(v) and k not in used)
        self.meta['unused_state_keys'] = unused[:20]; self.meta['unused_state_key_count'] = len(unused)
        if sparse_mask: self.meta['sparse_masks_matched'] = sum(n in used for n in masks)
        self.meta['parameter_count'] = int(sum(v.numel() for v in state.values() if torch.is_tensor(v)))
        end = os.fstat(fd)
        self.meta['source_stable_during_load'] = (end.st_ino, end.st_size, end.st_mtime_ns) == (ident.st_ino, ident.st_size, ident.st_mtime_ns)
        del state, ck, emb, masks
        gc.collect()
        os.close(fd)                             # release the mapping/inode before the next checkpoint rotation
        if monitor: monitor.ckpt_open = False
        self.meta['load_s'] = round(time.time() - t0, 1)

    # ---- forward pieces
    def hidden_seq(self, t):
        import torch.nn.functional as F
        x = self.in_proj(F.embedding(t.to(self.device), self.embed).float())
        for st in self.stages:
            x, _ = st.prefill(x)
        return x

    def logits(self, h):
        import torch.nn.functional as F
        return F.linear(h.to(self.tied_w.dtype), self.tied_w).float() * self.tied_scale + self.tied_bias

    def prefill(self, ids):
        import torch, torch.nn.functional as F
        t = torch.tensor(ids, dtype=torch.long, device=self.device)
        x = self.in_proj(F.embedding(t, self.embed).float())
        caches = []
        for st in self.stages:
            x, c = st.prefill(x); caches.append(c)
        return self.out_proj(rms_norm(x[-1:], self.norm)), caches

    def step(self, tid, caches):
        import torch, torch.nn.functional as F
        t = torch.tensor([tid], dtype=torch.long, device=self.device)
        x = self.in_proj(F.embedding(t, self.embed).float())
        for i, st in enumerate(self.stages):
            x, caches[i] = st.step(x, caches[i])
        return self.out_proj(rms_norm(x, self.norm))


# ----------------------------------------------------------------------------- held-out CE (same as diag_local.py)
def heldout(model, ids_path, nwin, win, code_windows):
    import torch
    ids = json.loads(Path(ids_path).read_text())
    nwin = min(nwin, len(ids) // win)
    per, alln = [], []
    t0 = time.time()
    with torch.inference_mode():
        for w in range(nwin):
            t = torch.tensor(ids[w * win:(w + 1) * win], dtype=torch.long)
            x = model.hidden_seq(t)
            h = model.out_proj(rms_norm(x[:-1], model.norm))
            lp = model.logits(h).log_softmax(-1)
            nll = -lp.gather(1, t[1:].to(lp.device)[:, None]).squeeze(1).float().cpu()
            per.append(float(nll.mean())); alln.append(nll)
            emit({'event': 'heldout_window', 'window': w, 'ce': round(per[-1], 4), 'code': w in code_windows, 'secs': round(time.time() - t0, 1)})
    allcat = torch.cat(alln)
    prose = [c for i, c in enumerate(per) if i not in code_windows]
    code = [c for i, c in enumerate(per) if i in code_windows]
    return {'windows': nwin, 'win_len': win, 'heldout_sha256': hashlib.sha256(Path(ids_path).read_bytes()).hexdigest(),
            'ce_all': round(float(allcat.mean()), 4), 'ce_prose': round(sum(prose) / max(1, len(prose)), 4),
            'n_prose': len(prose), 'ce_code': round(sum(code) / max(1, len(code)), 4), 'code_windows': sorted(code_windows),
            'ce_by_window': [round(c, 4) for c in per], 'secs': round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------- generation
def sample(logits, history, a, gen):
    import torch
    s = logits.float().clone()
    if a.repetition_penalty != 1.0 and history:
        for tid in set(history[-128:]):
            s[tid] = s[tid] / a.repetition_penalty if s[tid] > 0 else s[tid] * a.repetition_penalty
    n = a.no_repeat_ngram
    if n >= 2 and len(history) >= n - 1:
        prefix = history[-(n - 1):]
        banned = {history[i + n - 1] for i in range(len(history) - n + 1) if history[i:i + n - 1] == prefix}
        if banned:
            s[list(banned)] = float('-inf')
    if a.temperature <= 0:
        return int(s.argmax())
    p = (s / a.temperature).softmax(-1)
    if a.top_k > 0:
        kth = torch.topk(p, a.top_k).values[-1]; p[p < kth] = 0
    if 0 < a.top_p < 1:
        vals, idx = torch.sort(p, descending=True)
        cum = vals.cumsum(0) / vals.sum()
        drop = cum > a.top_p; drop[1:] = drop[:-1].clone(); drop[0] = False
        p[idx[drop]] = 0
    return int(torch.multinomial(p / p.sum(), 1, generator=gen).item())


def generate(model, tok, prompt, a, gen):
    import torch
    ids = list(tok.encode(prompt).ids) or [1]
    eos = {i for i in (tok.token_to_id(x) for x in ('<|endoftext|>', '<|eot_id|>', '</s>', '<eos>')) if i is not None}
    out = []
    with torch.inference_mode():
        t0 = time.perf_counter()
        h, caches = model.prefill(ids)
        t1 = time.perf_counter()
        for i in range(a.max_new):
            tid = sample(model.logits(h)[0], ids + out, a, gen)
            if tid in eos and not a.ignore_eos:
                break
            out.append(tid)
            if i + 1 < a.max_new:
                h = model.step(tid, caches)
        t2 = time.perf_counter()
    n = len(out)
    return {'prompt': prompt, 'completion': tok.decode(out, skip_special_tokens=False), 'prompt_tokens': len(ids),
            'new_tokens': n, 'prefill_s': round(t1 - t0, 3), 'decode_s': round(t2 - t1, 3),
            'decode_tok_s': round(n / max(1e-9, t2 - t1), 2), 'total_tok_s': round(n / max(1e-9, t2 - t0), 2)}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    ap.add_argument('--precision', choices=('int8', 'bf16', 'fp32'), default=None, help='default int8 on CPU, bf16 on CUDA')
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--allow-gpu', action='store_true', help='required with --device cuda (GPU is normally owned by the trainer)')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--prompt', action='append', default=[])
    ap.add_argument('--max-new', type=int, default=48)
    ap.add_argument('--temperature', type=float, default=0.0, help='0 = greedy')
    ap.add_argument('--top-p', type=float, default=1.0)
    ap.add_argument('--top-k', type=int, default=0)
    ap.add_argument('--repetition-penalty', type=float, default=1.0)
    ap.add_argument('--no-repeat-ngram', type=int, default=0)
    ap.add_argument('--ignore-eos', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--heldout', action='store_true')
    ap.add_argument('--heldout-ids', default=DEFAULT_HELDOUT)
    ap.add_argument('--windows', type=int, default=24)
    ap.add_argument('--win-len', type=int, default=512)
    ap.add_argument('--code-windows', default=DEFAULT_CODE_WINDOWS)
    ap.add_argument('--sparse-mask', action='store_true', help="apply the checkpoint's 4-of-8 sparse topology masks (not used by the existing CPU evaluator)")
    ap.add_argument('--min-mem-gib', type=float, default=25.0, help='refuse to start below this MemAvailable (cgroup-aware)')
    ap.add_argument('--mem-floor-gib', type=float, default=10.0, help='abort if MemAvailable drops below this')
    ap.add_argument('--disk-floor-gib', type=float, default=10.0, help='abort if disk free drops below this while the checkpoint is mapped')
    ap.add_argument('--no-safety', action='store_true')
    ap.add_argument('--json-out', default='')
    a = ap.parse_args()

    if a.device == 'cuda' and not a.allow_gpu:
        ap.error('--device cuda requires --allow-gpu')
    if a.device == 'cpu':
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    prec = a.precision or ('int8' if a.device == 'cpu' else 'bf16')
    if a.device == 'cuda' and prec == 'int8':
        ap.error('int8 dynamic quantisation is CPU-only; use --precision bf16/fp32 on CUDA')
    try:
        os.nice(19)
    except OSError:
        pass
    import torch
    torch.set_num_threads(max(1, a.threads))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    if 'x86' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'x86'

    ckpt = Path(a.ckpt)
    mon = Monitor(str(ckpt.resolve().parent), int(a.mem_floor_gib * 2**30), int(a.disk_floor_gib * 2**30))
    pre = {'event': 'preflight', 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'mem_available_gib': round(mem_available() / 2**30, 2), 'disk_free_gib': round(disk_free(str(ckpt.resolve().parent)) / 2**30, 2),
           'tmp_present': Path(str(ckpt.resolve()) + '.tmp').exists(), 'precision': prec, 'device': a.device, 'threads': a.threads}
    emit(pre)
    if not a.no_safety:
        if pre['tmp_present']:
            raise SystemExit('checkpoint save in progress (.tmp present); retry after it completes')
        if pre['mem_available_gib'] < a.min_mem_gib:
            raise SystemExit(f"MemAvailable {pre['mem_available_gib']} GiB < required {a.min_mem_gib} GiB")
        mon.start()

    model = RPV16(ckpt, prec, a.device, a.sparse_mask, mon if not a.no_safety else None)
    result = {'schema': 'agillm.rpv16.singlefile-infer.v1', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'model': model.meta}
    emit({'event': 'loaded', **model.meta, **(mon.summary() if not a.no_safety else {})})

    if a.heldout:
        cw = {int(x) for x in a.code_windows.split(',') if x.strip()}
        result['heldout'] = heldout(model, a.heldout_ids, a.windows, a.win_len, cw)
        emit({'event': 'heldout', 'step': model.meta['step'], **result['heldout']})
    if a.prompt:
        tok = load_tokenizer(a.tokenizer)
        gen = torch.Generator().manual_seed(a.seed)
        result['generations'] = []
        for p in a.prompt:
            g = generate(model, tok, p, a, gen)
            result['generations'].append(g); emit({'event': 'generation', **g})
        dec = [g for g in result['generations'] if g['new_tokens']]
        if dec:
            result['decode_tok_s_mean'] = round(sum(g['new_tokens'] for g in dec) / sum(g['decode_s'] for g in dec), 2)
    if not a.no_safety:
        mon.stop(); result['resources'] = mon.summary()
    emit({'event': 'done', **{k: v for k, v in result.items() if k in ('resources', 'decode_tok_s_mean', 'script_sha256')}})
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
