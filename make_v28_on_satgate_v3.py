#!/usr/bin/env python3
"""make_v28_varblock.py — derive the v28.0 "variable-block SAT" runtime from the live
satgate_v3 runtime (sha d9accb92..., itself derived from retain_activations_v2 989a7ee5) with anchored, exactly-once edits.

Owner directive (Scott, 2026-09-12 01:35 UK): "SAT VAR IS NOT SAT VAR IF JUST DOING 1 TOKEN
OR 2 TOKEN ONLY. IT HAS TO CHOOSE TRAINING OBJECTIVE, BALANCE SPEED AND INTELLIGENCE."

What changes (all additive, checkpoint-compatible, hot-switchable):
  1. SATHead gains  shift_emb[SATVAR_KMAX, d] (zero-init -> size-2 blocks reproduce the
     fixed shift-2 head bit-exactly) and a detached regret head predicting log E[CE_d]
     for d=1..SATVAR_KMAX from the block's last hidden state (bias = measured prior).
  2. The local DBlock SAT objective samples a random block partition (sizes 1..K from a
     hot-configurable distribution) instead of the fixed shift-2 partition; slot j of a
     block predicts the token j+1 beyond the block's view through proj(h + E_{j+1}).
     Size-1 blocks ARE the AR objective, size-2 blocks ARE the old SAT-fixed objective.
  3. A new full-stack "satvar" anchor (random partition crop through the whole stack)
     trains the same objective on the serving graph and reports per-d CE.
  4. Decode: --mode sat --var chooses the stride per step by
        m* = argmax_m  lambda*(m-1) - sum_{d=2..m}(CEhat_d - CEhat_1) - mu*[m > n_last]
     with CEhat from the regret head once trained, else from the slot entropies; growth
     beyond the last block merges the trailing tokens into one recomputed block.
  5. Optimizer: two new tail groups (satvar_shift, satvar_regret) so the mature
     checkpoint's optimizer indices are unchanged; both excluded from cosine/governor.
The fixed SAT anchor, the legacy 1-vs-2 gate and every other contract are untouched.
"""
import hashlib, re, sys, pathlib

SRC = pathlib.Path(sys.argv[1])
DST = pathlib.Path(sys.argv[2])
EXPECTED_SRC_SHA = "d9accb928ad2208aaee7150422a83247db9c252b3ac2c39d49ed429a4bb1aa7f"  # satgate_v3 live runtime

text = SRC.read_text(encoding="utf8")
actual = hashlib.sha256(text.encode("utf8")).hexdigest()
if actual != EXPECTED_SRC_SHA and "--allow-other-base" not in sys.argv:
    raise SystemExit(f"base runtime sha mismatch: {actual} != {EXPECTED_SRC_SHA}")

edits = []


def edit(name, old, new, count=1):
    edits.append((name, old, new, count))


# ───────────────────────── 1. constants + helpers (after EMIT_LAMBDA) ─────────────────────────
edit("constants", "EMIT_LAMBDA = 0.1\n", r'''EMIT_LAMBDA = 0.1

# ═══════════════════ v28 variable-block SAT (owner directive 2026-09-12) ═══════════════════
# A SAT block of size n is n consecutive tokens that attend to each other and to every
# earlier block (block-causal).  Slot j of a block (0-based) predicts the token j+1 beyond
# the block's last visible token through the tied projection of (h + shift_emb[j]).
# n=1 is the AR objective, n=2 the historical fixed SAT objective; the variable objective
# samples n in 1..SATVAR_KMAX so the model learns every stride and a regret head learns
# how much each extra drafted token costs, which is what lets decode trade speed for
# intelligence per position instead of a hard-coded 1-or-2 gate.
SATVAR_KMAX = 8
SATVAR_REGRET_HIDDEN = 256
# Measured per-slot held-out CE on checkpoint 2874393 (fixed block-2 slots 6.33/6.86, larger
# untrained blocks 7.4-7.9).  Used only as the regret head's initial bias (pessimistic prior).
SATVAR_REGRET_PRIOR_CE = [6.3, 6.9, 7.6, 7.8, 7.9, 8.0, 8.0, 8.1]
SATVAR_SCHEMA = "agillm44.satvar.variable-block.v1"
SATVAR_DEFAULT_BLOCK_PROBS = "1:0.20,2:0.40,3:0.20,4:0.20"
_SATVAR_PARAM_PREFIXES = ("shift_emb", "stride_regret.", "satvar_regret_updates")


def _satvar_is_param_key(key) -> bool:
    key = str(key)
    return any(key == p or key.startswith(p) for p in _SATVAR_PARAM_PREFIXES)


def _satvar_enabled(args) -> bool:
    """Hot-switchable: hot_config dblock_satvar_enabled (0/1) overrides the CLI default."""
    cli = int(getattr(args, "dblock_satvar_enabled", 1) or 0)
    if bool(getattr(args, "repair_mode", False)):
        return bool(cli)
    try:
        cfg = get_hot_config()
    except Exception:
        return bool(cli)
    return bool(_hot_int_from_config(cfg, ["dblock_satvar_enabled", "satvar_enabled"], cli))


def _satvar_parse_block_probs(spec, kmax=None):
    """'1:0.2,2:0.4,3:0.2,4:0.2' -> normalized list indexed by size-1 (length kmax)."""
    kmax = int(SATVAR_KMAX if kmax is None else kmax)
    probs = [0.0] * kmax
    for item in str(spec or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        size, _, weight = item.partition(":")
        size = int(size)
        weight = float(weight) if weight else 1.0
        if 1 <= size <= kmax and weight > 0.0 and math.isfinite(weight):
            probs[size - 1] += weight
    total = sum(probs)
    if total <= 0.0:
        probs = [0.0] * kmax
        probs[1 if kmax >= 2 else 0] = 1.0
        total = 1.0
    return [p / total for p in probs]


_satvar_block_probs_seen = {}


def _satvar_block_probs(args):
    spec = str(getattr(args, "dblock_satvar_block_probs", SATVAR_DEFAULT_BLOCK_PROBS) or SATVAR_DEFAULT_BLOCK_PROBS)
    if not bool(getattr(args, "repair_mode", False)):
        try:
            cfg = get_hot_config()
            for key in ("dblock_satvar_block_probs", "satvar_block_probs"):
                val = cfg.get(key) if isinstance(cfg, dict) else None
                if val is None and isinstance(cfg, dict) and isinstance(cfg.get("dblock"), dict):
                    val = cfg["dblock"].get(key)
                if val:
                    spec = str(val)
                    break
        except Exception:
            pass
    probs = _satvar_parse_block_probs(spec)
    key = id(args)
    if _satvar_block_probs_seen.get(key) != spec:
        _satvar_block_probs_seen[key] = spec
        print(f"[satvar] block size distribution {spec} -> {[round(p, 3) for p in probs]}", flush=True)
    return probs


def satvar_sample_sizes(T, probs, rng=None):
    """i.i.d. block sizes covering exactly T positions (last block truncated to fit)."""
    T = int(T)
    rng = rng or random
    sizes = []
    total = 0
    population = [i + 1 for i, p in enumerate(probs) if p > 0.0]
    weights = [p for p in probs if p > 0.0]
    while total < T:
        n = int(rng.choices(population, weights=weights, k=1)[0])
        n = min(n, T - total)
        sizes.append(n)
        total += n
    return sizes


def satvar_partition_arrays(sizes, T, device=None):
    """Per-position block id, slot index (0-based), block size, block-end position (vectorized)."""
    T = int(T)
    sizes_t = torch.as_tensor([int(s) for s in sizes], dtype=torch.long, device=device)
    if int(sizes_t.sum()) != T:
        raise ValueError(f"partition sizes sum {int(sizes_t.sum())} != T {T}")
    block_id = torch.repeat_interleave(torch.arange(sizes_t.numel(), device=sizes_t.device), sizes_t)
    bsize = torch.repeat_interleave(sizes_t, sizes_t)
    ends = torch.cumsum(sizes_t, 0) - 1
    starts = ends - sizes_t + 1
    bend = torch.repeat_interleave(ends, sizes_t)
    slot = torch.arange(T, device=sizes_t.device) - torch.repeat_interleave(starts, sizes_t)
    return block_id, slot, bsize, bend


def satvar_partition_mask(sizes, T, device=None, dtype=torch.float32):
    """Dense block-causal mask for an arbitrary partition: q attends k iff block(k) <= block(q)."""
    block_id, _, _, _ = satvar_partition_arrays(sizes, T, device=device or DEV)
    allow = block_id.view(1, -1) <= block_id.view(-1, 1)
    return torch.where(allow, torch.zeros((), device=allow.device, dtype=dtype),
                       torch.full((), float("-inf"), device=allow.device, dtype=dtype)).unsqueeze(0).unsqueeze(0)


def satvar_loss_index(sizes, T, loss_mask_row=None, device=None):
    """Positions that have a target (t + size < T): returns (pos, d_index, tgt_pos, block_end)."""
    block_id, slot, bsize, bend = satvar_partition_arrays(sizes, T, device=device)
    pos = torch.arange(T, device=block_id.device)
    tgt = pos + bsize
    keep = tgt < T
    if loss_mask_row is not None:
        lm = loss_mask_row.to(device=block_id.device, dtype=torch.bool)
        keep = keep & lm[tgt.clamp(max=T - 1)]
    keep = keep & (slot < SATVAR_KMAX)
    return pos[keep], slot[keep], tgt[keep], bend[keep]


@torch.no_grad()
def _satvar_token_ce_nograd(h, W, tgt, vchunk=8192):
    """Exact per-token CE (fp32 log-sum-exp, streamed over the vocab), no autograd.

    Autocast is disabled inside: under bf16/fp16 autocast the matmul would otherwise come back
    in the low-precision dtype and the fp32 index_put below would fail (v28.0 crashed on the
    first live SAT step with 'Index put requires the source and destination dtypes match').
    """
    hf = h.detach().float()
    Wf = W.detach()
    N = hf.size(0)
    V = Wf.size(0)
    m = torch.full((N,), -1e30, device=hf.device, dtype=torch.float32)
    s = torch.zeros(N, device=hf.device, dtype=torch.float32)
    zt = torch.zeros(N, device=hf.device, dtype=torch.float32)
    with torch.autocast(device_type=("cuda" if hf.is_cuda else "cpu"), enabled=False):
        for c in range(0, V, vchunk):
            lg = (hf @ Wf[c:c + vchunk].float().T).float()
            cm = lg.max(1).values
            nm = torch.maximum(m, cm)
            s = s * torch.exp(m - nm) + torch.exp(lg - nm[:, None]).sum(1)
            m = nm
            ic = (tgt >= c) & (tgt < c + vchunk)
            zt[ic] = lg[ic, tgt[ic] - c].to(zt.dtype)
    return (m + torch.log(s)) - zt


def _satvar_gather(sat_h, hidden, ids, loss_mask, sizes, max_tokens, rng=None):
    """Gather (hidden + E_d, target) pairs for a partition across all rows, sampled to max_tokens.

    Returns sat_hidden [N,d], sat_targets [N], used, available, d_index [N], end_pos [N], row [N].
    """
    B, T, D = hidden.shape
    block_id, slot, bsize, bend = satvar_partition_arrays(sizes, T, device=hidden.device)
    pos_t = torch.arange(T, device=hidden.device)
    tgt_t = pos_t + bsize
    keep_t = (tgt_t < T) & (slot < SATVAR_KMAX)
    if loss_mask is not None:
        lm = loss_mask.to(device=hidden.device, dtype=torch.bool)
        lm_at_tgt = lm.gather(1, tgt_t.clamp(max=T - 1).view(1, T).expand(B, T))
        keep_bt = keep_t.view(1, T) & lm_at_tgt
    else:
        keep_bt = keep_t.view(1, T).expand(B, T)
    idx = torch.nonzero(keep_bt, as_tuple=False)
    available = int(idx.size(0))
    if available <= 0:
        raise ValueError("no variable-block SAT targets available")
    max_tokens = int(max_tokens or 0)
    if 0 < max_tokens < available:
        idx = idx.index_select(0, torch.randint(available, (max_tokens,), device=idx.device))
    row = idx[:, 0]
    pos = idx[:, 1]
    d_idx = slot.index_select(0, pos)
    tgt = tgt_t.index_select(0, pos)
    bend_sel = bend.index_select(0, pos)
    flat = hidden.reshape(B * T, D)
    sel = sat_h.slot_hidden(flat.index_select(0, row * T + pos), d_idx)
    targets = ids.reshape(-1).index_select(0, row * T + tgt)
    return sel.contiguous(), targets.contiguous(), int(targets.numel()), available, d_idx, bend_sel, row


def _satvar_regret_loss(sat_h, hidden, ids, loss_mask, sizes, W, max_blocks=256):
    """Regress log E[CE_d] from the block's last hidden state (detached) for every observed slot.

    Per-slot CE is computed exactly without autograd; the regret head is the only module that
    receives gradient here, so the trunk/projection contracts are unchanged.  Returns
    (loss, info) where info carries per-d mean CE on the sampled blocks for telemetry.
    """
    B, T, D = hidden.shape
    dev = hidden.device
    sizes_t = torch.as_tensor([int(s) for s in sizes], dtype=torch.long, device=dev)
    ends = torch.cumsum(sizes_t, 0) - 1
    starts = ends - sizes_t + 1
    valid_block = (ends + sizes_t) < T                    # every slot of the block has a target
    cand = torch.nonzero(valid_block, as_tuple=False).flatten()
    if cand.numel() == 0:
        return None, {"blocks": 0}
    M_ = int(max(1, max_blocks))
    pick = cand.index_select(0, torch.randint(int(cand.numel()), (M_,), device=dev))
    rows = torch.randint(B, (M_,), device=dev)
    n_use = torch.clamp(sizes_t.index_select(0, pick), max=SATVAR_KMAX)   # slots per picked block
    blk = torch.repeat_interleave(torch.arange(M_, device=dev), n_use)
    offs = torch.cumsum(n_use, 0) - n_use
    d_idx = torch.arange(int(n_use.sum()), device=dev) - torch.repeat_interleave(offs, n_use)
    pos = starts.index_select(0, pick).index_select(0, blk) + d_idx
    tgt = pos + sizes_t.index_select(0, pick).index_select(0, blk)
    row = rows.index_select(0, blk)
    if loss_mask is not None:
        keep = loss_mask.reshape(-1).to(torch.bool).index_select(0, row * T + tgt)
        pos, d_idx, tgt, row, blk = (x[keep] for x in (pos, d_idx, tgt, row, blk))
        if pos.numel() == 0:
            return None, {"blocks": 0}
    flat = hidden.reshape(B * T, D)
    with torch.no_grad():
        hs = sat_h.slot_hidden(flat.index_select(0, row * T + pos), d_idx)
        ce = _satvar_token_ce_nograd(hs, W, ids.reshape(-1).index_select(0, row * T + tgt))
    # regret head input: the block's LAST slot hidden state (it has seen the whole block)
    end_pos = ends.index_select(0, pick)                      # [M_]
    h_last = flat.index_select(0, rows * T + end_pos)          # [M_, D]
    pred = sat_h.regret_log_ce(h_last)                         # [M_, KMAX]
    target_log = torch.log(ce.clamp_min(1e-3))
    pred_sel = pred[blk, d_idx]
    loss = F.smooth_l1_loss(pred_sel, target_log, beta=0.5)
    with torch.no_grad():
        info = {"blocks": int(M_), "slots": int(pos.numel()),
                "regret_mae_log": round(float((pred_sel - target_log).abs().mean()), 4),
                "ce_by_d": {}, "count_by_d": {}, "pred_ce_by_d": {}}
        for d in range(SATVAR_KMAX):
            m = d_idx == d
            cnt = int(m.sum())
            if cnt > 0:
                info["ce_by_d"][str(d + 1)] = round(float(ce[m].mean()), 4)
                info["count_by_d"][str(d + 1)] = cnt
                info["pred_ce_by_d"][str(d + 1)] = round(float(pred_sel[m].exp().mean()), 4)
    return loss, info


def _satvar_size_hist(sizes):
    hist = {}
    for n in sizes:
        hist[str(int(n))] = hist.get(str(int(n)), 0) + 1
    return hist


def _satvar_choose_stride(ce_hat, n_avail, lam, mu, kmax, remaining):
    """Deterministic speed/quality trade: maximize lam*(m-1) - sum_{d=2..m} max(0, CE_d - CE_1) - mu*[m>n_avail]."""
    best_m, best_u = 1, 0.0
    base = float(ce_hat[0])
    penalty = 0.0
    for m in range(2, int(min(kmax, remaining)) + 1):
        penalty += max(0.0, float(ce_hat[m - 1]) - base)
        u = float(lam) * (m - 1) - penalty - (float(mu) if m > int(n_avail) else 0.0)
        if u > best_u:
            best_m, best_u = m, u
    return best_m, best_u
''')

# ───────────────────────── 2. SATHead ─────────────────────────
edit("sathead", '''        self.gate_conf = _SATGateConf() if mode == "var" else None
    def forward(self, h_last):
''', '''        self.gate_conf = _SATGateConf() if mode == "var" else None
        # v28 variable-block SAT (see SATVAR_SCHEMA): per-distance shift embeddings and a
        # detached regret head.  Zero-init shift rows keep size-2 blocks bit-identical to
        # the historical fixed shift-2 head; the regret bias starts at the measured prior.
        self.satvar_d = int(d)
        self.shift_emb = nn.Parameter(torch.zeros(SATVAR_KMAX, d))
        self.stride_regret = nn.Sequential(
            nn.Linear(d, SATVAR_REGRET_HIDDEN), nn.GELU(), nn.Linear(SATVAR_REGRET_HIDDEN, SATVAR_KMAX))
        self.register_buffer("satvar_regret_updates", torch.zeros((), dtype=torch.long), persistent=True)
        self.satvar_init_()

    def satvar_init_(self):
        """Deterministic fresh init for the v28 tensors (also used when a checkpoint lacks them)."""
        with torch.no_grad():
            self.shift_emb.zero_()
            lin0, lin1 = self.stride_regret[0], self.stride_regret[2]
            g = torch.Generator(device="cpu").manual_seed(20260912)
            bound = 1.0 / math.sqrt(float(self.satvar_d))
            lin0.weight.copy_((torch.rand(lin0.weight.shape, generator=g) * 2.0 - 1.0) * bound)
            lin0.bias.zero_()
            lin1.weight.zero_()
            lin1.bias.copy_(torch.log(torch.tensor(SATVAR_REGRET_PRIOR_CE, dtype=torch.float32)))
            self.satvar_regret_updates.zero_()

    def slot_hidden(self, h, d_index):
        """h [N,d] plus the shift embedding of its distance index (0-based: d-1)."""
        return h + self.shift_emb.index_select(0, d_index.to(self.shift_emb.device)).to(h.dtype)

    def slot_logits(self, h, d_index):
        return self.proj(self.slot_hidden(h, d_index))

    def regret_log_ce(self, h_last):
        """Predicted log E[CE_d], d=1..SATVAR_KMAX, from the block's last hidden state (detached)."""
        w = self.stride_regret[0].weight
        return self.stride_regret(h_last.detach().to(w.dtype)).float()

    def forward(self, h_last):
''')

# ───────────────────────── 3. local DBlock SAT objective ─────────────────────────
edit("local-mask", '''    if run_sat:
        smask = M.sat_mask(T, structured=M.use_structured_masks(args))
        _t = _profile_tic(prof)
''', '''    if run_sat:
        _satvar_on = _satvar_enabled(args) and not M.use_structured_masks(args)
        if _satvar_on:
            _satvar_sizes = satvar_sample_sizes(T, _satvar_block_probs(args))
            smask = satvar_partition_mask(_satvar_sizes, T, device=ids.device)
        else:
            _satvar_sizes = None
            smask = M.sat_mask(T, structured=M.use_structured_masks(args))
        _t = _profile_tic(prof)
''')

edit("local-loss", '''        sat_ctx = Ds[:, :-SATB]
        sat_tgt = ids_s[:, SATB:]
        if sat_ctx.size(1) == 0 or sat_ctx.size(1) != sat_tgt.size(1):
            sat_ctx = Ds[:, :-1]
            sat_tgt = ids_s[:, 1:]
        sat_loss_mask = None
        if lm_s is not None:
            sat_loss_mask = lm_s[:, SATB:] if sat_ctx.size(1) == lm_s[:, SATB:].size(1) else lm_s[:, 1:]
        sat_hidden, sat_targets, sat_used, sat_total = _sample_sat_pair_loss_inputs(
            sat_ctx, sat_tgt, _dblock_loss_token_cap(args, "sat"),
            sat_loss_mask, block=SATB,
        )
        # SAT-variable admission gate is NOT trained by ordinary pretraining.
        # The old all-ones target contradicted the verifier contract; a later
        # full-vocab pseudo-label attempt was also prohibitively expensive at
        # production B/T. Gate supervision stays in bounded ORPO verifier loss.
        with M.amp(args.amp):
            satf = fused_ce(sat_hidden, _dblock_local_head_weight(sat_h), sat_targets)
            satv = 0.0
            sat_raw = satf
            local_diagnostics["objectives"]["sat"] = {
                "input": {
                    "sha256": local_diagnostics["batch"]["sha256"],
                    "source": "whole_batch",
                },
                "selected_targets": _dblock_token_boundary_diagnostics(sat_targets),
                "selected_target_count": int(sat_used),
                "available_target_count": int(sat_total),
                "sampling": "complete_sat_blocks_with_replacement",
                "sat_block_size": int(SATB),
                "selected_block_count": int(sat_used // max(1, SATB)),
                "available_block_count": int(sat_total // max(1, SATB)),
            }
            sat = sat_weight * w * sat_raw
            sat_raw_val, sat_val = _dblock_scalar_values(sat_raw, sat)
''', '''        if _satvar_on:
            # v28 variable-block SAT: slot j of every block predicts the token j+1 beyond the
            # block through proj(h + E_{j+1}); the detached regret head learns E[CE_d].
            sat_hidden, sat_targets, sat_used, sat_total, _sv_d, _sv_end, _sv_row = _satvar_gather(
                sat_h, Ds, ids_s, lm_s, _satvar_sizes, _dblock_loss_token_cap(args, "sat"))
            _sv_regret_w = _dblock_hot_float(args, "dblock_satvar_regret_weight", 0.05, min_value=0.0, max_value=1.0)
            _sv_regret_blocks = _dblock_fullstack_anchor_int(args, "dblock_satvar_regret_blocks", 256)
            with M.amp(args.amp):
                satf = fused_ce(sat_hidden, _dblock_local_head_weight(sat_h), sat_targets)
                satv = 0.0
                sat_raw = satf
                _sv_regret_loss, _sv_regret_info = (None, {"blocks": 0})
                if _sv_regret_w > 0.0 and _sv_regret_blocks > 0:
                    _sv_regret_loss, _sv_regret_info = _satvar_regret_loss(
                        sat_h, Ds.detach(), ids_s, lm_s, _satvar_sizes,
                        _dblock_local_head_weight(sat_h), max_blocks=_sv_regret_blocks)
                local_diagnostics["objectives"]["sat"] = {
                    "input": {
                        "sha256": local_diagnostics["batch"]["sha256"],
                        "source": "whole_batch",
                    },
                    "selected_targets": _dblock_token_boundary_diagnostics(sat_targets),
                    "selected_target_count": int(sat_used),
                    "available_target_count": int(sat_total),
                    "sampling": "variable_block_partition_with_replacement",
                    "schema": SATVAR_SCHEMA,
                    "block_size_hist": _satvar_size_hist(_satvar_sizes),
                    "mean_block_size": round(float(T) / max(1, len(_satvar_sizes)), 4),
                    "selected_d_hist": {str(d + 1): int((_sv_d == d).sum()) for d in range(SATVAR_KMAX) if int((_sv_d == d).sum()) > 0},
                    "regret": _dblock_json_copy(_sv_regret_info),
                    "regret_weight": float(_sv_regret_w),
                }
                sat = sat_weight * w * sat_raw
                if _sv_regret_loss is not None and torch.is_tensor(_sv_regret_loss) and bool(torch.isfinite(_sv_regret_loss)):
                    sat = sat + float(_sv_regret_w) * _sv_regret_loss
                    with torch.no_grad():
                        sat_h.satvar_regret_updates += 1
                sat_raw_val, sat_val = _dblock_scalar_values(sat_raw, sat)
                if _dblock_audit_due(state, args):
                    print("[dblock-satvar] " + json.dumps({
                        "step": int(state.get("step", 0)), "block": int(bi), "ce": round(float(sat_raw_val), 4),
                        "targets": int(sat_used), "available": int(sat_total),
                        "sizes": _satvar_size_hist(_satvar_sizes),
                        "d_hist": local_diagnostics["objectives"]["sat"]["selected_d_hist"],
                        "regret": _sv_regret_info, "regret_updates": int(sat_h.satvar_regret_updates.item()),
                    }, sort_keys=True, separators=(",", ":"), default=str), flush=True)
            del _sv_d, _sv_end, _sv_row, _sv_regret_loss
        else:
            sat_ctx = Ds[:, :-SATB]
            sat_tgt = ids_s[:, SATB:]
            if sat_ctx.size(1) == 0 or sat_ctx.size(1) != sat_tgt.size(1):
                sat_ctx = Ds[:, :-1]
                sat_tgt = ids_s[:, 1:]
            sat_loss_mask = None
            if lm_s is not None:
                sat_loss_mask = lm_s[:, SATB:] if sat_ctx.size(1) == lm_s[:, SATB:].size(1) else lm_s[:, 1:]
            sat_hidden, sat_targets, sat_used, sat_total = _sample_sat_pair_loss_inputs(
                sat_ctx, sat_tgt, _dblock_loss_token_cap(args, "sat"),
                sat_loss_mask, block=SATB,
            )
            # SAT-variable admission gate is NOT trained by ordinary pretraining.
            # The old all-ones target contradicted the verifier contract; a later
            # full-vocab pseudo-label attempt was also prohibitively expensive at
            # production B/T. Gate supervision stays in bounded ORPO verifier loss.
            with M.amp(args.amp):
                satf = fused_ce(sat_hidden, _dblock_local_head_weight(sat_h), sat_targets)
                satv = 0.0
                sat_raw = satf
                local_diagnostics["objectives"]["sat"] = {
                    "input": {
                        "sha256": local_diagnostics["batch"]["sha256"],
                        "source": "whole_batch",
                    },
                    "selected_targets": _dblock_token_boundary_diagnostics(sat_targets),
                    "selected_target_count": int(sat_used),
                    "available_target_count": int(sat_total),
                    "sampling": "complete_sat_blocks_with_replacement",
                    "sat_block_size": int(SATB),
                    "selected_block_count": int(sat_used // max(1, SATB)),
                    "available_block_count": int(sat_total // max(1, SATB)),
                }
                sat = sat_weight * w * sat_raw
                sat_raw_val, sat_val = _dblock_scalar_values(sat_raw, sat)
''')

# ───────────────────────── 4. full-stack satvar anchor ─────────────────────────
edit("anchor-def", '''def _nat_boundary_ids(mask_id=None):
    """Return the pinned EOS/PAD and active NAT-mask IDs used as hard boundaries."""
''', '''def _dblock_fullstack_satvar_anchor(core, sat_h, scaler, args, ids, state, loss_mask=None):
    """v28: variable-block SAT anchor through the whole serving stack.

    A random block partition (sizes 1..SATVAR_KMAX from the hot distribution) is applied to a
    few crops; every slot predicts its distance-conditioned target and the detached regret
    head regresses the observed per-slot CE.  Finite-checked like the other anchors; no spike
    machinery (its CE is bounded by construction) and no effect on the fixed SAT contract.
    """
    M = _agillm41_sys.modules[__name__]
    every = _dblock_fullstack_anchor_int(args, "dblock_fullstack_satvar_every", 3)
    offset = _dblock_fullstack_anchor_int(args, "dblock_fullstack_satvar_offset", 0)
    tokens = _dblock_fullstack_anchor_int(args, "dblock_fullstack_satvar_tokens", 256)
    rows = max(1, _dblock_fullstack_anchor_int(args, "dblock_fullstack_satvar_rows", 2))
    weight = _dblock_hot_float(args, "dblock_fullstack_satvar_weight", 0.10, min_value=0.0, max_value=3.0)
    regret_w = _dblock_hot_float(args, "dblock_satvar_regret_weight", 0.05, min_value=0.0, max_value=1.0)
    step = int(state.get("step", 0))
    enabled = _satvar_enabled(args) and not M.use_structured_masks(args)
    due = (
        enabled and sat_h is not None and every > 0 and tokens > 0 and weight > 0.0
        and _dblock_fullstack_anchor_due(step, every, offset)
    )
    if not due:
        return {"ran": False, "due": False, "finite": True, "raw": 0.0, "weighted": 0.0, "tokens": 0}
    seq_len = min(512, max(64, int(tokens)))
    if int(ids.size(1)) < seq_len:
        return {"ran": False, "due": True, "finite": True, "reason": "short_sequence", "tokens": 0}
    rows = min(rows, int(ids.size(0)))
    row0 = (step + 3) % int(ids.size(0))
    max_start = max(0, int(ids.size(1)) - seq_len)
    start0 = 0 if max_start == 0 else ((step * 97879 + row0 * 1093 + 29) % (max_start + 1))
    rows_sel = [(row0 + k * 7919) % int(ids.size(0)) for k in range(rows)]
    starts = [start0] + [(0 if max_start == 0 else ((start0 + k * 1009) % (max_start + 1))) for k in range(1, rows)]
    anchor_ids = torch.stack([ids[r, s:s + seq_len] for r, s in zip(rows_sel, starts)], dim=0)
    anchor_lm = None
    if loss_mask is not None:
        anchor_lm = torch.stack([loss_mask[r, s:s + seq_len] for r, s in zip(rows_sel, starts)], dim=0)
        if not bool(anchor_lm.any()):
            return {"ran": False, "due": True, "finite": True, "reason": "empty_loss_mask", "tokens": 0}
    sizes = satvar_sample_sizes(seq_len, _satvar_block_probs(args))
    with _dblock_deterministic_anchor_context(core, anchor_ids):
        _dblock_clear_moe_aux_stash(core)
        mask = satvar_partition_mask(sizes, seq_len, device=anchor_ids.device)
        h = _dblock_fullstack_hidden(core, anchor_ids, mask, args)
        W = sat_h.proj.weight
        hidden, targets, used, available, d_idx, _e, _r = _satvar_gather(
            sat_h, h, anchor_ids, anchor_lm, sizes, rows * seq_len)
        raw = fused_ce(hidden, W, targets)
        regret_loss, regret_info = (None, {"blocks": 0})
        if regret_w > 0.0:
            regret_loss, regret_info = _satvar_regret_loss(
                sat_h, h.detach(), anchor_ids, anchor_lm, sizes, W.detach(), max_blocks=128)
        raw_value = float(raw.detach())
        regret_value = 0.0 if regret_loss is None else float(regret_loss.detach())
        finite = math.isfinite(raw_value) and math.isfinite(regret_value)
        weighted = float(weight) * raw
        if regret_loss is not None and finite:
            weighted = weighted + float(regret_w) * regret_loss
        weighted_value = float(weighted.detach())
        finite = finite and math.isfinite(weighted_value)
        if finite:
            scaler.scale(weighted).backward()
            if regret_loss is not None:
                with torch.no_grad():
                    sat_h.satvar_regret_updates += 1
        _dblock_clear_moe_aux_stash(core)
    with torch.no_grad():
        d_hist = {str(d + 1): int((d_idx == d).sum()) for d in range(SATVAR_KMAX) if int((d_idx == d).sum()) > 0}
    info = {
        "ran": True, "due": True, "finite": finite, "raw": raw_value, "weighted": weighted_value,
        "regret_loss": regret_value, "tokens": int(used), "available": int(available),
        "rows": int(rows), "seq_len": int(seq_len), "sizes": _satvar_size_hist(sizes),
        "d_hist": d_hist, "regret": regret_info, "weight": float(weight), "regret_weight": float(regret_w),
        "regret_updates": int(sat_h.satvar_regret_updates.item()), "schema": SATVAR_SCHEMA,
    }
    state["fullstack_satvar_anchor_attempts"] = int(state.get("fullstack_satvar_anchor_attempts", 0)) + 1
    state["fullstack_satvar_anchor_last"] = dict(info, step=step)
    if _dblock_audit_due(state, args):
        print("[dblock-satvar-anchor] " + json.dumps(dict(info, step=step), sort_keys=True, separators=(",", ":"), default=str), flush=True)
    del mask, anchor_ids, h, hidden, targets, raw, weighted, regret_loss
    return info


def _nat_boundary_ids(mask_id=None):
    """Return the pinned EOS/PAD and active NAT-mask IDs used as hard boundaries."""
''')

edit("anchor-call", '''    state["training_science_last_receipt"]["anchors"]["nat"] = (
        _dblock_target_route_anchor_audit("nat", completed=False)
    )
    nat_anchor_info = _dblock_fullstack_nat_anchor(
        core, nat_h, scaler, args, ids, state, loss_mask=loss_mask
    )
''', '''    satvar_anchor_info = _dblock_fullstack_satvar_anchor(
        core, sat_h, scaler, args, ids, state, loss_mask=loss_mask
    )
    if satvar_anchor_info.get("ran") and not satvar_anchor_info.get("finite", False):
        opt.zero_grad(set_to_none=True)
        _dblock_clear_moe_aux_stash(core)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[dblock-satvar-anchor] non-finite; skipped optimizer step", flush=True)
        _profile_toc(state, "step_total", _step_t)
        _profile_step_done(state, args)
        _update_stats(state, bi, raw_avg_val, args, objective=objective, trained=False)
        state["training_science_last_receipt"]["route_outcome"] = "satvar_anchor_nonfinite"
        _dblock_emit_target_route_receipt(state, args)
        return raw_avg_val
    if satvar_anchor_info.get("ran"):
        local_diagnostics["satvar_anchor"] = _dblock_json_copy(satvar_anchor_info)

    state["training_science_last_receipt"]["anchors"]["nat"] = (
        _dblock_target_route_anchor_audit("nat", completed=False)
    )
    nat_anchor_info = _dblock_fullstack_nat_anchor(
        core, nat_h, scaler, args, ids, state, loss_mask=loss_mask
    )
''')

# ───────────────────────── 5. optimizer groups (tail) + adapter allow-list + LR exclusions ─────────────────────────
edit("optgroups", '''        add(sat_h.gate_conf.parameters(), conf_lr, "sat_variable_gate_conf")
    return groups
''', '''        add(sat_h.gate_conf.parameters(), conf_lr, "sat_variable_gate_conf")
    # v28 variable-block SAT tensors: appended after every historical group so a pre-v28
    # checkpoint's optimizer indices are unchanged (fresh moments via _hc_adapt_opt_state).
    _sv_shift = [p for n, p in sat_h.named_parameters() if n.startswith("shift_emb")]
    _sv_regret = [p for n, p in sat_h.named_parameters() if n.startswith("stride_regret.")]
    if _sv_shift:
        add(_sv_shift, float(getattr(args, "satvar_shift_lr", 1.0e-4) or 1.0e-4), "satvar_shift")
    if _sv_regret:
        add(_sv_regret, float(getattr(args, "satvar_regret_lr", 3.0e-4) or 3.0e-4), "satvar_regret")
    return groups
''')
edit("optgroups-exclude-sat-head-branch", '''    if getattr(sat_h, "gate", None) is None:
        add(sat_h.parameters(), lr_head, "sat_head")
    else:
''', '''    if getattr(sat_h, "gate", None) is None:
        add((p for n, p in sat_h.named_parameters() if not _satvar_is_param_key(n)), lr_head, "sat_head")
    else:
''')
edit("opt-adapter-allow", '''        allowed = {"hc_core", "repo_core", "sat_variable_gate_conf"}
''', '''        allowed = {"hc_core", "repo_core", "sat_variable_gate_conf", "satvar_shift", "satvar_regret"}
''')
edit("lr-cosine-exclude", '''                if str(_lrg.get("agillm43_role") or "") in ("sat_variable_gate", "sat_variable_gate_conf"):
                    # The 1-vs-2 gate was introduced/reset late in training and
''', '''                if str(_lrg.get("agillm43_role") or "") in ("sat_variable_gate", "sat_variable_gate_conf", "satvar_shift", "satvar_regret"):
                    # The 1-vs-2 gate was introduced/reset late in training and
''')
edit("lr-override-exclude", '''            for _lrg in opt.param_groups:
                if str(_lrg.get("agillm43_role") or "") in ("sat_variable_gate", "sat_variable_gate_conf"):
                    continue
                _lrg["lr"] = max(float(_lrg["lr"]), float(_lrov))
''', '''            for _lrg in opt.param_groups:
                if str(_lrg.get("agillm43_role") or "") in ("sat_variable_gate", "sat_variable_gate_conf", "satvar_shift", "satvar_regret"):
                    continue
                _lrg["lr"] = max(float(_lrg["lr"]), float(_lrov))
''')

# ───────────────────────── 6. inference head loader tolerates fresh v28 tensors ─────────────────────────
edit("infer-load", '''    loaded = module.load_state_dict(patched, strict=False)
    missing = [key for key in loaded.missing_keys if key not in zero_filled]
    conf_missing = [key for key in missing if key.startswith("gate_conf.")]
    missing = [key for key in missing if not key.startswith("gate_conf.")]
    if missing:
        raise RuntimeError(f"{name} checkpoint missing required keys: " + ", ".join(missing[:12]))
''', '''    loaded = module.load_state_dict(patched, strict=False)
    missing = [key for key in loaded.missing_keys if key not in zero_filled]
    satvar_missing = [key for key in missing if _satvar_is_param_key(key)]
    missing = [key for key in missing if not _satvar_is_param_key(key)]
    conf_missing = [key for key in missing if key.startswith("gate_conf.")]
    missing = [key for key in missing if not key.startswith("gate_conf.")]
    if missing:
        raise RuntimeError(f"{name} checkpoint missing required keys: " + ", ".join(missing[:12]))
    if satvar_missing and hasattr(module, "satvar_init_"):
        # pre-v28 checkpoint: fresh deterministic init of the variable-block SAT tensors
        module.satvar_init_()
        print(f"[infer-compat] {name}: fresh v28 satvar init for " + ", ".join(satvar_missing[:4]), flush=True)
''')

# ───────────────────────── 7. decode: variable-stride sampler ─────────────────────────
edit("decode-helpers", '''def _agillm43_sat_stride(gate, variable, greedy):
    """Use deterministic confidence-gated stride two; never sample admission."""
''', '''def _satvar_policy_active(args, sat_h) -> bool:
    policy = str(getattr(args, "satvar_policy", "auto") or "auto").strip().lower()
    if policy in ("off", "legacy", "gate"):
        return False
    return bool(getattr(args, "var", False)) and sat_h is not None and hasattr(sat_h, "shift_emb")


def _satvar_truncate_kvs(kvs, keep):
    """Drop the trailing cached positions (tuple caches) before a merge-recompute."""
    if kvs is None:
        return None
    out = []
    for kv in kvs:
        if isinstance(kv, KVBuffer):
            kv.length = int(min(kv.length, keep))
            out.append(kv)
        else:
            k, v = kv
            out.append((k[:, :, :keep], v[:, :, :keep]))
    return out


def _satvar_generate(core, ar_h, sat_h, ids, args, prompt_len, min_new, nat_mask_id):
    """Variable-stride SAT decoding (v28).

    The committed sequence is a partition into blocks; every block is forwarded once (all
    its tokens attend to each other and to the whole prefix) and its slot states predict
    the next tokens through proj(h + E_d).  Each step chooses the stride
        m* = argmax_m lambda*(m-1) - sum_{d=2..m} max(0, CEhat_d - CEhat_1) - mu*[m > n_last]
    where CEhat comes from the regret head once it has enough updates (else from the slot
    entropies).  Growth beyond the last block recomputes the trailing tokens as one block
    (a valid partition, so in-distribution) at the cost of one extra pass (mu).
    """
    kmax = max(1, min(int(getattr(args, "satvar_kmax", 4) or 4), SATVAR_KMAX))
    lam = float(getattr(args, "satvar_lambda", 1.0) or 0.0)
    # A merge (re-forwarding the trailing tokens as one block) costs one extra forward now but the
    # larger block keeps paying off on later steps; charge half a pass so 1->2 growth can break even.
    mu = float(getattr(args, "satvar_merge_cost", None) if getattr(args, "satvar_merge_cost", None) is not None else 0.5 * lam)
    min_updates = int(getattr(args, "satvar_regret_min_updates", 2000) or 0)
    policy = str(getattr(args, "satvar_policy", "auto") or "auto").strip().lower()
    use_regret = policy == "regret" or (policy == "auto" and int(sat_h.satvar_regret_updates.item()) >= min_updates)
    greedy = bool(getattr(args, "greedy", False))
    device = ids.device
    # commit the prompt as size-2 blocks (the best-trained regime) with a final block of up to kmax
    # tokens so the first stride decision has every option available without a merge.
    L = int(ids.size(1))
    sizes = []
    rem = L
    tail = min(kmax, L)
    while rem > tail:
        n = min(2 if kmax >= 2 else 1, rem - tail)
        sizes.append(n)
        rem -= n
    if rem > 0:
        sizes.append(rem)
    mask = satvar_partition_mask(sizes, L, device=device)
    h, kvs = core(ids, mask, use_cache=True, total_seq_len=L)
    n_last = sizes[-1]
    h_last = h[0, -n_last:]
    added = 0
    stop = False
    trace = {"stride_hist": {}, "core_forwards": 1, "merges": 0, "policy": "regret" if use_regret else "entropy",
             "lambda": lam, "mu": mu, "kmax": kmax, "chosen_utility_sum": 0.0}
    while added < int(args.max_new) and not stop:
        remaining = int(args.max_new) - added
        n_avail = int(h_last.size(0))
        # candidate slot logits for d = 1..min(n_avail, kmax)
        n_cand = min(n_avail, kmax)
        d_idx = torch.arange(n_cand, device=device)
        cand_h = h_last[:n_cand]
        logits_all = sat_h.slot_logits(cand_h, d_idx).float()
        logits_all[..., int(nat_mask_id)] = -1e9
        if use_regret:
            ce_hat = sat_h.regret_log_ce(h_last[-1:]).float().exp()[0].tolist()
        else:
            lp = logits_all.log_softmax(-1)
            ent = (-(lp.exp() * lp).sum(-1)).tolist()
            ce_hat = ent + [float("inf")] * (SATVAR_KMAX - len(ent))
        m, util = _satvar_choose_stride(ce_hat, n_avail, lam, mu, kmax, remaining)
        trace["chosen_utility_sum"] += float(util)
        if m > n_avail:
            # merge-recompute: the trailing m committed tokens become one block
            keep = L - m
            kvs = _satvar_truncate_kvs(kvs, keep)
            blk_ids = ids[:, keep:]
            blk_mask = torch.zeros((1, 1, m, L), device=device, dtype=torch.float32)
            h_blk, kvs = core(blk_ids, blk_mask, kv_caches=kvs, use_cache=True, total_seq_len=L)
            h_last = h_blk[0]
            trace["core_forwards"] += 1
            trace["merges"] += 1
            d_idx = torch.arange(m, device=device)
            logits_all = sat_h.slot_logits(h_last[:m], d_idx).float()
            logits_all[..., int(nat_mask_id)] = -1e9
        new_tokens = []
        for i in range(m):
            logits = logits_all[i:i + 1].clone()
            logits = _apply_penalties(logits, ids, args.penalty_last_n, args.repetition_penalty,
                                      args.presence_penalty, args.frequency_penalty)
            logits = _suppress_eos(logits, args, added < min_new)
            nxt = _sample(logits, args.temperature, args.top_k, args.top_p, args.min_p, greedy)
            new_tokens.append(nxt)
            ids = torch.cat([ids, nxt], 1)
            added += 1
            if EOS is not None and not getattr(args, "ignore_eos", False) and int(nxt.item()) == int(EOS):
                stop = True
                break
            if added >= int(args.max_new):
                break
        emitted = len(new_tokens)
        trace["stride_hist"][str(emitted)] = trace["stride_hist"].get(str(emitted), 0) + 1
        if stop or added >= int(args.max_new):
            break
        new_ids = torch.cat(new_tokens, dim=1)
        L = int(ids.size(1))
        blk_mask = torch.zeros((1, 1, emitted, L), device=device, dtype=torch.float32)
        h_blk, kvs = core(new_ids, blk_mask, kv_caches=kvs, use_cache=True, total_seq_len=L)
        h_last = h_blk[0]
        trace["core_forwards"] += 1
    gen = int(ids.size(1)) - int(prompt_len)
    trace["generated_tokens"] = gen
    trace["tokens_per_forward"] = round(gen / max(1, trace["core_forwards"]), 4)
    trace["mean_stride"] = round(sum(int(k) * v for k, v in trace["stride_hist"].items()) / max(1, sum(trace["stride_hist"].values())), 4)
    return ids, trace


def _agillm43_sat_stride(gate, variable, greedy):
    """Use deterministic confidence-gated stride two; never sample admission."""
''')

edit("decode-branch", '''    else:
        cached_len = ids.size(1)
        block_stream_kv = block_stream and _block_stream_kv_cache_enabled(args)
        if block_stream_kv:
            h, kvs = _block_stream_forward_cached(
                core,
                ids,
                sat_mask(ids.size(1), structured=use_structured_masks(args)),
                None,
                cached_len,
                args,
            )
''', '''    elif args.mode == "sat" and _satvar_policy_active(args, sat_h):
        # v28 variable-stride SAT: the stride is chosen per step (speed vs. intelligence).
        ids, _satvar_trace = _satvar_generate(core, ar_h, sat_h, ids, args, prompt_len, min_new, int(NAT_MASK_ID))
        sat_stride_hist = {int(k): int(v) for k, v in _satvar_trace["stride_hist"].items()}
        sat_core_forwards = int(_satvar_trace["core_forwards"])
        sat_var_stride1 = int(sat_stride_hist.get(1, 0))
        sat_ar_realign = 0
        if bool(getattr(args, "sat_trace", False)) or bool(getattr(args, "satvar_trace", False)):
            print("[satvar-trace] " + json.dumps(_satvar_trace, sort_keys=True), flush=True)
    else:
        cached_len = ids.size(1)
        block_stream_kv = block_stream and _block_stream_kv_cache_enabled(args)
        if block_stream_kv:
            h, kvs = _block_stream_forward_cached(
                core,
                ids,
                sat_mask(ids.size(1), structured=use_structured_masks(args)),
                None,
                cached_len,
                args,
            )
''')

# ───────────────────────── 8. CLI flags ─────────────────────────
edit("cli-train", '''    tr.add_argument("--dblock_sat_loss_tokens", type=int, default=0,
''', '''    tr.add_argument("--dblock_satvar_enabled", type=int, default=1,
                    help="v28: 1 = the local SAT objective and the satvar anchor use random variable block partitions (hot: dblock_satvar_enabled); 0 = legacy fixed shift-2 only.")
    tr.add_argument("--dblock_satvar_block_probs", default=SATVAR_DEFAULT_BLOCK_PROBS,
                    help="v28: block size distribution 'size:weight,...' for the variable SAT partition (hot: dblock_satvar_block_probs).")
    tr.add_argument("--dblock_satvar_regret_weight", type=float, default=0.05,
                    help="v28: weight of the detached regret-head regression (hot). 0 disables the regret head.")
    tr.add_argument("--dblock_satvar_regret_blocks", type=int, default=256,
                    help="v28: blocks per local SAT step whose slots supervise the regret head (hot).")
    tr.add_argument("--dblock_fullstack_satvar_every", type=int, default=3,
                    help="v28: run the full-stack variable-block SAT anchor every N DBlock steps (hot); 0 disables.")
    tr.add_argument("--dblock_fullstack_satvar_offset", type=int, default=0)
    tr.add_argument("--dblock_fullstack_satvar_tokens", type=int, default=256)
    tr.add_argument("--dblock_fullstack_satvar_rows", type=int, default=2)
    tr.add_argument("--dblock_fullstack_satvar_weight", type=float, default=0.10)
    tr.add_argument("--satvar_shift_lr", type=float, default=1.0e-4,
                    help="v28: fixed LR of the per-distance shift embeddings (excluded from cosine/governor like the gate).")
    tr.add_argument("--satvar_regret_lr", type=float, default=3.0e-4,
                    help="v28: fixed LR of the regret head.")
    tr.add_argument("--dblock_sat_loss_tokens", type=int, default=0,
''')
edit("cli-infer", '''    inf.add_argument("--var", action="store_true", default=None)
    inf.add_argument("--no-var", dest="var", action="store_false")
''', '''    inf.add_argument("--var", action="store_true", default=None)
    inf.add_argument("--no-var", dest="var", action="store_false")
    inf.add_argument("--satvar_policy", choices=["auto", "regret", "entropy", "legacy"], default="auto",
                     help="v28 --mode sat --var stride policy: auto = regret head when trained else slot entropy; legacy = the old 1-vs-2 gate.")
    inf.add_argument("--satvar_kmax", type=int, default=4, help="v28: maximum stride (<= SATVAR_KMAX).")
    inf.add_argument("--satvar_lambda", type=float, default=1.0,
                     help="v28: value of one saved forward pass in nats (speed vs intelligence dial; 0 = always stride 1).")
    inf.add_argument("--satvar_merge_cost", type=float, default=None,
                     help="v28: extra nats charged when the stride exceeds the last block (merge recompute); default = 0.5*lambda.")
    inf.add_argument("--satvar_regret_min_updates", type=int, default=2000,
                     help="v28: regret head updates required before auto policy trusts it.")
    inf.add_argument("--satvar_trace", action="store_true", help="v28: print the [satvar-trace] JSON.")
''')

# ───────────────────────── 9. checkpoint payload provenance ─────────────────────────
edit("payload-schema", '''            payload["agillm43_training_profile"] = "intelligence-v25-satvar-pretrain"
''', '''            payload["agillm43_training_profile"] = "intelligence-v25-satvar-pretrain"
            payload["agillm44_satvar_schema"] = SATVAR_SCHEMA
''')

# ───────────────────────── apply ─────────────────────────
out = text
for name, old, new, count in edits:
    n = out.count(old)
    if n != count:
        raise SystemExit(f"edit {name!r}: anchor occurs {n} times, expected {count}")
    out = out.replace(old, new)
out = out.replace(
    "# AGILLM44-SEMANTIC-REPO-SATVAR-20260911: Sakana RePo adaptation; identity NoPE retrofit; SAT-var gate remains detached.",
    "# AGILLM44-SEMANTIC-REPO-SATVAR-20260911: Sakana RePo adaptation; identity NoPE retrofit; SAT-var gate remains detached.\n"
    "# AGILLM44-SATVAR-VARIABLE-BLOCK-20260912 (v28.0): variable-block SAT objective + regret-head stride policy (owner directive).",
    1)
DST.write_text(out, encoding="utf8")
compile(out, str(DST), "exec")
print("base  ", actual)
print("output", hashlib.sha256(out.encode("utf8")).hexdigest(), DST, len(out.splitlines()), "lines")
