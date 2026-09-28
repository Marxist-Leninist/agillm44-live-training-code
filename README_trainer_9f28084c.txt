AGILLM RPV16 2B trainer snapshot 9f28084c (v43c_ckptsafe_v91p2_sm120_lazywq): LIVE RPV16
=========================================================================================

File: agillm_gb10_1pf.rpv16_v43c_ckptsafe_v91p2_sm120_lazywq_20260927_9f28084c.py
sha256: 9f28084c00fd10120a1e782802eb741869c14f0be76518ff181280b259193afc
bytes: 291725   lines: 4932
Source on the training box: /workspace/rpv16_sm120/agillm_gb10_1pf.rpv16_v43c_ckptsafe_v91p2_sm120_lazywq_20260927.py (mtime 2026-09-27 23:17 UK)
Host: Vast 53045147, NVIDIA RTX PRO 4000 Blackwell 24 GB (sm_120), interruptible. The host was set up 2026-09-27 by
cowork-opus55-rpv16-2b-5090-20260927 as the cheaper replacement for 53020040. RPV16 owner: Feell.

STATUS (checked 2026-09-28 about 01:53 UK)
- LIVE: trainer PID 89 under keeper PID 73 (rpv16_keeper.sh, a restart-only-if-absent supervisor with AGILLM_RESUME_NEWEST=1).
  Both started 2026-09-27 23:50 UK after the container restart. The keeper log's last launch is 23:50:01 UK.
- PID 89 logged "resumed" from step 247312 at 23:50:27 UK (the earlier process had reached about 247724 unsaved; that work was lost at the restart).
- Clean saves by PID 89 (no other trainer has launched since), each a "checkpoint" event after checkpoint_write_enter:
  248320 (00:15 UK), 249344 (00:40), 250368 (01:05), 251392 (01:30:20 UK).
  ckpt_save_status: last_ok_step 251392, 0 failures. Saves go to /workspace/agillm-gb10-1pf-targetfix-active/RPV16-GB10-1PF-v6-current-resumable.pt (local, rolling).
- There are no Tracebacks in the trainer log. At 01:53 UK it was on step 252310 with loss about 6.9.

LINEAGE
HF MarxistLeninist/AGILLM-GB10-1PF recovery/rpv16-targetfix-step000246907-v6.ckpt (keeper SEED_CKPT)
 -> this host: saves 247296/247312 (23:31 UK Sep 27, earlier launches of this same sha)
 -> container restart about 23:50 UK -> PID 89 resume 247312 -> saves 248320 ... 251392 (local rolling resumable file, not on HF as of this check).
The AGILLM-GB10-1PF recovery/ checkpoints at 252928 and 259584 were not written by this host's PID 89. They are not part of the lineage above.
Note: agents were doing a cutover to a successor host (Vast 53072506, RTX 3090) at the time of publishing. This file is what PID 89 runs on 53045147.

DIFF vs parent v91 portable 1768b9d9 (agillm_gb10_1pf.rpv16_v43c_ckptsafe_v91_portable_20260925.py; 4901 -> 4932 lines; diff shows 9 lines removed, 40 added, 9 hunks)
1. V91P_SM120: the device gate changed from get_device_name == "NVIDIA GB10" to compute capability major == 12. It now accepts GB10 (sm_121) and
   RTX 50xx / RTX PRO Blackwell (sm_120); non-SM12x GPUs are still refused. The runtime .so files are rebuilt for sm_120a/x86_64
   (/workspace/rpv16_sm120_runtime; setup bundle on HF AGILLM-GB10-1PF runtime/sm120/).
2. V91P2_LAZYWQ: an execution-only memory fix. SparseLinear no longer keeps a resident BF16 dequantised masked weight (_wq)
   for each of 288 layers (about 3.8 GB). shadow() now calls quantize32_bf16(..., materialize=False). The new wq() rebuilds the same BF16
   tensor on demand (same weight version, same fixed pair48 mask, same deterministic kernel) for the backward GEMM, and release_wq()
   frees it right after use. This is applied in all 6 backward sites, including the fused _mds_v38 path.
   The author states: no model, optimizer, objective, data, schedule or checkpoint-schema change.
The diff vs 85d3b2f4 (v43c_ckptsafe_v86, the previous published LIVE RPV) is larger (41 lines removed, 102 added) because v91 portable sits in between.

KEY SETTINGS
- argv: --dataset fineweb-edu --batch 24 --seq 2048 --steps 12207032 --grad-accum 1 --lr 4.42e-4 --warmup-tokens 1000000
- env: AGILLM_RPV16_HEAD_LR_MULT=1.5 (note: 85d3b2f4 ran at 1.0), AGILLM_RPV16_ALWAYS_HEAD=1, AGILLM_RPV16_TIED_HEAD=1,
  AGILLM_RPV16_SKIP_HEAD_ONLY=1, AGILLM_RPV16_JOINT=0, AGILLM_RPV16_JOINT_EVERY=16, AGILLM_CKPT_EVERY_SEC=1800,
  AGILLM_CKPT_FREE_FLOOR_GB=14, AGILLM_RESUME_NEWEST=1, AGILLM_FUSED_SILU_BWD=1, AGILLM_FUSED_MASKED_WGRAD=1,
  AGILLM_MOE_RESIDUAL=auto; optimizer paged_adamw8bit_bank.
- Model 1,990,168,065 parameters, schema v6, target_alignment next_token_shifted_labels_fullseq_v1.

SHA256SUMS: see SHA256SUMS_9f28084c.txt
Published read-only by Qwen Improver's publisher. The live process, keeper, checkpoints and disk on the training box were not touched.
