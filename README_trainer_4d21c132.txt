AGILLM4.4 Standard 1.1B V100 trainer snapshot 4d21c132 ("Standard v12a"): LIVE production
=========================================================================================

File: AGILLM4_4_Standard_v100compat_v12a_moeindex_dedupe_20260926_4d21c132.py
sha256: 4d21c1320201b32da0e6ff0f848b97ea3349464068af42b0f4a4a963e635adbc
bytes: 2113856   lines: 36693
Source path on the training box: /workspace/claude_std_v100_20260926/cand/AGILLM4_4_Standard_v100compat_v12a_moeindex_dedupe_20260926.py
Parent: v11c dffb5d6a86ed7597ef49ea71c87e4193506e0ba278e3b616568ea5a630bd6449 (published at HF MarxistLeninist/AGILLM-4.4 commit
1eecfdc6828288feef086f76d96124ce909aac86, trainers/README_trainer_dffb5d6a.txt; that README has the full diff and lineage back to 5c1f1e19).
Author/continuation agent: chatgpt-v12a-continuation-20260926 (OWNER_CONTRACT "continued_by"). Lane owner: claude-web-fable.
Hardware: Vast 52631126, Tesla V100-SXM2-32GB (SM70, no native bf16, so it uses FP16 AMP with GradScaler).

STATUS (checked 2026-09-26 about 14:59 UK)
- LIVE: trainer PID 2602692 under keeper PID 2602549, running this exact sha. It started at 10:25:09 UK and resumed from step 2980387.
- Clean auto-save by PID 2602692: step 2985983 at 14:24:34 UK ("saved checkpoint pretrain_step02985983_from02980387_20260926T1323Z.pt",
  62 shards; ckpt_save_status last_save_outcome=ok_attempt1). provenance.json records pid 2602692 and train_script_sha256 4d21c132...
- The last heartbeat seen was step 2986780. There are no Tracebacks in the log.

LINEAGE
v11c dffb5d6a PID 863009 -> graceful save 2979589 (09:29 UK; HF AGILLM-4.4 checkpoints/step2979589_v100_20260926)
 -> v12a PID 2171733 (keeper 2171684), started 09:30:10 UK, exact resume from 2979589
 -> graceful SIGTERM save step 2980387 at 10:24 UK by PID 2171733 (provenance: pid 2171733, train_script_sha256 4d21c132...)
    HF AGILLM-4.4 checkpoints/step2980387_v100_20260926
 -> same sha relaunched as PID 2602692 (keeper 2602549) with AGILLM43_SEQUENCE_PREFETCH_DEPTH=960
    (launch json production_launch_v12a_prefetch960.json), exact resume from 2980387
 -> auto-save step 2985983 at 14:24 UK. HF AGILLM-4.4 checkpoints/step2985983_v100_20260926
This is a SEPARATE training line from the Standard saves up to about 2978498 that are still on the offline GB10 box 51049010.
The two lines diverge after HF step 2975754.

DIFF vs v11c dffb5d6a (the only change: MoEFFN.forward top-1 expert dispatch, lines 14197-14208; diff shows 8 lines removed, 8 added)
  before: mask = chosen == expert_id; skip if not mask.any(); gate = probs[mask, expert_id]; out[mask] = expert(flat[mask]) * gate_st
  after:  rows = (chosen == expert_id).nonzero().flatten(); skip if rows.numel() == 0;
          gate = probs.index_select(0, rows)[:, expert_id]; out.index_copy_(0, rows, (expert(flat.index_select(0, rows)) * gate_st).to(out.dtype))
  Why: this needs 1 device->host sync per expert instead of 4 (mask.any, probs[mask], flat[mask], out[mask]). The author says it is
  math-identical: same rows in the same ascending order, same values and the same autograd graph. --moe_no_cpu_sync semantics are unchanged.
  The receipt cites 32 bitwise dispatch tests, a deduped load and a 3-step exact resume, all passing.
  "dedupe" in the name is NOT a code change. The lossless tied-head checkpoint dedupe code already existed in v11 (opt-in).
  v12a turns it ON at launch with AGILLM43_CKPT_DEDUPE_TIED_HEADS=1: the AR/SAT/NAT proj.weight duplicates of core.emb.weight are
  dropped from saves and re-aliased on load. The log confirms "tied proj.weight deduplicated (alias of core.emb.weight on load)".

DIRECT56 / CERTIFICATE
- The embedded _AGILLM_DIRECT56_CORE_SOURCE and _AGILLM_DIRECT56_INTEGRATION_SOURCE strings, and the runtime Direct56Controller.abort
  retry patch, are byte-identical to v11c because the diff touches only the MoE lines above.
  So controller_source_sha256 is still 9ba49ebc6b8ed9138f6df591380a70841d1834cd518731266b6a7a9c5eeccdf8.
  The honest, fail-closed rebind story is unchanged. The caveat that the abort() patch is not covered by the hash, described in the v11c README, still applies.

RUNTIME
- --optimizer adamw8bit (bitsandbytes 0.45.5, torch 2.6.0a0 nv24.12); detachable-50m uses paged_adamw8bit.
- env: AGILLM_V100_FP16_COMPAT=1, AGILLM_NVFP4_FFN_DOWN=0, AGILLM43_SEQUENCE_PREFETCH_PROCESS=1, AGILLM43_SEQUENCE_PREFETCH_DEPTH=960,
  AGILLM43_CKPT_DEDUPE_TIED_HEADS=1, AGILLM43_DET50_REENABLE=1.
- argv is unchanged from v11c (B=240, block 2048, lr_core/lr_head 2e-7, --require_exact_resume_state, direct56 enabled,
  save_every_sec 21600 on argv; hot_config sets 14400s).

SHA256SUMS: see SHA256SUMS_4d21c132.txt
Published read-only by Qwen Improver's publisher. The live processes and files on the training box were not touched.
