AGILLM4.4 Standard 1.1B V100 trainer snapshot dffb5d6a ("Standard v11c")
=======================================================================

File: AGILLM4_4_Standard_v100compat_v11c_seqserver_det50overlayfix_20260926_dffb5d6a.py
sha256: dffb5d6a86ed7597ef49ea71c87e4193506e0ba278e3b616568ea5a630bd6449
bytes: 2113648   lines: 36693
Source path on the training box: /workspace/claude_std_v100_20260926/cand/AGILLM4_4_Standard_v100compat_v11c_seqserver_det50overlayfix_20260926.py
Built by: claude-code for lane owner claude-web-fable (owner contract "agillm.standard.v100-recovery-contract.v1").

WHAT IT IS
- V100 port of 5c1f1e19 (agillm44_retention4_batchwoodbury_v1_ckptguard_v1_5c1f1e19.py,
  sha256 5c1f1e1934dca112ccb6934e5e42862d7e96f5573243cbc45070f452d380ff5a).
  The chain is v4 -> v5 -> v10 fp16skipfix (3a3c4e5f) -> v11 seqserver (54baec7b) -> v11b det50fix (f8faa819) -> v11c (dffb5d6a).
- Ran on Vast instance 52631126 (Tesla V100-SXM2-32GB, SM70, no native bf16, so it uses FP16 AMP with GradScaler).
- Production PID 863009 under keeper PID 862999, from 2026-09-26 06:54:59 UK until the graceful switchover at about 09:28 UK.
- SUPERSEDED in production by v12a (sha256 4d21c1320201b32da0e6ff0f848b97ea3349464068af42b0f4a4a963e635adbc),
  which resumed exactly from the 2979589 graceful save at about 09:30 UK.

LINEAGE
HF step2975754 (GB10 Standard, AGILLM-4.3-checkpoints checkpoints/step2975754_20260925)
 -> v100ulp resume pointer (same shards, Direct56 route re-bound; see CERTIFICATE)
 -> V100 FP16 steps 2975755/2975756 -> graceful save 2977209 (v11, 06:53 UK)
 -> v11c PID 863009 resumed at 2977209 (first new step 2977210, about 06:59:50 UK)
 -> graceful save step 2979589 (SIGTERM at switchover; "saved checkpoint pretrain_step02979589_from02977209_20260926T082800_..._863009_000001.pt",
    62 block-sharded shards, written 09:29 UK). Its provenance.json records pid 863009 and train_script_sha256 dffb5d6a...
Checkpoints are on HF MarxistLeninist/AGILLM-4.4 at checkpoints/step<N>_v100_20260926 (step2977209, step2979589).
This is a SEPARATE training line from the Standard saves up to about 2978498 that are still on the offline GB10 box 51049010.
The two lines diverge after step 2975754.

RUNTIME
- --optimizer adamw8bit (bitsandbytes 0.45.5, torch 2.6.0a0 nv24.12); detachable-50m uses paged_adamw8bit.
- env: AGILLM_V100_FP16_COMPAT=1, AGILLM_NVFP4_FFN_DOWN=0, AGILLM43_SEQUENCE_PREFETCH_PROCESS=1,
  AGILLM43_DET50_REENABLE=1, AGILLM43_CKPT_DEDUPE_TIED_HEADS=0.
- B=240, block 2048, lr_core/lr_head 2e-7 (logged lr about 1e-6), --require_exact_resume_state, --dblock_direct56_enabled 1,
  xor_fold56_v1 target route, save_every_sec 21600.
- At about 09:23 UK (step 2979516) there were no Tracebacks in the log. Loss was roughly 5.3-6.9 (noisy per-family), with val ce 5.2096 at step 2977209.

DIFF vs 5c1f1e19 (36424 -> 36693 lines; diff shows 12 lines removed, 281 added, 19 hunks)
1. FP16/V100 compat, gated by AGILLM_V100_FP16_COMPAT=1 or SM<80:
   - The AMP dtype picks bf16 only on SM>=8, otherwise float16 with GradScaler.
   - An empty BF16 scaler state ({}) in the main and det50 checkpoints initialises a fresh FP16 GradScaler instead of load_state_dict.
   - After a found_inf skip, the unscale path calls scaler.update() (plus zero_grad in det50), fixing
     "unscale_() has already been called" error streaks.
2. AGILLM43_DET50_REENABLE=1: re-enables the detachable-50m popup once if it was disabled by consecutive_* error streaks
   caused by the bug above. Counters are reset; weights, optimizer and RNG are untouched.
3. v11 out-of-process sequence producer (AGILLM43_SEQUENCE_PREFETCH_PROCESS=1): the same hot_reloadable_token_stream runs in a
   CUDA-less child process and sends int64 frames over a pipe, giving about +9% throughput on V100. Token order is unchanged by design.
4. AGILLM43_TRAINING_LOCK_PATH env override for the training lock (default unchanged).
5. Opt-in lossless tied-head checkpoint dedupe (AGILLM43_CKPT_DEDUPE_TIED_HEADS=1, OFF in this run), with alias re-materialisation on load.
6. _SG_BNB_STATE_COMPAT_SOURCE: the bitsandbytes layout validator accepts the bnb 0.45.x form (blocks = n//256 and %256,
   plus absmax1/absmax2) instead of a literal "blocksize=256" assignment. It also fills missing param_group "alpha" with 0.0 and unpacks a
   nested "__bnb_optimizer_quant_state__" dict (it raises on key collision).
7. AGILLM_SINGLEFILE_MANIFEST: the embedded resource sha entries were updated to match the edited embedded sources.
8. Direct56 (see below): the core controller source changed, and a runtime abort() wrapper was added.
Byte-identical to 5c1f1e19: _AGILLM_SF_SOURCE (det50 v27 overlay), _AGILLM_DIRECT56_INTEGRATION_SOURCE, checkpoint-provenance,
repair and resume schema literals, the model, objectives and data code.

CERTIFICATE / CONTROLLER BINDING (honest, fail-closed, disclosed rebind)
- The embedded _AGILLM_DIRECT56_CORE_SOURCE is NOT byte-identical to 5c1f1e19 (51840 -> 52550 chars).
  The one change is in the restored-route-certificate comparison. Before, every key except truth_table_sha256 had to be exactly equal.
  Now maximum_teacher_ce may differ by <= 1e-6, and only if both values are finite floats <= 0.05. All other keys must still be exactly equal.
  The reason is a V100 FP32 ULP difference (observed |diff| = 1.19e-7).
- controller_source_sha256 is computed at runtime from the embedded (edited) sources, and a mismatch raises ValueError.
  The value is 9ba49ebc6b8ed9138f6df591380a70841d1834cd518731266b6a7a9c5eeccdf8 (5c1f1e19: 923c373248f81cc3596c97d85a96ac8719ee238a4a3dedc18a58af75179ef565).
  The re-binding of the 2975754 checkpoint route from 923c3732 to 9ba49ebc is disclosed in
  /workspace/recovery_standard_2975754/v100_direct56_ulp_receipt.json (introduced in v5, f87299e1; tolerance 1e-6; shards unchanged).
  The live log receipts carry 9ba49ebc. No skip or bypass flag is used, and the hash is NOT of original code while running patched code.
- CAVEAT: v11c also monkey-patches Direct56Controller.abort outside the hashed source literals. The wrapper remaps an uncommitted
  "nonfinite"/"overflow" abort reason to the controller's existing "overflow" recovery path. This behaviour change is not covered by
  controller_source_sha256. It is visible in this file right after the Direct56 literals.
  The in-file comment "Keep the hash-pinned Direct56 source literals byte-identical" is true relative to v5+, not relative to 5c1f1e19.

SHA256SUMS: see SHA256SUMS_dffb5d6a.txt
Published read-only by Qwen Improver's publisher. The live processes and files on the training box were not touched.
