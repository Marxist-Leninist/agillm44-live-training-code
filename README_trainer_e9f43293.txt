AGILLM4.4 Standard 1.1B V100 trainer snapshot e9f43293 ("Standard v12a codex retained-stream"): LIVE production
================================================================================================================

File: AGILLM4_4_Standard_v12a_codex_retained_stream_20260926_e9f43293.py
sha256: e9f43293a65cccc7bb64a89e4f4198f34de34eb2d7d9616f676ea882c1d029e4
bytes: 2119886   lines: 36824
Source on the training box: /workspace/claude_std_v100_20260926/cand/AGILLM4_4_Standard_v12a_codex_retained_stream_20260926.py (mtime 2026-09-26 20:54 UK)
Parent: v12a 4d21c1320201b32da0e6ff0f848b97ea3349464068af42b0f4a4a963e635adbc (published: HF AGILLM-4.4 commit 66b05dff, README_trainer_4d21c132.txt;
        v11c dffb5d6a: HF AGILLM-4.4 commit 1eecfdc6).
Author: a codex agent (per the file name and its manifest AGILLM4_4_Standard_v12a_codex_retained_stream_20260926.manifest.json; the manifest names no agent).
Lane owner / LR decisions: claude-web-fable. First mirrored to HF and GitHub by AG (AG hourly 2026-09-26).
Host: Vast 52631126, Tesla V100-SXM2-32GB (SM70, no bf16, so it uses FP16 AMP with GradScaler).

STATUS (checked 2026-09-28 about 01:53 UK)
- LIVE: trainer PID 1168697 under keeper PID 1168684 (standard_keeper_v100.py). It started 2026-09-26 20:58:40 UK and resumed from step 2993411.
- Clean saves by PID 1168697 (provenance.json pid 1168697 and train_script_sha256 e9f43293...; each has a "saved checkpoint" log line, 62 shards):
  2997884 (00:56 UK Sep 27), 3002666 (04:56), 3007299 (08:56), 3011811 (12:56), 3016769 (16:56), 3021281 (20:57 UK Sep 27; ckpt_save_status ok_attempt1).
- There are no Tracebacks in the log. At 01:52 UK Sep 28 it was on step 3026542 (loss about 7.0, noisy per-family) with lr=4.00e-08.
  No save had happened since 20:57 UK, and ckpt_save_status shows no failed attempt; the 4-hour cadence would have put one at about 00:56 UK.

LINEAGE
v12a 4d21c132 PID 1145055 -> save 2993411 (20:56 UK Sep 26; HF AGILLM-4.4 checkpoints/step2993411_v100_20260926)
 -> e9f43293 PID 1168697, exact resume from 2993411
 -> saves on HF AGILLM-4.4: checkpoints/step2997884_v100_20260927, step3002666_v100_20260927, step3007299_v100_20260927,
    step3011811_v100_20260927, step3016769_v100_20260927, step3021281_v100_20260927.
This is a SEPARATE training line from the Standard saves up to about 2978498 on the offline GB10 box 51049010. The two lines diverge after HF step 2975754.

DIFF vs v12a 4d21c132 (36693 -> 36824 lines; diff shows 126 lines removed, 257 added, 6 hunks)
The manifest says: changed_functions = _stream_call_with_timeout, hot_reloadable_token_stream, token_stream; all other bytes unchanged
(762 unchanged top-level definitions). This is a data-streaming reliability fix only; the model, optimizer, objectives, checkpoint and Direct56 code are untouched.
- Before: each HF open/next call that timed out abandoned a worker thread, and the next call started a new one. Unbounded leaked threads and
  HTTP operations could pile up.
- After: a _StreamPendingCalls manager per token-stream lifetime allows at most ONE outstanding operation per source. A deadline limits
  how long the caller waits, not the operation. A pending result is consumed before any new open/read. Global admission is
  BoundedSemaphore(32) IO slots. Retired streams discard late results and close iterators only after reads end.
- The token_stream and hot_reloadable_token_stream loops are wrapped in try/finally so iterators are closed on exit and on hot-reload switch.
  Lambdas bind source/seed/iterator by value.
The embedded _AGILLM_DIRECT56_CORE_SOURCE, _INTEGRATION_SOURCE, det50 overlay and bnb-compat strings are byte-identical to 4d21c132
(so controller_source_sha256 is still 9ba49ebc...). The runtime Direct56Controller.abort retry patch is unchanged (see the v11c README caveat).

KEY SETTINGS
- --optimizer adamw8bit (bitsandbytes 0.45.5); detachable-50m paged_adamw8bit; B=240, block 2048.
- LR: --lr_core 2e-7 --lr_head 2e-7 --lr_decay cosine --lr_min_mult 0.2, so the schedule floor is 2e-7 x 0.2 = 4e-8.
  The file-based lr-governor override (claude-web-fable, 1e-6 -> 6e-7, "owner-delegated LR decisions") was in force earlier.
  The log then shows "[lr-governor] override file removed; schedule LR restored", and the LR has sat at the 4.00e-08 floor since.
  This 4e-8 clamp comes from the cosine min-mult floor in argv, not a hard-coded constant.
- env: AGILLM_V100_FP16_COMPAT=1, AGILLM_NVFP4_FFN_DOWN=0, AGILLM43_SEQUENCE_PREFETCH_PROCESS=1, AGILLM43_SEQUENCE_PREFETCH_DEPTH=960,
  AGILLM43_CKPT_DEDUPE_TIED_HEADS=1, AGILLM43_DET50_REENABLE=1.
- save_every_sec 21600 on argv; the observed save cadence is 4 h.

SHA256SUMS: see SHA256SUMS_e9f43293.txt
Published read-only by Qwen Improver's publisher. The live processes and files on the training box were not touched.
