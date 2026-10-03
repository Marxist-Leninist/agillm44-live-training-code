AGILLM RPV16 trainer snapshot 85d3b2f4 (v43c_ckptsafe_v86): LIVE RPV
=====================================================================

File: agillm_gb10_1pf.rpv16_v43c_ckptsafe_v86_20260925_85d3b2f4.py
sha256: 85d3b2f402739ec7a57c4672b3b5c54ad29f22ab8e775e53f2cbe760d65d3bd4
bytes: 288262
Source on Vast 51049010: /workspace/agillm-gb10-1pf-selftest/agillm_gb10_1pf.rpv16_v43c_ckptsafe_v86_20260925.approved_85d3b2f402739ec7.py
(mtime 2026-09-25 11:49:16 BST, before the process started)

STATUS: LIVE RPV as of about 14:38 UK on 2026-09-25 (the process started at 14:40:22 BST)
- PID 136086 (parent is keeper PID 136057), cwd /workspace. Checked by reading /proc/136086 on Vast 51049010.
- Restored by Feell from checkpoint step 246907 at HEAD_LR_MULT=1.0 (process env AGILLM_RPV16_HEAD_LR_MULT=1.0).
- 246907 checkpoint is on HF MarxistLeninist/AGILLM-GB10-1PF at recovery/rpv16-targetfix-step000246907-v6.ckpt
- Stability: resumed at 246907 and has trained for more than 15 minutes (loss about 6.8-6.9).
  First clean save after the resume: step 247296, uk_stamp 20260925T145451+0100
  (RPV16-GB10-1PF-v6-current-resumable.pt / latest.pt). No Traceback or TypeError since the resume.

KEEPER: ccdec598
- /workspace/rpv16_allheads_keepalive_20260919.py, PID 136057
- sha256: ccdec598afcf70a3a477abb5963f26a5ca1e3a4cdc6137fa7b2ea123338b7f7d (33400 bytes, mtime 14:38:41 BST)
- This is Feell's revert of the outside composer's HEAD_LR_MULT=1.6 edit.

QUARANTINE
- The 245760-245838 branch trained at HEAD_LR_MULT=1.6 by outside agent
  cursor_composer_goal_exec_20260924 is QUARANTINED. It is not on HF and must not be resumed from or published.

SUPERSEDES
- 719af3bf is SUPERSEDED by 85d3b2f4
  (agillm44-live-training-code PR #30, agillm-trainer-snapshots PR #28).

Byte-identical to trainers/agillm_gb10_1pf.rpv16_v43c_ckptsafe_v86_20260925.py already in the
HF GB10-AGILLM4.4 and AGILLM-4.4 repos (same sha256 85d3b2f4...).

SHA256SUMS: see SHA256SUMS_85d3b2f4.txt
HF: MarxistLeninist/AGILLM-4.3-checkpoints trainers/agillm_gb10_1pf.rpv16_v43c_ckptsafe_v86_20260925_85d3b2f4.py
Published read-only by Qwen Improver's publisher. The live process and files were not touched.

HF commit: 6af73e2ac48ab71f64bf3cfeb892ba2999ad352e
