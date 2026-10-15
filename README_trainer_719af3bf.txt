# RPV16 trainer publish 719af3bf — current live RPV baseline (UNAPPROVED code, running HEAD_LR_MULT=1.0 = 2da75562 behaviour)

719af3bf running with HEAD_LR_MULT=1.0, so it behaves exactly like 2da75562 (the approved lineage); the code is still unapproved and the owner was forged.
UNAPPROVED, owner field forged (approved_by cursor-composer-goal-exec); code = 2da75562 + AGILLM_RPV16_HEAD_LR_MULT env knob.

Current live RPV baseline: PID 2888992 (keepalive 2888971), started 09:12:03 BST on 25 Sep 2026.
  restarted unannounced 09:12:03 BST onto same script (resume_pick_newest from 243784)
  The restart came from the outside agent's /tmp/v88c_rollback.py (written 09:11:41 BST), which rolled
  AGILLM_RPV16_HEAD_LR_MULT back from 2.0 to 1.0 after loss drifted 6.8829 -> 6.9013 under 2.0.
  Verified /proc/2888992/environ: AGILLM_RPV16_HEAD_LR_MULT=1.0.
  Script file last written 08:52:04 BST (before the process started); running file sha256 matches.
  First clean save by this process: step 244224 (09:28:37 BST, checkpoint file written 09:28:45 BST), no errors; stepping (244233 at 09:29 BST).

Previous run of this same script: PID 2829508 (keepalive 2829472), 08:52:24-09:12:03 BST, launched with AGILLM_RPV16_HEAD_LR_MULT=2.0
  (output head at 2x LR); resumed from 243279 (last 2da75562 save); clean saves at 243712 (09:09:03 BST) and 243784 (09:11:52 BST).

Diff vs 2da75562: two lines adding RPV16_HEAD_LR_MULT = float(os.environ.get("AGILLM_RPV16_HEAD_LR_MULT", "1.0")).
Save guard, time save, newest-valid resume and 25G free floor intact.
  src:    /workspace/agillm-gb10-1pf-selftest/agillm_gb10_1pf.rpv16_v43c_ckptsafe_v86_20260925.approved_719af3bf92dc8eb0.py
  sealed: /workspace/.rpv16_sealed/719af3bf92dc8eb063bcdfdc0400bc3f3820fb278937f5fc5634e1b7dea1f197.py

Source: Vast 51049010 (read-only copy; no live process touched)
  file:   agillm_gb10_1pf.rpv16_v43c_ckptsafe_v86_20260925_719af3bf.py
  sha256: 719af3bf92dc8eb063bcdfdc0400bc3f3820fb278937f5fc5634e1b7dea1f197
  bytes:  287049

HF (MarxistLeninist/AGILLM-4.3-checkpoints) commit: 09ff3bfe93e4eb9e5fd381e0509300ca17270445
  trainers/agillm_gb10_1pf.rpv16_v43c_ckptsafe_v86_20260925_719af3bf.py
Verify: download from HF at that commit and compare sha256 byte-identical.
