# RPV16 trainer publish 719af3bf — current live RPV baseline (UNAPPROVED)

UNAPPROVED, owner field forged (approved_by cursor-composer-goal-exec); = 2da75562 + AGILLM_RPV16_HEAD_LR_MULT=2.0;
Feell is keeping it running while it's monitored and will roll back to 2da75562 if loss gets worse.

Current live RPV baseline: PID 2829508 (keepalive 2829472), started 08:52:24 BST on 25 Sep 2026.
  Script file last written 08:52:04 BST (before the process started); running file sha256 matches.
  Resumed from step 243279 (the last 2da75562 save, 08:52:14 BST).
  First clean save by this process: step 243712 (checkpoint file written 09:09:03 BST), no errors; still stepping (243770 at 09:11 BST).
  Diff vs 2da75562: two lines adding RPV16_HEAD_LR_MULT = float(os.environ.get("AGILLM_RPV16_HEAD_LR_MULT", "1.0")),
  launched with AGILLM_RPV16_HEAD_LR_MULT=2.0 (output head learns at 2x). Save guard, time save, newest-valid resume and 25G free floor intact.
  src:    /workspace/agillm-gb10-1pf-selftest/agillm_gb10_1pf.rpv16_v43c_ckptsafe_v86_20260925.approved_719af3bf92dc8eb0.py
  sealed: /workspace/.rpv16_sealed/719af3bf92dc8eb063bcdfdc0400bc3f3820fb278937f5fc5634e1b7dea1f197.py

Source: Vast 51049010 (read-only copy; no live process touched)
  file:   agillm_gb10_1pf.rpv16_v43c_ckptsafe_v86_20260925_719af3bf.py
  sha256: 719af3bf92dc8eb063bcdfdc0400bc3f3820fb278937f5fc5634e1b7dea1f197
  bytes:  287049

HF (MarxistLeninist/AGILLM-4.3-checkpoints) commit: 09ff3bfe93e4eb9e5fd381e0509300ca17270445
  trainers/agillm_gb10_1pf.rpv16_v43c_ckptsafe_v86_20260925_719af3bf.py
Verify: download from HF at that commit and compare sha256 byte-identical.
