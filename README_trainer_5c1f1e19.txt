# Standard trainer publish 5c1f1e19 — batchwoodbury_v1 ckptguard_v1

Status: staged save-guard, goes live at next keeper restart.
NOT the code currently running: live Standard PID 2365908 (started before the swap) still runs 71404ce2 in memory.

Date (UTC): 2026-09-25
Source: Vast 51049010 (read-only copy; no live process touched)
  src:    /workspace/direct56_batch_woodbury_20260920/agillm44_retention4_batchwoodbury_v1_ckptguard_v1.py
          (agillm44_retention4_batchwoodbury_v1.py in that dir is now a symlink to it, since 06:32 UTC;
           the previous file is kept as agillm44_retention4_batchwoodbury_v1.py.orig_71404ce2)
  file:   agillm44_retention4_batchwoodbury_v1_ckptguard_v1_5c1f1e19.py
  sha256: 5c1f1e1934dca112ccb6934e5e42862d7e96f5573243cbc45070f452d380ff5a
  bytes:  2093538   (patch of Standard 71404ce2)

HF (MarxistLeninist/AGILLM-4.3-checkpoints) commit: 95b7ce756ad3bccfec1d8935ad13428bb4c2e5d2
  trainers/agillm44_retention4_batchwoodbury_v1_ckptguard_v1_5c1f1e19.py
Verify: download from HF at that commit and compare sha256 byte-identical.
