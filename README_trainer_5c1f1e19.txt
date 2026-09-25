# Standard trainer publish 5c1f1e19 — batchwoodbury_v1 ckptguard_v1

Status: LIVE since ~07:53 BST on 25 Sep 2026 as Standard PID 2604047, B=240 (--batch_size 240),
resumed from step 2975744 with exact-resume verified.
  Process started 07:49:54 BST (06:49:54 UTC); script file last written 07:31:03 BST (06:31:03 UTC), before start.
  Log evidence (/workspace/dual_training_20260920/standard_trainer.log):
    [continuation-state] exact resume coverage required and verified
    [pretrain] resume counters: step=2975744 seen_tok=322403667968 current_B=240 current_L=2048
    [ckpt-guard] v1 active
It replaced Standard 71404ce2 (previous PID 2365908, now exited).

Source: Vast 51049010 (read-only copy; no live process touched)
  src:    /workspace/direct56_batch_woodbury_20260920/agillm44_retention4_batchwoodbury_v1_ckptguard_v1.py
          (launch path agillm44_retention4_batchwoodbury_v1.py in that dir is a symlink to it)
  file:   agillm44_retention4_batchwoodbury_v1_ckptguard_v1_5c1f1e19.py
  sha256: 5c1f1e1934dca112ccb6934e5e42862d7e96f5573243cbc45070f452d380ff5a
  bytes:  2093538   (patch of Standard 71404ce2)

HF (MarxistLeninist/AGILLM-4.3-checkpoints) commit: 95b7ce756ad3bccfec1d8935ad13428bb4c2e5d2
  trainers/agillm44_retention4_batchwoodbury_v1_ckptguard_v1_5c1f1e19.py
Verify: download from HF at that commit and compare sha256 byte-identical.
