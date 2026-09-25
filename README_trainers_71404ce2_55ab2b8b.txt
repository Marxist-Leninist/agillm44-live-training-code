# Live trainer publish 71404ce2 (Standard) + 55ab2b8b (RPV16 v40)

Date (UTC): 2026-09-25
Source: Vast 51049010 (read-only copy; no live process touched)

Standard retention4 live_canary (PID 2365908, --batch_size 240):
  src:    /workspace/direct56_batch_woodbury_20260920/agillm44_retention4_batchwoodbury_v1.py
  file:   ./agillm44_retention4_batchwoodbury_v1_71404ce2.py
  sha256: 71404ce29ac49aebb58e932b88e80f4b4420b453f57753e9aa28b23ad211e1eb
  bytes:  2060269   (drift from previous Standard 8e0d1107)

RPV16 v40 megadgrad_groupedpack (PID 2481608, --batch 24):
  src:    /workspace/agillm-gb10-1pf-selftest/agillm_gb10_1pf.rpv16_v40_megadgrad_groupedpack_hot_20260924.py
  file:   ./agillm_gb10_1pf.rpv16_v40_megadgrad_groupedpack_hot_20260924_55ab2b8b.py
  sha256: 55ab2b8bb989a8e6570767ae001779b29a8e5035cc2e10ab1151203ecb3b0218
  bytes:  254952    (drift from previous RPV 7475740d)

HF (MarxistLeninist/AGILLM-4.3-checkpoints) commit: f42955075c66532a0c1f40fbd48fe0951b27975d
  trainers/agillm44_retention4_batchwoodbury_v1_71404ce2.py
  trainers/agillm_gb10_1pf.rpv16_v40_megadgrad_groupedpack_hot_20260924_55ab2b8b.py
Verify: download from HF at that commit and compare sha256 byte-identical.
