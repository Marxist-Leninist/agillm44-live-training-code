# RPV16 trainer publish ecd23720 (history) — v43c ckptsafe

v43c, fixes the v43b double-reason save TypeError (7dd2b8d3, never published); live PID 2654750 08:06-08:24 BST on 25 Sep 2026;
first clean save 242688 (08:21:34 BST); killed by an outside in-place edit (v87-skiphead overwrite of the running file at 08:23:49 BST,
restart at 08:23:53 while the step-242747 save was in progress, leaving a partial .tmp).
  sealed copy: /workspace/.rpv16_sealed/ecd23720f26cae8a5d97d938fe27eaf3c730ff69a573627a9b7635a889a2388c.py
  also: agillm_gb10_1pf.rpv16_v43c_ckptsafe_v86_20260925.approved_ecd23720f26cae8a.py.bak-v87-skiphead-20260925T072349Z

Source: Vast 51049010 (read-only copy; no live process touched)
  file:   agillm_gb10_1pf.rpv16_v43c_ckptsafe_v86_20260925_ecd23720.py
  sha256: ecd23720f26cae8a5d97d938fe27eaf3c730ff69a573627a9b7635a889a2388c
  bytes:  286611

HF (MarxistLeninist/AGILLM-4.3-checkpoints) commit: f586622b5a2d153b47d9db2a1804cccc822c4e6f
  trainers/agillm_gb10_1pf.rpv16_v43c_ckptsafe_v86_20260925_ecd23720.py
Verify: download from HF at that commit and compare sha256 byte-identical.
