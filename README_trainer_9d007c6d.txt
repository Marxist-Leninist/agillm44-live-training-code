AGILLM RPV16 2B trainer snapshot 9d007c6d (v43c_ckptsafe_v91p5_emuint8): LIVE RPV16 on RTX 3090
==============================================================================================

File: agillm_gb10_1pf.rpv16_v43c_ckptsafe_v91p5_emuint8_20260928_9d007c6d.py
sha256: 9d007c6d14527a464b14e1bf91cf9666b04cdf61ef8ecbdf2c8811d099419ab1
bytes: 305574   lines: 5214
Source on the training box: /workspace/rpv16_sm120/agillm_gb10_1pf.rpv16_v43c_ckptsafe_v91p5_emuint8_20260928.py (mtime 2026-09-28 01:31 UK)
Host: Vast 53075549, NVIDIA GeForce RTX 3090 24 GB (France, on-demand, label rpv16-2b-3090-ondemand-final).
Published read-only by Qwen Improver's publisher. The live training process and save-dir on the box were not touched.

STATUS (checked 2026-09-28 about 03:03 UK)
- LIVE on Vast 53075549 with finite loss (635+ train events; recent loss about 6.85-7.10).
- Resumed from step 252804 (log resume / resume_pick_newest on RPV16-GB10-1PF-v6-current-resumable.pt).
  That file is the PRO 4000 (Vast 53045147) final cutover save; also present locally at
  /workspace/cutover_france_final_6b7e/step252804/. Against that final save, 0 steps were lost.
- First clean save by this lineage on the 3090: step 253177 (checkpoint_time_trigger saved:true;
  ckpt_save_status.json last_ok_step 253177). Current step was about 253439 when probed.

WHY THIS PUBLISH (full, not notes-only)
- SHA 9d007c6d does NOT match the portable v91p5 already under runtime/portable/ (fb09c47a...).
- Previous LIVE RPV trainer 9f28084c (v91p2_sm120_lazywq) only runs on Blackwell (sm_120 gate).
  After the Sep 28 3090 cutover it stopped on PRO 4000 Vast 53045147 (see README_trainer_9f28084c.txt
  cutover note). This file is the portable-capable v91p5 lineage with emulated NVFP4 and exact INT8
  block matmul (filename ...v91p5_emuint8...); it runs on 3090, not Blackwell-only.

LINEAGE / RESUME CHECKPOINT
PRO 4000 Vast 53045147 final save ~252804 was reportedly uploaded to a private HF commit f13e071c
(not confirmed on public MarxistLeninist/AGILLM-4.4 or AGILLM-4.3-checkpoints heads at publish time;
no public path matching step252804 / cutover / f13e071c was found under those repos).
Local cutover path on the 3090 box: /workspace/cutover_france_final_6b7e/step252804/.
This trainer resumed exactly from that 252804 resumable and continued on 53075549.

HF checkpoint home for AGILLM Standard/RPV work:
  MarxistLeninist/AGILLM-4.4 and mirror MarxistLeninist/AGILLM-4.3-checkpoints.

SHA256SUMS: see SHA256SUMS_9d007c6d.txt
