# AGILLM4.4 live training code

This standalone repository contains the exact, unmodified single-file Python trainer used by the live AGILLM4.4 production process at the verification time below.

- Verified live: 2026-09-06 15:51 UTC (vast.ai RTX 3090, instance 47086549)
- Live supervisor PID: `3653017`
- Live trainer PID: `3653107`
- Source file: `agillm43_singlefile_intelligence_v27_16_6_deepctx_20260906.py`
- SHA-256: `f7abca9aaef97e7690765f9c7f00e04b65151f700bbb959b5cf053126d21ba0f`
- Size: 1,937,250 bytes

What changed since the previous snapshot (v27.12, 2026-09-04):

- v27.16 "deepctx": DiffusionBlocks one-sublayer local training now takes the exact top-of-stack gradient (`--dblock_clean_context e2e`: frozen stack below the selected block, frozen activation-checkpointed stack above), multi-row full-stack AR/SAT/NAT composition anchors (`--dblock_fullstack_ar_rows`, `--dblock_fullstack_anchor_cap`), `--dblock_local_rows`, depth-anchor receipts.
- v27.16.2/.3: NAT-anchor overlap guard lifted when the AR anchor runs every step; AR-protection projection reference on the clean-context path.
- v27.16.4: adaptive vocab chunk in the streaming fused cross-entropy (memory).
- v27.16.5: fused cross-entropy backward fix. The gradient matmuls ran under an fp16 autocast after pre-scaling (softmax - onehot) by 1/N; with bf16 AMP (no GradScaler) that underflowed most of the softmax tail at large target counts, biasing the hidden-state gradient (measured cosine 0.70-0.79 vs the exact gradient at N~32k). Now the unscaled softmax goes through bf16 matmuls and the 1/N scale is applied in fp32 afterwards (cosine 1.000).
- v27.16.6: multi-row full-stack SAT/NAT composition anchors. The NAT anchor had been one 128-token crop (64 masked targets) every third step at weight <= 0.10 against the AR anchor's 32,752 targets every step, and the full held-out NAT metrics regressed while AR improved; the NAT/SAT anchors can now batch several crops per call (hot ints `dblock_fullstack_nat_rows` / `dblock_fullstack_sat_rows`) and the SAT/NAT weight ceiling is hot-configurable (`dblock_fullstack_aux_weight_max`). Live hot config: NAT 16 crops every step at weight 1.0, SAT 16 crops, AR anchor weight 2.0.

The historical `agillm43` filename is retained deliberately so the published file remains byte-for-byte identical to the running source.

This is a point-in-time source snapshot. It contains code only: no model weights, checkpoints, datasets, API credentials, or private keys. The trainer includes production-specific default paths and optional external-provider integrations; configure those for your own environment before running it.

No licence has been added because the source snapshot itself does not declare one.
