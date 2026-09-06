# AGILLM4.4 live training code

This standalone repository contains the exact, unmodified single-file Python trainer used by the live AGILLM4.4 production process at the verification time below.

- Verified live: 2026-09-06 15:00 UTC (vast.ai RTX 3090, instance 47086549)
- Live supervisor PID: `3450805`
- Live trainer PID: `3450881`
- Source file: `agillm43_singlefile_intelligence_v27_16_5_deepctx_20260906.py`
- SHA-256: `1d972134d55333a99c305f5642602aa81c729e44c019c1fcadf1f83b64660641`
- Size: 1,934,229 bytes

What changed since the previous snapshot (v27.12, 2026-09-04):

- v27.16 "deepctx": DiffusionBlocks one-sublayer local training now takes the exact top-of-stack gradient (`--dblock_clean_context e2e`: frozen stack below the selected block, frozen activation-checkpointed stack above), multi-row full-stack AR/SAT/NAT composition anchors (`--dblock_fullstack_ar_rows`, `--dblock_fullstack_anchor_cap`), `--dblock_local_rows`, depth-anchor receipts.
- v27.16.2/.3: NAT-anchor overlap guard lifted when the AR anchor runs every step; AR-protection projection reference on the clean-context path.
- v27.16.4: adaptive vocab chunk in the streaming fused cross-entropy (memory).
- v27.16.5: fused cross-entropy backward fix. The gradient matmuls ran under an fp16 autocast after pre-scaling (softmax - onehot) by 1/N; with bf16 AMP (no GradScaler) that underflowed most of the softmax tail at large target counts, biasing the hidden-state gradient (measured cosine 0.70-0.79 vs the exact gradient at N~32k). Now the unscaled softmax goes through bf16 matmuls and the 1/N scale is applied in fp32 afterwards (cosine 1.000).

The historical `agillm43` filename is retained deliberately so the published file remains byte-for-byte identical to the running source.

This is a point-in-time source snapshot. It contains code only: no model weights, checkpoints, datasets, API credentials, or private keys. The trainer includes production-specific default paths and optional external-provider integrations; configure those for your own environment before running it.

No licence has been added because the source snapshot itself does not declare one.
