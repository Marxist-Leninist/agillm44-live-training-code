# AGILLM4.4 live training code

This standalone repository contains the exact, unmodified single-file Python trainer used by the live AGILLM4.4 production process at the verification time below.

- Verified live: 2026-09-06 18:40 UTC (vast.ai RTX 3090, instance 47086549)
- Live supervisor PID: `3798273`
- Live trainer PID: `3798343`
- Source file: `agillm43_singlefile_intelligence_v27_16_8_det50_packet_20260906.py`
- SHA-256: `2178bdeb5c6f18b89ba169a43ad9d5c0128e48c81d3111a7787ecc41462b9d8a`
- Size: 1,939,932 bytes

What changed since the previous snapshot (v27.12, 2026-09-04):

- v27.16 "deepctx": DiffusionBlocks one-sublayer local training now takes the exact top-of-stack gradient (`--dblock_clean_context e2e`: frozen stack below the selected block, frozen activation-checkpointed stack above), multi-row full-stack AR/SAT/NAT composition anchors (`--dblock_fullstack_ar_rows`, `--dblock_fullstack_anchor_cap`), `--dblock_local_rows`, depth-anchor receipts.
- v27.16.2/.3: NAT-anchor overlap guard lifted when the AR anchor runs every step; AR-protection projection reference on the clean-context path.
- v27.16.4: adaptive vocab chunk in the streaming fused cross-entropy (memory).
- v27.16.5: fused cross-entropy backward fix. The gradient matmuls ran under an fp16 autocast after pre-scaling (softmax - onehot) by 1/N; with bf16 AMP (no GradScaler) that underflowed most of the softmax tail at large target counts, biasing the hidden-state gradient (measured cosine 0.70-0.79 vs the exact gradient at N~32k). Now the unscaled softmax goes through bf16 matmuls and the 1/N scale is applied in fp32 afterwards (cosine 1.000).
- v27.16.6: multi-row full-stack SAT/NAT composition anchors. The NAT anchor had been one 128-token crop (64 masked targets) every third step at weight <= 0.10 against the AR anchor's 32,752 targets every step, and the full held-out NAT metrics regressed while AR improved; the NAT/SAT anchors can now batch several crops per call (hot ints `dblock_fullstack_nat_rows` / `dblock_fullstack_sat_rows`) and the SAT/NAT weight ceiling is hot-configurable (`dblock_fullstack_aux_weight_max`). Live hot config: NAT 16 crops every step at weight 1.0, SAT 16 crops, AR anchor weight 2.0.
- v27.16.7 (another agent, 16:22 UTC): detachable-50M teacher-packet overlay only — a masked NAT capture no longer erases a clean packet emitted earlier in the same mature commit (under v27.16.6 the NAT anchor, now running every step with masked ids, had discarded every packet, so the detachable-50M student trained on no steps at all). Mature-model training semantics unchanged.
- v27.16.8 (18:19 UTC, live): detachable-50M teacher-packet fix, overlay literals only. Every remaining student error (`AR mature teacher-logit contract is invalid`, 101/101) fell on the SAT-anchor steps: the overlay re-emitted a packet from the 16-crop SAT capture without slicing to the student's batch (anchor_ids [16,128], teacher_logits [16,64,V]) and that packet replaced the canonical 1-row AR packet; the consumer caps ids at `schedule["batch"]` (1) but not the teacher logits, so `size(0)` 16 != 1. Now (1) a SAT/NAT capture never replaces a current, clean, valid packet (AR-packet precedence), (2) SAT/NAT packets are sliced to the student rows before the logits are computed, (3) the consumer caps `_active_teacher_logits` to the same rows. Verified live: 0 student errors and AR-student commits on every step residue over the first 67 steps (previously error_skipped on every third step). The `AGILLM_SINGLEFILE_MANIFEST` literal is intentionally unchanged (pinned derivation proof); the folded overlay sources are the `_AGILLM_SF_SOURCE` string literals after the `# Folded module:` markers.

### How the live process is launched (exact command lines, environment, hot config, LR override)

Both command lines are the ones the supervisor actually launched (reconstructed token-for-token from the launch provenance of the live process, which records `train_argv`; the only differences between the two spans are the trainer file, the `--resume` checkpoint, `--data_seed`, and the one-shot optimizer reset). 197 flags each. The hot-config values (block below) override the `--dblock_fullstack_*` flags at runtime; the LR override file overrides the `--lr_*` schedule.

<details>
<summary><b>Steps 2785394 → 2786801 — <code>agillm43_singlefile_intelligence_v27_16_5_deepctx_20260906.py</code> (launched 2026-09-06 12:08 UTC, one-shot optimizer reset)</b></summary>

```bash
python -u /workspace/agillm43_production_v14/runtime_candidates/deepctx_v27_16_20260906/agillm43_singlefile_intelligence_v27_16_5_deepctx_20260906.py train \
  --preset agillm4_floor \
  --block 2048 \
  --source 'HuggingFaceFW/fineweb-edu:sample-10BT|1.60,HuggingFaceFW/fineweb:CC-MAIN-2024-10|0.50,wikimedia/wikipedia:20231101.en|1.10,allenai/c4:en|0.40,Skylion007/openwebtext|0.35,HuggingFaceTB/cosmopedia:web_samples_v2|1.30,HuggingFaceTB/smollm-corpus:cosmopedia-v2|1.30,EleutherAI/proof-pile-2:all|1.50,allenai/dolma:v1_6-sample|0.50,codeparrot/codeparrot-clean|1.25' \
  --nat_mask_token_id 2 \
  --tie_weights \
  --tie_kv \
  --moe_ffn \
  --moe_experts 2 \
  --moe_top_k 1 \
  --moe_mlp_mult 4 \
  --moe_shared_experts 1 \
  --moe_shared_mlp_mult 2 \
  --moe_aux_coef 0.01 \
  --moe_z_coef 0.001 \
  --amp \
  --grad_checkpoint \
  --attn_backend sdpa \
  --sublinear_window 128 \
  --sublinear_stride 128 \
  --sublinear_max_anchors 128 \
  --sublinear_chunk 128 \
  --sublinear_sinks 4 \
  --sublinear_recent_anchors 64 \
  --no-sublinear_pooled_landmarks \
  --anchor_stride 256 \
  --anchor_max 2048 \
  --anchor_position -1 \
  --alibi_mode corrected \
  --alibi_scale 0.0 \
  --optimizer adamw8bit \
  --weight_decay 0.0 \
  --lr_decay cosine \
  --lr_decay_tokens 49000954880 \
  --lr_min_mult 0.2 \
  --lr_schedule_reset_on_resume \
  --lr_warmup_tokens 10000000 \
  --lr_warmup_min_mult 0.1 \
  --dblock \
  --dblock_schedule roundrobin \
  --dblock_router_ramp_steps 64 \
  --dblock_warmup_steps 14 \
  --dblock_checkpoint_skip_tail 0 \
  --dblock_loop_layers 0 \
  --dblock_loop_start 0 \
  --dblock_loop_cond_scale 1.0 \
  --dblock_sigma_curriculum_steps 2000 \
  --dblock_sigma_sampling lognormal \
  --dblock_sigma_stratified \
  --dblock_sigma_min 0.002 \
  --dblock_sigma_max 2.0 \
  --dblock_sigma_pmean -1.2 \
  --dblock_sigma_pstd 1.2 \
  --dblock_edm_wmax 5.0 \
  --dblock_ar_prob 0.8 \
  --dblock_sat_prob 0.1 \
  --dblock_nat_prob 0.1 \
  --dblock_ar_loss_tokens 8192 \
  --dblock_sat_loss_tokens 4096 \
  --dblock_nat_loss_tokens 4096 \
  --dblock_nat_embed_noise_mode off \
  --dblock_nat_embed_noise_scale 1.0 \
  --nat_loss_weight 1.0 \
  --nat_mask_ratio 0.5 \
  --nat_max_tokens 0 \
  --nat_span_mask_prob 0.35 \
  --nat_suffix_mask_prob 0.20 \
  --nat_span_max_tokens 0 \
  --loss_spike_skip 0.0 \
  --sat_every 1 \
  --nat_every 1 \
  --dblock_fullstack_anchor_spike_skip 0.0 \
  --dblock_fullstack_anchor_max_ce 0.0 \
  --dblock_anchor_softcap_mult 4.0 \
  --dblock_anchor_softcap_min_ce 25.0 \
  --dblock_anchor_softcap_max_ce 50.0 \
  --dblock_fullstack_ar_every 3 \
  --dblock_fullstack_ar_offset 0 \
  --dblock_fullstack_ar_tokens 256 \
  --dblock_fullstack_sat_every 3 \
  --dblock_fullstack_sat_offset 1 \
  --dblock_fullstack_sat_tokens 128 \
  --dblock_fullstack_nat_every 3 \
  --dblock_fullstack_nat_offset 2 \
  --dblock_fullstack_nat_tokens 128 \
  --dblock_fullstack_nat_mask_id -1 \
  --ckpt_codec block-sharded-zstd \
  --async_update_every_steps 0 \
  --dblock_router_hidden 64 \
  --dblock_router_heads 4 \
  --dblock_router_layers 2 \
  --dblock_router_lr 0.0005 \
  --dblock_explore 0.08 \
  --dblock_max_stale_steps 64 \
  --dblock_max_count_skew 1.35 \
  --dblock_stale_bonus 0.35 \
  --dblock_undertrain_bonus 0.25 \
  --dblock_activation_offload_min_mb 1.0 \
  --nat_document_boundary_aware \
  --dblock_fullstack_anchor_deterministic_eval \
  --dblock_fullstack_anchor_spike_retry_limit 3 \
  --dblock_local_spike_retry_limit 3 \
  --dblock_fullstack_nat_no_valid_crop_limit 8 \
  --target_tokens 400000000000 \
  --lr_schedule_reanchor_on_resume \
  --save_every_sec 1800 \
  --delta_max_keep 0 \
  --max_ckpts 2 \
  --disk_free_floor_gb 12.0 \
  --heartbeat_every_sec 30 \
  --dblock_fullstack_ar_weight 1.0 \
  --dblock_fullstack_sat_weight 0.1 \
  --dblock_fullstack_nat_weight 0.1 \
  --dblock_local_aux_warn_ce 0.0 \
  --dblock_local_aux_max_ce 0.0 \
  --lr_schedule_origin_tokens 85002387456 \
  --dblock_stop_after_commits 0 \
  --no-oom_auto_backoff \
  --dblock_objective_mode committed_9_1_1 \
  --dblock_ar_weight 0.25 \
  --dblock_sat_weight 0.25 \
  --dblock_nat_weight 0.25 \
  --dblock_aux_grad_ratio 0.5 \
  --dblock_min_ar_share 0.8 \
  --lr_core 2e-7 \
  --lr_head 2e-7 \
  --oom_known_safe_batch 56 \
  --dblock_checkpoint_stride 2 \
  --empty_cache_every_steps 0 \
  --dblock_log_every 25 \
  --cuda_max_reserved_mib 0 \
  --cuda_min_free_mib 0 \
  --dblock_clean_context e2e \
  --dblock_local_rows 2 \
  --dblock_clean_context_noise_scale 0.0 \
  --dblock_depth_anchor_every 50 \
  --dblock_fullstack_anchor_cap 16376 \
  --dblock_fullstack_ar_rows 8 \
  --detachable_50m_upgrade_multimode \
  --detachable_50m_upgrade_logit_distill \
  --detachable_50m_upgrade_sat_gate_v2713 \
  --detachable_50m_create_from_legacy \
  --detachable_50m_seed_from_mature \
  --detachable_50m_lr 0.0001 \
  --detachable_50m_weight_decay 0.0 \
  --detachable_50m_update_mode sublayer \
  --detachable_50m_batch 1 \
  --detachable_50m_tokens 64 \
  --detachable_50m_loss_tokens 64 \
  --detachable_50m_every 1 \
  --detachable_50m_teacher_every 1 \
  --detachable_50m_distill_weight 1.0 \
  --detachable_50m_logit_distill_alpha 0.30 \
  --detachable_50m_logit_distill_temperature 1.0 \
  --detachable_50m_logit_distill_tokens 64 \
  --detachable_50m_hidden_distill_weight 0.0 \
  --detachable_50m_anchor_every 128 \
  --detachable_50m_anchor_tokens 32 \
  --detachable_50m_seed 20260830 \
  --detachable_50m_log_every 10 \
  --detachable_50m_telemetry_keep 128 \
  --weight_accel_mode shadow \
  --weight_accel_dir /workspace/agillm43_production_v14/production_quality_v18_hard_spike_gates/weight_trajectory_accel \
  --weight_accel_every 1 \
  --weight_accel_train_every 4 \
  --weight_accel_record_every 14 \
  --weight_accel_state_every 256 \
  --detachable_50m \
  --detachable_50m_optimizer paged_adamw8bit \
  --delta_every_steps 0 \
  --delta_every_sec 0 \
  --resume /workspace/agillm43_production_v14/production_quality_v18_hard_spike_gates/checkpoints_continue400B_from200B/pretrain_step02785394_deepctx_v27165rb3_1294da4151a4a9ba.pt \
  --save_dir /workspace/agillm43_production_v14/production_quality_v18_hard_spike_gates/checkpoints_continue400B_from200B \
  --data_seed 2785436 \
  --batch_size 60 \
  --dblock_blocks 14 \
  --dblock_direct56_enabled 1 \
  --dblock_layers_per_update 0 \
  --dblock_router shadow \
  --dblock_router_blend 0.0 \
  --dblock_router_full_anchor_every 128 \
  --dblock_router_train_layer_policy cyclic \
  --dblock_router_train_layers_per_update 1 \
  --dblock_sublayer_mode off \
  --dblock_target_route_activate_from_checkpoint 1 \
  --dblock_target_route_activation_checkpoint_step 2717509 \
  --dblock_target_route_activation_parent_pointer_sha256 04638cfbc19c0f9db6e6dfa03b2509d50a0bf3c56048a2cbd6e67154bc85c827 \
  --dblock_target_route_offset 1 \
  --dblock_target_route_policy xor_fold56_v1 \
  --dblock_train_layer_policy cyclic \
  --dblock_train_layers_per_update 1 \
  --dblock_train_sublayer_policy cyclic \
  --dblock_train_sublayers_per_update 1 \
  --granularity_label one-sublayer-local-grad-direct56-teacher-v27.16 \
  --val_every_sec 600 \
  --val_file /workspace/agillm43_heldout_v3/token_ids.json \
  --val_sha256 513b30203ffd18167655deb2d9cd610541df87d1ea0b8a0f9241c58e68703d21 \
  --reset_optimizer_on_resume
```

</details>

<details>
<summary><b>Steps 2786801 → 2787024 — <code>agillm43_singlefile_intelligence_v27_16_6_deepctx_20260906.py</code> (launched 2026-09-06 15:38 UTC, exact optimizer-state resume)</b></summary>

```bash
python -u /workspace/agillm43_production_v14/runtime_candidates/deepctx_v27_16_20260906/agillm43_singlefile_intelligence_v27_16_6_deepctx_20260906.py train \
  --preset agillm4_floor \
  --block 2048 \
  --source 'HuggingFaceFW/fineweb-edu:sample-10BT|1.60,HuggingFaceFW/fineweb:CC-MAIN-2024-10|0.50,wikimedia/wikipedia:20231101.en|1.10,allenai/c4:en|0.40,Skylion007/openwebtext|0.35,HuggingFaceTB/cosmopedia:web_samples_v2|1.30,HuggingFaceTB/smollm-corpus:cosmopedia-v2|1.30,EleutherAI/proof-pile-2:all|1.50,allenai/dolma:v1_6-sample|0.50,codeparrot/codeparrot-clean|1.25' \
  --nat_mask_token_id 2 \
  --tie_weights \
  --tie_kv \
  --moe_ffn \
  --moe_experts 2 \
  --moe_top_k 1 \
  --moe_mlp_mult 4 \
  --moe_shared_experts 1 \
  --moe_shared_mlp_mult 2 \
  --moe_aux_coef 0.01 \
  --moe_z_coef 0.001 \
  --amp \
  --grad_checkpoint \
  --attn_backend sdpa \
  --sublinear_window 128 \
  --sublinear_stride 128 \
  --sublinear_max_anchors 128 \
  --sublinear_chunk 128 \
  --sublinear_sinks 4 \
  --sublinear_recent_anchors 64 \
  --no-sublinear_pooled_landmarks \
  --anchor_stride 256 \
  --anchor_max 2048 \
  --anchor_position -1 \
  --alibi_mode corrected \
  --alibi_scale 0.0 \
  --optimizer adamw8bit \
  --weight_decay 0.0 \
  --lr_decay cosine \
  --lr_decay_tokens 49000954880 \
  --lr_min_mult 0.2 \
  --lr_schedule_reset_on_resume \
  --lr_warmup_tokens 10000000 \
  --lr_warmup_min_mult 0.1 \
  --dblock \
  --dblock_schedule roundrobin \
  --dblock_router_ramp_steps 64 \
  --dblock_warmup_steps 14 \
  --dblock_checkpoint_skip_tail 0 \
  --dblock_loop_layers 0 \
  --dblock_loop_start 0 \
  --dblock_loop_cond_scale 1.0 \
  --dblock_sigma_curriculum_steps 2000 \
  --dblock_sigma_sampling lognormal \
  --dblock_sigma_stratified \
  --dblock_sigma_min 0.002 \
  --dblock_sigma_max 2.0 \
  --dblock_sigma_pmean -1.2 \
  --dblock_sigma_pstd 1.2 \
  --dblock_edm_wmax 5.0 \
  --dblock_ar_prob 0.8 \
  --dblock_sat_prob 0.1 \
  --dblock_nat_prob 0.1 \
  --dblock_ar_loss_tokens 8192 \
  --dblock_sat_loss_tokens 4096 \
  --dblock_nat_loss_tokens 4096 \
  --dblock_nat_embed_noise_mode off \
  --dblock_nat_embed_noise_scale 1.0 \
  --nat_loss_weight 1.0 \
  --nat_mask_ratio 0.5 \
  --nat_max_tokens 0 \
  --nat_span_mask_prob 0.35 \
  --nat_suffix_mask_prob 0.20 \
  --nat_span_max_tokens 0 \
  --loss_spike_skip 0.0 \
  --sat_every 1 \
  --nat_every 1 \
  --dblock_fullstack_anchor_spike_skip 0.0 \
  --dblock_fullstack_anchor_max_ce 0.0 \
  --dblock_anchor_softcap_mult 4.0 \
  --dblock_anchor_softcap_min_ce 25.0 \
  --dblock_anchor_softcap_max_ce 50.0 \
  --dblock_fullstack_ar_every 3 \
  --dblock_fullstack_ar_offset 0 \
  --dblock_fullstack_ar_tokens 256 \
  --dblock_fullstack_sat_every 3 \
  --dblock_fullstack_sat_offset 1 \
  --dblock_fullstack_sat_tokens 128 \
  --dblock_fullstack_nat_every 3 \
  --dblock_fullstack_nat_offset 2 \
  --dblock_fullstack_nat_tokens 128 \
  --dblock_fullstack_nat_mask_id -1 \
  --ckpt_codec block-sharded-zstd \
  --async_update_every_steps 0 \
  --dblock_router_hidden 64 \
  --dblock_router_heads 4 \
  --dblock_router_layers 2 \
  --dblock_router_lr 0.0005 \
  --dblock_explore 0.08 \
  --dblock_max_stale_steps 64 \
  --dblock_max_count_skew 1.35 \
  --dblock_stale_bonus 0.35 \
  --dblock_undertrain_bonus 0.25 \
  --dblock_activation_offload_min_mb 1.0 \
  --nat_document_boundary_aware \
  --dblock_fullstack_anchor_deterministic_eval \
  --dblock_fullstack_anchor_spike_retry_limit 3 \
  --dblock_local_spike_retry_limit 3 \
  --dblock_fullstack_nat_no_valid_crop_limit 8 \
  --target_tokens 400000000000 \
  --lr_schedule_reanchor_on_resume \
  --save_every_sec 1800 \
  --delta_max_keep 0 \
  --max_ckpts 2 \
  --disk_free_floor_gb 12.0 \
  --heartbeat_every_sec 30 \
  --dblock_fullstack_ar_weight 1.0 \
  --dblock_fullstack_sat_weight 0.1 \
  --dblock_fullstack_nat_weight 0.1 \
  --dblock_local_aux_warn_ce 0.0 \
  --dblock_local_aux_max_ce 0.0 \
  --lr_schedule_origin_tokens 85002387456 \
  --dblock_stop_after_commits 0 \
  --no-oom_auto_backoff \
  --require_exact_resume_state \
  --dblock_objective_mode committed_9_1_1 \
  --dblock_ar_weight 0.25 \
  --dblock_sat_weight 0.25 \
  --dblock_nat_weight 0.25 \
  --dblock_aux_grad_ratio 0.5 \
  --dblock_min_ar_share 0.8 \
  --lr_core 2e-7 \
  --lr_head 2e-7 \
  --oom_known_safe_batch 56 \
  --dblock_checkpoint_stride 2 \
  --empty_cache_every_steps 0 \
  --dblock_log_every 25 \
  --cuda_max_reserved_mib 0 \
  --cuda_min_free_mib 0 \
  --dblock_clean_context e2e \
  --dblock_local_rows 2 \
  --dblock_clean_context_noise_scale 0.0 \
  --dblock_depth_anchor_every 50 \
  --dblock_fullstack_anchor_cap 16376 \
  --dblock_fullstack_ar_rows 8 \
  --detachable_50m_upgrade_multimode \
  --detachable_50m_upgrade_logit_distill \
  --detachable_50m_upgrade_sat_gate_v2713 \
  --detachable_50m_create_from_legacy \
  --detachable_50m_seed_from_mature \
  --detachable_50m_lr 0.0001 \
  --detachable_50m_weight_decay 0.0 \
  --detachable_50m_update_mode sublayer \
  --detachable_50m_batch 1 \
  --detachable_50m_tokens 64 \
  --detachable_50m_loss_tokens 64 \
  --detachable_50m_every 1 \
  --detachable_50m_teacher_every 1 \
  --detachable_50m_distill_weight 1.0 \
  --detachable_50m_logit_distill_alpha 0.30 \
  --detachable_50m_logit_distill_temperature 1.0 \
  --detachable_50m_logit_distill_tokens 64 \
  --detachable_50m_hidden_distill_weight 0.0 \
  --detachable_50m_anchor_every 128 \
  --detachable_50m_anchor_tokens 32 \
  --detachable_50m_seed 20260830 \
  --detachable_50m_log_every 10 \
  --detachable_50m_telemetry_keep 128 \
  --weight_accel_mode shadow \
  --weight_accel_dir /workspace/agillm43_production_v14/production_quality_v18_hard_spike_gates/weight_trajectory_accel \
  --weight_accel_every 1 \
  --weight_accel_train_every 4 \
  --weight_accel_record_every 14 \
  --weight_accel_state_every 256 \
  --detachable_50m \
  --detachable_50m_optimizer paged_adamw8bit \
  --delta_every_steps 0 \
  --delta_every_sec 0 \
  --resume /workspace/agillm43_production_v14/production_quality_v18_hard_spike_gates/checkpoints_continue400B_from200B/pretrain_step02786801_deepctx_v27166_20335946f23a02d9.pt \
  --save_dir /workspace/agillm43_production_v14/production_quality_v18_hard_spike_gates/checkpoints_continue400B_from200B \
  --data_seed 2786843 \
  --batch_size 60 \
  --dblock_blocks 14 \
  --dblock_direct56_enabled 1 \
  --dblock_layers_per_update 0 \
  --dblock_router shadow \
  --dblock_router_blend 0.0 \
  --dblock_router_full_anchor_every 128 \
  --dblock_router_train_layer_policy cyclic \
  --dblock_router_train_layers_per_update 1 \
  --dblock_sublayer_mode off \
  --dblock_target_route_activate_from_checkpoint 1 \
  --dblock_target_route_activation_checkpoint_step 2717509 \
  --dblock_target_route_activation_parent_pointer_sha256 04638cfbc19c0f9db6e6dfa03b2509d50a0bf3c56048a2cbd6e67154bc85c827 \
  --dblock_target_route_offset 1 \
  --dblock_target_route_policy xor_fold56_v1 \
  --dblock_train_layer_policy cyclic \
  --dblock_train_layers_per_update 1 \
  --dblock_train_sublayer_policy cyclic \
  --dblock_train_sublayers_per_update 1 \
  --granularity_label one-sublayer-local-grad-direct56-teacher-v27.16 \
  --val_every_sec 600 \
  --val_file /workspace/agillm43_heldout_v3/token_ids.json \
  --val_sha256 513b30203ffd18167655deb2d9cd610541df87d1ea0b8a0f9241c58e68703d21
```

</details>

<details>
<summary><b>Environment of the trainer process (identical for both spans; <code>AGILLM43_ONESHOT_RESET_OPT_CKPT</code> was additionally set to the step-2785394 checkpoint path for the first launch only)</b></summary>

```bash
AGILLM43_DBLOCK_CC_NOISE_SCALE=0.0
AGILLM43_DBLOCK_CLEAN_CONTEXT=e2e
AGILLM43_DBLOCK_DEPTH_ANCHOR_EVERY=50
AGILLM43_DBLOCK_FULLSTACK_ANCHOR_CAP=16376
AGILLM43_DBLOCK_FULLSTACK_AR_ROWS=8
AGILLM43_DBLOCK_LOCAL_ROWS=2
AGILLM43_DBLOCK_TARGET_ROUTE_ACTIVATE_FROM_CHECKPOINT=1
AGILLM43_DBLOCK_TARGET_ROUTE_ACTIVATION_CHECKPOINT_STEP=2717509
AGILLM43_DBLOCK_TARGET_ROUTE_ACTIVATION_PARENT_POINTER_SHA256=04638cfbc19c0f9db6e6dfa03b2509d50a0bf3c56048a2cbd6e67154bc85c827
AGILLM43_DBLOCK_TARGET_ROUTE_OFFSET=1
AGILLM43_DBLOCK_TARGET_ROUTE_POLICY=xor_fold56_v1
AGILLM43_DECOMPRESS_CACHE=0
AGILLM43_DETACHABLE50M_ANCHOR_EVERY=128
AGILLM43_DETACHABLE50M_ANCHOR_TOKENS=32
AGILLM43_DETACHABLE50M_BATCH=1
AGILLM43_DETACHABLE50M_CREATE_FROM_LEGACY=1
AGILLM43_DETACHABLE50M_DISTILL_WEIGHT=1.0
AGILLM43_DETACHABLE50M_ENABLED=1
AGILLM43_DETACHABLE50M_EVERY=1
AGILLM43_DETACHABLE50M_GATE_V2713_START_CHECKPOINT_SHA256=39f442fc0e6f9fad6f875e067f63db67f33a41a53e8cbc49ce1591010bb4aec3
AGILLM43_DETACHABLE50M_HIDDEN_DISTILL_WEIGHT=0.0
AGILLM43_DETACHABLE50M_LOGIT_DISTILL_ALPHA=0.30
AGILLM43_DETACHABLE50M_LOGIT_DISTILL_TEMPERATURE=1.0
AGILLM43_DETACHABLE50M_LOGIT_DISTILL_TOKENS=64
AGILLM43_DETACHABLE50M_LOG_EVERY=10
AGILLM43_DETACHABLE50M_LOSS_TOKENS=64
AGILLM43_DETACHABLE50M_LR=0.0001
AGILLM43_DETACHABLE50M_OPTIMIZER=paged_adamw8bit
AGILLM43_DETACHABLE50M_SEED=20260830
AGILLM43_DETACHABLE50M_SEED_FROM_MATURE=1
AGILLM43_DETACHABLE50M_TEACHER_EVERY=1
AGILLM43_DETACHABLE50M_TELEMETRY_KEEP=128
AGILLM43_DETACHABLE50M_TOKENS=64
AGILLM43_DETACHABLE50M_UPDATE_MODE=sublayer
AGILLM43_DETACHABLE50M_UPGRADE_LOGIT_DISTILL=1
AGILLM43_DETACHABLE50M_UPGRADE_MULTIMODE=1
AGILLM43_DETACHABLE50M_UPGRADE_SAT_GATE_V2713=1
AGILLM43_DETACHABLE50M_WEIGHT_DECAY=0.0
AGILLM43_SEQUENCE_PREFETCH_DEPTH=256
AGILLM43_SINGLEFILE_PROFILE=intelligence-v25-satvar-pretrain
AGILLM43_WEIGHT_ACCEL_MODE=shadow
AGILLM_DATASETS_STREAMING_READ_MAX_RETRIES=2
AGILLM_DATASETS_STREAMING_READ_RETRY_INTERVAL=1
AGILLM_DATASET_AGENT_ROUTER=0
AGILLM_DATASET_HOTLOAD_CONFIG=/workspace/agillm43_dataset_hotload.json
AGILLM_DATASET_HOT_RELOAD_CHECK_TOKENS=2048
AGILLM_DATASET_HOT_RELOAD_SEC=5
AGILLM_DATASET_NN_ROUTER=1
AGILLM_DATASET_ROUTER_BLEND=0.25
AGILLM_DATASET_ROUTER_EXPLORE=0.03
AGILLM_DATASET_ROUTER_LR=0.005
AGILLM_DATASET_ROUTER_MIN_SCORE=0.15
AGILLM_DATASET_ROUTER_SHARPNESS=2.0
AGILLM_ENABLE_CUPY=0
AGILLM_HF_HUB_DOWNLOAD_TIMEOUT=30
AGILLM_HOT_CONFIG=/workspace/agillm43_production_v14/contracts/hot_config_owner_v25.json
AGILLM_HOT_CONFIG_SHA256=''
AGILLM_MAX_EXAMPLE_CHARS=32768
AGILLM_MAX_EXAMPLE_TOKENS=4096
AGILLM_REMOTE_SHARDS_PER_ITER=4
AGILLM_STREAM_NEXT_TIMEOUT_SEC=60
AGILLM_STREAM_OPEN_TIMEOUT_SEC=120
AGILLM_STREAM_SOCKET_TIMEOUT_SEC=30
AGILLM_STREAM_SOURCE_FATAL_COOLDOWN_SEC=600
AGILLM_STREAM_SOURCE_MAX_COOLDOWN_SEC=120
AGILLM_SYNTHETIC_TOKENIZER=0
CUDA_MODULE_LOADING=LAZY
CUDA_VERSION=12.1.1
CUDA_VISIBLE_DEVICES=0
HF_HUB_DOWNLOAD_TIMEOUT=30
HF_HUB_ETAG_TIMEOUT=10
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
PYTORCH_VERSION=2.2.1
```

The supervisor also carried these persistent `set_arg` overrides (already applied in the command lines above) via `AGILLM43_SUPERVISE_SET_ARGS_JSON`:

```json
{
  "--batch_size": "60",
  "--dblock_blocks": "14",
  "--dblock_direct56_enabled": "1",
  "--dblock_layers_per_update": "0",
  "--dblock_router": "shadow",
  "--dblock_router_blend": "0.0",
  "--dblock_router_full_anchor_every": "128",
  "--dblock_router_train_layer_policy": "cyclic",
  "--dblock_router_train_layers_per_update": "1",
  "--dblock_sublayer_mode": "off",
  "--dblock_target_route_activate_from_checkpoint": "1",
  "--dblock_target_route_activation_checkpoint_step": "2717509",
  "--dblock_target_route_activation_parent_pointer_sha256": "04638cfbc19c0f9db6e6dfa03b2509d50a0bf3c56048a2cbd6e67154bc85c827",
  "--dblock_target_route_offset": "1",
  "--dblock_target_route_policy": "xor_fold56_v1",
  "--dblock_train_layer_policy": "cyclic",
  "--dblock_train_layers_per_update": "1",
  "--dblock_train_sublayer_policy": "cyclic",
  "--dblock_train_sublayers_per_update": "1",
  "--granularity_label": "one-sublayer-local-grad-direct56-teacher-v27.16",
  "--val_every_sec": "600",
  "--val_file": "/workspace/agillm43_heldout_v3/token_ids.json",
  "--val_sha256": "513b30203ffd18167655deb2d9cd610541df87d1ea0b8a0f9241c58e68703d21"
}
```

</details>

<details>
<summary><b>Hot config <code>contracts/hot_config_owner_v25.json</code> — rev 11 (in force 2785394 → 2786801) and rev 12 (2786801 → 2787024), verbatim values</b></summary>

rev 11 (updated 2026-09-06T09:52:23Z):

```json
{
  "dblock_fullstack_anchor_cap": 32752,
  "dblock_fullstack_ar_every": 1,
  "dblock_fullstack_ar_offset": 0,
  "dblock_fullstack_ar_rows": 16,
  "dblock_fullstack_ar_tokens": 32752,
  "dblock_fullstack_ar_weight": 3.0,
  "dblock_fullstack_nat_weight": 0.1,
  "dblock_fullstack_sat_weight": 0.1,
  "dblock_local_rows": 2,
  "revision": 11,
  "save_every_sec": 3600
}
```

rev 12 (updated 2026-09-06T15:34:47Z; change receipt: rev11->rev12 (15:36Z, effective when runtime v27.16.6 sha f7abca9a is live): NAT composition anchor 1 crop -> 16 crops x 128 tokens (1024 masked targets) on EVERY step (was every 3rd), weight 0.10 -> 1.0 (aux ceiling 0.10 -> 1.0); SAT composition anchor 1 crop -> 16 crops (2016 shift-2 targets) every 3rd step (weight stays the contract's 0.10; SAT-variable gate supervision unchanged, first crop); AR anchor weight 3.0 -> 2.0. v27.16.5 ignores the new keys and clamps NAT weight to 0.10 until the hop lands.):

```json
{
  "dblock_fullstack_anchor_cap": 32752,
  "dblock_fullstack_ar_every": 1,
  "dblock_fullstack_ar_offset": 0,
  "dblock_fullstack_ar_rows": 16,
  "dblock_fullstack_ar_tokens": 32752,
  "dblock_fullstack_ar_weight": 2.0,
  "dblock_fullstack_aux_weight_max": 1.0,
  "dblock_fullstack_nat_every": 1,
  "dblock_fullstack_nat_rows": 16,
  "dblock_fullstack_nat_weight": 1.0,
  "dblock_fullstack_sat_rows": 16,
  "dblock_fullstack_sat_weight": 0.1,
  "dblock_local_rows": 2,
  "revision": 12,
  "save_every_sec": 3600
}
```

</details>

<details>
<summary><b>LR override file <code>/workspace/agillm43_lr_override.json</code> — semantics and the governor's writes</b></summary>

Effective LR = `max(schedule, lr × clamp((seen_tok − since_seen_tok) / ramp_tokens, 0, 1))`, re-read by the trainer every 10 s. The governor (`heldout_monitor_v2.py`: FLUSH_NOW → CPU fp32 held-out probe every 20 min) raises `lr` ×1.5 only after three consecutive improving probes, halves it when a probe exceeds the best by 0.03, quarters it above best + 0.30, cap 2e-5, floor 1e-6.

| when (UTC) | write |
|---|---|
| 12:08:08Z launch @2785394 | 5e-6, ramp_tokens 20,000,000 from since_seen_tok 275,980,769,280 |
| 12:48:54Z escalate @2785598 | 7.5e-6 (since_seen_tok 275,999,170,133, ramp 10,000,000 → continuous from 5e-6) |
| 13:33:57Z escalate @2785924 | 1.125e-5 (since_seen_tok 276,041,440,853, ramp 10,000,000) |
| 14:34:02Z escalate @2786373 | 1.6875e-5 (since_seen_tok 276,096,613,973, ramp 10,000,000) |
| 15:34:47Z escalate @2786788 | 2e-5 = governor cap (since_seen_tok 276,145,223,940, ramp 10,000,000) |
| 17:01:06Z backoff @2787120 (after this checkpoint) | 1e-5 |

Fields of the file (current contents, governor-managed): `lr`, `ramp_tokens`, `since_seen_tok`, `phase` = `short_ramp_then_hold`, `stage` = `owner_lr_fix_v2_after_rollback`, `owner_directed` = true, plus the governor block (`cap` 2e-05, `floor` 1e-06).

</details>

The live v27.16.8 process was launched 2026-09-06 18:19 UTC with the identical environment and flags, the trainer file swapped to `agillm43_singlefile_intelligence_v27_16_8_det50_packet_20260906.py`, `--resume` pointing at the derived package `pretrain_step02787566_deepctx_v27168_c0e8ba7500480fe2.pt` (step 2787566, exact optimizer-state resume, `--data_seed 2787608`).

The checkpoint trained by these two launches is published as `checkpoints/step2787024_20260906/` in [MarxistLeninist/AGILLM-4.3-checkpoints](https://huggingface.co/MarxistLeninist/AGILLM-4.3-checkpoints), whose README carries the held-out results and a plain-language summary of every flag group.

The historical `agillm43` filename is retained deliberately so the published file remains byte-for-byte identical to the running source.

This is a point-in-time source snapshot. It contains code only: no model weights, checkpoints, datasets, API credentials, or private keys. The trainer includes production-specific default paths and optional external-provider integrations; configure those for your own environment before running it.

No licence has been added because the source snapshot itself does not declare one.
