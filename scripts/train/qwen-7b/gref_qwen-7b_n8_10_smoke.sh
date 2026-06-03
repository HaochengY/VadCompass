#!/usr/bin/env bash
set -euo pipefail

cd /root/fuxian/SegCompass

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=true
export NCCL_IB_DISABLE=1
export PYTHONPATH="/root/fuxian/SegCompass:${PYTHONPATH:-}"
export SEGCOMPASS_REF_POS_TOKEN="<|object_ref_start|>"

PY="/root/autodl-tmp/verl_env/bin/python"
RUN_FLAG="smoke10"
RUN_NAME="gref_qwen-7b_n8_10_smoke"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/get_hf/Qwen2.5-VL-3B-Instruct}"

ARGS=(
  data.train_files="data/sht_video_10"
  data.video_embed_dir=""
  data.prompt_key=text
  data.rollout_batch_size=10
  data.max_prompt_length=1400
  data.max_response_length=256
  data.shuffle=false
  worker.llm_version="qwen-2.5"
  worker.actor.model.model_path="${MODEL_PATH}"
  worker.actor.model.init_sae_ckpt=null
  worker.actor.model.k_slots=1
  worker.sae.d_in=2048
  worker.sae.hook_layer=13
  worker.supp_bf16=true
  worker.rollout.n=1
  worker.rollout.tensor_parallel_size=1
  worker.rollout.max_num_seqs=16
  worker.rollout.gpu_memory_utilization=0.45
  worker.rollout.dtype=bfloat16
  worker.rollout.enable_chunked_prefill=true
  worker.actor.global_batch_size=10
  worker.actor.micro_batch_size_per_device_for_update=1
  worker.actor.micro_batch_size_per_device_for_experience=1
  worker.actor.optim.base_lr=1.0e-6
  worker.actor.seg_loss_coef=0.3
  worker.actor.conf_loss_coef=0.2
  worker.actor.kl_loss_coef=0.2
  worker.actor.entropy_coef=0.0
  trainer.n_gpus_per_node=1
  trainer.total_episodes=1
  trainer.save_freq=-1
  trainer.val_before_train=false
  trainer.logger='["console"]'
  trainer.save_checkpoint_path="checkpoints/${RUN_NAME}/${RUN_FLAG}"
  trainer.load_checkpoint_path=null
  config="scripts/initial.yaml"
)

PYTHONUNBUFFERED=1 "$PY" -u -m verl.trainer.main "${ARGS[@]}" 2>&1 | tee -a "print_log_smoke10.txt"
