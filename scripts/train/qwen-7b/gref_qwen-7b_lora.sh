#!/usr/bin/env bash
set -euo pipefail

usage(){
  echo "Usage: ROOT_DIR=/abs/path RAY_TMPDIR=/abs/path PY=/abs/python DATA_TRAIN_FILES=/abs/data MODEL_PATH=/abs/model INIT_SAE_CKPT=/abs/sae.pt OUTPUT_DIR=/abs/out $0"
  exit 1
}
require_abs(){
  local name="$1"; local value="${!name:-}"
  [ -n "$value" ] || { echo "Error: $name is required." >&2; usage; }
  case "$value" in /*) ;; *) echo "Error: $name must be absolute: $value" >&2; exit 1;; esac
}
for name in ROOT_DIR RAY_TMPDIR PY DATA_TRAIN_FILES MODEL_PATH INIT_SAE_CKPT OUTPUT_DIR; do
  require_abs "$name"
done
[ -d "$ROOT_DIR" ] || { echo "Error: ROOT_DIR does not exist: $ROOT_DIR" >&2; exit 1; }
[ -x "$PY" ] || { echo "Error: PY is not executable: $PY" >&2; exit 1; }
[ -e "$DATA_TRAIN_FILES" ] || { echo "Error: DATA_TRAIN_FILES does not exist: $DATA_TRAIN_FILES" >&2; exit 1; }
[ -e "$MODEL_PATH" ] || { echo "Error: MODEL_PATH does not exist: $MODEL_PATH" >&2; exit 1; }
[ -f "$INIT_SAE_CKPT" ] || { echo "Error: INIT_SAE_CKPT does not exist: $INIT_SAE_CKPT" >&2; exit 1; }
if [ -n "${LOAD_CHECKPOINT_PATH:-}" ]; then
  require_abs LOAD_CHECKPOINT_PATH
  [ -d "$LOAD_CHECKPOINT_PATH" ] || {
    echo "Error: LOAD_CHECKPOINT_PATH does not exist: $LOAD_CHECKPOINT_PATH" >&2
    exit 1
  }
fi

cd "$ROOT_DIR"
mkdir -p "$RAY_TMPDIR" "$OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RAY_TMPDIR
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${ROOT_DIR}/wandb}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export SEGCOMPASS_REF_POS_TOKEN="${SEGCOMPASS_REF_POS_TOKEN:-<|object_ref_start|>}"

SUPP_BF16="${SUPP_BF16:-false}"
if [ "$SUPP_BF16" = "true" ]; then
  export VLLM_ATTENTION_BACKEND=FLASH_ATTN
  DTYPE=bfloat16
  CHUNKED_PREFILL=true
else
  export VLLM_ATTENTION_BACKEND=SDPA
  export VLLM_USE_TRITON=0
  export XFORMERS_FORCE_DISABLE_TRITON=1
  DTYPE=float16
  CHUNKED_PREFILL=false
fi

RUN_FLAG="${RUN_FLAG:-lora_$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-qwen_vad_lora}"
ARGS=(
  data.train_files="$DATA_TRAIN_FILES"
  data.video_embed_dir=""
  data.prompt_key=text
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
  data.max_response_length="${MAX_RESPONSE_LENGTH:-256}"
  data.shuffle="${DATA_SHUFFLE:-true}"

  worker.hybrid_engine=true
  worker.llm_version="${LLM_VERSION:-qwen-2.5}"
  worker.supp_bf16="${SUPP_BF16}"
  worker.actor.model.model_path="$MODEL_PATH"
  worker.actor.model.init_sae_ckpt="$INIT_SAE_CKPT"
  worker.actor.model.k_slots="${K_SLOTS:-1}"
  worker.actor.model.use_lora=true
  worker.actor.model.lora_rank="${LORA_R:-16}"
  worker.actor.model.lora_alpha="${LORA_ALPHA:-32}"
  worker.actor.model.lora_dropout="${LORA_DROPOUT:-0.05}"
  worker.actor.model.lora_target_modules="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

  worker.sae.d_in="${D_IN:-3584}"
  worker.sae.d_sae="${D_SAE:-65536}"
  worker.sae.hook_layer="${HOOK_LAYER:-13}"

  worker.rollout.n="${ROLLOUT_N:-2}"
  worker.rollout.tensor_parallel_size="${TP_SIZE:-1}"
  worker.rollout.max_num_seqs="${MAX_NUM_SEQS:-8}"
  worker.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.45}"
  worker.rollout.dtype="$DTYPE"
  worker.rollout.enable_chunked_prefill="$CHUNKED_PREFILL"
  worker.rollout.limit_images="${LIMIT_IMAGES:-0}"
  worker.rollout.limit_videos="${LIMIT_VIDEOS:-1}"

  data.rollout_batch_size="${ROLLOUT_BATCH_SIZE:-2}"
  worker.actor.global_batch_size="${GLOBAL_BATCH_SIZE:-2}"
  worker.actor.micro_batch_size_per_device_for_update="${MICRO_UPDATE:-1}"
  worker.actor.micro_batch_size_per_device_for_experience="${MICRO_EXPERIENCE:-1}"
  worker.actor.optim.base_lr="${BASE_LR:-1.6e-6}"
  worker.actor.seg_loss_coef="${SEG_LOSS_COEF:-0.3}"
  worker.actor.conf_loss_coef="${CONF_LOSS_COEF:-0.2}"
  worker.actor.kl_loss_coef="${KL_LOSS_COEF:-0.2}"
  worker.actor.adjust_loss_step="${ADJUST_LOSS_STEP:-1500}"
  worker.actor.entropy_coef="${ENTROPY_COEF:-0.0}"
  worker.actor.dice_loss_coef_new="${DICE_LOSS_COEF_NEW:-2.0}"
  worker.actor.focal_loss_coef_new="${FOCAL_LOSS_COEF_NEW:-5.0}"

  trainer.n_gpus_per_node="${N_GPUS:-1}"
  trainer.total_episodes="${TOTAL_EPISODES:-1}"
  trainer.save_freq="${SAVE_FREQ:--1}"
  trainer.val_before_train=false
  trainer.logger='["console"]'
  trainer.save_checkpoint_path="$OUTPUT_DIR"
  trainer.load_checkpoint_path="${LOAD_CHECKPOINT_PATH:-null}"
  config="scripts/initial.yaml"
)

export PATH="$(dirname "$PY"):${PATH}"
LOG_PATH="${OUTPUT_DIR}/print_log_${RUN_NAME}_${RUN_FLAG}.txt"

echo "Root: $ROOT_DIR"
echo "Python: $PY"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Trainer GPUs: ${N_GPUS:-1}"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_TRAIN_FILES"
echo "Output: $OUTPUT_DIR"
echo "LoRA: r=${LORA_R:-16}, alpha=${LORA_ALPHA:-32}, dropout=${LORA_DROPOUT:-0.05}"
echo "Hybrid actor-rollout: true"
echo "Log: $LOG_PATH"

PYTHONUNBUFFERED=1 "$PY" -u -m verl.trainer.main "${ARGS[@]}" 2>&1 | tee -a "$LOG_PATH"
