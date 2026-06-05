#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/fuxian/SegCompass"
RUN_FLAG="default"
RUN_NAME="qwen_vad_a100_n8"
PY="/root/autodl-tmp/verl_env/bin/python"

DATA_PATH="data/sht_video_10"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
INIT_SAE_CKPT="${INIT_SAE_CKPT:-sae_checkpoints/sae_qwen-7b_L13/default/ep_6.pt}"

BF16=true
N_GPUS=8
K_SLOTS=1
TOTAL_EPISODES=4
SAVE_FREQ=400
ROLLOUT_N=8
ROLLOUT_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=16
MICRO_UPDATE=2
MICRO_EXPERIENCE=8
TP_SIZE=4
MAX_NUM_SEQS=64
GPU_MEMORY_UTILIZATION=0.6
BASE_LR=1.6e-6

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/train/qwen-7b/qwen_vad_a100_n8.sh [options]

Options:
  --root DIR                 Project root. Default: /root/fuxian/SegCompass
  --python PATH              Python executable. Default: /root/autodl-tmp/verl_env/bin/python
  --data PATH                HF dataset path. Default: data/sht_video_10
  --model PATH               Qwen2.5-VL model path/name.
  --sae PATH|null            SAE checkpoint path, or null to disable loading.
  --run-flag NAME            Checkpoint/log suffix. Default: default
  --bf16 true|false          Use bf16/FLASH_ATTN on A100. Default: true
  --n-gpus N                 Number of visible GPUs used by trainer. Default: 8
  --k-slots N                Number of VAD slots. Default: 1
  --total-episodes N         Default: 4
  --save-freq N              Default: 400
  --rollout-n N              Default: 8
  --rollout-batch-size N     Default: 16
  --global-batch-size N      Default: 16
  --micro-update N           Default: 2
  --micro-experience N       Default: 8
  --tp-size N                vLLM tensor parallel size. Default: 4
  --max-num-seqs N           Default: 64
  --gpu-mem-util FLOAT       Default: 0.6
  --base-lr FLOAT            Default: 1.6e-6
  -h, --help                 Show this help.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT_DIR="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    --data) DATA_PATH="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --sae) INIT_SAE_CKPT="$2"; shift 2 ;;
    --run-flag) RUN_FLAG="$2"; shift 2 ;;
    --bf16) BF16="$2"; shift 2 ;;
    --n-gpus) N_GPUS="$2"; shift 2 ;;
    --k-slots) K_SLOTS="$2"; shift 2 ;;
    --total-episodes) TOTAL_EPISODES="$2"; shift 2 ;;
    --save-freq) SAVE_FREQ="$2"; shift 2 ;;
    --rollout-n) ROLLOUT_N="$2"; shift 2 ;;
    --rollout-batch-size) ROLLOUT_BATCH_SIZE="$2"; shift 2 ;;
    --global-batch-size) GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
    --micro-update) MICRO_UPDATE="$2"; shift 2 ;;
    --micro-experience) MICRO_EXPERIENCE="$2"; shift 2 ;;
    --tp-size) TP_SIZE="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --gpu-mem-util) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --base-lr) BASE_LR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export SEGCOMPASS_REF_POS_TOKEN="${SEGCOMPASS_REF_POS_TOKEN:-<|object_ref_start|>}"

if [ -n "${RAY_TMPDIR:-}" ]; then
  mkdir -p "$RAY_TMPDIR"
fi

if [ "$BF16" = "true" ]; then
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

prefix() {
  case "$1" in
    ""|null) printf '%s' "$1" ;;
    /*) printf '%s' "$1" ;;
    *) printf '%s/%s' "$ROOT_DIR" "$1" ;;
  esac
}

SAE_ARG=null
if [ -n "$INIT_SAE_CKPT" ] && [ "$INIT_SAE_CKPT" != "null" ]; then
  SAE_ARG="$(prefix "$INIT_SAE_CKPT")"
fi

ARGS=(
  data.train_files="$(prefix "$DATA_PATH")"
  data.video_embed_dir=""
  data.prompt_key=text
  data.rollout_batch_size="${ROLLOUT_BATCH_SIZE}"
  data.max_prompt_length=1400
  data.max_response_length=256
  data.shuffle=true

  worker.llm_version="qwen-2.5"
  worker.actor.model.model_path="${MODEL_PATH}"
  worker.actor.model.init_sae_ckpt="${SAE_ARG}"
  worker.actor.model.k_slots="${K_SLOTS}"
  worker.sae.d_in=3584
  worker.sae.hook_layer=13
  worker.supp_bf16="${BF16}"

  worker.rollout.n="${ROLLOUT_N}"
  worker.rollout.tensor_parallel_size="${TP_SIZE}"
  worker.rollout.max_num_seqs="${MAX_NUM_SEQS}"
  worker.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}"
  worker.rollout.dtype="${DTYPE}"
  worker.rollout.enable_chunked_prefill="${CHUNKED_PREFILL}"

  worker.actor.global_batch_size="${GLOBAL_BATCH_SIZE}"
  worker.actor.micro_batch_size_per_device_for_update="${MICRO_UPDATE}"
  worker.actor.micro_batch_size_per_device_for_experience="${MICRO_EXPERIENCE}"
  worker.actor.optim.base_lr="${BASE_LR}"
  worker.actor.seg_loss_coef=0.3
  worker.actor.conf_loss_coef=0.2
  worker.actor.kl_loss_coef=0.2
  worker.actor.adjust_loss_step=1500
  worker.actor.entropy_coef=0.0

  trainer.n_gpus_per_node="${N_GPUS}"
  trainer.total_episodes="${TOTAL_EPISODES}"
  trainer.save_freq="${SAVE_FREQ}"
  trainer.val_before_train=false
  trainer.logger='["console"]'
  trainer.save_checkpoint_path="checkpoints/${RUN_NAME}/${RUN_FLAG}"
  trainer.load_checkpoint_path=null
  config="scripts/initial.yaml"
)

LOG_PATH="print_log_${RUN_NAME}_${RUN_FLAG}.txt"

echo "Root: $ROOT_DIR"
echo "Python: $PY"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Trainer GPUs: $N_GPUS"
echo "Model: $MODEL_PATH"
echo "Data: $(prefix "$DATA_PATH")"
echo "Vision features: on-the-fly Qwen2.5-VL ViT"
echo "SAE ckpt: $SAE_ARG"
echo "Log: $LOG_PATH"

PYTHONUNBUFFERED=1 "$PY" -u -m verl.trainer.main "${ARGS[@]}" 2>&1 | tee -a "$LOG_PATH"
