#!/usr/bin/env bash
set -euo pipefail

######## Devices ########
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

######## bind attention backend with bf16 support ########
SUPP_BF16="${SUPP_BF16:-false}"    # RTX 4080 SUPER is safer with SDPA/fp16 here.

if [ "$SUPP_BF16" = "true" ]; then export VLLM_ATTENTION_BACKEND=FLASH_ATTN; DTYPE=bfloat16;
else export VLLM_ATTENTION_BACKEND=SDPA; export VLLM_USE_TRITON=0; export XFORMERS_FORCE_DISABLE_TRITON=1; DTYPE=float16; fi

######## function tools ########
ROOT_DIR="${ROOT_DIR:-/root/fuxian/SegCompass}"
RUN_FLAG="${RUN_FLAG:-default}"
RUN_NAME="${RUN_NAME:-qwen_vad_7b_n2}"
usage(){ echo "Usage: $0 [-r|--root DIR] [-f|--run_flag RUN_FLAG]"; exit 1; }
while [ $# -gt 0 ]; do case "$1" in
  -r|--root|-f|--run_flag)
    [ $# -ge 2 ] || usage; key="$1"; val="$2"; shift 2;
    case "$key" in -r|--root) ROOT_DIR="$val";; -f|--run_flag) RUN_FLAG="$val";; esac ;;
  *) usage ;; esac; done
prefix(){
  case "$1" in
    ""|null) printf '%s' "$1" ;;
    /*) printf '%s' "$1" ;;
    *) printf '%s/%s' "${ROOT_DIR%/}" "$1" ;;
  esac
}
cd "$ROOT_DIR"

######## Disable NCCL InfiniBand/RDMA and wandb ########
export NCCL_IB_DISABLE=1; export WANDB_MODE=offline; export WANDB_DIR="$(prefix 'wandb')"
export RAY_TMPDIR="${RAY_TMPDIR:-/root/autodl-tmp/ray}"
export SEGCOMPASS_REF_POS_TOKEN="${SEGCOMPASS_REF_POS_TOKEN:-<|object_ref_start|>}"
mkdir -p "$RAY_TMPDIR"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
INIT_SAE_CKPT="${INIT_SAE_CKPT:-sae_checkpoints/sae_qwen-7b_L13/default/ep_6.pt}"
DATA_TRAIN_FILES="${DATA_TRAIN_FILES:-data/sht_video_10}"
D_IN="${D_IN:-3584}"
D_SAE="${D_SAE:-65536}"
HOOK_LAYER="${HOOK_LAYER:-13}"
K_SLOTS="${K_SLOTS:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
MICRO_UPDATE="${MICRO_UPDATE:-1}"
MICRO_EXPERIENCE="${MICRO_EXPERIENCE:-1}"
TP_SIZE="${TP_SIZE:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
N_GPUS="${N_GPUS:-2}"
TOTAL_EPISODES="${TOTAL_EPISODES:-1}"
SAVE_FREQ="${SAVE_FREQ:--1}"

######## Grouped CLI args ########
ARGS=(
  ### paths ###
  data.train_files="$(prefix "${DATA_TRAIN_FILES}")"
  data.video_embed_dir=""
  data.prompt_key=text
  data.max_prompt_length=${MAX_PROMPT_LENGTH}
  data.max_response_length=${MAX_RESPONSE_LENGTH}
  data.shuffle=true
  worker.actor.model.model_path="${MODEL_PATH}"
  worker.actor.model.init_sae_ckpt="$(if [ "${INIT_SAE_CKPT}" = "null" ]; then printf 'null'; else prefix "${INIT_SAE_CKPT}"; fi)"

  trainer.save_checkpoint_path="$(prefix "checkpoints/${RUN_NAME}/${RUN_FLAG}")"
  trainer.load_checkpoint_path=null
  config="scripts/initial.yaml"

  ### multi-slots and SAE ###
  worker.sae.d_in=${D_IN}
  worker.sae.d_sae=${D_SAE}
  worker.llm_version="qwen-2.5"
  worker.actor.model.k_slots=${K_SLOTS}
  worker.sae.hook_layer=${HOOK_LAYER}

  ### batch size ###   (global_batch_size * n) / nnodes 要被 micro_batch_size_per_device 整除
  worker.rollout.n=${ROLLOUT_N}
  data.rollout_batch_size=${ROLLOUT_BATCH_SIZE}
  worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE}
  worker.actor.micro_batch_size_per_device_for_update=${MICRO_UPDATE}
  worker.actor.micro_batch_size_per_device_for_experience=${MICRO_EXPERIENCE}

  ### rollout ###
  worker.rollout.tensor_parallel_size=${TP_SIZE}
  worker.rollout.max_num_seqs=${MAX_NUM_SEQS}
  worker.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}
  worker.rollout.dtype=${DTYPE}
  worker.rollout.enable_chunked_prefill=${SUPP_BF16}
  worker.rollout.limit_images=0
  worker.rollout.limit_videos=1

  ### trainer ###
  trainer.n_gpus_per_node=${N_GPUS}
  trainer.total_episodes=${TOTAL_EPISODES}
  trainer.save_freq=${SAVE_FREQ}
  trainer.val_before_train=false
  trainer.logger='["console"]'
  worker.supp_bf16=${SUPP_BF16}

  ### loss coef###
  worker.actor.optim.base_lr=1.6e-6
  worker.actor.seg_loss_coef=0.3
  worker.actor.conf_loss_coef=0.2
  worker.actor.kl_loss_coef=0.2
  worker.actor.adjust_loss_step=1500
  worker.actor.entropy_coef=0.0
  worker.actor.dice_loss_coef_new=2.0
  worker.actor.focal_loss_coef_new=5.0
)

PY="$(prefix 'segcompass/bin/python')"; [ -x "$PY" ] || PY=/root/autodl-tmp/verl_env/bin/python
export PATH="$(dirname "$PY"):${PATH}"
LOG_PATH="$(prefix "print_log_${RUN_NAME}_${RUN_FLAG}.txt")"

echo "Root: $ROOT_DIR"
echo "Python: $PY"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Trainer GPUs: $N_GPUS"
echo "Model: $MODEL_PATH"
echo "Data: $(prefix "${DATA_TRAIN_FILES}")"
echo "Vision features: on-the-fly Qwen2.5-VL ViT"
echo "SAE ckpt: $(if [ "${INIT_SAE_CKPT}" = "null" ]; then printf 'null'; else prefix "${INIT_SAE_CKPT}"; fi)"
echo "Run: ${RUN_NAME}/${RUN_FLAG}"
echo "Log: $LOG_PATH"

set -x
PYTHONUNBUFFERED=1 "$PY" -u -m verl.trainer.main "${ARGS[@]}" 2>&1 | tee -a "$LOG_PATH"
