#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"

usage() {
  cat <<'USAGE'
Usage:
  ROOT_DIR=/absolute/path/to/SegCompass \
  PY=/absolute/path/to/python \
  MODEL_PATH=/absolute/path/to/model \
  DATA_TRAIN_FILES=/absolute/path/to/dataset \
  OUTPUT_DIR=/absolute/path/to/output \
  bash scripts/train/qwen-7b/qwen_lora_sft.sh

Required path variables:
  ROOT_DIR, PY, MODEL_PATH, DATA_TRAIN_FILES, OUTPUT_DIR
USAGE
}

require_path() {
  local name="$1"
  local value="${!name:-}"
  if [ -z "$value" ]; then
    echo "Error: $name is required and must be an absolute path." >&2
    usage
    exit 1
  fi
  case "$value" in
    /*) ;;
    *)
      echo "Error: $name must be an absolute path: $value" >&2
      exit 1
      ;;
  esac
}

require_path ROOT_DIR
require_path PY
require_path MODEL_PATH
require_path DATA_TRAIN_FILES
require_path OUTPUT_DIR

[ -d "$ROOT_DIR" ] || { echo "Error: ROOT_DIR does not exist: $ROOT_DIR" >&2; exit 1; }
[ -x "$PY" ] || { echo "Error: PY is not executable: $PY" >&2; exit 1; }
[ -e "$MODEL_PATH" ] || { echo "Error: MODEL_PATH does not exist: $MODEL_PATH" >&2; exit 1; }
[ -e "$DATA_TRAIN_FILES" ] || {
  echo "Error: DATA_TRAIN_FILES does not exist: $DATA_TRAIN_FILES" >&2
  exit 1
}

export ROOT_DIR PY MODEL_PATH DATA_TRAIN_FILES OUTPUT_DIR CUDA_VISIBLE_DEVICES
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

if [ -z "$NPROC_PER_NODE" ]; then
  IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
  NPROC_PER_NODE="${#devices[@]}"
fi
if [ "$NPROC_PER_NODE" -lt 1 ]; then
  echo "NPROC_PER_NODE must be at least 1." >&2
  exit 1
fi

ARGS=(
  -m verl.trainer.lora_sft
  --model "$MODEL_PATH"
  --data "$DATA_TRAIN_FILES"
  --output-dir "$OUTPUT_DIR"
  --text-key "${TEXT_KEY:-text}"
  --response-key "${RESPONSE_KEY:-}"
  --video-key "${VIDEO_KEY:-video}"
  --image-key "${IMAGE_KEY:-image}"
  --ref-token "${REF_TOKEN:-<|object_ref_start|>}"
  --max-length "${MAX_LENGTH:-2048}"
  --max-pixels "${MAX_PIXELS:-602112}"
  --min-pixels "${MIN_PIXELS:-3136}"
  --video-fps "${VIDEO_FPS:-1.0}"
  --epochs "${EPOCHS:-1}"
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
  --learning-rate "${LEARNING_RATE:-2e-4}"
  --save-steps "${SAVE_STEPS:-100}"
  --logging-steps "${LOGGING_STEPS:-1}"
  --lora-r "${LORA_R:-16}"
  --lora-alpha "${LORA_ALPHA:-32}"
  --lora-dropout "${LORA_DROPOUT:-0.05}"
  --target-modules "${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
)

if [ "${BF16:-true}" = "false" ]; then
  ARGS+=(--no-bf16)
fi
if [ "${GRADIENT_CHECKPOINTING:-true}" = "false" ]; then
  ARGS+=(--no-gradient-checkpointing)
fi
if [ -n "${MAX_STEPS:-}" ]; then
  ARGS+=(--max-steps "$MAX_STEPS")
fi
if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
  ARGS+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

cd "$ROOT_DIR"
mkdir -p "$OUTPUT_DIR"

"$PY" -c 'import datasets, peft, qwen_vl_utils, torch, transformers' || {
  echo "Missing dependencies in $PY." >&2
  echo "Install the versions from ${ROOT_DIR}/requirements.txt, including datasets and peft." >&2
  exit 1
}

echo "GPUs: $CUDA_VISIBLE_DEVICES (processes: $NPROC_PER_NODE)"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_TRAIN_FILES"
echo "Output: $OUTPUT_DIR"

if [ "$NPROC_PER_NODE" -eq 1 ]; then
  exec "$PY" "${ARGS[@]}"
fi

exec "$PY" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  "${ARGS[@]}"
