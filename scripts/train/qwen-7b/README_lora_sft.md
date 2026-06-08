# Qwen-VL LoRA SFT

This entry is independent from the existing PPO/SegCompass launchers.

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
PY=/path/to/python \
ROOT_DIR=/path/to/SegCompass \
MODEL_PATH=/path/to/Qwen2.5-VL \
DATA_TRAIN_FILES=/path/to/dataset \
OUTPUT_DIR=/path/to/output \
bash scripts/train/qwen-7b/qwen_lora_sft.sh
```

Multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PY=/path/to/python \
ROOT_DIR=/path/to/SegCompass \
MODEL_PATH=/path/to/Qwen2.5-VL \
DATA_TRAIN_FILES=/path/to/dataset \
OUTPUT_DIR=/path/to/output \
bash scripts/train/qwen-7b/qwen_lora_sft.sh
```

The process count is inferred from `CUDA_VISIBLE_DEVICES`. Override it with
`NPROC_PER_NODE` when needed.

All path variables are mandatory and must be absolute:
`ROOT_DIR`, `PY`, `MODEL_PATH`, `DATA_TRAIN_FILES`, and `OUTPUT_DIR`.

The dataset may contain `response`, `answer`, `output`, or `assistant`. Use
`RESPONSE_KEY=...` for a different field. If no response field exists but
`frame_labels` is present, the trainer creates an anomaly-frame-range response
and appends `REF_TOKEN`.

Important variables:

- `PER_DEVICE_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`
- `LEARNING_RATE`, `EPOCHS`, `MAX_STEPS`
- `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`, `TARGET_MODULES`
- `MAX_LENGTH`, `MAX_PIXELS`, `MIN_PIXELS`, `VIDEO_FPS`
- `BF16`, `GRADIENT_CHECKPOINTING`, `ATTN_IMPLEMENTATION`
- `OUTPUT_DIR`, `RESUME_FROM_CHECKPOINT`

The selected Python environment must contain the repository requirements,
especially `datasets`, `peft`, `qwen-vl-utils`, `torch`, and `transformers`.
