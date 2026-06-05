# Qwen2.5-VL VAD Training on 8x A100

This directory contains the Qwen2.5-VL training entrypoints for the VAD version of SegCompass.

The old SAM segmentation path has been removed. The VAD path extracts visual features on the fly from the Qwen2.5-VL ViT during the normal model forward pass. It no longer uses `segment_anything`, SAM checkpoints, `prompt_encoder`, or `mask_decoder`.

## Files

- `qwen_vad_a100_n8.sh`: recommended 8x A100 launch script.
- `gref_qwen-7b_n8_10_smoke.sh`: legacy 1-GPU smoke script for small data/debugging.
- `gref_qwen-7b_n8.sh`: compatibility path now wired to the 2-GPU SHT VAD setup.

## Expected Workspace

Run from the project root:

```bash
cd /root/fuxian/SegCompass
```

Recommended directory layout:

```text
/root/fuxian/SegCompass
├── data/
│   ├── sht_video_10/
│   └── <your_vad_train_dataset>/
├── scripts/
│   └── initial.yaml
├── sae_checkpoints/
└── checkpoints/
```

The recommended dataset format contains video inputs that `qwen_vl_utils.process_vision_info` can load through the Qwen processor, plus frame-level labels.

The code still accepts historical precomputed `video_embed` / `video_emb_path` fields, but the A100 launch script does not require or expose a feature directory.

For the VAD path, frame supervision should be available through one of:

```text
frame_labels
frame_label
frame_targets
labels
label
```

## Environment Requirements

Use an environment where PyTorch CUDA is compatible with the machine driver.

Before launching, verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
```

Expected on the 8x A100 machine:

```text
torch.cuda.is_available() == True
torch.cuda.device_count() == 8
```

If the driver is CUDA 12.8, do not use a `cu130` PyTorch build. Install a CUDA 12.x compatible PyTorch/vLLM stack.

## Launch

Minimal 8x A100 run:

```bash
bash scripts/train/qwen-7b/qwen_vad_a100_n8.sh \
  --data data/<your_vad_train_dataset> \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --sae sae_checkpoints/sae_qwen-7b_L13/default/ep_6.pt \
  --run-flag vad_a100
```

2x A100 sanity run on physical CUDA devices 1 and 2:

```bash
CUDA_VISIBLE_DEVICES=1,2 bash scripts/train/qwen-7b/qwen_vad_a100_n8.sh \
  --data data/<your_vad_train_dataset> \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --sae null \
  --run-flag sanity_cuda1_2 \
  --n-gpus 2 \
  --k-slots 1 \
  --tp-size 2 \
  --rollout-n 1 \
  --rollout-batch-size 2 \
  --global-batch-size 2 \
  --micro-update 1 \
  --micro-experience 1 \
  --total-episodes 1 \
  --save-freq -1
```

If you do not want to load an SAE checkpoint:

```bash
bash scripts/train/qwen-7b/qwen_vad_a100_n8.sh \
  --data data/<your_vad_train_dataset> \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --sae null
```

Default outputs:

```text
checkpoints/qwen_vad_a100_n8/<run_flag>/
print_log_qwen_vad_a100_n8_<run_flag>.txt
wandb/
```

## Important Defaults

The launch script defaults to:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
trainer.n_gpus_per_node=8
worker.rollout.tensor_parallel_size=4
worker.rollout.n=8
data.rollout_batch_size=16
worker.actor.global_batch_size=16
worker.supp_bf16=true
worker.rollout.dtype=bfloat16
VLLM_ATTENTION_BACKEND=FLASH_ATTN
WANDB_MODE=offline
```

For A100, keep `--bf16 true`.

## Common Issues

### No GPUs Found

If NCCL reports:

```text
ProcessGroupNCCL is only supported with GPUs, no GPUs found
```

check that PyTorch CUDA matches the driver:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

### Ray Temporary Directory Is Full

If Ray reports `/tmp/ray` over 95% full, set a larger temp directory before launch:

```bash
export RAY_TMPDIR=/root/autodl-tmp/ray
mkdir -p "$RAY_TMPDIR"
```

The launch script uses `RAY_TMPDIR` if it is set.

### Historical SAM Errors

The VAD path should not import SAM. If an old log or old checkout fails with one of these strings, you are likely not running the current VAD entrypoint:

```text
segment_anything
sam_model_registry
init_sam_ckpt
prompt_encoder
mask_decoder
sam_embed
```

The current VAD launch scripts do not require these modules or checkpoints.
