import argparse
import functools
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage


def timed_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"Time: {elapsed:.6f}s\n")
        return result

    return wrapper


def _load_video_frames_uniform(video_path, num_frames=24):
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError(f"video has no readable frames: {video_path}")

        indices = np.linspace(0, max(total - 1, 0), int(num_frames)).round().astype(np.int64)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(PILImage.fromarray(frame))
        cap.release()
    except ModuleNotFoundError:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_pattern = os.path.join(tmpdir, "frame_%04d.jpg")
            vf = f"fps={int(num_frames)}/3"
            cmd = [
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(video_path),
                "-vf", vf, "-frames:v", str(int(num_frames)), out_pattern,
            ]
            subprocess.run(cmd, check=True)
            frames = [PILImage.open(p).convert("RGB") for p in sorted(Path(tmpdir).glob("frame_*.jpg"))]

    if not frames:
        raise RuntimeError(f"failed to decode any frame from video: {video_path}")
    while len(frames) < int(num_frames):
        frames.append(frames[-1].copy())
    return frames[: int(num_frames)]


@timed_function
def gen_qwen25vl_video_emb_from_sht(
        dataset_dir="/root/autodl-tmp/sht/window3s_step1s_dataset",
        save_anno_dir="/root/autodl-tmp/sht/qwen25vl_video_emb_test_hf",
        save_video_embed_dir="/root/autodl-tmp/sht/qwen25vl_video_emb_test_embeds",
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        split="train",
        num_videos=10,
        num_frames=24,
        prompt="Find anomalous events in this video.",
        device="cuda:0",
        attn_implementation="sdpa",
        overwrite=False,
):
    from datasets import Dataset, Features, Sequence, Value
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dataset_dir = Path(dataset_dir)
    clips_dir = dataset_dir / "clips" / split
    labels_dir = dataset_dir / "frame_labels" / split
    save_anno_dir = Path(save_anno_dir)
    save_video_embed_dir = Path(save_video_embed_dir)

    if overwrite:
        shutil.rmtree(save_anno_dir, ignore_errors=True)
        shutil.rmtree(save_video_embed_dir, ignore_errors=True)
    save_video_embed_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(clips_dir.glob("*.mp4"))[: int(num_videos)]
    if not video_paths:
        raise FileNotFoundError(f"no mp4 videos found under {clips_dir}")

    run_device = device if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    ).to(run_device).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    visual_encoder = getattr(model, "visual", None) or model.model.visual

    data_items = []
    with torch.inference_mode():
        for idx, video_path in enumerate(video_paths):
            stem = video_path.stem
            label_path = labels_dir / f"{stem}.npy"
            if not label_path.exists():
                raise FileNotFoundError(f"missing frame label for {video_path}: {label_path}")

            frames = _load_video_frames_uniform(video_path, num_frames=num_frames)
            frame_labels = np.load(label_path).astype(np.float32)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt},
                ],
            }]
            raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            proc = processor(text=[raw_prompt], videos=[frames], return_tensors="pt")

            pixel_values_videos = proc["pixel_values_videos"].to(model.device)
            video_grid_thw = proc["video_grid_thw"].to(model.device)
            if dtype != torch.float32:
                pixel_values_videos = pixel_values_videos.to(dtype)

            video_emb = visual_encoder(pixel_values_videos, grid_thw=video_grid_thw)
            if hasattr(video_emb, "last_hidden_state"):
                video_emb = video_emb.last_hidden_state
            elif isinstance(video_emb, (tuple, list)):
                video_emb = video_emb[0]
            video_emb = video_emb.detach().to(torch.float16).cpu()

            embed_path = save_video_embed_dir / f"{stem}.pt"
            torch.save({
                "video_emb": video_emb,
                "video_grid_thw": video_grid_thw.detach().cpu(),
                "video_path": str(video_path),
                "num_frames": int(num_frames),
                "model_path": model_path,
            }, embed_path)

            item = {
                "video": str(video_path),
                "video_filename": video_path.name,
                "text": prompt,
                "frame_label_path": str(label_path),
                "frame_labels": frame_labels.tolist(),
                "video_emb_path": str(embed_path),
                "video_grid_thw": video_grid_thw.detach().cpu().reshape(-1).to(torch.int64).tolist(),
                "num_frames": int(num_frames),
                "num_video_tokens": int(video_emb.shape[0]),
            }
            data_items.append(item)
            print(
                f"[{idx + 1}/{len(video_paths)}] {video_path.name}: "
                f"video_emb={tuple(video_emb.shape)} grid={item['video_grid_thw']}",
                flush=True,
            )

    features = Features({
        "video": Value("string"),
        "video_filename": Value("string"),
        "text": Value("string"),
        "frame_label_path": Value("string"),
        "frame_labels": Sequence(Value("float32")),
        "video_emb_path": Value("string"),
        "video_grid_thw": Sequence(Value("int64")),
        "num_frames": Value("int32"),
        "num_video_tokens": Value("int32"),
    })
    dataset = Dataset.from_list(data_items, features=features)
    dataset.save_to_disk(str(save_anno_dir))
    print(f"saved hf dataset: {save_anno_dir}")
    print(f"saved video embeddings: {save_video_embed_dir}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sht_dataset_dir", default="/root/autodl-tmp/sht/window3s_step1s_dataset")
    parser.add_argument("--save_anno_dir", default="/root/autodl-tmp/sht/qwen25vl_video_emb_test_hf")
    parser.add_argument("--save_video_embed_dir", default="/root/autodl-tmp/sht/qwen25vl_video_emb_test_embeds")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_videos", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=24)
    parser.add_argument("--prompt", default="Find anomalous events in this video.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    gen_qwen25vl_video_emb_from_sht(
        dataset_dir=args.sht_dataset_dir,
        save_anno_dir=args.save_anno_dir,
        save_video_embed_dir=args.save_video_embed_dir,
        model_path=args.model_path,
        split=args.split,
        num_videos=args.num_videos,
        num_frames=args.num_frames,
        prompt=args.prompt,
        device=args.device,
        attn_implementation=args.attn_implementation,
        overwrite=args.overwrite,
    )
