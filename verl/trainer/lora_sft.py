"""LoRA supervised fine-tuning for text, image, or video instruction data."""

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model name or local model directory.")
    parser.add_argument("--data", required=True, help="Dataset saved by datasets.save_to_disk, JSON, or JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--response-key", default="")
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--video-key", default="video")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--ref-token", default="<|object_ref_start|>")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-pixels", type=int, default=602112)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated Linear module suffixes.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", default="")
    return parser.parse_args()


def load_training_dataset(path: str):
    if os.path.isdir(path):
        dataset = load_from_disk(path)
        if hasattr(dataset, "keys") and "train" in dataset:
            dataset = dataset["train"]
        return dataset
    extension = os.path.splitext(path)[1].lower()
    if extension not in {".json", ".jsonl"}:
        raise ValueError(f"Unsupported dataset path: {path}")
    return load_dataset("json", data_files=path, split="train")


def first_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value)


def positive_ranges(labels: List[Any]) -> str:
    indices = [i for i, value in enumerate(labels) if float(value) > 0.5]
    if not indices:
        return "none"
    ranges = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            ranges.append((start, previous))
            start = index
        previous = index
    ranges.append((start, previous))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def get_response(row: Dict[str, Any], response_key: str, ref_token: str) -> str:
    candidate_keys = [response_key] if response_key else ["response", "answer", "output", "assistant"]
    for key in candidate_keys:
        if key and key in row and row[key] not in (None, ""):
            return first_text(row[key])
    if "frame_labels" in row:
        ranges = positive_ranges(row["frame_labels"])
        return (
            f"<think>The anomalous frame ranges are {ranges}.</think>\n"
            f"Here is the reference position: {ref_token}"
        )
    raise KeyError(
        "No response field was found. Pass --response-key or provide frame_labels "
        "for automatic anomaly-range targets."
    )


@dataclass
class QwenVLCollator:
    processor: Any
    text_key: str
    response_key: str
    image_key: str
    video_key: str
    system_prompt: str
    ref_token: str
    max_length: int
    max_pixels: int
    min_pixels: int
    video_fps: float

    def _messages(self, row: Dict[str, Any], include_answer: bool) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        video = row.get(self.video_key)
        image = row.get(self.image_key)
        if video:
            content.append(
                {
                    "type": "video",
                    "video": video,
                    "max_pixels": self.max_pixels,
                    "min_pixels": self.min_pixels,
                    "fps": self.video_fps,
                }
            )
        elif image:
            content.append(
                {
                    "type": "image",
                    "image": image,
                    "max_pixels": self.max_pixels,
                    "min_pixels": self.min_pixels,
                }
            )
        content.append({"type": "text", "text": first_text(row[self.text_key])})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": content},
        ]
        if include_answer:
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": get_response(row, self.response_key, self.ref_token),
                        }
                    ],
                }
            )
        return messages

    def _encode(self, messages: List[Dict[str, Any]], add_generation_prompt: bool) -> Dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        video_kwargs = {
            key: value[0] if isinstance(value, list) and len(value) == 1 else value
            for key, value in video_kwargs.items()
        }
        encoded = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **video_kwargs,
        )
        keep_batch_dim = {"image_grid_thw", "video_grid_thw"}
        return {
            key: value if key in keep_batch_dim else value.squeeze(0)
            for key, value in encoded.items()
        }

    def __call__(self, rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        samples = []
        for row in rows:
            prompt = self._encode(self._messages(row, include_answer=False), add_generation_prompt=True)
            full = self._encode(self._messages(row, include_answer=True), add_generation_prompt=False)
            labels = full["input_ids"].clone()
            labels[: min(prompt["input_ids"].numel(), labels.numel())] = -100
            if torch.all(labels == -100):
                raise ValueError(
                    "The assistant response was fully truncated. Increase --max-length "
                    "or reduce the media resolution/number of frames."
                )
            full["labels"] = labels
            samples.append(full)

        tokenizer = self.processor.tokenizer
        padding_side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        batch = tokenizer.pad(
            [
                {
                    "input_ids": sample["input_ids"],
                    "attention_mask": sample["attention_mask"],
                }
                for sample in samples
            ],
            padding=True,
            return_tensors="pt",
        )
        tokenizer.padding_side = padding_side
        padded_labels = torch.full_like(batch["input_ids"], -100)
        for index, sample in enumerate(samples):
            length = sample["labels"].numel()
            padded_labels[index, :length] = sample["labels"]
        batch["labels"] = padded_labels

        for key in ("pixel_values", "pixel_values_videos"):
            values = [sample[key] for sample in samples if key in sample]
            if values:
                batch[key] = torch.cat(values, dim=0)
        for key in ("image_grid_thw", "video_grid_thw"):
            values = [sample[key] for sample in samples if key in sample]
            if values:
                batch[key] = torch.cat(values, dim=0)
        return batch


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if int(os.environ.get("RANK", "0")) == 0:
        model.print_trainable_parameters()

    dataset = load_training_dataset(args.data)
    collator = QwenVLCollator(
        processor=processor,
        text_key=args.text_key,
        response_key=args.response_key,
        image_key=args.image_key,
        video_key=args.video_key,
        system_prompt=args.system_prompt,
        ref_token=args.ref_token,
        max_length=args.max_length,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        video_fps=args.video_fps,
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collator)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
