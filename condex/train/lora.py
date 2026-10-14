"""LoRA fine-tune of the judge (PLAN.md Phase 3 step 2). Cluster only (torch, transformers, peft).

Recipe: Qwen3-VL-8B-Instruct, LoRA on the language model's attention and MLP projections, vision
encoder frozen; input = the packet (images + text) as a chat; target = the verdict JSON; loss =
next-token cross-entropy on the target tokens only. One adapter per input condition, same recipe,
same splits; seeds from the CLI.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

LM_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
IGNORE = -100


@dataclass
class TrainConfig:
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    lr: float = 1e-4
    epochs: int = 2
    grad_accum: int = 16
    warmup_frac: float = 0.03
    weight_decay: float = 0.0
    max_len: int = 8192
    seed: int = 0
    eval_every: int = 200  # optimizer steps


def chat_messages(row: dict, with_target: bool) -> list[dict]:
    content = [{"type": "image"} if c["type"] == "image" else {"type": "text", "text": c["text"]} for c in row["content"]]
    msgs = [{"role": "system", "content": [{"type": "text", "text": row["system"]}]}, {"role": "user", "content": content}]
    if with_target:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": row["target"]}]})
    return msgs


def mask_labels(input_ids: list[int], prompt_len: int) -> list[int]:
    """Loss on the target tokens only: everything before the assistant answer is ignored."""
    return [IGNORE] * prompt_len + list(input_ids[prompt_len:])


def encode(processor, row: dict, root: Path, max_len: int) -> dict:
    from PIL import Image

    images = [Image.open(root / c["path"]).convert("RGB") for c in row["content"] if c["type"] == "image"]
    prompt = processor.apply_chat_template(chat_messages(row, False), tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template(chat_messages(row, True), tokenize=False, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise ValueError("chat template: the prompt is not a prefix of prompt+answer")
    enc_p = processor(text=[prompt], images=images or None, return_tensors="pt")
    enc = processor(text=[full], images=images or None, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    if len(ids) > max_len:
        raise ValueError(f"{row['id']}: {len(ids)} tokens > max_len {max_len}")
    import torch

    enc["labels"] = torch.tensor([mask_labels(ids, enc_p["input_ids"].shape[1])])
    return dict(enc)


def train(train_rows: list[dict], val_rows: list[dict], root: Path, out_dir: Path, cfg: TrainConfig, log=print) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
                                             target_modules=LM_TARGETS, task_type="CAUSAL_LM"))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bad = [n for n, p in model.named_parameters() if p.requires_grad and "visual" in n]
    if bad:
        raise RuntimeError(f"vision parameters are trainable: {bad[:3]}")
    log(f"trainable params: {n_train / 1e6:.1f}M")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = math.ceil(len(train_rows) * cfg.epochs / cfg.grad_accum)
    warm = max(1, int(cfg.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))

    def run(rows, train_mode):
        loss_sum, n = 0.0, 0
        for r in rows:
            batch = {k: v.to(model.device) for k, v in encode(processor, r, root, cfg.max_len).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            if train_mode:
                (loss / cfg.grad_accum).backward()
            loss_sum += float(loss)
            n += 1
            yield loss_sum / n

    history, step, t0 = [], 0, time.time()
    best = math.inf
    for epoch in range(cfg.epochs):
        rows = list(train_rows)
        random.shuffle(rows)
        model.train()
        for i, avg in enumerate(run(rows, True), 1):
            if i % cfg.grad_accum == 0 or i == len(rows):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    log(json.dumps({"epoch": epoch, "step": step, "train_loss": round(avg, 4), "lr": sched.get_last_lr()[0],
                                    "elapsed_s": round(time.time() - t0)}))
                if val_rows and step % cfg.eval_every == 0:
                    model.eval()
                    with torch.no_grad():
                        v = list(run(val_rows, False))[-1]
                    history.append({"step": step, "val_loss": v})
                    log(json.dumps(history[-1]))
                    if v < best:
                        best = v
                        model.save_pretrained(out_dir / "best")
                    model.train()
        model.save_pretrained(out_dir / f"epoch{epoch}")
    if val_rows:
        model.eval()
        with torch.no_grad():
            v = list(run(val_rows, False))[-1]
        history.append({"step": step, "val_loss": v})
        if v < best:
            best = v
            model.save_pretrained(out_dir / "best")
    summary = {"config": asdict(cfg), "trainable_params": n_train, "optimizer_steps": step, "history": history,
               "best_val_loss": best if val_rows else None, "train_rows": len(train_rows), "val_rows": len(val_rows)}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=1))
    return summary
