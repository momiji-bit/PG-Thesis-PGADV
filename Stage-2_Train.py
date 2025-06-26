#!/usr/bin/env python
# coding: utf-8
"""
Prompt-Learner 微调示例：
    * 仅训练 Qwen2.5-VL 的 prompt_learner
    * 输入：视频 + 问句 “Is there any anomaly in the video?”
    * 标签：label.csv 中的 description 列
    * 评估指标：生成文本 == 标签 的准确率
"""

from __future__ import annotations
import logging, os, argparse, glob, re, warnings

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from anomaly_qwen2_5_vl import (
    Anomaly_Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)
from qwen_vl_utils import process_vision_info

logging.getLogger("transformers.video_processing_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------- 数据集 ---------------------------- #
class VideoDescDataset(Dataset):
    """读取 (video, description) 对，用于多模态文本生成训练"""

    def __init__(
        self,
        video_dir: str,
        csv_path: str,
        processor: Qwen2_5_VLProcessor,
        # prompt: str = "Is there any anomaly in the video?",
        prompt: str = "视频中有任何异常行为吗?（只回答：有异常或没有异常）",
        fps: float = 24.0,
    ):
        self.video_dir = video_dir
        self.processor = processor
        self.prompt = prompt
        self.fps = fps

        # 1. 先加载csv，建立从视频名到描述的索引
        df = pd.read_csv(csv_path)
        # dict: {video_name: description}
        self.video2desc = dict(zip(df['video'], df['description']))

        # 2. 遍历video_dir下所有视频文件
        self.video_names = [
            f for f in os.listdir(video_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx: int):
        video_name = self.video_names[idx]
        video_path = os.path.join(self.video_dir, video_name)

        # 3. 用视频名查找csv描述
        if video_name not in self.video2desc:
            raise ValueError(f"视频 {video_name} 在 csv 文件中找不到描述")

        description = self.video2desc[video_name]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "fps": self.fps},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        return {
            "messages": messages,
            "label_text": description,
        }


# ------------------------- batch 拼接 ------------------------- #
def collate_fn(batch, processor):
    # ---------- 1. 多模态 prompt ----------
    messages = [b["messages"] for b in batch]
    prompt_texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )

    inputs = processor(
        text=prompt_texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )                       # 得到 input_ids / attention_mask / vision_tensors …

    # ---------- 2. 构造 labels ----------
    # 2-1 把标签文字转成 token
    label_ids = processor.tokenizer(
        [b["label_text"] for b in batch],
        padding=True,
        return_tensors="pt",
    ).input_ids                         # shape = (batch, ans_len)

    # 2-2 创建与 input_ids 同形状的全 -100 tensor
    labels = torch.full_like(inputs["input_ids"], -100)

    # 2-3 只把最后几个位置替换成真正的答案 token
    for i in range(labels.size(0)):
        ans_len = (label_ids[i] != processor.tokenizer.pad_token_id).sum()
        labels[i, -ans_len:] = label_ids[i, :ans_len]

    inputs["labels"] = labels          # 与 logits 对齐
    return inputs, [b["label_text"] for b in batch]



# ------------------------- 训练 / 验证 ------------------------- #
def train_one_epoch(
    model,
    processor,
    loader,
    optimizer,
    device,
    epoch: int,
    grad_clip: float | None = 1.0,
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Train E{epoch}", ncols=120)

    for step, (inputs, targets) in enumerate(pbar):
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        outputs = model(**inputs, use_cache=False)  # forward + CE loss
        loss = outputs.loss

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.model.prompt_learner.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # 统计 loss
        total_loss += loss.item()

        # 生成文本评估训练准确率（粗略）
        gen_ids = model.generate(
            **{k: v for k, v in inputs.items() if k != "labels"}, max_new_tokens=128
        )
        preds = processor.batch_decode(
            gen_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print('---')
        print(preds)
        print(targets)
        print('---')
        for p, g in zip(preds, targets):
            correct += int(p.strip() == g.strip())
            total += 1

        pbar.set_postfix(loss=f"{total_loss/(step+1):.4f}", acc=f"{correct/total:.2%}")

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, processor, device):
    model.eval()
    correct, total = 0, 0
    for inputs, targets in tqdm(loader, desc="Val", ncols=120):
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        gen_ids = model.generate(**inputs, max_new_tokens=32)
        preds = processor.batch_decode(
            gen_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for p, g in zip(preds, targets):
            correct += int(p.strip() == g.strip())
            total += 1
    return correct / max(total, 1)


# ------------------------- Checkpoint ------------------------- #
def save_ckpt(path: str, epoch: int, model, optimizer):
    torch.save(
        {
            "epoch": epoch,
            "prompt_learner": model.model.prompt_learner.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_ckpt(path: str, model, optimizer):
    if not os.path.isfile(path):
        return 0
    ckpt = torch.load(path, map_location="cpu")
    model.model.prompt_learner.load_state_dict(ckpt["prompt_learner"])
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except ValueError:
        pass
    print(f"[Resume] Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"] + 1  # 下一轮开始的 epoch


# ------------------------- 主函数 ------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="dataset/train")
    parser.add_argument("--val_dir", default="dataset/val")
    parser.add_argument("--gt_csv", default="dataset/labels.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_dir", default="ckpts_prompt")
    parser.add_argument("--resume_ckpt", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. 模型与处理器 -------------------------------------------------------
    model = Anomaly_Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Geo/Anomaly_Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(
        "Geo/Anomaly_Qwen2.5-VL-7B-Instruct"
    )

    # 冻结全部，再解冻 prompt_learner
    model.requires_grad_(False)
    for p in model.model.prompt_learner.parameters():
        p.requires_grad_(True)

    # 2. 数据 ----------------------------------------------------------------
    train_set = VideoDescDataset(args.train_dir, args.gt_csv, processor)
    val_set = VideoDescDataset(args.val_dir, args.gt_csv, processor)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, processor),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, processor),
        pin_memory=True,
    )

    # 3. 优化器 --------------------------------------------------------------
    optimizer = optim.AdamW(model.model.prompt_learner.parameters(), lr=args.lr)

    # 4. 恢复训练 ------------------------------------------------------------
    start_epoch = 1
    if args.resume_ckpt:
        start_epoch = load_ckpt(args.resume_ckpt, model, optimizer)

    # 5. 训练循环 ------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_acc = train_one_epoch(
            model, processor, train_loader, optimizer, device, epoch
        )
        val_acc = evaluate(model, val_loader, processor, device)

        print(
            f"[Epoch {epoch}]  loss: {train_loss:.4f}  "
            f"train_acc: {train_acc:.2%}  val_acc: {val_acc:.2%}"
        )

        ckpt_path = os.path.join(args.save_dir, f"epoch{epoch:02d}.pth")
        save_ckpt(ckpt_path, epoch, model, optimizer)

    print("Training finished!")


if __name__ == "__main__":
    main()
