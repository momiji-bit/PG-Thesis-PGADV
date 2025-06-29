from __future__ import annotations
import logging
logging.getLogger("transformers.video_processing_utils").setLevel(logging.ERROR)

import os, glob, json, math, random, warnings, argparse, re
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from anomaly_qwen2_5_vl import (
    Anomaly_Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------- 数据集 ---------------------------- #
class VideoMaskDataset(Dataset):
    def __init__(
            self,
            train_dir: str,
            gt_dir: str,
            file_ids: list[str] | None = None,
            processor: Qwen2_5_VLProcessor | None = None,
            target_size: tuple[int, int] = (34, 62),
            prompt: str = "Is there any anomaly in the video？",
            fps: float = 24.0,
            max_frames: int = 24,
    ):
        self.train_dir = train_dir
        self.gt_dir = gt_dir
        self.file_ids = (
            file_ids
            if file_ids is not None
            else [
                os.path.splitext(os.path.basename(p))[0]
                for p in sorted(glob.glob(os.path.join(train_dir, "*.mp4")))
            ]
        )
        self.processor = processor
        self.target_size = target_size
        self.prompt = prompt
        self.fps = fps
        self.max_frames = max_frames

    def __len__(self):
        return len(self.file_ids)

    # ————— 帮助函数 ————— #
    @staticmethod
    def _load_video(path, max_frames, to_rgb=True):
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        while len(frames) < max_frames:  # 帧不足则循环最后一帧填充
            frames.append(frames[-1].copy())
        return np.stack(frames, axis=0)  # (T, H, W, 3)

    @staticmethod
    def _frames2_12(frames: np.ndarray):
        """两帧合并为一帧（取平均，形状 (12, H, W, 3)）"""
        frames = frames.reshape(12, 2, *frames.shape[1:])
        return frames.mean(axis=1)

    @staticmethod
    def _merge_mask(mask24: np.ndarray):
        """把 24 帧 mask → 12 帧，取逻辑或（binary）"""
        mask24 = mask24.astype(np.uint8).reshape(12, 2, *mask24.shape[1:])
        return mask24.max(axis=1)  # (12, H, W)

    def __getitem__(self, idx: int):
        file_id = self.file_ids[idx]
        video_path = os.path.join(self.train_dir, f"{file_id}.mp4")
        mask_path = os.path.join(self.gt_dir, f"{file_id}.npy")

        frames24 = self._load_video(video_path, self.max_frames)  # (24, H, W, 3)
        masks24 = np.load(mask_path)                              # (24, H, W)

        frames24 = frames24[: self.max_frames]
        masks24 = masks24[: self.max_frames]

        frames12 = self._frames2_12(frames24)  # (12, H, W, 3)
        masks12  = self._merge_mask(masks24)   # (12, H, W)

        # resize mask 到 34×62
        masks12_small = np.stack(
            [
                cv2.resize(
                    m.astype(np.uint8),
                    self.target_size[::-1],  # cv2 需要 (W, H)
                    interpolation=cv2.INTER_NEAREST,
                )
                for m in masks12
            ],
            axis=0,
        )  # (12, 34, 62)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "fps": self.fps},
                    {"type": "text",  "text": self.prompt},
                ],
            }
        ]

        sample = {
            "messages": messages,
            "target": torch.from_numpy(masks12_small).long(),  # (12, H, W)
            "file_id": file_id,
        }
        return sample


# collate_fn：把 message → processor 输入，拼 batch
def collate_fn(batch, processor):
    messages = [b["messages"] for b in batch]
    file_ids = [b["file_id"] for b in batch]
    targets  = [b["target"]  for b in batch]  # (12, 34, 62)

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    targets = torch.stack(targets, dim=0)  # (B, 12, 34, 62)
    return inputs, targets, file_ids


# ---------------------------- 训练函数 ---------------------------- #

def train_one_epoch(
        model,
        loader,
        optimizer,
        scaler,
        criterion,
        device,
        epoch: int,
        accum_iter=1,
        max_norm=1.0,
        pred_save_dir: str | None = None,
        global_step_start: int = 0,
        video_dir: str | None = None,
        save_dir: str | None = None,
        save_every_steps: int = 500,
        best_miou: float = 0.0,
        # ==== 新增：动态 LR 参数 ====
        lr_min: float = 1e-6,
        lr_max: float = 1e-3,
        lr_up_factor: float = 1.05,
        lr_down_factor: float = 0.7,
):
    """单个 epoch 训练，并在每 `save_every_steps` 步保存一次 checkpoint"""
    model.train()
    running_loss = 0

    # ===== 新增：记录步内 LR & best mIoU（训练阶段）=====
    current_lr = optimizer.param_groups[0]["lr"]
    best_miou_step = 0.0

    # 用于统计训练过程中已观察样本的 mIoU
    inter, union = 0, 0
    curve_save_path = os.path.join('..', 'Phase_1_Train_curve.png') if pred_save_dir is not None else None
    loss_list, miou_list, step_list = [], [], []

    progress = tqdm(loader, desc=f"train(E{epoch})", ncols=160)

    for step, (inputs, targets, file_ids) in enumerate(progress):
        current_global_step = global_step_start + step

        # ---------- forward / backward ---------- #
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        targets = targets.to(device, non_blocking=True)       # (B, 12, H, W)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
            _ = model.model.language_model(
                **{k: inputs[k] for k in ["input_ids", "attention_mask"]}
            )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = model(**inputs, use_cache=False, return_dict=True)
            preds  = model.model.anomaly_maps                # (B*T, 2, H, W)
            labels = targets.view(-1, *targets.shape[2:])    # (B*T, H, W)
            loss   = criterion(preds, labels) / accum_iter

        # 反向传播 & 梯度累积
        scaler.scale(loss).backward()
        if (step + 1) % accum_iter == 0:
            if max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.model.anomaly_maps_decoder.parameters(), max_norm
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accum_iter      # 还原真实 loss

        # ---------- 计算并累积 mIoU ---------- #
        preds_cls = preds.argmax(1).view_as(targets)  # (B, 12, H, W)
        inter += ((preds_cls == 1) & (targets == 1)).sum().item()
        union += ((preds_cls == 1) | (targets == 1)).sum().item()
        miou_so_far = inter / (union + 1e-8)

        # ====== 新增：根据 mIoU 动态调整学习率（逐 step）======
        if miou_so_far > best_miou_step + 1e-6:          # 有可见提升
            current_lr = min(current_lr * lr_up_factor, lr_max)
            best_miou_step = miou_so_far
        else:                                            # 无提升 → 降 LR
            current_lr = max(current_lr * lr_down_factor, lr_min)

        for g in optimizer.param_groups:
            g["lr"] = current_lr
        # ===============================================

        # ========== 更新进度条显示 ==========
        avg_loss_so_far = running_loss / (step + 1)
        progress.set_postfix(
            loss=f"{loss.item():.8f}",
            avg_loss=f"{avg_loss_so_far:.8f}",
            mIoU=f"{miou_so_far:.4f}",
            lr=f"{current_lr:.6e}",          # ==== 新增 ====
        )

        # ----------- 保存预测可视化 ----------- #
        if pred_save_dir is not None:
            B, T, H, W = targets.shape  # (B, 12, 34, 62)

            prob_maps = torch.softmax(preds, dim=1)[:, 1].view_as(targets).cpu().detach().numpy()  # (B,12,H,W)
            gts = targets.cpu().detach().numpy()  # (B,12,H,W)

            vis_orig, vis_gts, vis_preds = [], [], []
            orig_size = None  # 原始视频帧尺寸 (H', W')

            for i, fid in enumerate(file_ids):
                if video_dir is not None:
                    video_path = os.path.join(video_dir, f"{fid}.mp4")
                    frames24 = VideoMaskDataset._load_video(video_path, 24, to_rgb=False)

                    for t in range(T):
                        f1 = frames24[2*t]
                        f2 = frames24[min(2*t+1, len(frames24)-1)]
                        blend = cv2.addWeighted(f1, 0.5, f2, 0.5, 0)
                        vis_orig.append(blend)
                        if orig_size is None:
                            orig_size = (blend.shape[1], blend.shape[0])

                for t in range(T):
                    gt = (gts[i, t] * 255).astype(np.uint8)
                    gt_color = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
                    if orig_size is not None:
                        gt_color = cv2.resize(gt_color, orig_size, interpolation=cv2.INTER_NEAREST)
                    vis_gts.append(gt_color)

                    heat = (prob_maps[i, t] * 255).astype(np.uint8)
                    pred_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                    if orig_size is not None:
                        pred_color = cv2.resize(pred_color, orig_size, interpolation=cv2.INTER_LINEAR)
                    vis_preds.append(pred_color)

            row_orig = cv2.hconcat(vis_orig)   if vis_orig else None
            row_gt   = cv2.hconcat(vis_gts)
            row_pred = cv2.hconcat(vis_preds)
            rows     = [r for r in (row_orig, row_gt, row_pred) if r is not None]
            comp_img = cv2.vconcat(rows)

            save_path = os.path.join(pred_save_dir, f"step{current_global_step:06d}.png")
            cv2.imwrite(save_path, comp_img)

        # ----------- 每 `save_every_steps` 保存 checkpoint ----------- #
        if (current_global_step + 1) % save_every_steps == 0 and save_dir is not None:
            ckpt_path = os.path.join(save_dir, f"step{current_global_step + 1:06d}.pth")
            torch.save({
                "epoch": epoch,
                "global_step": current_global_step + 1,
                "state_dict": model.model.anomaly_maps_decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_miou": best_miou,
                "scaler": scaler.state_dict() if scaler is not None else None,
            }, ckpt_path)
            # 将最新 checkpoint 复制/链接为 latest.pth，方便快速载入
            latest_link = os.path.join(save_dir, "latest.pth")
            try:
                if os.path.islink(latest_link) or os.path.exists(latest_link):
                    os.remove(latest_link)
                os.symlink(os.path.basename(ckpt_path), latest_link)
            except Exception:
                # Windows 下无符号链接权限时复制文件
                import shutil
                shutil.copy2(ckpt_path, latest_link)

        # 累加loss和mIoU
        loss_list.append(loss.item())
        miou_list.append(miou_so_far)
        step_list.append(current_global_step)

        if curve_save_path is not None:
            plt.figure(figsize=(10, 4))
            ax1 = plt.gca()
            ax1.plot(step_list, loss_list, label='Loss', color='red')
            ax1.set_xlabel('Step')
            ax1.set_yscale('log')
            ax1.set_ylabel('Loss')
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx()
            ax2.plot(step_list, miou_list, label='mIoU', color='blue')  # 设置红色
            ax2.set_ylabel('mIoU')

            # 只为 ax2 设置 legend，放左下角
            ax2.legend(loc='lower left')

            plt.title('Training Curve: Loss & mIoU')
            plt.tight_layout()
            plt.savefig(curve_save_path)
            plt.close()


    progress.close()
    avg_loss = running_loss / len(loader)
    next_global_step = global_step_start + len(loader)
    return avg_loss, next_global_step


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    inter, union = 0, 0

    progress = tqdm(loader, desc="val", ncols=120)

    for inputs, targets, _ in progress:
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        targets = targets.to(device, non_blocking=True)  # (B, 12, 34, 62)
        _ = model(**inputs, use_cache=False, return_dict=True)
        preds = model.model.anomaly_maps.argmax(1).view_as(targets)  # (B, 12, H, W)

        inter += ((preds == 1) & (targets == 1)).sum().item()
        union += ((preds == 1) | (targets == 1)).sum().item()

        current_miou = inter / (union + 1e-8)
        progress.set_postfix(mIoU=f"{current_miou:.4f}")

    progress.close()
    return inter / (union + 1e-8)


# ---------------------------- Checkpoint Resume ---------------------------- #

def load_latest_checkpoint(save_dir: str, model, optimizer, scaler):
    """自动加载 save_dir 中最新的 step checkpoint。如果没有，则返回初始状态"""
    # 先尝试 latest.pth（符号链接或复制）
    latest_link = os.path.join(save_dir, "latest.pth")
    if os.path.exists(latest_link):
        ckpt_path = latest_link
    else:
        ckpts = glob.glob(os.path.join(save_dir, "step*.pth"))
        if not ckpts:
            return 0, 1, 0.0  # global_step, start_epoch, best_miou
        ckpt_path = max(ckpts, key=lambda p: int(re.findall(r"step(\d+).pth", p)[0]))

    print(f"[Checkpoint] Loading from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.model.anomaly_maps_decoder.load_state_dict(ckpt["state_dict"])
    # optimizer.load_state_dict(ckpt["optimizer"])

    if scaler is not None and ckpt.get("scaler") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except ValueError:
            pass  # 忽略不匹配的 scaler

    global_step = ckpt.get("global_step", 0)
    start_epoch = ckpt.get("epoch", 1)
    best_miou = ckpt.get("best_miou", 0.0)

    print(f"[Checkpoint] Resumed at epoch {start_epoch}, global_step {global_step}, best_miou {best_miou:.5%}")
    return global_step, start_epoch, best_miou


# ---------------------------- 主函数 ---------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="./dataset/train")
    parser.add_argument("--val_dir",   default="./dataset/val")
    parser.add_argument("--gt_dir",    default="./dataset/gt")
    parser.add_argument("--epochs",    type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr",   type=float, default=1e-5)   # 保留原始初始 LR
    # ==== 新增：动态 LR 的上下限与调节系数 ====
    parser.add_argument("--lr_min",         type=float, default=1e-8, help="动态调整时的最小学习率下界")
    parser.add_argument("--lr_max",         type=float, default=1e-3, help="动态调整时的最大学习率上界")
    parser.add_argument("--lr_up_factor",   type=float, default=1.001, help="mIoU 提升时，LR 乘以该系数")
    parser.add_argument("--lr_down_factor", type=float, default=0.995, help="mIoU 未提升时，LR 乘以该系数（<1）")
    parser.add_argument("--accum_iter", type=int, default=1)
    parser.add_argument("--save_dir",   default="./ckpts")
    parser.add_argument("--pred_save_dir", default="./preds")
    parser.add_argument("--save_every_steps", type=int, default=100, help="每多少 step 保存一次 checkpoint")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    if args.pred_save_dir is not None:
        os.makedirs(args.pred_save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 模型 & 处理器 ---------- #
    model = Anomaly_Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "../Geo/Anomaly_Qwen2.5-VL-7B-Instruct",
        torch_dtype="auto",
        device_map="auto",
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(
        "../Geo/Anomaly_Qwen2.5-VL-7B-Instruct"
    )

    # 只训练 anomaly_maps_decoder
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.model.anomaly_maps_decoder.parameters():
        p.requires_grad_(True)

    # ---------- 数据 ---------- #
    train_set = VideoMaskDataset(
        args.train_dir,
        args.gt_dir,
        processor=processor,
    )
    val_set = VideoMaskDataset(
        args.val_dir,
        args.gt_dir,
        processor=processor,
    )

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

    # ---------- 训练组件 ---------- #
    optimizer = optim.AdamW(
        model.model.anomaly_maps_decoder.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=False)

    # ---------- 尝试恢复 ---------- #
    global_step, start_epoch, best_miou = load_latest_checkpoint(args.save_dir, model, optimizer, scaler)

    # ----------- 训练循环 ----------- #
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
            epoch=epoch,
            accum_iter=args.accum_iter,
            pred_save_dir=args.pred_save_dir,
            global_step_start=global_step,
            video_dir=args.train_dir,
            save_dir=args.save_dir,
            save_every_steps=args.save_every_steps,
            best_miou=best_miou,
            # ==== 新增：把 CLI 参数透传给 train_one_epoch ====
            lr_min=args.lr_min,
            lr_max=args.lr_max,
            lr_up_factor=args.lr_up_factor,
            lr_down_factor=args.lr_down_factor,
        )

        print(f"train loss: {train_loss:.5f}")

        miou = evaluate(model, val_loader, device)
        print(f"val  mIoU: {miou:.5%}")

        # 保存以 epoch 命名的 checkpoint
        ckpt_path = os.path.join(args.save_dir, f"epoch{epoch:02d}_miou{miou:.4f}.pth")
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "state_dict": model.model.anomaly_maps_decoder.state_dict(),
                "miou": miou,
                "optimizer": optimizer.state_dict(),
                "best_miou": best_miou,
                "scaler": scaler.state_dict() if scaler is not None else None,
            },
            ckpt_path,
        )

        # 更新 best
        if miou > best_miou:
            best_miou = miou
            best_path = os.path.join(args.save_dir, "best.pth")
            torch.save(model.model.anomaly_maps_decoder.state_dict(), best_path)
            print(f"** New best mIoU! model saved to {best_path}")

    print("Training finished!")


if __name__ == "__main__":
    main()
