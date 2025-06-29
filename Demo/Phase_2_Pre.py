#!/usr/bin/env python
# coding: utf-8
"""
Track anomaly targets in binary mask sequences (.npy files) and report their
9-grid region WITH integer pixel coordinates, plus the mask resolution.

输出示例
--------
{
  "01_0014-168_191.npy": {
    "size": [256, 192],          # ← 新增字段 (H, W)
    "frame number": 24,
    "object number": 1,
    "object 1": [
      "left_(42,85)",
      "left_(44,86)",
      ...
      "middle_(128,96)"
    ]
  },
  ...
}
"""

import argparse
import glob
import json
import os
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.ndimage import center_of_mass, label
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 9-宫格辅助
# -----------------------------------------------------------------------------
REGION_NAMES = (
    "top-left", "top", "top-right",
    "left", "middle", "right",
    "bottom-left", "bottom", "bottom-right",
)


def _centroid_region(y: float, x: float, h: int, w: int) -> str:
    """判定 (y, x) 所在的九宫格区域名。"""
    row = int(3 * y / h)  # 0, 1, 2
    col = int(3 * x / w)  # 0, 1, 2
    idx = row * 3 + col
    return REGION_NAMES[idx]


# -----------------------------------------------------------------------------
# 目标追踪
# -----------------------------------------------------------------------------

def _connected_components(frame: np.ndarray) -> Tuple[np.ndarray, int]:
    """对二值帧做连通域标记，返回 (labeled, n_components)。"""
    structure = np.ones((3, 3), dtype=int)  # 8 连通
    labeled, nlab = label(frame, structure=structure)
    return labeled, nlab


class Tracker:
    """基于质心距离的简单多目标跟踪器。"""

    def __init__(self, dist_thresh: float = 30.0):
        self.next_id: int = 1
        self.dist_thresh = dist_thresh
        self.tracks: Dict[int, List[Optional[str]]] = defaultdict(list)  # id → 序列
        self._prev_centroids: Dict[int, Tuple[float, float]] = {}

    # .................................................................
    def update(self,
               centroids: List[Tuple[float, float]],
               labels:    List[str]):
        """
        按最近邻分配 *centroids* 到已有轨迹或新轨迹。

        labels 已包含 "region_(x,y)"。
        """
        if not self._prev_centroids:
            # 第一帧：每个质心开启一条轨迹
            for c, lab in zip(centroids, labels):
                tid = self._new_track()
                self._prev_centroids[tid] = c
                self.tracks[tid].append(lab)
            return

        # 计算代价矩阵（欧氏距离）
        prev_ids = list(self._prev_centroids.keys())
        prev_pts = np.array([self._prev_centroids[i] for i in prev_ids])
        curr_pts = np.array(centroids) if centroids else np.empty((0, 2))

        if curr_pts.size and prev_pts.size:
            dists = np.linalg.norm(prev_pts[:, None, :] - curr_pts[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(dists)
        else:
            dists = np.empty((0, 0))
            row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)

        matched_prev, matched_curr = set(), set()

        # 距离阈值内的配对
        for r, c in zip(row_ind, col_ind):
            if dists[r, c] <= self.dist_thresh:
                tid = prev_ids[r]
                self._prev_centroids[tid] = centroids[c]
                self.tracks[tid].append(labels[c])
                matched_prev.add(r)
                matched_curr.add(c)

        # 前一帧未匹配轨迹 → 本帧缺失
        for r, tid in enumerate(prev_ids):
            if r not in matched_prev:
                self.tracks[tid].append(None)

        # 本帧未匹配质心 → 新轨迹
        for c, (cent, lab) in enumerate(zip(centroids, labels)):
            if c not in matched_curr:
                tid = self._new_track()
                self._prev_centroids[tid] = cent
                existing_len = len(next(iter(self.tracks.values())))
                self.tracks[tid].extend([None] * (existing_len - 1))
                self.tracks[tid].append(lab)

    # .................................................................
    def _new_track(self) -> int:
        tid = self.next_id
        self.next_id += 1
        return tid

    # .................................................................
    def finalize(self, total_frames: int):
        """所有轨迹补足到 *total_frames* 长度。"""
        for lst in self.tracks.values():
            lst.extend([None] * (total_frames - len(lst)))


# -----------------------------------------------------------------------------
# 主处理函数
# -----------------------------------------------------------------------------

def process_mask_array(mask: np.ndarray) -> OrderedDict:
    """对 (T,H,W) 掩码跟踪并返回结果 dict。"""
    T, H, W = mask.shape
    tracker = Tracker(dist_thresh=max(H, W) / 10)  # 距离阈值 = 10% 较大边

    for t in range(T):
        labeled, nlab = _connected_components(mask[t])
        if nlab == 0:
            # 无目标帧：所有活动轨迹补 None
            for v in tracker.tracks.values():
                v.append(None)
            continue

        centroids = center_of_mass(np.ones_like(labeled), labeled, range(1, nlab + 1))
        centroids = [(y, x) for y, x in centroids]

        regions = [_centroid_region(y, x, H, W) for y, x in centroids]

        labels = [
            f"{reg}_({int(x)},{int(y)})"
            for (y, x), reg in zip(centroids, regions)
        ]

        tracker.update(centroids, labels)

    tracker.finalize(T)

    # 组装结果
    result = OrderedDict()
    result["size"] = [int(H), int(W)]          # ← 新增
    result["frame number"] = T
    result["object number"] = len(tracker.tracks)
    for i, (_, positions) in enumerate(sorted(tracker.tracks.items()), 1):
        result[f"object {i}"] = positions
    return result


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Track anomaly targets in *.npy mask files.")
    parser.add_argument("--directory",
                        default="dataset/gt",
                        help="Directory containing *.npy files (default: dataset/gt)")
    parser.add_argument("--output",
                        default="dataset/track.json",
                        help="Path to write JSON results (default: dataset/track.json)")
    args = parser.parse_args()

    npy_files = sorted(glob.glob(os.path.join(args.directory, "*.npy")))
    if not npy_files:
        raise FileNotFoundError("No *.npy files found in the specified directory.")

    all_results = OrderedDict()

    for path in tqdm(npy_files, desc="Processing"):
        name = os.path.basename(path)
        masks = np.load(path)
        if masks.ndim != 3:
            raise ValueError(f"{name} is not a (T,H,W) array, got shape {masks.shape}")
        all_results[name] = process_mask_array(masks)

    # 输出
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(all_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
