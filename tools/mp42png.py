import cv2
import os
import numpy as np
from PIL import Image

def export_video_frames_as_png(video_path, save_path='output.png', resize=None, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        count += 1
        if count < 12:
            continue
        if not ret:
            break
        if resize:
            frame = cv2.resize(frame, resize)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and count >= max_frames:
            break

    cap.release()

    if not frames:
        raise ValueError("未能读取到任何帧")

    # 横向拼接
    concat_image = np.hstack(frames)  # shape: (H, W * T, 3)

    # 导出为PNG
    Image.fromarray(concat_image).save(save_path)
    print(f"保存成功: {save_path}")

# 示例调用
video_path = "masked_black.mp4"  # 替换为你的MP4路径
export_video_frames_as_png(video_path, save_path="../all_frames.png")
