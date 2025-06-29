import itertools
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from anomaly_qwen2_5_vl import Qwen2_5_VLProcessor

# ────────────────────────────────────
# 0. 基本配置
# ────────────────────────────────────
processor = Qwen2_5_VLProcessor.from_pretrained(
    "Geo/Anomaly_Qwen2.5-VL-7B-Instruct"
)
tokenizer = processor.tokenizer

# 0.1 事件类别（中英文一一对应，位置索引保持一致）
CLASS_NAMES_EN = [
    "person running", "people fighting", "person loitering", "person jumping",
    "object being thrown", "crowd gathering", "person riding bicycle",
    "person skateboarding", "person falling down", "abandoned object",
    "vehicle on walkway", "anomaly",
]
CLASS_NAMES_ZH = [
    "有人奔跑", "多人打斗", "有人徘徊", "有人跳跃",
    "有物体被抛掷", "人群聚集", "有人骑自行车",
    "有人玩滑板", "有人摔倒", "有遗弃物",
    "有车辆在走道上", "异常事件",
]
# 事件名称根据语言返回
EVENT_NAME_BY_LANG = {"en": CLASS_NAMES_EN, "zh": CLASS_NAMES_ZH}

# 0.2 状态短语（normal / abnormal）模板
STATE_PHRASES_BY_LANG = {
    "en": [
        [  # normal = 0
            "no {} present",
            "normal campus scene without {}",
            "ordinary surveillance video free of {}",
            "calm campus walkway without {}",
            "surveillance video without {}",
            "CCTV footage free of {}",
            "scene devoid of {}",
        ],
        [  # abnormal = 1
            "{}",
            "surveillance video of {}",
            "CCTV footage showing {}",
            "{} happening on campus",
            "anomalous event: {}",
            "security camera captures {}",
            "unusual incident on campus - {}",
        ],
    ],
    "zh": [
        [  # normal
            "当前无{}发生",
            "正常的校园场景，没有{}",
            "普通的监控视频，没有{}",
            "平静的校园通道，没有{}",
            "监控视频中没有{}",
            "闭路电视画面没有{}",
            "场景中没有{}",
        ],
        [  # abnormal
            "{}",
            "监控视频中出现{}",
            "闭路电视画面显示{}",
            "校园内发生了{}",
            "异常事件：{}",
            "安全摄像头拍到{}",
            "校园异常事件 - {}",
        ],
    ],
}

# 0.3 句子级（完整描述）模板
SENT_TEMPLATES_BY_LANG = {
    "en": [
        "a surveillance video of {}.",
        "a CCTV footage of the {}.",
        "a grainy security-camera clip showing {}.",
        "a low-resolution campus video containing {}.",
        "a short surveillance clip where {} occurs.",
    ],
    "zh": [
        "一段关于{}的监控视频。",
        "一段包含{}的闭路电视画面。",
        "一段模糊的监控视频，显示{}。",
        "一段低分辨率的校园视频，包含{}。",
        "一段简短的监控片段，发生了{}。",
    ],
}


# ────────────────────────────────────
# 1. 工具函数
# ────────────────────────────────────
def encode_prompts(
        prompts: List[str], tokenizer
) -> Dict[str, torch.Tensor]:
    """将一批文本编码成张量 Dict。"""
    out = tokenizer(
        prompts, padding=True, truncation=True, return_tensors="pt"
    )
    return {k: v for k, v in out.items()}


def pad_and_cat(tensors: List[torch.Tensor], max_len: int, pad_val: int):
    """补齐到统一长度并在 dim=0 拼接。"""
    return torch.cat(
        [F.pad(t, (0, max_len - t.shape[1]), value=pad_val) for t in tensors],
        dim=0,
    )


def stack_prompts(
        prompt_dict: Dict[
            str, Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ],
        tokenizer,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    将所有 normal / abnormal 的编码结果打包成两个批次。
    prompt_dict 的值是 (normal_tensor_dict, abnormal_tensor_dict)
    """
    normal_ids, normal_attn = [], []
    ab_ids, ab_attn = [], []

    for normal_d, ab_d in prompt_dict.values():
        normal_ids.append(normal_d["input_ids"])
        normal_attn.append(normal_d["attention_mask"])
        ab_ids.append(ab_d["input_ids"])
        ab_attn.append(ab_d["attention_mask"])

    max_len = max(
        max(t.shape[1] for t in normal_ids + ab_ids),
        1,
    )
    pad_id = tokenizer.pad_token_id

    normal_batch = {
        "input_ids": pad_and_cat(normal_ids, max_len, pad_id),
        "attention_mask": pad_and_cat(normal_attn, max_len, 0),
    }
    abnormal_batch = {
        "input_ids": pad_and_cat(ab_ids, max_len, pad_id),
        "attention_mask": pad_and_cat(ab_attn, max_len, 0),
    }
    return normal_batch, abnormal_batch


# ────────────────────────────────────
# 2. 构建全部语言组合的提示
# ────────────────────────────────────
prompt_sentences = {}  # key -> (normal_dict, abnormal_dict)

# 2.1 语言组合：模板语言 × 事件名称语言
lang_pairs = [("en", "en"), ("zh", "zh"), ("zh", "en"), ("en", "zh")]

for idx_event, (evt_en, evt_zh) in enumerate(
        zip(CLASS_NAMES_EN, CLASS_NAMES_ZH)
):
    # 当前事件名按语言索引
    evt_name_by_lang = {"en": evt_en, "zh": evt_zh}

    for tmpl_lang, evt_lang in lang_pairs:
        combo_key = f"{evt_en}__tmpl-{tmpl_lang}__evt-{evt_lang}"
        normal_ab_dicts = []  # 0: normal, 1: abnormal

        for state_idx in (0, 1):  # normal / abnormal
            # 2.1 把占位符替换成具体事件名称
            state_phrases = [
                phr.format(evt_name_by_lang[evt_lang])
                for phr in STATE_PHRASES_BY_LANG[tmpl_lang][state_idx]
            ]
            # 2.2 用句子级模板拼装
            full_sents = [
                tpl.format(sp)
                for sp, tpl in itertools.product(
                    state_phrases, SENT_TEMPLATES_BY_LANG[tmpl_lang]
                )
            ]
            # 2.3 编码
            tensor_dict = encode_prompts(full_sents, tokenizer)
            normal_ab_dicts.append(tensor_dict)

        prompt_sentences[combo_key] = tuple(normal_ab_dicts)

# ────────────────────────────────────
# 3. 打包成最终批次
# ────────────────────────────────────
normal_batch, anomaly_batch = stack_prompts(prompt_sentences, tokenizer)

# 测试打印形状
print(
    "Normal batch:",
    {k: v.shape for k, v in normal_batch.items()},
    "\nAbnormal batch:",
    {k: v.shape for k, v in anomaly_batch.items()},
)

