import json
import ast
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 用于进度条

# ---------------- 配置参数 ----------------
INPUT_FILENAME = "../dataset/track.json"  # 输入文件名
OUTPUT_FILENAME = "../dataset/dataset_track.json"  # 输出文件名
DEEPSEEK_API_KEY = "sk-4aa22d364c584955a739ab25a4ba2078"  # 替换为你的 DeepSeek API 密钥
DEEPSEEK_MODEL = "deepseek-chat"            # 使用的模型
API_URL = "https://api.deepseek.com/v1/chat/completions"

MAX_WORKERS = 16   # 并发线程数量，可根据网络 / CPU 适当调整
MAX_RETRY = 10     # 单条请求最大重试次数
# -----------------------------------------


def call_deepseek_api(prompt: str) -> str:
    """调用 DeepSeek API 并返回响应内容（出现异常会抛出）"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2000
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def build_prompt(value: dict) -> str:
    """根据目标信息构造提示词"""
    base_prompt = (
        "请根据目标在每一帧的九宫格位置（左上、上、右上、左、中、右、左下、下、右下）以及坐标序列，为每个目标生成3句多样化的自然语言描述。每句应包含：\n"
        "- 目标出现的帧区间（如第1帧到第24帧）。\n"
        "- 目标出现过的所有区域（按顺序），如“从左侧移动到中间”或“一直停留在中间区域”等。\n"
        "- 如果目标主要停留在一个区域内，请结合坐标变化描述为“小幅度移动”、“几乎没有移动”或“基本静止”等。\n"
        "- 如果目标跨越多个区域，请突出“跨区域移动”、“从左侧快速移动到右侧”等大幅度运动特征。\n"
        "- 最终输出不要有方括号，每个目标输出3种不同表述但意思相近的句子，格式为列表。\n"
        "- 小幅度移动例句：“目标在左上区域内小幅度向下移动”，“该物体一直停留在中间区域，位置变化不大”，“目标基本静止在右下角区域”。\n"
        "- 不要出现原始坐标信息，通过自然语言描述移动。(0, 0)表示视频图像左上角。原始视频size会在下面的数据中。\n"
        "输出格式（python列表）：\n"
        "['目标1从第1帧到第24帧，最初位于左侧区域，随后小幅度移动到中间，整体运动较为平缓。','该目标1出现在1到24帧，主要活动于左侧，后期逐步进入中间区域，移动速度较慢。','在第1到24帧之间，目标1从左区缓慢移动至中区，运动范围不大。']\n"
        "或有多个目标，一句话中先说第一个目标，再说第二个目标，仍旧是三句话组成python列表。\n\n"
    )
    return base_prompt + str(value)


def process_single_item(item):
    """处理单条数据，返回 (key, processed_value)"""
    key, value = item
    # 若无目标，直接返回空列表
    if value.get("object number", 0) == 0:
        return key, []

    prompt = build_prompt(value)
    last_err = None
    for _ in range(MAX_RETRY):
        try:
            result = call_deepseek_api(prompt)
            processed_value = ast.literal_eval(result)
            if isinstance(processed_value, list):
                return key, processed_value
        except Exception as e:
            last_err = e  # 记录最后一次异常，继续重试
    # 达到最大重试次数仍失败
    print(f"【警告】{key} 连续{MAX_RETRY}次API结果格式不正确，已跳过。 错误: {last_err}")
    return key, []


def process_data():
    """主函数：加载数据并并发调用 API 处理，并实时写入 JSON"""
    # 1. 读取输入文件
    with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_items = len(data)
    print(f"开始处理 {total_items} 条数据... (并发 {MAX_WORKERS} 线程)")

    # 2. 并发处理
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_item, item): item[0] for item in data.items()}
        for future in tqdm(as_completed(futures), total=total_items, desc="处理进度"):
            key, processed_value = future.result()
            results[key] = processed_value

            # —— 实时写入 JSON ——
            # 主线程顺序写入，避免并发写文件产生冲突
            with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"处理完成! 最终结果已保存至: {OUTPUT_FILENAME}")


if __name__ == "__main__":
    process_data()