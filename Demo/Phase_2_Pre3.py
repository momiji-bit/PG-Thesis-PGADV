import json
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

INPUT_FILE = "../dataset/dataset_track.json"
OUTPUT_FILE = "../dataset/dataset_track_zh_en.json"
DEEPSEEK_API_KEY = "sk-4aa22d364c584955a739ab25a4ba2078"
DEEPSEEK_MODEL = "deepseek-chat"
API_URL = "https://api.deepseek.com/v1/chat/completions"
THREADS = 6  # 可根据你的带宽/接口QPS限制调整

write_lock = Lock()

def translate(text, retry=3):
    prompt = f"请将以下文本翻译为英文，不要有任何注释，只输出英文：\n{text}"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 2048
    }
    for i in range(retry):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"翻译失败，第{i+1}次重试: {e}")
            time.sleep(2)
    return "[翻译失败]"

def translate_list(key, desc_list):
    # 跳过空列表
    if not desc_list:
        return key, desc_list
    # 检查最后一个是否为英文（ASCII），避免重复
    if desc_list and all(ord(c) < 128 for c in desc_list[-1]):
        print(f"{key} 已有英文翻译，跳过")
        return key, desc_list
    new_list = list(desc_list)
    for text in desc_list:
        print(f"{key} 正在翻译: {text}")
        en_text = translate(text)
        print(f"{key} 翻译结果: {en_text}")
        new_list.append(en_text)
    return key, new_list

def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 统计所有需要处理的任务
    tasks = [(k, v) for k, v in data.items() if v and not (v and all(ord(c) < 128 for c in v[-1]))]
    print(f"共需处理 {len(tasks)} 个条目（非空且未翻译）")

    # 主循环
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = []
        for key, desc_list in tasks:
            futures.append(executor.submit(translate_list, key, desc_list))

        # 实时保存：每完成一个key就写文件
        for future in as_completed(futures):
            key, new_list = future.result()
            with write_lock:
                data[key] = new_list
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"{key} 已写入文件")

if __name__ == "__main__":
    main()
