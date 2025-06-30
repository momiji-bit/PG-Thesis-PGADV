# TGADV: Text-guided Anomaly Detection in Videos

<p align="center">
  <a href="https://arxiv.org/abs/" target="_blank"><img src="https://img.shields.io/badge/arXiv-Upcoming-red?logo=arxiv"></a>
  <a href="https://huggingface.co/datasets/Geo2425/ShanghaiTech_Campus" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange"></a>
  <a href="https://huggingface.co/Geo2425/Anomaly_Qwen2.5-VL-7B-Instruct" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange"></a>
  <a href="https://drive.google.com/drive/folders/1cIISTK_XLcwCBgUw9wfyF8ABrh26nZuV?usp=sharing" target="_blank"><img src="https://img.shields.io/badge/Google%20Drive-Decoder-blue?logo=google-drive"></a>
</p>



We propose an anomaly detection framework based on the Qwen2.5-VL-7B-Instruct multimodal model. Initially, the visual encoder and text embedding components of Qwen2.5-VL-7B-Instruct are utilized to extract features from anomalous videos and corresponding textual prompts indicating 'normal' or 'abnormal' conditions. Subsequently, our custom anomaly decoder aligns these visual and textual features, producing pixel-level anomaly heatmaps for each video frame. These heatmaps are then encoded to pinpoint anomaly regions across predefined spatial positions—top, bottom, left, right, center, top-left, top-right, bottom-left, bottom-right—and to capture a global context for identifying the presence of anomalies within each frame. Furthermore, a dedicated learnable token is introduced to facilitate fine-tuning, enabling dimensional alignment between encoded heatmap representations and the large language model (LLM). Ultimately, leveraging a generated natural language dataset, we fine-tune the base LLM using Low-Rank Adaptation (LoRA), enhancing its capability for anomaly detection in video content.



## 🚧 Development Progress

- [x] **Phase 1:** Anomaly video decoding → spatiotemporal heatmap generation
- [x] **Phase 2:** Learnable prompts module construction; Fine-tuning dataset generation (video + description pairs); LoRA fine-tuning on Qwen2.5-VL
- [ ] **Phase 3:** Inference, evaluation, and visualization
- [ ] **Phase 4:** Deployment (demo interface + Hugging Face integration)



## 📦 1. Installation

```bash
git clone https://github.com/momiji-bit/PG-Thesis-TGADV.git
cd PG-Thesis-TGADV

```

```bash
conda create -n TGADV python=3.12
conda activate TGADV
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.52.4 accelerate qwen-vl-utils[decord] opencv-python matplotlib chardet ipywidgets

```



## 📂 2. Data preparation

### 2.1 Pretrained Model

```bash
pip install huggingface_hub
mkdir Geo/Anomaly_Qwen2.5-VL-7B-Instruct
huggingface-cli login
huggingface-cli download Geo2425/Anomaly_Qwen2.5-VL-7B-Instruct --local-dir Geo/Anomaly_Qwen2.5-VL-7B-Instruct

```

### 2.2 Heatmap Decoder

```bash
pip install gdown
mkdir ckpts
gdown --fuzzy https://drive.google.com/file/d/1nWqTMzWLorg2DX7WX51Czhe4ayY_MHC8/view?usp=sharing -O ckpts/step006800.pth

```

### 2.3 ShanghaiTech Campus Dataset

```bash
huggingface-cli download Geo2425/ShanghaiTech_Campus --local-dir dataset --repo-type dataset
unzip dataset/train.zip -d dataset/train
unzip dataset/val.zip -d dataset/val
unzip dataset/masked.zip -d dataset/masked
unzip dataset/gt.zip -d dataset/gt
```

## ✅ 3. Demo & Debug

- **Phase 1:** Please run `Phase_1_Demo.ipynb` to reproduce the **Anomaly video heatmap Decoder** experimental results.
- **🚧Phase 2:** Please run `Phase_2_Demo.ipynb` to reproduce the **Learnable Prompt Encoder** experimental results.
- **🚧Phase 3:**  ...
- **🚧Phase 4:**  ...



## 🏋️‍♂️ 4. Training & Testing





## 📧 5. Contact

For any questions, feel free to contact: Mr. Jihao Gu (jihao.gu.23@ucl.ac.uk).
