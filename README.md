# TGADV: Text-guided Anomaly Detection in Videos

<p align="center">
  <a href="https://arxiv.org/abs/" target="_blank"><img src="https://img.shields.io/badge/arXiv-Upcoming-red?logo=arxiv"></a>
  <a href="https://huggingface.co/datasets/Geo2425/ShanghaiTech_Campus" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange"></a>
</p>
We propose an anomaly detection framework based on the Qwen2.5-VL-7B-Instruct multimodal model. Initially, the visual encoder and text embedding components of Qwen2.5-VL-7B-Instruct are utilized to extract features from anomalous videos and corresponding textual prompts indicating 'normal' or 'abnormal' conditions. Subsequently, our custom anomaly decoder aligns these visual and textual features, producing pixel-level anomaly heatmaps for each video frame. These heatmaps are then encoded to pinpoint anomaly regions across predefined spatial positions—top, bottom, left, right, center, top-left, top-right, bottom-left, bottom-right—and to capture a global context for identifying the presence of anomalies within each frame. Furthermore, a dedicated learnable token is introduced to facilitate fine-tuning, enabling dimensional alignment between encoded heatmap representations and the large language model (LLM). Ultimately, leveraging a generated natural language dataset, we fine-tune the base LLM using Low-Rank Adaptation (LoRA), enhancing its capability for anomaly detection in video content.



## 🚧 Development Progress

- [x] **Phase 1:** Anomaly video decoding → spatiotemporal heatmap generation
- [x] **Phase 2.1:** Learnable prompts module construction
- [x] **Phase 2.2:** Fine-tuning dataset generation (video + description pairs)
- [x] **Phase 2.3:** LoRA fine-tuning on Qwen2.5-VL
- [ ] **Phase 3:** Inference, evaluation, and visualization
- [ ] **Phase 4:** Deployment (demo interface + Hugging Face integration)



## 📦 1. Installation

```bash
git https://github.com/momiji-bit/PG-Thesis-TGADV.git
cd PG-Thesis-TGADV

```



## 📂 2. Data preparation

### 2.1 Pretrained Model

```
```

### 2.2 Datasets (ShanghaiTech Campus)

```
```



## 🏋️‍♂️ 3. Training & Testing





## 📧 4. Contact

For any questions, feel free to contact: Mr. Jihao Gu (jihao.gu.23@ucl.ac.uk).
