# Phase 1: Anomaly video heatmap Decoder

<p align="center">
  <a href="https://arxiv.org/abs/" target="_blank"><img src="https://img.shields.io/badge/arXiv-Upcoming-red?logo=arxiv"></a>
  <a href="https://huggingface.co/datasets/Geo2425/ShanghaiTech_Campus" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange"></a>
  <a href="https://huggingface.co/Geo2425/Anomaly_Qwen2.5-VL-7B-Instruct" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange"></a>
  <a href="https://drive.google.com/drive/folders/1cIISTK_XLcwCBgUw9wfyF8ABrh26nZuV?usp=sharing" target="_blank"><img src="https://img.shields.io/badge/Google%20Drive-Decoder
     -blue?logo=google-drive"></a>
</p>



## 📦 1. Installation

```bash
git clone https://github.com/momiji-bit/PG-Thesis-TGADV.git
cd PG-Thesis-TGADV

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

### 2.3 ShanghaiTech Campus Dataset (Optional) 

```bash
mkdir dataset
huggingface-cli download Geo2425/ShanghaiTech_Campus --local-dir dataset --repo-type dataset
unzip 
```



## ✅ 3. Demo & Debug



## 📧 4. Contact

For any questions, feel free to contact: Mr. Jihao Gu (jihao.gu.23@ucl.ac.uk).
