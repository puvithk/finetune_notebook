# 🚚 Proof of Delivery (POD) VLM Pipeline

Comprehensive repository containing fine-tuning and inference pipelines for Proof of Delivery (POD) document extraction and categorization using Vision-Language Models (Qwen3-VL-8B & Qwen2.5-VL-7B).

---

## 📑 Table of Contents
1. [Inference Guide (`inference.ipynb`)](#-1-inference-guide-inferenceipynb)
2. [Fine-Tuning: Official Hugging Face (`fine-tuning-official.ipynb`)](#-2-fine-tuning-official-hugging-face-fine-tuning-officialipynb)
3. [Fine-Tuning: Unsloth Stack (`fine-tuning-using-unsloth.ipynb`)](#-3-fine-tuning-unsloth-stack-fine-tuning-using-unslothipynb)
4. [Dataset & Expected JSON Output Schema](#-4-dataset--output-schema)
5. [Visual Workflow Architecture](#-5-visual-workflow-architecture)

---

## 🔍 1. Inference Guide (`inference.ipynb`)

[`inference.ipynb`](file:///C:/Users/puvit/Downloads/Infereance%20Test/inference.ipynb) is a flexible, single-entrypoint notebook designed to test both base and fine-tuned models across both runtime stacks.

### 🎛️ Key Configuration Toggles (Cell 1)
```python
# Select Pipeline: "unsloth" or "official"
USE_PIPELINE = "unsloth"

# Toggle fine-tuned LoRA adapter vs original base model
USE_FINETUNED = True

# Dataset source (Supports attached Kaggle datasets or auto-download via API)
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/puvithk/pod-clasification-v2"

# LoRA Weights / Checkpoint repository
LORA_PATH = "puvith/qwen3-8b-pod-analyzer-v3"
```

### ✨ Features
* **Zero-Setup Kaggle API Fallback**: If the dataset isn't attached directly to `/kaggle/input/`, the notebook automatically pulls and unzips it via `kaggle datasets download`.
* **Side-by-Side Visual Inspection**: Generates matplotlib visualizations displaying the POD image alongside the extracted structured JSON.
* **Unified API**: Exposes a single `predict_pod(image_input)` interface whether you use Unsloth or Official Transformers.

---

## 🚀 2. Fine-Tuning: Official Hugging Face (`fine-tuning-official.ipynb`)

[`fine-tuning-official.ipynb`](file:///C:/Users/puvit/Downloads/Infereance%20Test/fine-tuning-official.ipynb) trains **Qwen/Qwen3-VL-8B-Instruct** using standard, upstream Hugging Face libraries without proprietary wrappers.

### 🛠️ Tech Stack
* `transformers >= 4.57.0`
* `peft` (QLoRA, `r=16`, `lora_alpha=32`, language projection targets)
* `trl` (`SFTTrainer`)
* `bitsandbytes` (4-bit NF4 Quantization)

### ⚙️ Key Details
* **Resolution Control**: Explicit pixel budget constraints (`MIN_PIXELS = 128*128`, `MAX_PIXELS = 512*512`) to prevent GPU OOM on single 16GB T4 instances.
* **Custom Collator**: Custom `QwenVLDataCollator` that performs label masking (calculating cross-entropy loss exclusively on assistant response tokens).
* **Fix for VLM Loss**: Custom loss routing override to prevent chunked cross-entropy errors on `Qwen3VLCausalLMOutputWithPast`.

---

## ⚡ 3. Fine-Tuning: Unsloth Stack (`fine-tuning-using-unsloth.ipynb`)

[`fine-tuning-using-unsloth.ipynb`](file:///C:/Users/puvit/Downloads/Infereance%20Test/fine-tuning-using-unsloth.ipynb) trains **Qwen2.5-VL-7B-Instruct** with optimized memory and compute performance using Unsloth.

### 🛠️ Tech Stack
* `unsloth` (`FastVisionModel`, `UnslothVisionDataCollator`)
* `qwen-vl-utils`
* `trl` (`SFTTrainer`)

### ⚙️ Key Details
* **2x Faster & Lower VRAM**: Uses pre-quantized 4-bit base weights (`unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit`).
* **Full Multimodal Training**: Trains vision encoder layers (`finetune_vision_layers=True`), language decoder, MLP, and attention heads simultaneously.
* **Export Options**: Export as lightweight LoRA adapter (~200MB), merged 16-bit model (~14GB), or GGUF.

---

## 📊 4. Dataset & Output Schema

### 📋 Classification Priority Rules
1. **Physical Paper Damage** (torn, ripped, missing sections) $\rightarrow$ `MANUAL_CHECK_REQUIRED`
2. **Damage + Short** mentioned in remarks $\rightarrow$ `ISSUE_POD_DAMAGED_AND_SHORT`
3. **Damage** mentioned in remarks $\rightarrow$ `ISSUE_POD_DAMAGED`
4. **Shortage** mentioned in remarks $\rightarrow$ `ISSUE_POD_SHORT`
5. **Seal/Stamp + Signature** present $\rightarrow$ `CLEAN_POD_SEAL_AND_SIGNATURE`
6. **Seal/Stamp Only** $\rightarrow$ `CLEAN_POD_ONLY_SEAL`
7. **Signature Only** $\rightarrow$ `CLEAN_POD_ONLY_SIGNATURE`
8. **Neither Seal nor Signature** $\rightarrow$ `NO_SIGNATURE_NO_STAMP`

### 📄 Target JSON Structure
```json
{
  "cnNumber": "531264008095",
  "hasSignature": true,
  "hasStamp": true,
  "hasHandwriting": false,
  "imageQualityPassed": true,
  "remarksText": null,
  "deliveryDate": "2026-07-24",
  "categoryReason": "The POD contains both a valid recipient signature and company stamp.",
  "confidenceScore": 0.95,
  "podCategory": "CLEAN_POD_SEAL_AND_SIGNATURE",
  "limit_exceed": false
}
```

---

## 📐 5. Visual Workflow Architecture

* [Official HF Fine-Tuning Diagram](file:///C:/Users/puvit/Downloads/Infereance%20Test/official-finetuning-pipeline.svg)
* [Unsloth Fine-Tuning Diagram](file:///C:/Users/puvit/Downloads/Infereance%20Test/unsloth-finetuning-pipeline.svg)

---

## 🚀 Quick Start on Kaggle

1. Select **GPU T4 x2** accelerator in your Kaggle notebook settings.
2. Enable **Internet Access** in notebook settings.
3. For fine-tuning, run either [`fine-tuning-official.ipynb`](file:///C:/Users/puvit/Downloads/Infereance%20Test/fine-tuning-official.ipynb) or [`fine-tuning-using-unsloth.ipynb`](file:///C:/Users/puvit/Downloads/Infereance%20Test/fine-tuning-using-unsloth.ipynb).
4. Run [`inference.ipynb`](file:///C:/Users/puvit/Downloads/Infereance%20Test/inference.ipynb) to validate predictions and benchmark accuracy against your dataset.
