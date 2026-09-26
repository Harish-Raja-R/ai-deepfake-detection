# 🎯 Final Dataset Selection: Fast-Track Academic Mini-Project

**Document ID:** `FINAL_DATASET_SELECTION.md`  
**Phase:** Phase 2A Fast-Track Dataset Strategy  
**Target:** 20-Mark Academic Mini-Project on a Local Windows Laptop  
**Status:** Dataset Formally Selected — Awaiting User Download Approval  

---

## 1. Selected Dataset Overview

To complete the academic mini-project locally today within the laptop's disk (~1.89 GB – 3.0 GB free) and compute constraints, the project has formally selected:

### **`garystafford/deepfake-audio-detection`** (Hugging Face)

* **Exact Dataset Name:** `garystafford/deepfake-audio-detection`
* **Source:** Hugging Face ([https://huggingface.co/datasets/garystafford/deepfake-audio-detection](https://huggingface.co/datasets/garystafford/deepfake-audio-detection))
* **Download Size:** **~574 MB** (compressed Parquet / audio bytes) / **~1.16 GB** raw extracted FLAC
* **Total Audio Files:** **1,866** audio files
* **REAL (Bonafide) Count:** **933** authentic human speech utterances
* **FAKE (Deepfake) Count:** **933** AI-synthesized speech utterances
* **Audio Format:** Lossless **FLAC**
* **Sampling Rate:** **16,000 Hz (16 kHz)**, mono, 16-bit
* **Duration:** Clean segmented speech chunks ranging from 2.5 to 13.0 seconds
* **Speaker / Generator Information:** Authentic speech from human speakers vs. synthetic speech generated across **6 distinct text-to-speech platforms** (including modern cloning leaders such as **ElevenLabs**, **Amazon Polly**, and **Hume AI**)
* **Train / Test Split:** Pre-configured for a standard **70% Training / 15% Validation / 15% Testing** stratified split (or reproducible seeded partition using our `src.dataset.split_dataset(seed=42)`)
* **License & Access:** **Apache 2.0** (Open Access, public HTTP download without authentication walls, API tokens, or login barriers)

---

## 2. Why This Dataset is the Optimal Choice for This Project

| Selection Criteria | Project Requirement | How `garystafford/deepfake-audio-detection` Fulfills It |
| :--- | :--- | :--- |
| **Download Size** | Strict limit: < 1 GB | **574 MB download**, consuming minimal bandwidth and fitting comfortably on the laptop drive. |
| **Sample Count** | Target: 500 – 5,000 samples | **1,866 files** — large enough for statistical significance and academic credibility, small enough for rapid feature extraction (< 5 minutes on CPU). |
| **Class Balance** | Prevent accuracy bias | **Exactly 50% Real / 50% Fake (933 vs 933)**, avoiding the 1:9 imbalance of raw ASVspoof. |
| **Acoustic Consistency** | Standardized 16 kHz audio | **100% native 16 kHz mono FLAC**, perfectly aligning with `AudioConfig(sample_rate=16000)` without resampling loss. |
| **Spectrogram & CNN Suitability** | High-fidelity spectral artifacts | Preserves vocoder and phase artifacts up to the 8 kHz Nyquist boundary, ideal for 128-band Mel-spectrogram 2D CNNs. |
| **Contemporary Academic Value** | Impressive 20-mark evaluation | Features modern generative AI voice cloning (**ElevenLabs**) rather than outdated 2019 concatenative vocoders. |
| **Access Ease** | Fast-track implementation | **Zero friction**: No Kaggle API credentials, credit cards, or institutional agreements required. |

---

## 3. Exact Ingestion Directory Architecture

When downloaded, the dataset will be placed into the existing raw data directory structure without modifying project conventions:

```text
AI_VOICE_DEEPFAKE_DETECTION/data/raw/
├── real/
│   ├── real_0001.flac
│   ├── ...
│   └── real_0933.flac
└── fake/
    ├── fake_0001.flac
    ├── ...
    └── fake_0933.flac
```

This layout is automatically and natively discovered by `src.dataset.scan_dataset()`.

---

## 4. Exact Download Instructions

The dataset can be ingested using either of the two methods below once approved:

### Method 1: Automated Script Ingestion (Recommended & Fastest)
A lightweight helper script will fetch the public Parquet file directly from Hugging Face Hub and unpack the 1,866 FLAC audio files into `data/raw/real/` and `data/raw/fake/` in under 2 minutes:

```bash
python -c "
import pandas as pd, io, soundfile as sf
from pathlib import Path

out_real = Path('data/raw/real')
out_fake = Path('data/raw/fake')
out_real.mkdir(parents=True, exist_ok=True)
out_fake.mkdir(parents=True, exist_ok=True)

url = 'https://huggingface.co/datasets/garystafford/deepfake-audio-detection/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet'
print('Downloading dataset parquet from Hugging Face...')
df = pd.read_parquet(url)
print(f'Loaded {len(df)} samples. Extracting FLAC audio files to data/raw/...')

for idx, row in df.iterrows():
    audio_bytes = row['audio']['bytes']
    label = row['label'] # 0: real, 1: fake
    target_dir = out_real if label == 0 else out_fake
    prefix = 'real' if label == 0 else 'fake'
    file_path = target_dir / f'{prefix}_{idx:04d}.flac'
    with open(file_path, 'wb') as f:
        f.write(audio_bytes)

print('Done! 933 REAL and 933 FAKE audio files extracted successfully.')
"
```

### Method 2: Via Hugging Face `datasets` Library
```bash
python -c "from datasets import load_dataset; ds = load_dataset('garystafford/deepfake-audio-detection')"
```

---

## 5. Next Steps for Today's Workflow

1. **User Approval:** User reviews and confirms proceeding with `garystafford/deepfake-audio-detection`.
2. **Phase 2 Ingestion & EDA:** Run ingestion into `data/raw/`, verify file hashes and sample rates, export `dataset_summary.csv`, and render the 7 required EDA plots in `results/figures/dataset/`.
3. **Phase 3 Feature Extraction & Training:** Extract 128-band Mel-spectrograms and train the baseline model (Random Forest / MLP) and the improved 2D CNN model.
4. **Phase 4 Evaluation & Reporting:** Generate ROC curves, confusion matrices, and final metrics (Accuracy, ROC-AUC, EER).
5. **Phase 5 Web App Verification:** Launch Streamlit UI for interactive demonstration.
