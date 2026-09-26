# ✅ Verified Dataset Audit: garystafford/deepfake-audio-detection

**Verification Date:** 2026-09-14  
**Source:** [Hugging Face Hub](https://huggingface.co/datasets/garystafford/deepfake-audio-detection)  
**License:** CC-BY-4.0 (Open Access)  
**Local Storage Path:** `data/raw/` (`data/raw/real/` and `data/raw/fake/`)  
**Audit Verdict:** ✅ **APPROVED WITH IDENTIFIED PREPROCESSING SAFEGUARDS**  

---

## 1. Verified Key Statistics

| Metric | Empirical Verified Value | Requirement / Benchmark | Notes |
| :--- | :--- | :--- | :--- |
| **Total Audio Samples** | **1,866** files | 500 – 5,000 samples | Stored directly in `data/raw/` |
| **Authentic Human (REAL)** | **933** files (50.0%) | 1:1 Class Balance | Prefix: `yt_` |
| **AI-Generated (FAKE)** | **933** files (50.0%) | 1:1 Class Balance | Prefixes: `po_`, `el_`, `hg_`, `hu_`, `lv_`, `sp_` |
| **Audio File Format** | **FLAC** (Lossless) | Uncompressed / Lossless | 100% valid headers |
| **Raw Sampling Rates** | **44,100 Hz** (REAL: 933) / **16,000 Hz** (FAKE: 933) | 16,000 Hz target | Standardized via `src/preprocessing.py` |
| **Audio Channels** | **1 (Mono)** | Mono | All single-channel |
| **Duration Range** | **2.50s to 12.53s** | 2.5s to 13.0s | Mean: 4.20s |
| **Mean Duration (± std)** | **4.20 ± 1.59 seconds** | Natural speech segments | Total: 2.17 hours |
| **Disk Space Consumed** | **562.74 MB** (590,074,814 bytes) | < 1.0 GB target | Fits easily in local storage |
| **Corrupted / Unreadable Files**| **0** (100% readable) | 0 corrupted | Verified with `soundfile` |
| **Cryptographic Duplicates** | **216** files (SHA-256 matched) | Audit finding | Upstream pass-2 segmentation artifacts |

---

## 2. Generator & Source Platform Breakdown

| Platform / Source | Category | File Prefix | Sample Count | Proportion |
| :--- | :--- | :--- | :--- | :--- |
| **YouTube Public Recordings** | Authentic Human Speech | `yt_` | 933 | 50.00% |
| **Speechify TTS** | Commercial AI Voice | `sp_` | 211 | 11.31% |
| **Amazon Polly** | Cloud Neural TTS | `po_` | 209 | 11.20% |
| **ElevenLabs Voice Cloning** | Deep Generative Voice | `el_` | 173 | 9.27% |
| **Luvvoice Web TTS** | Neural Web TTS | `lv_` | 156 | 8.36% |
| **Hume AI Empathic Voice** | Expressive Neural Voice | `hu_` | 116 | 6.22% |
| **Hexgrad Kokoro** | Lightweight Neural TTS | `hg_` | 68 | 3.64% |

---

## 3. Data Leakage & Integrity Audit Findings

### A. Sampling-Rate Discrepancy & Mitigation
* **Finding:** All 933 authentic human recordings are sampled at **44.1 kHz**, whereas all 933 AI-generated recordings are sampled at **16.0 kHz**.
* **Risk:** If fed raw to a classifier, the network could classify samples solely based on frequency bandwidth rather than synthesis vocoder artifacts.
* **Mitigation Enforced:** In `src/preprocessing.py`, all waveforms are automatically resampled to **16,000 Hz** via `librosa.resample(y, orig_sr=sr, target_sr=16000)` upon loading. This eliminates high-frequency bandwidth leakage.

### B. Cryptographic Deduplication
* **Finding:** 216 files share identical SHA-256 hashes with another file in the dataset.
* **Root Cause:** The dataset author employed a two-pass segmentation workflow (`Pass 1.5` concatenation and `Pass 2` sub-chunking), where some identical raw audio segments were indexed under both standard names (`yt_0006_part_049`) and sub-chunked names (`yt_0006_p2_part_050`).
* **Handling:** Deduplicated indices are recorded in `final_dataset_metadata.csv` and can be deduplicated during train/val/test splitting to guarantee strict independence.

### C. Label Integrity & Verification
* **Finding:** Every file belongs unambiguously to either `data/raw/real/` or `data/raw/fake/`.
* **Mapping:** `real` $\rightarrow$ 0 (`REAL`), `fake` $\rightarrow$ 1 (`FAKE`).
* **Class Ratio:** Exact 1.0 : 1.0 class ratio.

---

## 4. Visualizations Generated

The following plots were generated from the actual downloaded files and saved to `results/figures/dataset/`:
1. `class_distribution.png`: Exact 50/50 balance (933 vs 933).
2. `duration_distribution.png`: Speech chunk duration histogram and KDE curve.
3. `sampling_rate_distribution.png`: Raw sampling rate breakdown (16 kHz vs 44.1 kHz).
4. `example_real_waveform.png`: Authentic human speech waveform.
5. `example_fake_waveform.png`: Synthesized speech waveform.
6. `example_real_melspectrogram.png`: 128-band Log-Mel spectrogram for authentic speech.
7. `example_fake_melspectrogram.png`: 128-band Log-Mel spectrogram for synthetic speech.
