# ⚙️ Audio Preprocessing & Standardization Report

**Phase:** Phase 3 — Leakage-Safe Dataset Splitting & Audio Preprocessing  
**Standard Target:** 16,000 Hz, Single-Channel Mono, Peak Normalized, Fixed 4.0 Seconds (64,000 Samples)  
**Processed Storage:** `data/processed/audio/` (NumPy `.npy` format)  
**Verification Date:** 2026-10-03 19:30:47  

---

## 1. Actual Measured Dataset Partitioning Counts

All 1,866 audio files were partitioned using deterministic group-aware stratified partitioning (`seed=42`).
Every SHA-256 duplicate group is quarantined strictly within a single partition to ensure zero data leakage.

| Metric / Partition | Training (70%) | Validation (15%) | Testing (15%) | Overall Total |
| :--- | :--- | :--- | :--- | :--- |
| **Total Audio Samples** | **1,307** | **276** | **283** | **1,866** |
| **REAL Count** | 653 | 139 | 141 | 933 |
| **FAKE Count** | 654 | 137 | 142 | 933 |
| **REAL Proportion** | 49.96% | 50.36% | 49.82% | 50.00% |
| **FAKE Proportion** | 50.04% | 49.64% | 50.18% | 50.00% |
| **Unique Duplicate Groups** | 1,155 | 247 | 248 | 1,650 |
| **Cross-Partition Duplicate Groups** | **0** | **0** | **0** | **0** |

* **Total Duplicate Groups in Dataset:** 1,650
* **Cross-Partition Duplicate Count:** **0** (strictly zero duplicate leakage)

---

## 2. Standardization & Acoustic Transformation Parameters

| Preprocessing Step | Rule Applied | Target Specification | Leakage Mitigation Impact |
| :--- | :--- | :--- | :--- |
| **Resampling** | Band-limited Kaiser window Sinc interpolation | **16,000 Hz** uniform | Eliminates the 44.1 kHz (REAL) vs. 16.0 kHz (FAKE) sample-rate shortcut. |
| **Channel Conversion** | Single-channel averaging | **Mono (1D)** | Prevents stereo vs mono channel bias. |
| **Silence Trimming** | Leading and trailing silence trimmed at 20 dB | `librosa.effects.trim(top_db=20)` | Eliminates silent lead-in discrepancies between studio TTS and YouTube audio. |
| **Amplitude Normalization** | Maximum absolute peak normalization | Range $[-1.0, 1.0]$ | Prevents classifiers from keying on loudness or mastering gain variations. |
| **Duration Standardization** | Truncation / Zero-padding | **Exactly 4.0s (64,000 samples)** | Standardizes 2D spectrogram tensor dimensions for CNN input. |

* **Uniform Preprocessing Sample Rate:** **16,000 Hz**
* **Fixed Audio Duration:** **4.0 seconds** (64,000 samples)
* **Number of Processed Files:** **1,866**

---

## 3. Storage Efficiency & Resource Constraints

* **Storage Format:** 32-bit Float NumPy arrays (`.npy`)
* **Array Shape per Sample:** `(64000,)` float32
* **Storage per Audio File:** 64,000 samples × 4 bytes + 128 bytes header ≈ 250.1 KB
* **Total Processed Files:** **1,866** files
* **Total Preprocessing Output Size:** **455.79 MB** (477,934,848 bytes)
* **Storage Footprint Assessment:** At **~455 MB**, the processed cache consumes minimal disk space, well within the host laptop's 14.9 GB free capacity. No unnecessary duplicate copies of the raw FLAC files were created.

---

## 4. Generated Artifacts

### Metadata Files (`data/metadata/`)
* `train.csv` (1307 samples)
* `validation.csv` (276 samples)
* `test.csv` (283 samples)
* `split_integrity_report.json` (Machine-readable audit)
* `SPLIT_REPORT.md` (Detailed human-readable split analysis)
* `processed_metadata.csv` (1866 standardized entries)
* `PREPROCESSING_REPORT.md` (This document)

### Verification Figures (`results/figures/preprocessing/`)
1. `real_waveform_before.png`: Authentic speech waveform before preprocessing.
2. `fake_waveform_before.png`: Synthetic speech waveform before preprocessing.
3. `real_waveform_after.png`: Standardized authentic speech waveform (16 kHz, 4.0s).
4. `fake_waveform_after.png`: Standardized synthetic speech waveform (16 kHz, 4.0s).
5. `duration_before_vs_after.png`: Duration distribution shift from variable (2.5s–12.5s) to fixed 4.0s.
6. `samplerate_before_vs_after.png`: Complete elimination of the 44.1 kHz vs. 16 kHz sampling dichotomy.

---

## PHASE 3 STATUS

* Split created: **YES**
* Duplicate leakage: **0**
* Train samples: **1,307**
* Validation samples: **276**
* Test samples: **283**
* REAL/FAKE distribution: **Train (50.0% / 50.0%), Val (50.4% / 49.6%), Test (49.8% / 50.2%)**
* Processing completed: **YES**
* Preprocessing output size: **455.79 MB**
* Ready for feature extraction: **YES**
