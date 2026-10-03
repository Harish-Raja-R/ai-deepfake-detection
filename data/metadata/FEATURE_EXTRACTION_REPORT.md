# 📊 Feature Extraction & Integrity Verification Report

**Phase:** Phase 4 — Feature Extraction & Normalization  
**Feature Representation:** Log-Mel Spectrogram (Decibels, Per-Frequency-Bin z-score Normalized)  
**Input Tensor Dimensions:** `(128, 126, 1)` float32 (Mel Bands × Time Frames × Channels)  
**Storage Destination:** `data/processed/features/*.npy`  
**Verification Date:** 2026-10-03 19:34:31  
**Status:** ✅ PASSED (All 11 Integrity Checks Satisfied)  

---

## 1. Feature Dimensions & Tensor Specifications

| Parameter | Specification | Purpose / Rationale |
| :--- | :--- | :--- |
| **Audio Sample Rate** | 16,000 Hz | Matched uniform rate from Phase 3 (zero SR shortcut). |
| **Fixed Audio Duration** | 4.0 seconds (64,000 samples) | Guarantees identical time dimension for batching. |
| **FFT Window Length (`n_fft`)** | 1,024 samples (64 ms) | High frequency resolution across vocal formants. |
| **Hop Length (`hop_length`)** | 512 samples (32 ms) | 50% window overlap for smooth temporal continuity. |
| **Number of Mel Bins (`n_mels`)** | 128 bands | Perceptually scaled logarithmic frequency bands. |
| **Frequency Range (`fmin` - `fmax`)**| 20 Hz – 8,000 Hz | Covers full Nyquist bandwidth for 16 kHz audio. |
| **Decibel Dynamic Range (`top_db`)** | 80.0 dB | Standardized dynamic range via `librosa.power_to_db`. |
| **Calculated Time Frames** | **126 frames** | $1 + \lfloor 64,000 / 512 \rfloor = 126$ centered frames. |
| **Final Tensor Shape** | **`(128, 126, 1)`** | Ready for 2D Deep CNN (`Conv2D`) architectures. |
| **Tensor Data Type** | `float32` (4 bytes per element) | Standard numerical precision for neural training. |

---

## 2. Leakage-Safe Normalization Strategy

* **Learned Statistics Derivation:** Normalization parameters were calculated **STRICTLY from the 1,307 training samples** in `train.csv`.
* **Zero Leakage Rule:** Validation and test samples were **never** used during mean/std calculation.
* **Normalization Mode:** Per-Mel-bin z-score:
  $$\text{tensor}[m, t, 0] = \frac{\text{spec}[m, t] - \mu_{\text{train}}[m]}{\sigma_{\text{train}}[m] + 10^{-6}}$$
* **Training Global Baseline:** Mean = 1307 samples, Global Mean = -53.2496 dB, Global Std = 18.8576 dB.
* **Persisted Parameters:** `data/metadata/feature_normalization_stats.json`.

---

## 3. Extracted Feature Partition Distribution

| Partition | Expected Count | Extracted Count | REAL Samples | FAKE Samples | REAL % | FAKE % | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Training (70%)** | 1,307 | **1,307** | 653 | 654 | 49.96% | 50.04% | ✅ Matched |
| **Validation (15%)** | 276 | **276** | 139 | 137 | 50.36% | 49.64% | ✅ Matched |
| **Testing (15%)** | 283 | **283** | 141 | 142 | 49.82% | 50.18% | ✅ Matched |
| **Total** | 1,866 | **1,866** | 933 | 933 | 50.00% | 50.00% | ✅ Complete |

---

## 4. Integrity Validation Checklist (11 Checks)

1. [x] **Extracted Training Features Count:** Exactly **1,307** features.
2. [x] **Extracted Validation Features Count:** Exactly **276** features.
3. [x] **Extracted Test Features Count:** Exactly **283** features.
4. [x] **Zero Missing Feature Files:** All **1,866** files exist and are verified.
5. [x] **Zero Corrupted Feature Files:** Exactly **0** unreadable files.
6. [x] **Tensor Shape Homogeneity:** 100% of tensors match `(128, 126, 1)`.
7. [x] **Zero NaN Values:** **0** NaN values detected across all 30,094,848 elements.
8. [x] **Zero Infinite Values:** **0** infinite values detected.
9. [x] **Label Validity:** All labels are strictly bounded to REAL (0) and FAKE (1).
10. [x] **Partition ID Consistency:** Sample IDs across train, val, and test match Phase 3 splits with 100% fidelity.
11. [x] **Duplicate Group Isolation:** Exactly **0** duplicate groups cross partitions.

---

## 5. Storage Footprint & Resource Management

* **Array File Format:** Single-precision 32-bit NumPy array (`.npy`)
* **Storage per Sample:** 64,512 bytes data + 128 bytes header = 64,640 bytes (63.13 KiB / 64.64 KB)
* **Total Feature Files:** **1,866** files
* **Total Feature Storage Size:** **115.03 MB** (120,618,240 bytes)
* **Memory Footprint:** In-memory batch loading uses ~84.5 MB for training, making it exceptionally well suited for laptop CPU execution without RAM bottlenecks.

---

## 6. Generated Visualizations (`results/figures/features/`)

1. `real_log_mel_spectrogram.png`: Log-Mel energy representation of genuine human voice.
2. `fake_log_mel_spectrogram.png`: Log-Mel energy representation of synthetic speech.
3. `real_mfcc.png`: 40-coefficient cepstral matrix for human speech.
4. `fake_mfcc.png`: 40-coefficient cepstral matrix for AI synthetic voice.
5. `real_vs_fake_spectrogram_comparison.png`: Side-by-side harmonic and formant comparison.
6. `feature_distribution.png`: Mel-band energy profiles and statistical distribution density.

---

## PHASE 4 STATUS

* Feature extraction completed: **YES**
* Feature type: **Log-Mel Spectrogram (dB scale, training-set normalized)**
* Mel bands: **128**
* Time frames: **126**
* Final tensor shape: **(128, 126, 1)**
* Train features: **1,307**
* Validation features: **276**
* Test features: **283**
* NaN values: **0**
* Infinite values: **0**
* Missing features: **0**
* Feature storage size: **115.03 MB**
* Ready for CNN training: **YES**
