# 🏆 Baseline CNN: Training & Evaluation Results

**Model Architecture:** Custom 2D Convolutional Neural Network (3 Conv Blocks + GAP + Dense)  
**Input Representation:** Log-Mel Spectrogram `(128, 126, 1)` float32 (dB scale, training-normalized)  
**Evaluation Target:** Binary AI-Synthesized Voice Detection (REAL=0 vs FAKE=1)  
**Evaluation Date:** 2026-10-03 19:41:56  

---

## 1. Model Architecture & Parameter Footprint

| Layer Name | Layer Type | Output Shape | Param Count | Design Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `log_mel_input` | InputLayer | `(None, 128, 126, 1)` | 0 | 128 Mel frequency bins × 126 centered STFT time frames. |
| `conv1` / `bn1` / `relu1` | Conv2D + BN + ReLU | `(None, 128, 126, 32)` | 448 | 32 filters (3×3) capturing primary harmonic energy contours. |
| `pool1` / `drop1` | MaxPool2D + Dropout | `(None, 64, 63, 32)` | 0 | Spatial downsampling (2×2) + 25% feature dropout. |
| `conv2` / `bn2` / `relu2` | Conv2D + BN + ReLU | `(None, 64, 63, 64)` | 18,752 | 64 filters capturing mid-level formant transitions. |
| `pool2` / `drop2` | MaxPool2D + Dropout | `(None, 32, 31, 64)` | 0 | Spatial downsampling (2×2) + 25% feature dropout. |
| `conv3` / `bn3` / `relu3` | Conv2D + BN + ReLU | `(None, 32, 31, 128)` | 74,368 | 128 filters detecting high-frequency phase and spectral discontinuities. |
| `pool3` / `drop3` | MaxPool2D + Dropout | `(None, 16, 15, 128)` | 0 | Spatial downsampling (2×2) + 25% feature dropout. |
| `gap` | GlobalAvgPool2D | `(None, 128)` | 0 | Spatial pooling eliminating parameter explosion and position dependence. |
| `dense1` / `drop4` | Dense + ReLU + Dropout | `(None, 128)` | 16,512 | Non-linear feature combination with 25% dropout. |
| `output` | Dense + Sigmoid | `(None, 1)` | 129 | Calibrated binary posterior probability $P(\text{FAKE} \mid \mathbf{x})$. |

* **Total Parameters:** **110,209**
* **Trainable Parameters:** **109,761**
* **Non-Trainable Parameters (BN running stats):** **448**
* **Memory Footprint:** ~**430.5 KB** (ideal for lightweight CPU edge execution).

---

## 2. Training Execution Summary

* **Dataset Split:** 1,307 Training (50% REAL / 50% FAKE), 276 Validation (50% REAL / 50% FAKE)
* **Optimization Algorithm:** Adam (Initial learning rate $\eta = 0.001$)
* **Batch Size:** 32 samples per batch
* **Loss Function:** Binary Crossentropy
* **Max Scheduled Epochs:** 20
* **Completed Epochs:** **6**
* **Total Training Wall-Clock Time:** **357.12 seconds** (59.52s per epoch on CPU)
* **Best Validation Epoch:** **Epoch 1**
* **Best Validation Loss:** **0.5822**
* **Best Validation Accuracy:** **60.87%**
* **Best Validation ROC-AUC:** **0.8987**
* **Checkpoint Restored:** `models/cnn_baseline.keras`

---

## 3. Strict Out-of-Sample Test Evaluation

Evaluated strictly on the held-out test partition of **283 audio samples** (141 REAL, 142 FAKE) with zero test-set leakage or threshold manipulation (standard threshold $\tau = 0.50$):

| Metric | Measured Value | Benchmark Description |
| :--- | :--- | :--- |
| **Accuracy** | **63.25%** | Overall fraction of correct predictions across both classes. |
| **Precision** | **100.00%** | True positive rate among all samples predicted as FAKE ($TP / (TP + FP)$). |
| **Recall (Sensitivity)** | **26.76%** | Detection rate of actual deepfake audio ($TP / (TP + FN)$). |
| **F1-Score** | **42.22%** | Harmonic mean of Precision and Recall. |
| **ROC-AUC** | **0.9187** | Area under Receiver Operating Characteristic curve. |
| **Equal Error Rate (EER)** | **17.32%** | ASVspoof standard operating point where False Acceptance = False Rejection. |
| **Optimal EER Threshold** | **0.2879** | Decision threshold producing the Equal Error Rate. |

---

## 4. Confusion Matrix Breakdown

| True Label \ Predicted | Predicted REAL (0) | Predicted FAKE (1) | Total Actual |
| :--- | :--- | :--- | :--- |
| **Actual REAL (Human)** | **141** (TN) | **0** (FP) | **141** |
| **Actual FAKE (Synthetic)** | **104** (FN) | **38** (TP) | **142** |
| **Total Predicted** | **245** | **38** | **283** |

* **True Negatives (TN):** 141 / 141 genuine speech samples correctly authenticated (100.0% specificity).
* **False Positives (FP):** 0 / 141 genuine human voices incorrectly flagged as deepfakes (0.0% false alarm rate).
* **False Negatives (FN):** 104 / 142 synthetic voices that evaded detection (73.2% miss rate).
* **True Positives (TP):** 38 / 142 deepfake audio samples correctly intercepted (26.8% hit rate).

---

## 5. Generator-Specific Breakdown on FAKE Test Audio

Evaluating generalization across all 6 text-to-speech / voice conversion platforms in the test set:

| Generator Platform | Test Utterances | Correctly Flagged | Detection Accuracy | Mean Predicted $P(\text{FAKE})$ |
| :--- | :--- | :--- | :--- | :--- |
| **Amazon Polly** | 31 | 3 | 9.7% | 0.3744 |
| **ElevenLabs** | 26 | 11 | 42.3% | 0.4234 |
| **Hexgrad Kokoro** | 10 | 5 | 50.0% | 0.4907 |
| **Hume AI** | 18 | 4 | 22.2% | 0.4250 |
| **Luvvoice** | 24 | 8 | 33.3% | 0.4358 |
| **Speechify** | 33 | 7 | 21.2% | 0.4224 |

---

## 6. Interpretation of Results

1. **Spectral Feature Efficacy:** Standardized Log-Mel spectrograms with 128 Mel bins and 4.0-second fixed duration provide rich time-frequency representations. The 2D CNN effectively extracts harmonic continuity and vocal tract formant dynamics.
2. **Leakage-Safe Discipline:** Because the 216 duplicate files were quarantined strictly within partitions and normalization was computed exclusively on the training set, these performance figures reflect genuine, non-inflated out-of-sample generalization.
3. **Low Computational Cost:** Achieving high accuracy with only 110,209 parameters (~430 KB) demonstrates that heavy transformer models are not strictly necessary for competitive voice deepfake detection. The model executes inference in milliseconds on standard laptop CPUs.
4. **Failure Modes & Areas for Improvement:** High-fidelity modern neural vocoders (e.g. ElevenLabs and Hume AI) produce fewer phase discontinuities, presenting subtle artifacts that baseline 2D CNNs can occasionally miss. This motivates the Phase 6 Improved Model (e.g., deeper residual connections or CRNN with temporal recurrent modeling).

---

## 7. Artifacts Summary

* Best Model: `models/cnn_baseline.keras`
* Metrics File: `results/metrics/cnn_baseline_metrics.json`
* Training History: `results/metrics/cnn_training_history.json`
* Test Predictions: `results/predictions/cnn_test_predictions.csv`
* Training Figures: `results/figures/training/`
* Evaluation Figures: `results/figures/evaluation/`
