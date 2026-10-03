# 🧠 Improved CRNN: Training & Evaluation Results

**Model Architecture:** Convolutional Recurrent Neural Network (3 Conv Blocks + Permute/Reshape + Bidirectional GRU(64) + Dense)  
**Input Representation:** Log-Mel Spectrogram `(128, 126, 1)` float32 (dB scale, training-normalized)  
**Evaluation Target:** Binary AI-Synthesized Voice Detection (REAL=0 vs FAKE=1)  
**Evaluation Date:** 2026-10-03 19:55:14  

---

## 1. Model Architecture & Parameter Footprint

| Layer Name | Layer Type | Output Shape | Param Count | Design Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `audio_spectrogram` | InputLayer | `(None, 128, 126, 1)` | 0 | 128 Mel frequency bins × 126 STFT time frames. |
| `conv1` / `bn1` / `relu1` | Conv2D + BN + ReLU | `(None, 128, 126, 32)` | 448 | Low-level acoustic spectro-temporal feature extraction. |
| `pool1` / `drop1` | MaxPool2D + Dropout | `(None, 64, 63, 32)` | 0 | Spatial downsampling (2×2) + 25% feature dropout. |
| `conv2` / `bn2` / `relu2` | Conv2D + BN + ReLU | `(None, 64, 63, 64)` | 18,752 | Mid-level formant frequency modeling. |
| `pool2` / `drop2` | MaxPool2D + Dropout | `(None, 32, 31, 64)` | 0 | Spatial downsampling (2×2) + 25% feature dropout. |
| `conv3` / `bn3` / `relu3` | Conv2D + BN + ReLU | `(None, 32, 31, 128)` | 74,368 | High-level spectral discontinuity representation. |
| `pool3` / `drop3` | MaxPool2D + Dropout | `(None, 16, 15, 128)` | 0 | Spatial downsampling (2×2) + 25% feature dropout. |
| `time_as_seq` | Permute((2, 1, 3)) | `(None, 15, 16, 128)` | 0 | Exchanges axes so time (15 frames) is the sequence dimension. |
| `seq_reshape` | Reshape((-1, 2048)) | `(None, 15, 2048)` | 0 | Flattens frequency × channels (16 × 128 = 2048) into feature vectors. |
| `bi_gru` | Bidirectional(GRU(64)) | `(None, 128)` | 811,776 | Bidirectional recurrent context tracking forward & backward in time. |
| `dense1` / `drop4` | Dense(64) + Dropout | `(None, 64)` | 8,256 | Non-linear feature combination with 40% dropout. |
| `prediction_output` | Dense(1) + Sigmoid | `(None, 1)` | 65 | Calibrated posterior probability $P(\text{FAKE} \mid \mathbf{x})$. |

* **Total Parameters:** **913,665**
* **Trainable Parameters:** **913,217**
* **Non-Trainable Parameters (BN running stats):** **448**
* **Memory Footprint:** ~**3.49 MB** (lightweight for CPU training and inference).

---

## 2. Training Execution Summary

* **Dataset Split:** 1,307 Training (50% REAL / 50% FAKE), 276 Validation (50% REAL / 50% FAKE)
* **Optimization Algorithm:** Adam (Initial learning rate $\eta = 0.0005$)
* **Batch Size:** 32 samples per batch
* **Loss Function:** Binary Crossentropy
* **Max Scheduled Epochs:** 20
* **Completed Epochs:** **6**
* **Total Training Wall-Clock Time:** **297.86 seconds** (49.64s per epoch on CPU)
* **Best Validation Epoch:** **Epoch 1**
* **Best Validation Loss:** **1.2276**
* **Best Validation Accuracy:** **50.36%**
* **Best Validation ROC-AUC:** **0.8027**
* **Checkpoint Restored:** `models/crnn_improved.keras`

---

## 3. Strict Out-of-Sample Test Evaluation

Evaluated strictly on the held-out test partition of **283 audio samples** (141 REAL, 142 FAKE) with zero test-set leakage or threshold manipulation (standard threshold $\tau = 0.50$):

| Metric | Measured Value | Benchmark Description |
| :--- | :--- | :--- |
| **Accuracy** | **49.82%** | Overall fraction of correct predictions across both classes. |
| **Precision** | **0.00%** | True positive rate among all samples predicted as FAKE ($TP / (TP + FP)$). |
| **Recall (Sensitivity)** | **0.00%** | Detection rate of actual deepfake audio ($TP / (TP + FN)$). |
| **F1-Score** | **0.00%** | Harmonic mean of Precision and Recall. |
| **ROC-AUC** | **0.8303** | Area under Receiver Operating Characteristic curve. |
| **Equal Error Rate (EER)** | **28.62%** | ASVspoof standard operating point where False Acceptance = False Rejection. |
| **Optimal EER Threshold** | **0.0749** | Decision threshold producing the Equal Error Rate. |

---

## 4. Confusion Matrix Breakdown

| True Label \ Predicted | Predicted REAL (0) | Predicted FAKE (1) | Total Actual |
| :--- | :--- | :--- | :--- |
| **Actual REAL (Human)** | **141** (TN) | **0** (FP) | **141** |
| **Actual FAKE (Synthetic)** | **142** (FN) | **0** (TP) | **142** |
| **Total Predicted** | **283** | **0** | **283** |

* **True Negatives (TN):** 141 / 141 genuine speech samples correctly authenticated (100.0% specificity).
* **False Positives (FP):** 0 / 141 genuine human voices incorrectly flagged as deepfakes (0.0% false alarm rate).
* **False Negatives (FN):** 142 / 142 synthetic voices that evaded detection (100.0% miss rate).
* **True Positives (TP):** 0 / 142 deepfake audio samples correctly intercepted (0.0% hit rate).

---

## 5. Generator-Specific Breakdown on FAKE Test Audio

| Generator Platform | Test Utterances | Correctly Flagged | Detection Accuracy | Mean Predicted $P(\text{FAKE})$ |
| :--- | :--- | :--- | :--- | :--- |
| **Amazon Polly** | 31 | 0 | 0.0% | 0.0892 |
| **ElevenLabs** | 26 | 0 | 0.0% | 0.0986 |
| **Hexgrad Kokoro** | 10 | 0 | 0.0% | 0.0936 |
| **Hume AI** | 18 | 0 | 0.0% | 0.0966 |
| **Luvvoice** | 24 | 0 | 0.0% | 0.0991 |
| **Speechify** | 33 | 0 | 0.0% | 0.0979 |

---

## 6. Artifacts Summary

* Best Model Checkpoint: `models/crnn_improved.keras`
* Metrics File: `results/metrics/crnn_improved_metrics.json`
* Training History: `results/metrics/crnn_training_history.json`
* Test Predictions: `results/predictions/crnn_test_predictions.csv`
* Training Figures: `results/figures/improved_model/`
* Evaluation Figures: `results/figures/improved_model/`
* Model Comparison Report: `results/metrics/MODEL_COMPARISON.md`
