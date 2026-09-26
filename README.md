# 🎙️ AI Voice Deepfake Detection

An end-to-end, modular, and academically rigorous machine learning system designed to detect AI-generated/synthesized voice and deepfake speech from genuine human speech.

---

## 🚀 Run the Application

Launch the production-style Streamlit web interface:

```bash
python -m streamlit run app/app.py
```

*(Note: Using `python -m streamlit run app/app.py` avoids PATH resolution issues on Windows environments where the Python scripts directory is not configured in the system PATH).*

The interactive dashboard allows users to:
1. Upload speech audio files (`.wav`, `.flac`, `.mp3`).
2. Inspect acoustic properties (original sampling rate, channels, duration, format).
3. Listen via the embedded audio player.
4. Visualize standardized waveforms and 128-band Log-Mel spectrograms.
5. Receive deepfake verdicts ("REAL / BONAFIDE" vs "AI-GENERATED / DEEPFAKE") with confidence scores.
6. Toggle between the standard threshold ($\tau = 0.50$) and the biometric EER operating point ($\tau = 0.2879$).

---

## 🔄 Complete Project Pipeline

```text
Dataset (garystafford/deepfake-audio-detection: 1,866 utterances)
  ↓
Verification (100% audio integrity, 933 REAL @ 44.1 kHz, 933 FAKE @ 16 kHz across 6 TTS engines)
  ↓
Leakage-Safe Group Split (SHA-256 duplicate quarantine: 1,307 Train / 276 Val / 283 Test, zero leakage)
  ↓
Audio Preprocessing (Mono, 16 kHz, silence trimmed, peak-normalized, fixed 4.0s / 64,000 samples)
  ↓
Log-Mel Feature Extraction (128 Mel bands × 126 STFT frames, training-derived per-bin z-score norm)
  ↓
Model Training (Baseline 2D CNN: 110,209 params vs. Improved CRNN: 913,665 params)
  ↓
Empirical Evaluation (Test set N=283: ROC-AUC 0.9187 vs 0.8303, EER 17.32% vs 28.62%)
  ↓
Final Model Deployment (Baseline CNN selected for superior accuracy, AUC, EER & efficiency)
  ↓
Streamlit Real-Time Inference (Clean web dashboard with dual operating-point selection)
```

---

## 🏆 Final Model Performance Summary

Evaluated strictly on the held-out test partition (**283 audio samples**: 141 REAL human utterances, 142 FAKE synthetic utterances spanning 6 text-to-speech generators) with zero test-set leakage:

| Evaluation Metric | Baseline CNN (Deployed) | Improved CRNN | Comparison Verdict |
| :--- | :---: | :---: | :--- |
| **Model Architecture** | Custom 2D CNN (3 Conv + GAP + Dense) | CRNN (3 Conv + BiGRU + Dense) | CNN is 8.3× more parameter-efficient |
| **Parameters** | **110,209** (~430 KB) | **913,665** (~3.49 MB) | Baseline CNN (Lightweight edge footprint) |
| **CPU Training Time** | **261.96 seconds** | **404.88 seconds** | Baseline CNN (1.55× faster) |
| **Standard Accuracy ($\tau=0.50$)** | **63.25%** | **49.82%** | **CNN (+13.43%)** |
| **Standard Precision ($\tau=0.50$)** | **100.00%** | **0.00%** | **CNN (+100.00%)** |
| **Standard Recall ($\tau=0.50$)** | **26.76%** | **0.00%** | **CNN (+26.76%)** |
| **Standard F1-Score ($\tau=0.50$)** | **42.22%** | **0.00%** | **CNN (+42.22%)** |
| **ROC-AUC (Overall Ranking)** | **0.9187** | **0.8303** | **CNN (+0.0884)** |
| **Equal Error Rate (EER)** | **17.32%** | **28.62%** | **CNN (-11.30% lower error)** |
| **EER Operating Accuracy** | **82.69%** | **71.38%** | **CNN (+11.31%)** |
| **EER Operating F1-Score** | **82.81%** | **71.38%** | **CNN (+11.43%)** |

**Final Recommendation:** Deploy **Baseline 2D CNN** (`models/cnn_baseline.keras`). It strictly dominates the CRNN across all statistical, discriminative, and computational efficiency dimensions.

---

## 🗂️ Project Structure

```text
AI_VOICE_DEEPFAKE_DETECTION/
├── app/
│   └── app.py                      # Production-style Streamlit web application
├── data/
│   ├── raw/                        # Original verified audio (933 real, 933 fake)
│   ├── processed/
│   │   ├── audio/                  # Standardized 16 kHz 4.0s waveforms (.npy)
│   │   └── features/               # 128-band normalized Log-Mel spectrograms (.npy)
│   └── metadata/                   # Partition CSVs and training normalization statistics
├── models/
│   ├── cnn_baseline.keras          # Final recommended deployment model (110K params)
│   └── crnn_improved.keras         # CRNN model checkpoint (913K params)
├── results/
│   ├── figures/
│   │   ├── dataset/                # Class balance and duration distributions
│   │   ├── preprocessing/          # Waveform comparisons and spectral density
│   │   ├── features/               # Mel-spectrogram heatmaps and mean spectra
│   │   ├── training/               # Baseline CNN learning curves
│   │   ├── evaluation/             # Baseline CNN confusion matrix & ROC curve
│   │   ├── improved_model/         # CRNN learning curves, confusion matrix & ROC
│   │   ├── comparison/             # Head-to-head metric bars and ROC overlays
│   │   └── final_comparison/       # Publication-quality final figures (8 figures)
│   ├── metrics/
│   │   ├── cnn_baseline_metrics.json
│   │   ├── crnn_improved_metrics.json
│   │   ├── FINAL_MODEL_COMPARISON.csv
│   │   ├── FINAL_MODEL_COMPARISON.md
│   │   ├── MODEL_COMPARISON.md
│   │   └── FINAL_EVALUATION_REPORT.md
│   └── predictions/
│       ├── cnn_test_predictions.csv
│       └── crnn_test_predictions.csv
├── src/
│   ├── config.py                   # Centralized paths, seeds, and audio configurations
│   ├── dataset.py                  # Dataset indexing and split management
│   ├── preprocessing.py            # Audio resampling, trimming, and padding
│   ├── features.py                 # 128-band Mel-spectrogram & normalization
│   ├── baseline_model.py           # Baseline 2D CNN architecture
│   ├── improved_model.py           # Improved CRNN architecture
│   ├── predict.py                  # End-to-end inference engine (CLI & API)
│   ├── phase5_pipeline.py          # Phase 5 execution script
│   ├── phase6_pipeline.py          # Phase 6 execution script
│   └── phase7_evaluation.py        # Phase 7 final evaluation script
├── tests/
│   └── test_app_pipeline.py        # Automated test suite (8 unit tests)
├── requirements.txt
└── README.md
```

---

## 🧪 Automated Testing

Run the automated test suite to verify model loading, preprocessing, feature extraction, predictions, threshold policies, and error handling:

```bash
python -m pytest tests/
```

All 8 automated tests validate the end-to-end inference pipeline without requiring Streamlit UI interaction.

---

## ⚠️ Limitations & Ethical Disclaimers

1. **Academic Demonstration:** Developed as an academic mini-project demonstrating ML anti-spoofing methodologies.
2. **Not a Forensic System:** The model is not a certified forensic tool and should not be used as sole evidence in legal or high-stakes identity verification.
3. **Distribution Sensitivity:** Performance is optimal on speech characteristics similar to the training benchmark (`garystafford/deepfake-audio-detection`). Unseen neural vocoders, high background noise, or heavy telephone compression may impact accuracy.
4. **Probability vs. Proof:** A high model probability is a statistical correlation with synthetic acoustic markers, not absolute proof.
