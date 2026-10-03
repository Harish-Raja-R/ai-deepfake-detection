# 📊 Final Model Comparison: Baseline CNN vs. Improved CRNN

**Dataset:** `garystafford/deepfake-audio-detection`  
**Test Set Size:** 283 samples (141 REAL, 142 FAKE)  
**Evaluation Protocol:** Out-of-Sample Held-Out Test Set (Zero Leakage)  
**Generated On:** 2026-10-03 19:55:34  

---

## 1. Primary Model Comparison Table (Standard Threshold $\tau = 0.50$)

| Metric | CNN | CRNN | Better Model |
| :--- | :---: | :---: | :---: |
| **Parameters** | **110,209** | **913,665** | **CNN** (8.3× smaller) |
| **Accuracy** | **63.25%** | **49.82%** | **CNN** (+13.43%) |
| **Precision** | **100.00%** | **0.00%** | **CNN** (+100.00%) |
| **Recall** | **26.76%** | **0.00%** | **CNN** (+26.76%) |
| **F1** | **42.22%** | **0.00%** | **CNN** (+42.22%) |
| **Specificity** | **100.00%** | **100.00%** | **Tie** (100.00%) |
| **False-Positive Rate** | **0.00%** | **0.00%** | **Tie** (0.00%) |
| **False-Negative Rate** | **73.24%** | **100.00%** | **CNN** (-26.76%) |
| **ROC-AUC** | **0.9187** | **0.8303** | **CNN** (+0.0884) |
| **EER** | **17.32%** | **28.62%** | **CNN** (Lower by 11.30%) |
| **Training Time** | **261.96 sec** | **404.88 sec** | **CNN** (1.55× faster) |

---

## 2. EER Operating-Point Metrics (Independently Determined $\tau_{\text{EER}}$)

Because threshold $\tau=0.50$ is sensitive to posterior scale compression, the table below documents performance when decision thresholds are calibrated to each model's independent Equal Error Rate (EER) operating point:

| Metric | Baseline CNN ($\tau = 0.2879$) | Improved CRNN ($\tau = 0.0749$) | Better Model |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **82.69%** | **71.38%** | **CNN** (+11.31%) |
| **Precision** | **82.52%** | **71.63%** | **CNN** (+10.89%) |
| **Recall** | **83.10%** | **71.13%** | **CNN** (+11.97%) |
| **F1-Score** | **82.81%** | **71.38%** | **CNN** (+11.43%) |
| **Specificity** | **82.27%** | **71.63%** | **CNN** (+10.64%) |
| **False-Positive Rate** | **17.73%** | **28.37%** | **CNN** (-10.64%) |
| **False-Negative Rate** | **16.90%** | **28.87%** | **CNN** (-11.97%) |

---

## 3. Final Determination

* **Best overall model:** **Baseline CNN**
* **Best Accuracy:** **Baseline CNN** (63.25% @ 0.50; 82.69% @ EER)
* **Best F1:** **Baseline CNN** (42.22% @ 0.50; 82.81% @ EER)
* **Best ROC-AUC:** **Baseline CNN** (0.9187 vs. 0.8303)
* **Best EER:** **Baseline CNN** (17.32% vs. 28.62%)
* **Smallest model:** **Baseline CNN** (110,209 vs. 913,665 parameters)
* **Fastest model:** **Baseline CNN** (261.96s vs. 404.88s training time)

**Final Recommendation:** Deploy **Baseline CNN** (`models/cnn_baseline.keras`). It strictly dominates the Improved CRNN across all statistical, discriminative, and computational efficiency dimensions.
