"""
Evaluation and Metrics Module for AI Voice Deepfake Detection.

Calculates key classification metrics for speech deepfake detection:
- Accuracy, Precision, Recall, F1-Score
- Area Under ROC Curve (ROC-AUC)
- Equal Error Rate (EER) - industry benchmark for voice anti-spoofing
- Confusion Matrix and ROC/PR curve plots (saved to results/figures/)
- Metric reports exported to JSON (saved to results/metrics/)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.baseline_model import load_baseline_model
from src.config import config
from src.dataset import scan_dataset, split_dataset
from src.improved_model import load_improved_model
from src.train import extract_dataset_features


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) where False Positive Rate equals False Negative Rate.
    
    Args:
        y_true (np.ndarray): Binary ground truth (0 for real, 1 for fake).
        y_scores (np.ndarray): Predicted probability of being fake.
        
    Returns:
        Tuple[float, float]: (eer, threshold_at_eer)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    
    # EER is point where |fpr - fnr| is minimized
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    return float(eer), float(threshold)


def calculate_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics suite.
    
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_scores (np.ndarray): Predicted probabilities of fake class.
        threshold (float): Decision threshold for binary classification.
        
    Returns:
        Dict[str, Any]: Dictionary of calculated metrics.
    """
    y_pred = (y_scores >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = float("nan")

    try:
        eer, eer_threshold = compute_eer(y_true, y_scores)
    except Exception:
        eer, eer_threshold = float("nan"), float("nan")

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "equal_error_rate": float(eer),
        "eer_threshold": float(eer_threshold),
        "decision_threshold": float(threshold),
        "confusion_matrix": cm,
    }


def plot_evaluation_charts(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    model_name: str,
    output_dir: Optional[Union[str, Path]] = None,
    threshold: float = 0.5,
) -> Dict[str, Path]:
    """
    Plot and save evaluation visualizations:
    1. Confusion matrix heatmap.
    2. ROC Curve with EER point.
    3. Probability distribution of predictions.
    
    Returns:
        Dict[str, Path]: Paths to saved figures.
    """
    out_dir = Path(output_dir) if output_dir else config.paths.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real (Human)", "Fake (AI)"],
        yticklabels=["Real (Human)", "Fake (AI)"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = out_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    saved_paths["confusion_matrix"] = cm_path

    # 2. ROC Curve Plot
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
        auc = roc_auc_score(y_true, y_scores)
        eer, _ = compute_eer(y_true, y_scores)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Chance")
        plt.scatter([eer], [1 - eer], color="#dc2626", zorder=5, label=f"EER = {eer:.3f}")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FAR)")
        plt.ylabel("True Positive Rate (1 - FRR)")
        plt.title(f"ROC Curve & Equal Error Rate - {model_name}")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = out_dir / f"{model_name}_roc_curve.png"
        plt.savefig(roc_path, dpi=300)
        plt.close()
        saved_paths["roc_curve"] = roc_path
    except Exception as e:
        print(f"[Warning] Could not plot ROC curve: {e}")

    # 3. Probability Distribution Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(y_scores[y_true == 0], color="#10b981", label="Genuine Human Speech", kde=True, stat="density", bins=20)
    sns.histplot(y_scores[y_true == 1], color="#ef4444", label="AI Deepfake Speech", kde=True, stat="density", bins=20)
    plt.axvline(x=threshold, color="black", linestyle="--", label=f"Threshold ({threshold})")
    plt.title(f"Prediction Probability Distribution - {model_name}")
    plt.xlabel("Predicted Probability (P(Deepfake))")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    dist_path = out_dir / f"{model_name}_prob_distribution.png"
    plt.savefig(dist_path, dpi=300)
    plt.close()
    saved_paths["prob_dist"] = dist_path

    return saved_paths


def evaluate_model_pipeline(
    model_path: Union[str, Path],
    test_df: pd.DataFrame,
    feature_mode: str = "2d",
    feature_type: str = "melspectrogram",
    model_name: str = "model",
    save_results: bool = True,
) -> Dict[str, Any]:
    """
    Run full evaluation on a test set using a saved model checkpoint.
    """
    m_path = Path(model_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Model file not found: {m_path}")

    # 1. Extract test features
    X_test, y_test = extract_dataset_features(test_df, feature_mode=feature_mode, feature_type=feature_type)

    # 2. Predict probabilities
    if m_path.suffix in [".keras", ".h5"]:
        model = load_improved_model(m_path)
        probs = model.predict(X_test, verbose=0).flatten()
    else:
        model = load_baseline_model(m_path)
        probs = model.predict_proba(X_test)[:, 1]

    # 3. Compute metrics
    metrics = calculate_metrics(y_test, probs)
    print("\n--- Model Evaluation Results ---")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k:>20}: {v:.4f}")

    if save_results:
        # Save JSON metrics
        metrics_file = config.paths.metrics_dir / f"{model_name}_metrics.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to: {metrics_file}")

        # Save Visualizations
        plots = plot_evaluation_charts(y_test, probs, model_name=model_name)
        for name, p in plots.items():
            print(f"Saved chart [{name}]: {p}")

    return metrics


def main():
    """Command-line evaluation entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate AI Voice Deepfake Detection Model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model checkpoint")
    parser.add_argument("--data-dir", type=str, default=str(config.paths.raw_data_dir), help="Path to evaluation audio data")
    parser.add_argument("--metadata", type=str, default=None, help="Path to dataset metadata CSV")
    parser.add_argument("--feature-mode", type=str, default="2d", choices=["1d", "2d"])
    parser.add_argument("--feature-type", type=str, default="melspectrogram", choices=["melspectrogram", "mfcc"])
    parser.add_argument("--model-name", type=str, default="evaluation_run")
    args = parser.parse_args()

    config.initialize()
    df = scan_dataset(args.data_dir, metadata_path=args.metadata)
    if len(df) == 0:
        print("[Notice] No evaluation files found in dataset path.")
        return

    _, _, test_df = split_dataset(df, seed=config.training.seed)
    evaluate_model_pipeline(
        model_path=args.model_path,
        test_df=test_df,
        feature_mode=args.feature_mode,
        feature_type=args.feature_type,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
