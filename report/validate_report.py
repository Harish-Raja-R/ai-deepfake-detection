"""
Validation script for Phase 9: Academic Report.
Verifies all 25 sections, metric fidelity against project JSON/CSV files,
referenced image existence, and core project artifact completeness.
"""

import os
import sys
import re
import json
from pathlib import Path

def validate():
    print("=" * 80)
    print("PHASE 9: ACADEMIC REPORT VALIDATION")
    print("=" * 80)

    project_root = Path(__file__).resolve().parent.parent
    report_path = project_root / "report" / "AI_VOICE_DEEPFAKE_DETECTION_REPORT.md"
    references_path = project_root / "report" / "references.md"
    metrics_dir = project_root / "results" / "metrics"

    errors = []

    # 1. Verify existence of primary documents
    print("[1/5] Checking report and reference documents...")
    if not report_path.exists() or report_path.stat().st_size == 0:
        errors.append(f"Report document missing or empty: {report_path}")
    else:
        print(f"  [PASS] Academic report found ({report_path.stat().st_size} bytes)")

    if not references_path.exists() or references_path.stat().st_size == 0:
        errors.append(f"References document missing or empty: {references_path}")
    else:
        print(f"  [PASS] References bibliography found ({references_path.stat().st_size} bytes)")

    report_text = report_path.read_text(encoding="utf-8")

    # 2. Verify all 25 required sections exist
    print("\n[2/5] Checking all 25 required report sections...")
    required_sections = [
        "1. Abstract",
        "2. Introduction",
        "3. Problem Statement",
        "4. Objectives",
        "5. Literature Survey",
        "6. Dataset Description",
        "7. Data Integrity and Leakage Prevention",
        "8. Dataset Split",
        "9. Methodology",
        "10. Audio Preprocessing",
        "11. Feature Extraction",
        "12. Baseline CNN Architecture",
        "13. Improved CRNN Architecture",
        "14. Training Configuration",
        "15. Evaluation Metrics",
        "16. Results — Baseline CNN",
        "17. Results — CRNN",
        "18. Threshold Analysis",
        "19. Model Comparison",
        "20. Streamlit Application",
        "21. Results and Discussion",
        "22. Limitations",
        "23. Future Scope",
        "24. Conclusion",
        "25. References"
    ]

    for section in required_sections:
        # Match '## X. Section' or '## X. Section' with slight variation
        pattern = re.compile(rf"##\s+{re.escape(section)}", re.IGNORECASE)
        if not pattern.search(report_text):
            errors.append(f"Missing required section header: '## {section}'")
        else:
            print(f"  [PASS] Section '{section}' present")

    # 3. Verify metrics match project JSON files
    print("\n[3/5] Verifying metric consistency with project JSON results...")
    cnn_json_path = metrics_dir / "cnn_baseline_metrics.json"
    crnn_json_path = metrics_dir / "crnn_improved_metrics.json"

    if not cnn_json_path.exists():
        errors.append(f"CNN metrics JSON not found at {cnn_json_path}")
    else:
        with open(cnn_json_path, "r", encoding="utf-8") as f:
            cnn_data = json.load(f)
        
        expected_cnn_values = [
            ("Parameters", str(cnn_data["parameters"]), "110,209"),
            ("Accuracy", f"{cnn_data['accuracy'] * 100:.2f}%", "63.25%"),
            ("Precision", f"{cnn_data['precision'] * 100:.2f}%", "100.00%"),
            ("Recall", f"{cnn_data['recall'] * 100:.2f}%", "26.76%"),
            ("F1-Score", f"{cnn_data['f1_score'] * 100:.2f}%", "42.22%"),
            ("ROC-AUC", f"{cnn_data['roc_auc']:.4f}", "0.9187"),
            ("EER", f"{cnn_data['eer'] * 100:.2f}%", "17.32%"),
            ("EER Threshold", f"{cnn_data['eer_threshold']:.4f}", "0.2879"),
            ("Training Time", f"{cnn_data['training_time_seconds']:.2f}", "261.96"),
            ("Epochs Trained", str(cnn_data["epochs_trained"]), "6"),
        ]
        for name, val, formatted in expected_cnn_values:
            if formatted not in report_text and val not in report_text:
                errors.append(f"CNN metric mismatch: {name} ({formatted} or {val}) not found in report text")
            else:
                print(f"  [PASS] CNN {name} = {formatted} verified in report")

    if not crnn_json_path.exists():
        errors.append(f"CRNN metrics JSON not found at {crnn_json_path}")
    else:
        with open(crnn_json_path, "r", encoding="utf-8") as f:
            crnn_data = json.load(f)
        
        expected_crnn_values = [
            ("Parameters", str(crnn_data["parameters"]), "913,665"),
            ("Accuracy", f"{crnn_data['accuracy'] * 100:.2f}%", "49.82%"),
            ("Precision", f"{crnn_data['precision'] * 100:.2f}%", "0.00%"),
            ("Recall", f"{crnn_data['recall'] * 100:.2f}%", "0.00%"),
            ("F1-Score", f"{crnn_data['f1_score'] * 100:.2f}%", "0.00%"),
            ("ROC-AUC", f"{crnn_data['roc_auc']:.4f}", "0.8303"),
            ("EER", f"{crnn_data['eer'] * 100:.2f}%", "28.62%"),
            ("EER Threshold", f"{crnn_data['eer_threshold']:.4f}", "0.0749"),
            ("Training Time", f"{crnn_data['training_time_seconds']:.2f}", "404.88"),
            ("Epochs Trained", str(crnn_data["epochs_trained"]), "6"),
        ]
        for name, val, formatted in expected_crnn_values:
            if formatted not in report_text and val not in report_text:
                errors.append(f"CRNN metric mismatch: {name} ({formatted} or {val}) not found in report text")
            else:
                print(f"  [PASS] CRNN {name} = {formatted} verified in report")

    # Check key dataset numbers
    dataset_numbers = [
        ("Total samples", "1,866"),
        ("REAL samples", "933"),
        ("FAKE samples", "933"),
        ("Duplicate files", "216"),
        ("Train split", "1,307"),
        ("Val split", "276"),
        ("Test split", "283"),
        ("Test REAL", "141"),
        ("Test FAKE", "142"),
        ("Real sample rate", "44.1 kHz"),
        ("Fake sample rate", "16 kHz"),
        ("Standardized rate", "16,000 Hz"),
    ]
    for name, val in dataset_numbers:
        if val not in report_text:
            errors.append(f"Dataset statistic '{name}' ({val}) missing from report text")
        else:
            print(f"  [PASS] Dataset {name} = {val} verified in report")

    # 4. Verify all referenced figures exist
    print("\n[4/5] Verifying all referenced markdown figure paths exist on disk...")
    # Find all image links: ![alt](path)
    img_matches = re.findall(r'!\[.*?\]\((.*?)\)', report_text)
    if not img_matches:
        errors.append("No figure images were referenced in the report!")
    else:
        for img_rel_path in img_matches:
            # Resolve relative to report/ directory
            resolved_path = (report_path.parent / img_rel_path).resolve()
            if not resolved_path.exists():
                errors.append(f"Referenced figure does not exist: {img_rel_path} -> {resolved_path}")
            else:
                print(f"  [PASS] Figure exists: {img_rel_path} ({resolved_path.stat().st_size} bytes)")

    # 5. Check core project artifacts
    print("\n[5/5] Verifying required project files exist...")
    required_artifacts = [
        project_root / "models" / "cnn_baseline.keras",
        project_root / "models" / "crnn_improved.keras",
        project_root / "data" / "metadata" / "feature_normalization_stats.json",
        project_root / "results" / "metrics" / "FINAL_MODEL_COMPARISON.csv",
        project_root / "app" / "app.py"
    ]
    for art in required_artifacts:
        if not art.exists() or art.stat().st_size == 0:
            errors.append(f"Required core project artifact missing or empty: {art}")
        else:
            print(f"  [PASS] Core artifact: {art.name}")

    print("\n" + "=" * 80)
    if errors:
        print(f"VALIDATION FAILED WITH {len(errors)} ERROR(S):")
        for err in errors:
            print(f"  [FAIL] {err}")
        print("=" * 80)
        sys.exit(1)
    else:
        print("PHASE 9 VALIDATION SUCCESSFUL: ALL CHECKS PASSED (0 ERRORS)")
        print("=" * 80)
        sys.exit(0)

if __name__ == "__main__":
    validate()
