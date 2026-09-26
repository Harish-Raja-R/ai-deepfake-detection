"""
Inference & Prediction Pipeline for AI Voice Deepfake Detection.

Provides end-to-end inference functions to:
1. Process single audio files (.wav, .flac, .mp3).
2. Extract exact 128-band Log-Mel Spectrograms with training-set per-bin normalization.
3. Pass tensors to trained model checkpoints (default: models/cnn_baseline.keras).
4. Apply standard (0.50) or biometric EER (0.2879) decision thresholds.
5. Export batch predictions to results/predictions/.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from tensorflow import keras

from src.config import config
from src.features import extract_mel_spectrogram, load_normalization_stats, normalize_spectrogram
from src.preprocessing import preprocess_audio

SUPPORTED_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

DEFAULT_MODEL_PATH = config.paths.models_dir / "cnn_baseline.keras"
DEFAULT_STATS_PATH = config.paths.metadata_dir / "feature_normalization_stats.json"
STANDARD_THRESHOLD = 0.50
CNN_EER_THRESHOLD = 0.2879


class VoiceDeepfakeDetector:
    """End-to-end detector engine wrapping preprocessing, feature extraction, and CNN inference."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        stats_path: Optional[Union[str, Path]] = None,
        threshold: float = STANDARD_THRESHOLD,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        self.stats_path = Path(stats_path) if stats_path else DEFAULT_STATS_PATH
        if not self.stats_path.exists():
            raise FileNotFoundError(f"Feature normalization statistics not found: {self.stats_path}")

        self.threshold = float(threshold)
        self.stats = load_normalization_stats(self.stats_path)
        self.model = keras.models.load_model(str(self.model_path))

    def get_audio_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract acoustic file metadata from raw audio header."""
        path = Path(file_path)
        try:
            info = sf.info(str(path))
            return {
                "file_name": path.name,
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "duration_seconds": float(info.duration),
                "format": str(info.format),
                "subtype": str(info.subtype),
            }
        except Exception:
            # Fallback estimation
            import librosa
            y, sr = librosa.load(str(path), sr=None, mono=False)
            channels = 1 if y.ndim == 1 else y.shape[0]
            duration = len(y) / sr if channels == 1 else y.shape[1] / sr
            return {
                "file_name": path.name,
                "sample_rate": int(sr),
                "channels": int(channels),
                "duration_seconds": float(duration),
                "format": path.suffix.replace(".", "").upper(),
                "subtype": "UNKNOWN",
            }

    def process_and_extract(
        self, file_path: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute the exact training-time preprocessing and feature extraction pipeline:
        1. Mono conversion + 16 kHz resampling + silence trimming + peak normalization + 4.0s pad/trunc.
        2. 128-band Log-Mel Spectrogram extraction.
        3. Per-bin z-score normalization using training statistics.
        
        Returns:
            Tuple of:
            - waveform: shape (64000,) float32
            - mel_spec_db: shape (128, 126) float32 (decibel scale)
            - input_tensor: shape (1, 128, 126, 1) float32 (model ready)
        """
        # 1. Preprocess waveform
        waveform = preprocess_audio(file_path, audio_cfg=config.audio)
        if len(waveform) != config.audio.target_num_samples:
            raise ValueError(
                f"Waveform length {len(waveform)} != expected {config.audio.target_num_samples}"
            )

        # 2. Extract 128-band Mel-Spectrogram in dB
        mel_spec_db = extract_mel_spectrogram(waveform, audio_cfg=config.audio)
        if mel_spec_db.shape != (128, 126):
            raise ValueError(f"Mel-spectrogram shape {mel_spec_db.shape} != expected (128, 126)")

        # 3. Apply training-derived per-bin z-score normalization
        norm_spec = normalize_spectrogram(mel_spec_db, stats=self.stats, mode="per_bin")
        input_tensor = np.expand_dims(norm_spec, axis=0)  # Shape: (1, 128, 126, 1)

        return waveform, mel_spec_db, input_tensor

    def predict_file(
        self,
        file_path: Union[str, Path],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run complete deepfake detection on an audio file.
        
        Args:
            file_path: Path to target audio file (.wav, .flac, .mp3).
            threshold: Classification decision threshold (defaults to self.threshold).
            
        Returns:
            Dictionary containing prediction verdict, probabilities, metadata, and tensors.
        """
        thresh = float(threshold) if threshold is not None else self.threshold
        meta = self.get_audio_metadata(file_path)
        waveform, mel_spec_db, input_tensor = self.process_and_extract(file_path)

        # Run CNN inference
        raw_prob = float(self.model.predict(input_tensor, verbose=0).flatten()[0])
        prob_fake = float(np.clip(raw_prob, 0.0, 1.0))
        prob_real = float(1.0 - prob_fake)

        is_fake = prob_fake >= thresh
        verdict = "AI-GENERATED / DEEPFAKE" if is_fake else "REAL / BONAFIDE"
        confidence_pct = (prob_fake if is_fake else prob_real) * 100.0

        return {
            "file_path": str(Path(file_path).resolve()),
            "file_name": meta["file_name"],
            "metadata": meta,
            "waveform": waveform,
            "mel_spectrogram_db": mel_spec_db,
            "input_tensor": input_tensor,
            "probability_fake": round(prob_fake, 6),
            "probability_real": round(prob_real, 6),
            "probability_fake_pct": round(prob_fake * 100.0, 2),
            "probability_real_pct": round(prob_real * 100.0, 2),
            "is_deepfake": bool(is_fake),
            "verdict": verdict,
            "confidence_score_pct": round(confidence_pct, 1),
            "threshold_used": thresh,
            "model_name": "cnn_baseline",
        }

    def predict_batch(
        self,
        audio_dir_or_list: Union[str, Path, List[Union[str, Path]]],
        threshold: Optional[float] = None,
        save_csv_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Run detection on a directory of audio files or list of paths."""
        if isinstance(audio_dir_or_list, (str, Path)):
            folder = Path(audio_dir_or_list)
            file_paths = [p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_AUDIO_EXTS]
        else:
            file_paths = [Path(p) for p in audio_dir_or_list]

        results = []
        for p in file_paths:
            try:
                res = self.predict_file(p, threshold=threshold)
                # Keep serializable columns for dataframe
                results.append({
                    "file_name": res["file_name"],
                    "probability_fake": res["probability_fake"],
                    "probability_real": res["probability_real"],
                    "verdict": res["verdict"],
                    "is_deepfake": res["is_deepfake"],
                    "confidence_pct": res["confidence_score_pct"],
                    "duration_sec": res["metadata"]["duration_seconds"],
                    "sample_rate": res["metadata"]["sample_rate"],
                    "threshold": res["threshold_used"],
                })
            except Exception as e:
                print(f"[Warning] Inference error on {p.name}: {e}")

        df_results = pd.DataFrame(results)

        if save_csv_path:
            out_p = Path(save_csv_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            df_results.to_csv(out_p, index=False)
            print(f"Batch predictions exported to: {out_p}")

        return df_results


def main():
    """Command-line inference entrypoint."""
    parser = argparse.ArgumentParser(description="AI Voice Deepfake Inference Pipeline")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to model checkpoint")
    parser.add_argument("--audio-path", type=str, required=True, help="Path to audio file or folder")
    parser.add_argument("--threshold", type=float, default=STANDARD_THRESHOLD, help="Decision threshold (default: 0.50)")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to export batch predictions CSV")
    args = parser.parse_args()

    detector = VoiceDeepfakeDetector(
        model_path=args.model_path,
        threshold=args.threshold,
    )

    path_obj = Path(args.audio_path)
    if path_obj.is_file():
        result = detector.predict_file(path_obj)
        print("\n" + "=" * 50)
        print("AI VOICE DEEPFAKE DETECTION RESULT")
        print("=" * 50)
        print(f"File:                {result['file_name']}")
        print(f"Verdict:             {result['verdict']}")
        print(f"Model Confidence:    {result['confidence_score_pct']}%")
        print(f"Probability FAKE:    {result['probability_fake_pct']}%")
        print(f"Probability REAL:    {result['probability_real_pct']}%")
        print(f"Threshold:           {result['threshold_used']}")
        print("=" * 50)
    elif path_obj.is_dir():
        save_dest = args.output_csv or (config.paths.predictions_dir / "cli_batch_predictions.csv")
        df_preds = detector.predict_batch(path_obj, threshold=args.threshold, save_csv_path=save_dest)
        print(f"\nProcessed {len(df_preds)} audio files.")
    else:
        print(f"Error: Path does not exist: {args.audio_path}")


if __name__ == "__main__":
    main()
