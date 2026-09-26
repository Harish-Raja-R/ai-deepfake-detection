"""
Script to verify model prediction on real and fake benchmark audio through the exact Streamlit pipeline.
"""
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.predict import (
    VoiceDeepfakeDetector,
    DEFAULT_MODEL_PATH,
    DEFAULT_STATS_PATH,
    CNN_EER_THRESHOLD,
    STANDARD_THRESHOLD,
)
from app.app import plot_audio_waveform, plot_spectrogram

def test_pipeline():
    detector = VoiceDeepfakeDetector(DEFAULT_MODEL_PATH, DEFAULT_STATS_PATH)

    real_file = project_root / "data" / "raw" / "real" / "yt_0000_p2_part_167.flac"
    fake_file = project_root / "data" / "raw" / "fake" / "el_0001_c_part_002.flac"

    for sample_type, file_path in [("REAL", real_file), ("FAKE", fake_file)]:
        print("=" * 65)
        print(f"Testing {sample_type} file: {file_path.name}")
        
        # 1. Metadata extraction
        meta = detector.get_audio_metadata(file_path)
        channels = meta["channels"]
        channel_text = "Mono (1)" if channels == 1 else f"Stereo ({channels})"
        print(f"  Metadata: format={meta['format']}, sr={meta['sample_rate']} Hz, duration={meta['duration_seconds']:.2f}s, channels={channel_text}")
        
        # 2. Preprocessing and feature extraction
        waveform, mel_spec, tensor = detector.process_and_extract(file_path)
        print(f"  Waveform shape: {waveform.shape}, range: [{waveform.min():.2f}, {waveform.max():.2f}]")
        print(f"  Mel spec shape: {mel_spec.shape}, tensor shape: {tensor.shape}")
        assert waveform.shape == (64000,), f"Expected 64000 samples, got {waveform.shape}"
        assert mel_spec.shape == (128, 126), f"Expected (128, 126), got {mel_spec.shape}"
        assert tensor.shape == (1, 128, 126, 1), f"Expected (1, 128, 126, 1), got {tensor.shape}"
        
        # 3. Visualizations
        fig_w = plot_audio_waveform(waveform)
        fig_s = plot_spectrogram(mel_spec)
        assert fig_w is not None and fig_s is not None
        print("  Waveform and Spectrogram figures rendered successfully")
        
        # 4. Standard threshold (0.50) prediction
        pred_std = detector.predict_file(file_path, threshold=STANDARD_THRESHOLD)
        print(f"  [Standard tau=0.50] P(FAKE)={pred_std['probability_fake']:.4f}, Verdict={pred_std['verdict']}, Confidence={pred_std['confidence_score_pct']:.1f}%")
        assert 0.0 <= pred_std["probability_fake"] <= 1.0
        assert pred_std["verdict"] in ["REAL / BONAFIDE", "AI-GENERATED / DEEPFAKE"]
        
        # 5. EER threshold (0.2879) prediction
        pred_eer = detector.predict_file(file_path, threshold=CNN_EER_THRESHOLD)
        print(f"  [EER tau={CNN_EER_THRESHOLD:.4f}] P(FAKE)={pred_eer['probability_fake']:.4f}, Verdict={pred_eer['verdict']}, Confidence={pred_eer['confidence_score_pct']:.1f}%")
        assert 0.0 <= pred_eer["probability_fake"] <= 1.0
        assert pred_eer["verdict"] in ["REAL / BONAFIDE", "AI-GENERATED / DEEPFAKE"]

    print("=" * 65)
    print("ALL BENCHMARK AUDIO PREDICTIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_pipeline()
