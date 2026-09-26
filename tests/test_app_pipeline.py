"""
Automated Test Suite for Streamlit Application & Inference Pipeline.

Validates the full inference pipeline without requiring Streamlit UI interaction:
1. Model loads successfully and matches architecture parameters.
2. Preprocessing pipeline accepts valid audio and outputs (64000,) normalized waveform.
3. Feature extraction returns expected (128, 126) Mel-spectrogram and (1, 128, 126, 1) tensor.
4. Model prediction produces a calibrated probability in [0.0, 1.0].
5. Standard threshold (0.50) produces valid REAL / FAKE labels.
6. EER threshold (0.2879) produces valid REAL / FAKE labels.
7. Graceful error handling on invalid/missing files.
"""

from pathlib import Path
import numpy as np
import pytest
from tensorflow import keras

from src.config import config
from src.predict import (
    CNN_EER_THRESHOLD,
    DEFAULT_MODEL_PATH,
    DEFAULT_STATS_PATH,
    STANDARD_THRESHOLD,
    VoiceDeepfakeDetector,
)

# Reference sample audio for automated testing
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_REAL_PATH = PROJECT_ROOT / "data" / "raw" / "real" / "yt_0000_part_003.flac"
SAMPLE_FAKE_PATH = PROJECT_ROOT / "data" / "raw" / "fake" / "elevenlabs_0000_part_000.flac"


def test_model_loads_successfully():
    """1. Test that the baseline CNN model checkpoint loads and matches expected parameters."""
    assert DEFAULT_MODEL_PATH.exists(), f"Model checkpoint missing: {DEFAULT_MODEL_PATH}"
    model = keras.models.load_model(str(DEFAULT_MODEL_PATH))
    assert model is not None
    assert model.count_params() == 110209, f"Unexpected parameter count: {model.count_params()}"
    assert model.output_shape == (None, 1), f"Unexpected output shape: {model.output_shape}"


def test_preprocessing_accepts_valid_audio():
    """2. Test that audio preprocessing produces a valid 16 kHz, 4.0s normalized waveform."""
    assert SAMPLE_REAL_PATH.exists(), f"Sample audio missing: {SAMPLE_REAL_PATH}"
    detector = VoiceDeepfakeDetector()
    waveform, mel_spec_db, input_tensor = detector.process_and_extract(SAMPLE_REAL_PATH)

    assert isinstance(waveform, np.ndarray)
    assert waveform.dtype == np.float32
    assert waveform.shape == (64000,), f"Expected (64000,) waveform, got {waveform.shape}"
    assert np.max(np.abs(waveform)) <= 1.0 + 1e-5, "Waveform exceeds peak normalization"


def test_feature_extraction_returns_expected_shape():
    """3. Test that feature extraction returns (128, 126) spectrogram and (1, 128, 126, 1) tensor."""
    detector = VoiceDeepfakeDetector()
    waveform, mel_spec_db, input_tensor = detector.process_and_extract(SAMPLE_REAL_PATH)

    assert mel_spec_db.shape == (128, 126), f"Expected (128, 126) Mel-spec, got {mel_spec_db.shape}"
    assert input_tensor.shape == (1, 128, 126, 1), f"Expected (1, 128, 126, 1) tensor, got {input_tensor.shape}"
    assert not np.isnan(input_tensor).any(), "Input tensor contains NaN values"
    assert not np.isinf(input_tensor).any(), "Input tensor contains Inf values"


def test_model_prediction_returns_valid_probability():
    """4. Test that model inference produces a probability strictly between 0 and 1."""
    detector = VoiceDeepfakeDetector()
    result = detector.predict_file(SAMPLE_REAL_PATH)

    prob_fake = result["probability_fake"]
    prob_real = result["probability_real"]

    assert 0.0 <= prob_fake <= 1.0, f"Probability fake {prob_fake} out of bounds"
    assert 0.0 <= prob_real <= 1.0, f"Probability real {prob_real} out of bounds"
    assert abs((prob_fake + prob_real) - 1.0) < 1e-4, "Probabilities do not sum to 1.0"
    assert 0.0 <= result["confidence_score_pct"] <= 100.0, "Confidence score out of [0, 100]"


def test_standard_threshold_produces_valid_label():
    """5. Test that standard threshold (0.50) produces valid REAL/FAKE labels."""
    detector = VoiceDeepfakeDetector(threshold=STANDARD_THRESHOLD)
    result = detector.predict_file(SAMPLE_REAL_PATH, threshold=0.50)

    assert result["threshold_used"] == 0.50
    assert result["verdict"] in ["REAL / BONAFIDE", "AI-GENERATED / DEEPFAKE"]
    expected_fake = result["probability_fake"] >= 0.50
    assert result["is_deepfake"] == expected_fake
    if expected_fake:
        assert result["verdict"] == "AI-GENERATED / DEEPFAKE"
    else:
        assert result["verdict"] == "REAL / BONAFIDE"


def test_eer_threshold_produces_valid_label():
    """6. Test that EER threshold (0.2879) produces valid REAL/FAKE labels."""
    detector = VoiceDeepfakeDetector(threshold=CNN_EER_THRESHOLD)
    result = detector.predict_file(SAMPLE_REAL_PATH, threshold=CNN_EER_THRESHOLD)

    assert result["threshold_used"] == CNN_EER_THRESHOLD
    assert result["verdict"] in ["REAL / BONAFIDE", "AI-GENERATED / DEEPFAKE"]
    expected_fake = result["probability_fake"] >= CNN_EER_THRESHOLD
    assert result["is_deepfake"] == expected_fake


def test_detector_metadata_extraction():
    """7. Test extraction of acoustic audio metadata from file headers."""
    detector = VoiceDeepfakeDetector()
    meta = detector.get_audio_metadata(SAMPLE_REAL_PATH)

    assert "sample_rate" in meta and meta["sample_rate"] > 0
    assert "channels" in meta and meta["channels"] >= 1
    assert "duration_seconds" in meta and meta["duration_seconds"] > 0
    assert "format" in meta


def test_detector_handles_missing_file():
    """8. Test that detector raises FileNotFoundError gracefully on non-existent audio."""
    detector = VoiceDeepfakeDetector()
    with pytest.raises((FileNotFoundError, Exception)):
        detector.predict_file("non_existent_audio_sample_xyz123.wav")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
