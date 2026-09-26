"""
Audio Preprocessing Module for AI Voice Deepfake Detection.

Handles audio loading, mono conversion, silence trimming, normalization,
and fixed-length padding/truncation for reproducibility.
"""

from pathlib import Path
from typing import Optional, Union
import librosa
import numpy as np
import soundfile as sf

from src.config import AudioConfig, config


def load_audio(
    file_path: Union[str, Path],
    sr: Optional[int] = None,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file using librosa with soundfile fallback.
    
    Args:
        file_path (Union[str, Path]): Path to target audio file.
        sr (Optional[int]): Target sample rate. If None, uses original audio sample rate.
        mono (bool): Convert multi-channel audio to mono.
        
    Returns:
        tuple[np.ndarray, int]: (audio_waveform, sample_rate)
    """
    target_path = Path(file_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Audio file not found: {target_path}")

    target_sr = sr or config.audio.sample_rate

    try:
        y, sample_rate = librosa.load(str(target_path), sr=target_sr, mono=mono)
    except Exception:
        # Fallback to soundfile for certain non-standard headers
        y, sample_rate = sf.read(str(target_path))
        if mono and y.ndim > 1:
            y = np.mean(y, axis=1)
        if target_sr is not None and sample_rate != target_sr:
            y = librosa.resample(y, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr

    return y.astype(np.float32), sample_rate


def trim_silence(y: np.ndarray, top_db: float = 20.0) -> np.ndarray:
    """
    Trim leading and trailing silence from audio signal.
    
    Args:
        y (np.ndarray): Audio signal.
        top_db (float): The threshold (in decibels) below reference to consider as silence.
        
    Returns:
        np.ndarray: Trimmed audio signal.
    """
    if len(y) == 0:
        return y
    trimmed_y, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed_y if len(trimmed_y) > 0 else y


def normalize_audio(y: np.ndarray, method: str = "peak", eps: float = 1e-9) -> np.ndarray:
    """
    Normalize audio waveform amplitude.
    
    Args:
        y (np.ndarray): Audio signal array.
        method (str): 'peak' (normalize by max absolute value) or 'rms' (root-mean-square).
        eps (float): Numerical stability epsilon to avoid divide-by-zero.
        
    Returns:
        np.ndarray: Normalized audio signal.
    """
    if len(y) == 0:
        return y

    if method == "peak":
        peak = np.max(np.abs(y))
        if peak > eps:
            return y / peak
        return y
    elif method == "rms":
        rms = np.sqrt(np.mean(y ** 2))
        if rms > eps:
            return y / rms
        return y
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def pad_or_truncate(y: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Adjust audio length to exactly target_samples via zero-padding or central/tail truncation.
    
    Args:
        y (np.ndarray): Input audio array.
        target_samples (int): Desired exact number of samples.
        
    Returns:
        np.ndarray: Audio array of shape (target_samples,).
    """
    current_samples = len(y)
    if current_samples == target_samples:
        return y
    elif current_samples > target_samples:
        # Truncate to target length from start
        return y[:target_samples]
    else:
        # Zero-pad remaining samples at the end
        pad_width = target_samples - current_samples
        return np.pad(y, (0, pad_width), mode="constant", constant_values=0.0)


def preprocess_audio(
    file_path: Union[str, Path],
    audio_cfg: Optional[AudioConfig] = None,
    trim: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Complete audio preprocessing pipeline:
    1. Load audio with target sample rate and mono conversion.
    2. Optional silence trimming.
    3. Peak amplitude normalization.
    4. Deterministic fixed-length padding/truncation.
    
    Args:
        file_path (Union[str, Path]): Path to audio file.
        audio_cfg (Optional[AudioConfig]): Audio configuration. Defaults to config.audio.
        trim (bool): Whether to trim silence.
        normalize (bool): Whether to normalize amplitude.
        
    Returns:
        np.ndarray: Standardized audio waveform array of shape (target_num_samples,).
    """
    cfg = audio_cfg or config.audio
    
    # 1. Load
    y, _ = load_audio(file_path, sr=cfg.sample_rate, mono=True)
    
    # 2. Trim silence
    if trim:
        y = trim_silence(y, top_db=cfg.trim_db)
        
    # 3. Normalize
    if normalize:
        y = normalize_audio(y, method="peak")
        
    # 4. Pad or truncate to deterministic sample count
    y = pad_or_truncate(y, target_samples=cfg.target_num_samples)
    
    return y.astype(np.float32)
