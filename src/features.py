"""
Feature Extraction Module for AI Voice Deepfake Detection.

Extracts acoustic representations from preprocessed audio waveforms:
- Mel-Frequency Spectrograms (dB scale)
- MFCCs (with first and second order deltas)
- Spectral Contrast and Chroma
- Aggregated 1D statistical feature vectors for baseline models
- 2D/3D feature tensors formatted for convolutional neural networks
- Leakage-safe normalization strictly derived from training set statistics
- Reusable model-ready dataset loaders and batch generators
"""

import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import librosa
import numpy as np
import pandas as pd

from src.config import AudioConfig, config


def extract_mel_spectrogram(
    y: np.ndarray,
    audio_cfg: Optional[AudioConfig] = None,
) -> np.ndarray:
    """
    Extract log-scaled Mel-spectrogram in decibels (dB).
    
    Args:
        y (np.ndarray): Audio waveform signal.
        audio_cfg (Optional[AudioConfig]): Audio configuration parameters.
        
    Returns:
        np.ndarray: Mel-spectrogram in dB of shape (n_mels, time_steps).
    """
    cfg = audio_cfg or config.audio
    
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )
    # Convert power spectrogram to decibels
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=cfg.top_db)
    return mel_spec_db.astype(np.float32)


def extract_mfcc(
    y: np.ndarray,
    audio_cfg: Optional[AudioConfig] = None,
    include_deltas: bool = True,
) -> np.ndarray:
    """
    Extract MFCCs along with first (velocity) and second (acceleration) order deltas.
    
    Args:
        y (np.ndarray): Audio signal array.
        audio_cfg (Optional[AudioConfig]): Audio configuration.
        include_deltas (bool): Whether to append delta and delta-delta features.
        
    Returns:
        np.ndarray: MFCC feature matrix of shape (n_features, time_steps).
    """
    cfg = audio_cfg or config.audio
    
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=cfg.sample_rate,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    
    if not include_deltas:
        return mfcc.astype(np.float32)
        
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    stacked = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    return stacked.astype(np.float32)


def extract_statistical_features(
    y: np.ndarray,
    audio_cfg: Optional[AudioConfig] = None,
) -> np.ndarray:
    """
    Extract 1D summary statistical acoustic features (mean, std) across multiple domains.
    Designed for traditional ML baseline models (Logistic Regression, Random Forest, SVM).
    
    Features extracted:
    - MFCC statistics (mean, std)
    - Mel-spectrogram statistics (mean, std)
    - Spectral Centroid (mean, std)
    - Spectral Rolloff (mean, std)
    - Spectral Contrast (mean, std)
    - Zero Crossing Rate (mean, std)
    - Chroma STFT (mean, std)
    
    Args:
        y (np.ndarray): Audio signal array.
        audio_cfg (Optional[AudioConfig]): Audio configuration.
        
    Returns:
        np.ndarray: 1D concatenated feature vector.
    """
    cfg = audio_cfg or config.audio
    feature_list = []

    # 1. MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=cfg.sample_rate, n_mfcc=cfg.n_mfcc, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    feature_list.extend([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])

    # 2. Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    feature_list.extend([[np.mean(cent)], [np.std(cent)]])

    # 3. Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    feature_list.extend([[np.mean(rolloff)], [np.std(rolloff)]])

    # 4. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=cfg.hop_length)
    feature_list.extend([[np.mean(zcr)], [np.std(zcr)]])

    # 5. Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    feature_list.extend([np.mean(contrast, axis=1), np.std(contrast, axis=1)])

    # 6. Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    feature_list.extend([np.mean(chroma, axis=1), np.std(chroma, axis=1)])

    # Concatenate into 1D flat vector
    return np.hstack([f.flatten() for f in feature_list]).astype(np.float32)


def extract_2d_features(
    y: np.ndarray,
    audio_cfg: Optional[AudioConfig] = None,
    feature_type: str = "melspectrogram",
) -> np.ndarray:
    """
    Extract 2D time-frequency representation formatted for 2D Deep Learning models.
    Output shape: (freq_bins, time_frames, channels=1).
    
    Args:
        y (np.ndarray): Audio signal.
        audio_cfg (Optional[AudioConfig]): Audio configuration.
        feature_type (str): 'melspectrogram' or 'mfcc'.
        
    Returns:
        np.ndarray: 3D array of shape (freq_bins, time_frames, 1).
    """
    cfg = audio_cfg or config.audio
    
    if feature_type == "melspectrogram":
        feat = extract_mel_spectrogram(y, audio_cfg=cfg)
    elif feature_type == "mfcc":
        feat = extract_mfcc(y, audio_cfg=cfg, include_deltas=True)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    # Expand channel dimension: (height, width, 1)
    feat_3d = np.expand_dims(feat, axis=-1)
    return feat_3d.astype(np.float32)


def compute_training_normalization_stats(
    train_csv_path: Union[str, Path],
    audio_dir: Union[str, Path],
    output_json_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Calculate normalization statistics STRICTLY derived from the training partition.
    Never uses validation or test audio to prevent label or distribution leakage.
    
    Computes:
    - Per-Mel-bin mean: shape (128, 1)
    - Per-Mel-bin standard deviation: shape (128, 1)
    - Global scalar mean and standard deviation
    
    Args:
        train_csv_path: Path to train.csv.
        audio_dir: Path to preprocessed waveforms directory (data/processed/audio/).
        output_json_path: Optional destination to save statistics as JSON.
        
    Returns:
        Dict containing training normalization parameters.
    """
    train_df = pd.read_csv(train_csv_path)
    audio_path = Path(audio_dir)
    
    specs = []
    for sample_id in train_df["sample_id"]:
        y = np.load(audio_path / f"{sample_id}.npy")
        mel = extract_mel_spectrogram(y)
        specs.append(mel)
        
    stacked_specs = np.stack(specs, axis=0)  # (N_train, 128, 126)
    
    # Per-frequency-bin statistics across all training samples and time steps
    per_bin_mean = np.mean(stacked_specs, axis=(0, 2), keepdims=True)[0]  # (128, 1)
    per_bin_std = np.std(stacked_specs, axis=(0, 2), keepdims=True)[0]    # (128, 1)
    global_mean = float(np.mean(stacked_specs))
    global_std = float(np.std(stacked_specs))
    
    stats = {
        "n_train_samples": len(train_df),
        "n_mels": stacked_specs.shape[1],
        "n_time_frames": stacked_specs.shape[2],
        "global_mean": global_mean,
        "global_std": global_std,
        "per_bin_mean": per_bin_mean.flatten().tolist(),
        "per_bin_std": per_bin_std.flatten().tolist(),
    }
    
    if output_json_path:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)
            
    return stats


def load_normalization_stats(stats_json_path: Union[str, Path]) -> Dict[str, Any]:
    """Load previously calculated training normalization statistics."""
    with open(stats_json_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return stats


def normalize_spectrogram(
    mel_spec: np.ndarray,
    stats: Dict[str, Any],
    mode: str = "per_bin",
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Normalize a Mel-spectrogram using learned statistics from the training set.
    
    Args:
        mel_spec: 2D Mel-spectrogram of shape (128, 126).
        stats: Dictionary containing training normalization statistics.
        mode: 'per_bin' for per-frequency-band z-score, or 'global' for scalar z-score.
        eps: Epsilon to avoid division by zero.
        
    Returns:
        np.ndarray: Normalized 3D tensor of shape (128, 126, 1) float32.
    """
    if mode == "per_bin":
        mean = np.array(stats["per_bin_mean"], dtype=np.float32).reshape(-1, 1)
        std = np.array(stats["per_bin_std"], dtype=np.float32).reshape(-1, 1)
        norm_spec = (mel_spec - mean) / (std + eps)
    elif mode == "global":
        norm_spec = (mel_spec - stats["global_mean"]) / (stats["global_std"] + eps)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}. Choose 'per_bin' or 'global'.")
        
    # Format for 2D CNN (freq_bins, time_frames, 1)
    return np.expand_dims(norm_spec, axis=-1).astype(np.float32)


def load_split_features(
    split: str,
    meta_dir: Optional[Union[str, Path]] = None,
    features_dir: Optional[Union[str, Path]] = None,
    return_ids: bool = False,
    mmap_mode: Optional[str] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    Efficiently load feature tensors and labels for a specific partition ('train', 'validation', 'test').
    Maps 'REAL' -> 0, 'FAKE' -> 1.
    
    Args:
        split: 'train', 'validation', or 'test'.
        meta_dir: Directory containing partition CSVs. Defaults to data/metadata.
        features_dir: Directory containing extracted .npy features. Defaults to data/processed/features.
        return_ids: Whether to also return the list of sample IDs.
        mmap_mode: Optional memory-map mode for numpy loading (e.g. 'r').
        
    Returns:
        (X, y) or (X, y, sample_ids) where X has shape (N, 128, 126, 1) and y has shape (N,).
    """
    m_dir = Path(meta_dir) if meta_dir else config.paths.metadata_dir
    f_dir = Path(features_dir) if features_dir else config.paths.processed_data_dir / "features"
    
    csv_file = m_dir / f"{split}.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Partition metadata not found: {csv_file}")
        
    df = pd.read_csv(csv_file)
    sample_ids = df["sample_id"].tolist()
    labels = df["label"].map({"REAL": 0, "FAKE": 1}).values.astype(np.int32)
    
    tensors = []
    for sid in sample_ids:
        npy_path = f_dir / f"{sid}.npy"
        feat = np.load(npy_path, mmap_mode=mmap_mode)
        tensors.append(feat)
        
    X = np.stack(tensors, axis=0).astype(np.float32)
    
    if return_ids:
        return X, labels, sample_ids
    return X, labels


def get_model_ready_data(
    meta_dir: Optional[Union[str, Path]] = None,
    features_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load all three model-ready dataset partitions without memory duplication:
    Returns (X_train, y_train), (X_val, y_val), (X_test, y_test).
    """
    X_train, y_train = load_split_features("train", meta_dir=meta_dir, features_dir=features_dir)
    X_val, y_val = load_split_features("validation", meta_dir=meta_dir, features_dir=features_dir)
    X_test, y_test = load_split_features("test", meta_dir=meta_dir, features_dir=features_dir)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_batch_generator(
    split: str,
    meta_dir: Optional[Union[str, Path]] = None,
    features_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yields batches of features and labels lazily for CPU-constrained environments.
    Reads .npy feature files on-the-fly to keep memory footprint minimal.
    """
    m_dir = Path(meta_dir) if meta_dir else config.paths.metadata_dir
    f_dir = Path(features_dir) if features_dir else config.paths.processed_data_dir / "features"
    
    df = pd.read_csv(m_dir / f"{split}.csv")
    sample_ids = df["sample_id"].values
    labels = df["label"].map({"REAL": 0, "FAKE": 1}).values.astype(np.int32)
    n = len(df)
    indices = np.arange(n)
    
    rng = np.random.RandomState(seed)
    
    while True:
        if shuffle:
            rng.shuffle(indices)
            
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_x = np.stack([np.load(f_dir / f"{sample_ids[i]}.npy") for i in batch_idx], axis=0)
            batch_y = labels[batch_idx]
            yield batch_x.astype(np.float32), batch_y.astype(np.int32)


def create_tf_dataset(
    split: str,
    meta_dir: Optional[Union[str, Path]] = None,
    features_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Create a prefetching tf.data.Dataset for high-throughput TensorFlow training.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is required to create a tf.data.Dataset.")
        
    X, y = load_split_features(split, meta_dir=meta_dir, features_dir=features_dir)
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(y), seed=seed)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
