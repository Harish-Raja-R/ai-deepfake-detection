"""
Configuration Module for AI Voice Deepfake Detection.

Provides centralized configuration for:
- Random seeds and reproducibility
- Directory paths (configurable via environment variables or parameters)
- Audio preprocessing parameters
- Feature extraction settings
- Model training and evaluation hyperparameters
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seed across all libraries to ensure reproducible experiments.
    
    Args:
        seed (int): Random seed value. Defaults to 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Enable deterministic operations if supported
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except ImportError:
        pass


@dataclass
class AudioConfig:
    """Audio signal processing and feature extraction configuration."""
    sample_rate: int = 16000
    duration: float = 4.0  # seconds
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 512
    n_mfcc: int = 40
    fmin: float = 20.0
    fmax: Optional[float] = 8000.0
    top_db: float = 80.0  # Dynamic range for dB conversion
    trim_db: float = 20.0  # Threshold for silence trimming

    @property
    def target_num_samples(self) -> int:
        """Target length in audio samples."""
        return int(self.sample_rate * self.duration)


@dataclass
class TrainingConfig:
    """Hyperparameters for model training."""
    seed: int = 42
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.15
    test_split: float = 0.15
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6


@dataclass
class PathConfig:
    """Directory structure configuration with configurable base paths."""
    # Defaults to project root (parent of src/)
    base_dir: Path = field(default_factory=lambda: Path(os.getenv(
        "PROJECT_ROOT",
        Path(__file__).resolve().parent.parent
    )))

    # Raw, processed, and metadata directories
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    metadata_dir: Path = field(init=False)

    # Models and experiment outputs
    models_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    metrics_dir: Path = field(init=False)
    predictions_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = Path(os.getenv("DATA_DIR", self.base_dir / "data"))
        self.raw_data_dir = Path(os.getenv("RAW_DATA_DIR", self.data_dir / "raw"))
        self.processed_data_dir = Path(os.getenv("PROCESSED_DATA_DIR", self.data_dir / "processed"))
        self.metadata_dir = Path(os.getenv("METADATA_DIR", self.data_dir / "metadata"))

        self.models_dir = Path(os.getenv("MODELS_DIR", self.base_dir / "models"))
        self.results_dir = Path(os.getenv("RESULTS_DIR", self.base_dir / "results"))
        self.figures_dir = Path(os.getenv("FIGURES_DIR", self.results_dir / "figures"))
        self.metrics_dir = Path(os.getenv("METRICS_DIR", self.results_dir / "metrics"))
        self.predictions_dir = Path(os.getenv("PREDICTIONS_DIR", self.results_dir / "predictions"))

    def create_directories(self) -> None:
        """Ensure all required output directories exist."""
        for path in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.metadata_dir,
            self.models_dir,
            self.figures_dir,
            self.metrics_dir,
            self.predictions_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ProjectConfig:
    """Global configuration wrapper."""
    paths: PathConfig = field(default_factory=PathConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    label_map: Dict[str, int] = field(default_factory=lambda: {
        "real": 0,
        "genuine": 0,
        "human": 0,
        "bonafide": 0,
        "fake": 1,
        "deepfake": 1,
        "spoof": 1,
        "synthetic": 1,
        "ai_generated": 1,
    })

    def initialize(self) -> None:
        """Apply random seed and ensure directories exist."""
        set_seed(self.training.seed)
        self.paths.create_directories()


# Default singleton instance
config = ProjectConfig()
