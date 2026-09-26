"""
Improved Deep Learning Model Architectures for AI Voice Deepfake Detection.

Provides:
1. 2D Deep Convolutional Neural Network (CNN2D) with Batch Normalization,
   residual-style convolutions, and Global Average Pooling.
2. Convolutional Recurrent Neural Network (CRNN) with Bidirectional GRU
   for simultaneous local spectral feature extraction and temporal modeling.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import TrainingConfig, config


def build_cnn2d_model(
    input_shape: Tuple[int, int, int],
    training_cfg: Optional[TrainingConfig] = None,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    Construct an improved 2D Deep CNN for Mel-Spectrogram or MFCC image-like representations.
    
    Architecture highlights:
    - 4 Conv Blocks with progressive filter sizes (32 -> 64 -> 128 -> 256)
    - Batch Normalization and ReLU activations
    - SpatialDropout2D to prevent overfitting to local time-frequency clusters
    - GlobalAveragePooling2D to eliminate spatial collapse and keep parameter count low
    - Regularized classification head
    
    Args:
        input_shape (Tuple[int, int, int]): (freq_bins, time_steps, channels).
        training_cfg (Optional[TrainingConfig]): Hyperparameter settings.
        dropout_rate (float): Regularization dropout rate.
        
    Returns:
        keras.Model: Compiled 2D CNN model.
    """
    cfg = training_cfg or config.training

    inputs = layers.Input(shape=input_shape, name="spectrogram_input")

    # Block 1
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.ReLU(name="relu1_1")(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.ReLU(name="relu1_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.SpatialDropout2D(dropout_rate / 2, name="sdropout1")(x)

    # Block 2
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.ReLU(name="relu2_1")(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.ReLU(name="relu2_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.SpatialDropout2D(dropout_rate / 2, name="sdropout2")(x)

    # Block 3
    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.ReLU(name="relu3_1")(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.ReLU(name="relu3_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.SpatialDropout2D(dropout_rate, name="sdropout3")(x)

    # Block 4
    x = layers.Conv2D(256, kernel_size=(3, 3), padding="same", name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.ReLU(name="relu4_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)
    x = layers.SpatialDropout2D(dropout_rate, name="sdropout4")(x)

    # Global aggregation
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense_feat")(x)
    x = layers.Dropout(dropout_rate, name="dropout_head")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="prediction_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="improved_cnn2d")

    optimizer = keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_crnn_model(
    input_shape: Tuple[int, int, int] = (128, 126, 1),
    learning_rate: float = 0.0005,
    gru_units: int = 64,
    dropout_rate: float = 0.25,
) -> keras.Model:
    """
    Construct a Convolutional Recurrent Neural Network (CRNN) for Voice Deepfake Detection.
    
    Architecture:
    Input: (128, 126, 1)
    → Conv2D(32, (3, 3), same) → BatchNorm → ReLU → MaxPool2D(2, 2) → Dropout(0.25)
    → Conv2D(64, (3, 3), same) → BatchNorm → ReLU → MaxPool2D(2, 2) → Dropout(0.25)
    → Conv2D(128, (3, 3), same) → BatchNorm → ReLU → MaxPool2D(2, 2) → Dropout(0.25)
    → Permute((2, 1, 3)) + Reshape((-1, freq * channels)) [time as sequence]
    → Bidirectional(GRU(64, return_sequences=False))
    → Dense(64, ReLU) → Dropout(0.4)
    → Dense(1, Sigmoid)
    
    Args:
        input_shape: (freq_bins, time_steps, channels).
        learning_rate: Adam initial learning rate (0.0005).
        gru_units: Number of units in GRU layer.
        dropout_rate: Feature dropout rate.
        
    Returns:
        Compiled CRNN Keras Model.
    """
    inputs = layers.Input(shape=input_shape, name="audio_spectrogram")

    # Front-end Convolution Blocks
    # Block 1
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    # Block 2
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(dropout_rate, name="drop2")(x)

    # Block 3
    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(dropout_rate, name="drop3")(x)

    # Sequence conversion: time as sequence dimension, freq * channels as feature dimension
    x = layers.Permute((2, 1, 3), name="time_as_seq")(x)
    freq_dim = x.shape[2]
    chan_dim = x.shape[3]
    x = layers.Reshape((-1, freq_dim * chan_dim), name="seq_reshape")(x)

    # Recurrent Temporal Modeling
    x = layers.Bidirectional(layers.GRU(gru_units, return_sequences=False), name="bi_gru")(x)

    # Dense Classification Head
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.4, name="drop4")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="improved_crnn")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def save_improved_model(model: keras.Model, save_path: Union[str, Path]) -> None:
    """Save improved model in standard Keras format."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def load_improved_model(load_path: Union[str, Path]) -> keras.Model:
    """Load improved model from disk."""
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return keras.models.load_model(str(path))
