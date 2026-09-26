"""
Baseline Model Module for AI Voice Deepfake Detection.

Provides:
1. Scikit-Learn baseline pipelines (Random Forest / Logistic Regression / SVM)
   trained on 1D aggregated acoustic statistical features.
2. Lightweight Keras MLP baseline for neural comparison.
"""

from typing import Optional, Tuple, Union
import joblib
from pathlib import Path
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import TrainingConfig, config


def build_cnn_baseline(
    input_shape: Tuple[int, int, int] = (128, 126, 1),
    learning_rate: float = 0.001,
    dropout_rate: float = 0.25,
) -> keras.Model:
    """
    Construct the baseline 2D Convolutional Neural Network for Log-Mel Spectrogram deepfake detection.
    
    Architecture:
    Input (128, 126, 1)
    → Conv2D(32, (3, 3), padding='same') → BatchNorm → ReLU → MaxPool2D(2, 2) → Dropout(0.25)
    → Conv2D(64, (3, 3), padding='same') → BatchNorm → ReLU → MaxPool2D(2, 2) → Dropout(0.25)
    → Conv2D(128, (3, 3), padding='same') → BatchNorm → ReLU → MaxPool2D(2, 2) → Dropout(0.25)
    → GlobalAveragePooling2D
    → Dense(128, ReLU) → Dropout(0.25)
    → Dense(1, sigmoid)
    
    Args:
        input_shape: Dimensions of log-mel spectrogram (freq_bins, time_frames, channels).
        learning_rate: Initial Adam learning rate.
        dropout_rate: Regularization dropout probability.
        
    Returns:
        Compiled Keras Model.
    """
    inputs = layers.Input(shape=input_shape, name="log_mel_input")
    
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
    
    # Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_rate, name="drop4")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_baseline")
    
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


def build_sklearn_baseline(
    model_type: str = "rf",
    seed: int = 42,
) -> Pipeline:
    """
    Build a Scikit-Learn classification pipeline with standard scaling.
    
    Args:
        model_type (str): 'rf' for Random Forest, 'lr' for Logistic Regression.
        seed (int): Random state for reproducibility.
        
    Returns:
        Pipeline: Scikit-learn Pipeline object.
    """
    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )
    elif model_type == "lr":
        clf = LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'rf' or 'lr'.")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])
    return pipeline


def build_mlp_baseline(
    input_dim: int,
    training_cfg: Optional[TrainingConfig] = None,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    Construct a lightweight Multi-Layer Perceptron (MLP) baseline in Keras.
    
    Args:
        input_dim (int): Number of input statistical features.
        training_cfg (Optional[TrainingConfig]): Hyperparameter configuration.
        dropout_rate (float): Dropout probability for regularization.
        
    Returns:
        keras.Model: Compiled Keras binary classification model.
    """
    cfg = training_cfg or config.training

    inputs = layers.Input(shape=(input_dim,), name="audio_feature_input")
    x = layers.Dense(128, activation="relu", name="dense_1")(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    x = layers.Dense(32, activation="relu", name="dense_3")(x)
    x = layers.Dropout(dropout_rate / 2, name="dropout_3")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="prediction_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="baseline_mlp")

    optimizer = keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"), keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )
    return model


def save_baseline_model(
    model: Union[Pipeline, keras.Model],
    save_path: Union[str, Path],
) -> None:
    """Save baseline model to disk."""
    target_path = Path(save_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(model, Pipeline):
        joblib.dump(model, target_path)
    elif isinstance(model, keras.Model):
        model.save(str(target_path))
    else:
        raise TypeError(f"Unknown model type: {type(model)}")


def load_baseline_model(
    load_path: Union[str, Path],
    is_keras: bool = False,
) -> Union[Pipeline, keras.Model]:
    """Load baseline model from disk."""
    target_path = Path(load_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Model file not found: {target_path}")

    if is_keras or target_path.suffix in [".keras", ".h5"]:
        return keras.models.load_model(str(target_path))
    return joblib.load(target_path)
