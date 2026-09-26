"""
Training Orchestrator Module for AI Voice Deepfake Detection.

Provides full pipeline to:
1. Scan and split dataset (stratified, seeded).
2. Extract acoustic features (statistical for baseline, 2D tensors for CNN/CRNN).
3. Train baseline (Scikit-learn / Keras MLP) or improved (CNN2D / CRNN) models.
4. Save model checkpoints, training logs, and evaluation metrics.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from tensorflow import keras
from tqdm import tqdm

from src.baseline_model import (
    build_mlp_baseline,
    build_sklearn_baseline,
    save_baseline_model,
)
from src.config import config, set_seed
from src.dataset import scan_dataset, split_dataset
from src.features import extract_2d_features, extract_statistical_features
from src.improved_model import (
    build_cnn2d_model,
    build_crnn_model,
    save_improved_model,
)
from src.preprocessing import preprocess_audio


def extract_dataset_features(
    df: pd.DataFrame,
    feature_mode: str = "2d",  # '1d' for baseline stats, '2d' for spectrograms
    feature_type: str = "melspectrogram",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process all audio files in a DataFrame and extract features and labels.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'file_path' and 'label'.
        feature_mode (str): '1d' (statistical vectors) or '2d' (time-frequency matrices).
        feature_type (str): 'melspectrogram' or 'mfcc'.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X_features, y_labels)
    """
    X_list = []
    y_list = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {feature_mode} features"):
        file_path = row["file_path"]
        label = int(row["label"])
        
        try:
            # 1. Preprocess waveform
            waveform = preprocess_audio(file_path, audio_cfg=config.audio)
            
            # 2. Extract features
            if feature_mode == "1d":
                feats = extract_statistical_features(waveform, audio_cfg=config.audio)
            elif feature_mode == "2d":
                feats = extract_2d_features(waveform, audio_cfg=config.audio, feature_type=feature_type)
            else:
                raise ValueError(f"Unknown feature_mode: {feature_mode}")
                
            X_list.append(feats)
            y_list.append(label)
        except Exception as e:
            print(f"[Warning] Failed to process {file_path}: {e}")

    if len(X_list) == 0:
        raise RuntimeError("No features could be extracted from the dataset.")

    return np.array(X_list), np.array(y_list)


def train_sklearn_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_type: str = "rf",
    seed: int = 42,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Train and evaluate a Scikit-Learn baseline model on statistical acoustic features."""
    set_seed(seed)
    
    print(f"--- Training Scikit-Learn Baseline ({model_type.upper()}) ---")
    X_train, y_train = extract_dataset_features(train_df, feature_mode="1d")
    X_val, y_val = extract_dataset_features(val_df, feature_mode="1d")

    pipeline = build_sklearn_baseline(model_type=model_type, seed=seed)
    pipeline.fit(X_train, y_train)

    train_acc = pipeline.score(X_train, y_train)
    val_acc = pipeline.score(X_val, y_val)
    print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if output_path:
        save_baseline_model(pipeline, output_path)
        print(f"Saved baseline model to {output_path}")

    return {"train_acc": float(train_acc), "val_acc": float(val_acc)}


def train_deep_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_architecture: str = "cnn2d",  # 'cnn2d', 'crnn', or 'mlp'
    feature_type: str = "melspectrogram",
    epochs: int = 20,
    batch_size: int = 32,
    seed: int = 42,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Train a deep neural network (Keras MLP, CNN2D, or CRNN) on extracted acoustic features."""
    set_seed(seed)
    feature_mode = "1d" if model_architecture == "mlp" else "2d"

    print(f"--- Extracting Features for {model_architecture.upper()} ---")
    X_train, y_train = extract_dataset_features(train_df, feature_mode=feature_mode, feature_type=feature_type)
    X_val, y_val = extract_dataset_features(val_df, feature_mode=feature_mode, feature_type=feature_type)

    input_shape = X_train[0].shape
    print(f"Input feature shape: {input_shape}")

    # Build model
    if model_architecture == "mlp":
        model = build_mlp_baseline(input_dim=input_shape[0])
    elif model_architecture == "cnn2d":
        model = build_cnn2d_model(input_shape=input_shape)
    elif model_architecture == "crnn":
        model = build_crnn_model(input_shape=input_shape)
    else:
        raise ValueError(f"Unknown architecture: {model_architecture}")

    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.training.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.training.reduce_lr_factor,
            patience=config.training.reduce_lr_patience,
            min_lr=config.training.min_lr,
            verbose=1,
        ),
    ]

    if output_path:
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            filepath=str(output_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        )
        callbacks.append(checkpoint_cb)

    print(f"\n--- Starting Training for {epochs} Epochs ---")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if output_path and not Path(output_path).exists():
        save_improved_model(model, output_path)

    return model, history


def main():
    """Command-line training entrypoint."""
    parser = argparse.ArgumentParser(description="Train AI Voice Deepfake Detection Models")
    parser.add_argument("--data-dir", type=str, default=str(config.paths.raw_data_dir), help="Path to raw audio dataset")
    parser.add_argument("--metadata", type=str, default=None, help="Path to optional dataset metadata CSV/JSON")
    parser.add_argument("--model-type", type=str, default="cnn2d", choices=["rf", "lr", "mlp", "cnn2d", "crnn"], help="Model architecture")
    parser.add_argument("--feature-type", type=str, default="melspectrogram", choices=["melspectrogram", "mfcc"], help="Acoustic feature type")
    parser.add_argument("--epochs", type=int, default=config.training.epochs, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.training.batch_size, help="Training batch size")
    parser.add_argument("--seed", type=int, default=config.training.seed, help="Random seed")
    parser.add_argument("--output-model", type=str, default=None, help="Custom model checkpoint save path")
    args = parser.parse_args()

    config.initialize()
    set_seed(args.seed)

    print(f"Scanning dataset from: {args.data_dir}")
    df = scan_dataset(args.data_dir, metadata_path=args.metadata)
    print(f"Found {len(df)} audio samples.")
    if len(df) == 0:
        print("[Notice] No audio files detected. Place dataset in data/raw/ before initiating training.")
        return

    train_df, val_df, test_df = split_dataset(df, seed=args.seed)
    print(f"Dataset split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    default_save_name = f"{args.model_type}_{args.feature_type}.keras" if args.model_type in ["mlp", "cnn2d", "crnn"] else f"{args.model_type}_baseline.joblib"
    output_path = Path(args.output_model) if args.output_model else (config.paths.models_dir / default_save_name)

    if args.model_type in ["rf", "lr"]:
        train_sklearn_baseline(train_df, val_df, model_type=args.model_type, seed=args.seed, output_path=output_path)
    else:
        train_deep_model(
            train_df,
            val_df,
            model_architecture=args.model_type,
            feature_type=args.feature_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
