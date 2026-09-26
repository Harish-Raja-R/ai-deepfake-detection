"""
Dataset Ingestion and Management Module.

Handles dataset scanning, flexible metadata creation, train/validation/test splitting,
and data generator preparation without assuming rigid directory structures.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import config, set_seed

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def infer_label_from_name(name: str, label_map: Dict[str, int]) -> Optional[int]:
    """
    Infer label from folder name or file stem using configured label mapping.
    
    Args:
        name (str): Directory name or filename to inspect.
        label_map (Dict[str, int]): Mapping of string keywords to numeric labels.
        
    Returns:
        Optional[int]: Numeric label (0 for real, 1 for fake) or None if unmapped.
    """
    name_lower = name.lower()
    for key, val in label_map.items():
        if key in name_lower:
            return val
    return None


def scan_dataset(
    data_dir: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
    label_map: Optional[Dict[str, int]] = None,
    save_metadata_to: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Scans a directory for audio files and builds a structured metadata DataFrame.
    Supports:
    1. Metadata file (CSV or JSON) containing 'file_path' and 'label' (or customizable column names).
    2. Subfolder-based labels (e.g., data/raw/real/001.wav, data/raw/fake/002.wav).
    3. Filename-based patterns (e.g., 001_real.wav, 002_spoof.wav).
    
    Args:
        data_dir (Union[str, Path]): Base directory containing audio files.
        metadata_path (Optional[Union[str, Path]]): Optional path to pre-existing CSV/JSON metadata.
        label_map (Optional[Dict[str, int]]): Custom label mapping. Defaults to config.label_map.
        save_metadata_to (Optional[Union[str, Path]]): Destination to save the generated metadata DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with columns ['file_path', 'label', 'class_name', 'file_name']
    """
    data_path = Path(data_dir)
    active_label_map = label_map or config.label_map

    # Option 1: Load pre-existing metadata table if provided
    if metadata_path and Path(metadata_path).exists():
        meta_file = Path(metadata_path)
        if meta_file.suffix == ".csv":
            df = pd.read_csv(meta_file)
        elif meta_file.suffix == ".json":
            df = pd.read_json(meta_file)
        else:
            raise ValueError(f"Unsupported metadata format: {meta_file.suffix}. Use CSV or JSON.")
        
        # Standardize column naming if necessary
        if "file_path" not in df.columns and "path" in df.columns:
            df["file_path"] = df["path"]
            
        # Ensure file paths are absolute or resolved relative to data_path
        df["file_path"] = df["file_path"].apply(
            lambda p: str(data_path / p) if not Path(p).is_absolute() else str(p)
        )
        return df

    # Option 2 & 3: Scan directory recursively
    if not data_path.exists():
        raise FileNotFoundError(f"Provided dataset directory does not exist: {data_path}")

    records: List[Dict[str, Union[str, int]]] = []
    
    for file_path in data_path.rglob("*"):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            # Check parent folder first
            label = infer_label_from_name(file_path.parent.name, active_label_map)
            # If not detected from parent directory, check filename stem
            if label is None:
                label = infer_label_from_name(file_path.stem, active_label_map)

            # Map numeric label to descriptive name
            class_name = "real" if label == 0 else ("fake" if label == 1 else "unknown")

            records.append({
                "file_path": str(file_path.resolve()),
                "file_name": file_path.name,
                "label": label if label is not None else -1,
                "class_name": class_name,
            })

    df = pd.DataFrame(records)

    if save_metadata_to:
        save_path = Path(save_metadata_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

    return df


def split_dataset(
    df: pd.DataFrame,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split metadata DataFrame into reproducible Train, Validation, and Test partitions.
    
    Args:
        df (pd.DataFrame): Input metadata DataFrame containing at least 'file_path' and 'label'.
        val_split (float): Fraction of data for validation.
        test_split (float): Fraction of data for testing.
        seed (int): Random seed for reproducibility.
        stratify (bool): Whether to stratify by label.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    set_seed(seed)
    
    if len(df) == 0:
        raise ValueError("Cannot split an empty DataFrame.")

    total_eval = val_split + test_split
    if total_eval >= 1.0 or total_eval <= 0:
        raise ValueError("Sum of val_split and test_split must be between 0 and 1.")

    stratify_target = df["label"] if (stratify and "label" in df.columns and df["label"].nunique() > 1) else None

    # First split: Train vs (Val + Test)
    train_df, eval_df = train_test_split(
        df,
        test_size=total_eval,
        random_state=seed,
        stratify=stratify_target,
    )

    # Second split: Val vs Test
    relative_test_size = test_split / total_eval
    stratify_eval = eval_df["label"] if (stratify and "label" in eval_df.columns and eval_df["label"].nunique() > 1) else None

    val_df, test_df = train_test_split(
        eval_df,
        test_size=relative_test_size,
        random_state=seed,
        stratify=stratify_eval,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
