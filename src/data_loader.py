"""
Cyber Shield - Data Loader
Loads and merges all datasets into a unified binary format.
Columns: text, label (0 = Not Bullying, 1 = Bullying)
"""

import pandas as pd
import os
from src.utils import DATASETS_DIR, get_logger, timer

logger = get_logger("data_loader")


@timer
def load_cyberbullying_tweets(filepath: str = None) -> pd.DataFrame:
    """
    Load cyberbullying_tweets.csv and convert to binary labels.
    
    Original labels: age, ethnicity, gender, religion, other_cyberbullying, not_cyberbullying
    Mapping: not_cyberbullying → 0, everything else → 1
    """
    if filepath is None:
        filepath = os.path.join(DATASETS_DIR, "cyberbullying_tweets.csv")
    
    df = pd.read_csv(filepath)
    logger.info(f"  Loaded cyberbullying_tweets: {len(df)} rows")
    
    # Rename columns for consistency
    df = df.rename(columns={"tweet_text": "text"})
    
    # Map to binary: not_cyberbullying = 0, all others = 1
    df["label"] = df["cyberbullying_type"].apply(
        lambda x: 0 if x == "not_cyberbullying" else 1
    )
    
    logger.info(f"  Label distribution → Bullying: {df['label'].sum()}, "
                f"Not Bullying: {(df['label'] == 0).sum()}")
    
    return df[["text", "label"]]


@timer
def load_hate_speech_dataset(filepath: str = None) -> pd.DataFrame:
    """
    Load combined_hate_speech_dataset.csv.
    Uses hate_label column directly (already binary: 0/1).
    """
    if filepath is None:
        filepath = os.path.join(DATASETS_DIR, "combined_hate_speech_dataset.csv")
    
    df = pd.read_csv(filepath)
    logger.info(f"  Loaded hate_speech_dataset: {len(df)} rows")
    
    # Rename columns for consistency
    df = df.rename(columns={"hate_label": "label"})
    
    logger.info(f"  Label distribution → Bullying: {df['label'].sum()}, "
                f"Not Bullying: {(df['label'] == 0).sum()}")
    
    return df[["text", "label"]]


@timer
def load_jigsaw_toxic(filepath: str = None, sample_size: int = 20000) -> pd.DataFrame:
    """
    Load Jigsaw Toxic Comment dataset (train.csv).
    Creates binary label: 1 if any toxic column is flagged, else 0.
    Subsamples to balance the dataset.
    """
    if filepath is None:
        filepath = os.path.join(DATASETS_DIR, "train.csv")
    
    df = pd.read_csv(filepath)
    logger.info(f"  Loaded jigsaw_toxic: {len(df)} rows")
    
    # Rename for consistency
    df = df.rename(columns={"comment_text": "text"})
    
    # Create binary label: 1 if ANY toxic flag is set
    toxic_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    df["label"] = (df[toxic_cols].sum(axis=1) > 0).astype(int)
    
    # Subsample to keep dataset manageable
    positive = df[df["label"] == 1]
    negative = df[df["label"] == 0]
    
    # Take all positive samples, sample negative to match
    n_pos = len(positive)
    n_neg = min(len(negative), sample_size - n_pos)
    
    if n_neg > 0:
        negative_sample = negative.sample(n=n_neg, random_state=42)
        df_sampled = pd.concat([positive, negative_sample], ignore_index=True)
    else:
        df_sampled = positive.copy()
    
    logger.info(f"  Subsampled to {len(df_sampled)} rows → "
                f"Bullying: {df_sampled['label'].sum()}, "
                f"Not Bullying: {(df_sampled['label'] == 0).sum()}")
    
    return df_sampled[["text", "label"]]


@timer
def load_and_merge_all(save_unified: bool = True) -> pd.DataFrame:
    """
    Load all datasets and merge into a single unified DataFrame.
    
    Returns:
        DataFrame with columns: text, label
    """
    logger.info("=" * 60)
    logger.info("LOADING AND MERGING ALL DATASETS")
    logger.info("=" * 60)
    
    # Load each dataset
    df_cyber = load_cyberbullying_tweets()
    df_hate = load_hate_speech_dataset()
    df_jigsaw = load_jigsaw_toxic()
    
    # Merge all
    df_unified = pd.concat([df_cyber, df_hate, df_jigsaw], ignore_index=True)
    
    # Drop duplicates based on text
    initial_len = len(df_unified)
    df_unified = df_unified.drop_duplicates(subset=["text"]).reset_index(drop=True)
    logger.info(f"  Removed {initial_len - len(df_unified)} duplicate texts")
    
    # Drop rows with NaN text
    df_unified = df_unified.dropna(subset=["text"]).reset_index(drop=True)
    
    # Shuffle
    df_unified = df_unified.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info("─" * 60)
    logger.info(f"  UNIFIED DATASET: {len(df_unified)} rows")
    logger.info(f"  Bullying:     {df_unified['label'].sum()} "
                f"({df_unified['label'].mean()*100:.1f}%)")
    logger.info(f"  Not Bullying: {(df_unified['label'] == 0).sum()} "
                f"({(1 - df_unified['label'].mean())*100:.1f}%)")
    logger.info("─" * 60)
    
    # Save unified dataset
    if save_unified:
        unified_path = os.path.join(DATASETS_DIR, "unified_dataset.csv")
        df_unified.to_csv(unified_path, index=False)
        logger.info(f"  Saved unified dataset to: {unified_path}")
    
    return df_unified


if __name__ == "__main__":
    df = load_and_merge_all()
    print(f"\nUnified dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
