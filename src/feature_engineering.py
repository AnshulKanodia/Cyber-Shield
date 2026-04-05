"""
Cyber Shield - Feature Engineering
TF-IDF vectorization with n-gram support.
"""

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import MODELS_DIR, get_logger, timer

logger = get_logger("feature_engineering")


@timer
def build_tfidf_vectorizer(
    texts,
    max_features: int = 15000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
    save: bool = True,
) -> tuple:
    """
    Build and fit a TF-IDF vectorizer.
    
    Args:
        texts: Iterable of cleaned text strings
        max_features: Maximum number of features
        ngram_range: Tuple of (min_n, max_n) for n-gram range
        min_df: Minimum document frequency 
        max_df: Maximum document frequency (as proportion)
        sublinear_tf: Apply sublinear tf scaling (1 + log(tf))
        save: Whether to save the fitted vectorizer
    
    Returns:
        Tuple of (tfidf_matrix, vectorizer)
    """
    logger.info(f"  Building TF-IDF vectorizer:")
    logger.info(f"    max_features = {max_features}")
    logger.info(f"    ngram_range  = {ngram_range}")
    logger.info(f"    min_df       = {min_df}")
    logger.info(f"    max_df       = {max_df}")
    logger.info(f"    sublinear_tf = {sublinear_tf}")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b\w+\b",
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    logger.info(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    logger.info(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Show top features
    feature_names = vectorizer.get_feature_names_out()
    logger.info(f"  Sample features: {list(feature_names[:10])}")
    
    if save:
        vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
        joblib.dump(vectorizer, vec_path)
        logger.info(f"  Saved vectorizer to: {vec_path}")
    
    return tfidf_matrix, vectorizer


def load_vectorizer(filepath: str = None) -> TfidfVectorizer:
    """Load a previously saved TF-IDF vectorizer."""
    if filepath is None:
        filepath = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    
    vectorizer = joblib.load(filepath)
    logger.info(f"  Loaded vectorizer from: {filepath}")
    return vectorizer


def transform_texts(texts, vectorizer: TfidfVectorizer = None):
    """Transform texts using a fitted vectorizer."""
    if vectorizer is None:
        vectorizer = load_vectorizer()
    
    return vectorizer.transform(texts)
