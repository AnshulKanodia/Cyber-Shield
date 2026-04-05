"""
Cyber Shield - ML Models
SVM, Random Forest, and XGBoost classifiers for cyberbullying detection.
"""

import os
import joblib
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.utils import MODELS_DIR, get_logger, timer

logger = get_logger("models")


def get_models() -> dict:
    """
    Return a dictionary of model name → model instance.
    All models are configured with class balancing for imbalanced data.
    """
    models = {
        "SVM (LinearSVC)": CalibratedClassifierCV(
            LinearSVC(
                class_weight="balanced",
                max_iter=5000,
                C=1.0,
                random_state=42,
            ),
            cv=3,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }
    return models


@timer
def train_model(model, X_train, y_train, model_name: str):
    """
    Train a single model.
    
    Args:
        model: Scikit-learn compatible model
        X_train: Training features
        y_train: Training labels
        model_name: Name for logging
    
    Returns:
        Trained model
    """
    logger.info(f"  Training {model_name}...")
    
    # Handle XGBoost scale_pos_weight
    if isinstance(model, XGBClassifier):
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        if n_pos > 0:
            model.set_params(scale_pos_weight=n_neg / n_pos)
            logger.info(f"    XGBoost scale_pos_weight: {n_neg / n_pos:.2f}")
    
    model.fit(X_train, y_train)
    logger.info(f"  {model_name} training complete")
    
    return model


@timer
def cross_validate_model(model, X, y, model_name: str, cv: int = 5) -> dict:
    """
    Perform stratified k-fold cross validation.
    
    Returns:
        Dict with mean and std for each metric
    """
    logger.info(f"  Cross-validating {model_name} ({cv}-fold)...")
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = {}
    
    for metric in scoring_metrics:
        scores = cross_val_score(model, X, y, cv=skf, scoring=metric, n_jobs=-1)
        results[metric] = {
            "mean": scores.mean(),
            "std": scores.std(),
            "scores": scores.tolist(),
        }
        logger.info(f"    {metric:>12s}: {scores.mean():.4f} (±{scores.std():.4f})")
    
    return results


@timer
def train_all_models(X_train, y_train, save: bool = True) -> dict:
    """
    Train all models and save them.
    
    Returns:
        Dict of model_name → trained_model
    """
    logger.info("=" * 60)
    logger.info("TRAINING ALL MODELS")
    logger.info("=" * 60)
    
    models = get_models()
    trained_models = {}
    
    for name, model in models.items():
        model = train_model(model, X_train, y_train, name)
        trained_models[name] = model
        
        if save:
            # Create safe filename
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            model_path = os.path.join(MODELS_DIR, f"{safe_name}.joblib")
            joblib.dump(model, model_path)
            logger.info(f"  Saved {name} to: {model_path}")
    
    return trained_models


def load_model(model_name: str):
    """Load a saved model by name."""
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    model_path = os.path.join(MODELS_DIR, f"{safe_name}.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"  Loaded model: {model_name} from {model_path}")
    return model


def predict(text_features, model=None, model_name: str = None) -> dict:
    """
    Make a prediction with confidence score.
    
    Uses a confidence threshold to prevent false positives:
    - Only classify as "Bullying" if model is >= 60% confident
    - Below threshold → default to "Non-Bullying" (benefit of the doubt)
    
    Returns:
        Dict with 'label', 'confidence', 'severity', 'raw_probabilities'
    """
    BULLYING_THRESHOLD = 0.60  # Minimum confidence to flag as bullying
    
    if model is None:
        if model_name is None:
            model_name = "XGBoost"
        model = load_model(model_name)
    
    # Get prediction probability
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(text_features)[0]
        bullying_prob = float(proba[1])  # Probability of bullying (class 1)
        safe_prob = float(proba[0])      # Probability of safe (class 0)
        
        # Apply threshold: only flag as bullying if confidence is strong enough
        if bullying_prob >= BULLYING_THRESHOLD:
            prediction = 1
            confidence = bullying_prob
        else:
            prediction = 0
            confidence = safe_prob
    else:
        prediction = int(model.predict(text_features)[0])
        confidence = 1.0
        bullying_prob = float(prediction)
        safe_prob = 1.0 - bullying_prob
    
    # Determine severity based on confidence
    if prediction == 1:
        if confidence >= 0.85:
            severity = "high"
        elif confidence >= 0.65:
            severity = "medium"
        else:
            severity = "low"
    else:
        severity = "none"
    
    return {
        "label": "Bullying" if prediction == 1 else "Non-Bullying",
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "severity": severity,
        "bullying_probability": round(bullying_prob, 4),
        "safe_probability": round(safe_prob, 4),
    }

