"""
Cyber Shield - Training Pipeline
End-to-end script: load → preprocess → features → train → evaluate → report.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from src.data_loader import load_and_merge_all
from src.preprocessing import preprocess_dataframe
from src.feature_engineering import build_tfidf_vectorizer
from src.models import train_all_models
from src.evaluation import evaluate_all_models, generate_all_reports
from src.utils import get_logger, MODELS_DIR

import joblib

logger = get_logger("pipeline")


def main():
    """Run the full training pipeline."""
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║     CYBER SHIELD — Training Pipeline                    ║")
    logger.info("║     Cyberbullying Detection System                      ║")
    logger.info("╚" + "═" * 58 + "╝")
    
    # ── Step 1: Load & Merge Datasets ──────────────────────────────────────
    logger.info("\n📂 STEP 1: Loading & Merging Datasets")
    df = load_and_merge_all(save_unified=True)
    
    # ── Step 2: Preprocess Text ────────────────────────────────────────────
    logger.info("\n🧹 STEP 2: Preprocessing Text Data")
    df = preprocess_dataframe(df, text_col="text")
    
    # ── Step 3: Train/Test Split ───────────────────────────────────────────
    logger.info("\n✂️  STEP 3: Splitting Data (80/20)")
    X_text = df["cleaned_text"]
    y = df["label"]
    
    X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_text, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    logger.info(f"  Train: {len(X_text_train)} samples")
    logger.info(f"  Test:  {len(X_text_test)} samples")
    
    # ── Step 4: Feature Engineering ────────────────────────────────────────
    logger.info("\n🔧 STEP 4: Building TF-IDF Features")
    X_train, vectorizer = build_tfidf_vectorizer(X_text_train)
    X_test = vectorizer.transform(X_text_test)
    
    logger.info(f"  Train features shape: {X_train.shape}")
    logger.info(f"  Test features shape:  {X_test.shape}")
    
    # ── Step 5: Train Models ───────────────────────────────────────────────
    logger.info("\n🤖 STEP 5: Training Models")
    trained_models = train_all_models(X_train, y_train)
    
    # ── Step 6: Evaluate Models ────────────────────────────────────────────
    logger.info("\n📊 STEP 6: Evaluating Models")
    results = evaluate_all_models(trained_models, X_test, y_test)
    
    # ── Step 7: Generate Reports ───────────────────────────────────────────
    logger.info("\n📈 STEP 7: Generating Reports")
    summary_df = generate_all_reports(results, y_test)
    
    # ── Step 8: Save Best Model Info ───────────────────────────────────────
    best_model_name = summary_df.loc[summary_df["F1 Score"].idxmax(), "Model"]
    best_model_info = {
        "best_model_name": best_model_name,
        "metrics": {
            col: summary_df.loc[summary_df["F1 Score"].idxmax(), col]
            for col in summary_df.columns if col != "Model"
        },
    }
    
    info_path = os.path.join(MODELS_DIR, "best_model_info.joblib")
    joblib.dump(best_model_info, info_path)
    logger.info(f"\n  🏆 Best model: {best_model_name}")
    logger.info(f"  Saved best model info to: {info_path}")
    
    # ── Done ───────────────────────────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info("  Models saved to:  models/")
    logger.info("  Reports saved to: reports/")
    logger.info("═" * 60)
    
    return trained_models, results, summary_df


if __name__ == "__main__":
    main()
