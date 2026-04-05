"""
Cyber Shield - Model Evaluation
Metrics computation, confusion matrices, ROC curves, and comparison charts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from src.utils import REPORTS_DIR, get_logger, timer

logger = get_logger("evaluation")

# Style configuration
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
COLORS = {
    "SVM (LinearSVC)": "#6366f1",   # Indigo
    "Random Forest": "#10b981",      # Emerald
    "XGBoost": "#f59e0b",           # Amber
}


@timer
def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a single model with all metrics.
    
    Returns:
        Dict with all metric values
    """
    logger.info(f"  Evaluating {model_name}...")
    
    y_pred = model.predict(X_test)
    
    # Get probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred.astype(float)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    
    # Log results
    logger.info(f"    Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"    Precision: {metrics['precision']:.4f}")
    logger.info(f"    Recall:    {metrics['recall']:.4f}")
    logger.info(f"    F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Print classification report
    report = classification_report(
        y_test, y_pred,
        target_names=["Non-Bullying", "Bullying"],
    )
    logger.info(f"\n{report}")
    
    return metrics


@timer
def evaluate_all_models(models: dict, X_test, y_test) -> dict:
    """
    Evaluate all models and return results.
    
    Args:
        models: Dict of model_name → trained_model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dict of model_name → metrics_dict
    """
    logger.info("=" * 60)
    logger.info("EVALUATING ALL MODELS")
    logger.info("=" * 60)
    
    all_results = {}
    for name, model in models.items():
        all_results[name] = evaluate_model(model, X_test, y_test, name)
    
    return all_results


@timer
def plot_confusion_matrices(results: dict, y_test, save: bool = True):
    """Plot confusion matrix for each model side by side."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    fig.suptitle("Confusion Matrices — Model Comparison", 
                 fontsize=16, fontweight="bold", y=1.02)
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, metrics) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, metrics["y_pred"])
        
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Bullying", "Bullying"],
            yticklabels=["Non-Bullying", "Bullying"],
            ax=ax, cbar=False,
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_title(name, fontsize=13, fontweight="bold",
                     color=COLORS.get(name, "#333"))
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_xlabel("Predicted", fontsize=11)
    
    plt.tight_layout()
    
    if save:
        path = os.path.join(REPORTS_DIR, "confusion_matrices.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        logger.info(f"  Saved confusion matrices to: {path}")
    
    plt.close(fig)


@timer
def plot_roc_curves(results: dict, y_test, save: bool = True):
    """Plot ROC curves for all models overlaid."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, metrics in results.items():
        fpr, tpr, _ = roc_curve(y_test, metrics["y_proba"])
        auc_score = metrics["roc_auc"]
        
        ax.plot(
            fpr, tpr,
            label=f"{name} (AUC = {auc_score:.4f})",
            color=COLORS.get(name, None),
            linewidth=2.5,
        )
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        path = os.path.join(REPORTS_DIR, "roc_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        logger.info(f"  Saved ROC curves to: {path}")
    
    plt.close(fig)


@timer
def plot_metrics_comparison(results: dict, save: bool = True):
    """Plot bar chart comparing all metrics across models."""
    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    display_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metric_names))
    width = 0.25
    offsets = np.linspace(-width, width, len(results))
    
    for i, (name, metrics) in enumerate(results.items()):
        values = [metrics[m] for m in metric_names]
        bars = ax.bar(
            x + offsets[i], values, width * 0.9,
            label=name, color=COLORS.get(name, None),
            edgecolor="white", linewidth=0.5,
            alpha=0.9, zorder=3,
        )
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold",
            )
    
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y", zorder=0)
    
    plt.tight_layout()
    
    if save:
        path = os.path.join(REPORTS_DIR, "metrics_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        logger.info(f"  Saved metrics comparison to: {path}")
    
    plt.close(fig)


@timer
def generate_summary_report(results: dict, save: bool = True) -> pd.DataFrame:
    """Generate a summary DataFrame and save as CSV."""
    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    
    rows = []
    for name, metrics in results.items():
        row = {"Model": name}
        for m in metric_names:
            row[m.replace("_", " ").title()] = round(metrics[m], 4)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Determine best model by F1-score
    best_idx = df["F1 Score"].idxmax()
    best_model = df.loc[best_idx, "Model"]
    best_f1 = df.loc[best_idx, "F1 Score"]
    
    logger.info("─" * 60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("─" * 60)
    logger.info(f"\n{df.to_string(index=False)}")
    logger.info(f"\n  🏆 Best Model: {best_model} (F1 = {best_f1:.4f})")
    logger.info("─" * 60)
    
    if save:
        path = os.path.join(REPORTS_DIR, "model_comparison.csv")
        df.to_csv(path, index=False)
        logger.info(f"  Saved summary to: {path}")
    
    return df


@timer
def generate_all_reports(results: dict, y_test):
    """Generate all plots and reports."""
    logger.info("=" * 60)
    logger.info("GENERATING ALL REPORTS")
    logger.info("=" * 60)
    
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_metrics_comparison(results)
    summary_df = generate_summary_report(results)
    
    return summary_df
