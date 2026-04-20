"""
model.py
--------
Train a Random Forest classifier to predict the NEXT day's market regime
using today's technical features.

Key design choices:
- Walk-forward validation: the model is always trained on past data only
- We shift regime labels by -1 so we're predicting tomorrow's regime
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis
"""

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def prepare_dataset(features: pd.DataFrame, regimes: pd.Series) -> tuple:
    """
    Align features with next-day regime labels.

    We want to predict tomorrow's regime using today's features,
    so we shift regimes by -1 (one day into the future).

    Returns X (features), y (next-day regime labels), aligned dates.
    """
    # Shift labels: y[t] = regime at t+1 (tomorrow)
    y = regimes.shift(-1).dropna().astype(int)

    # Align features to valid label dates
    X = features.reindex(y.index).dropna()
    y = y.reindex(X.index)

    print(f"  Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"  Label distribution: {dict(y.value_counts().sort_index())}")
    return X, y


def walk_forward_eval(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """
    Walk-forward (time-series) cross-validation.

    Unlike k-fold, this respects the temporal order of data:
    we always train on past, test on future. This prevents lookahead bias.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_preds = []
    all_true = []
    fold_scores = []

    print(f"\n  Running {n_splits}-fold walk-forward validation...")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale features (important for some feature types)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Simple RF for cross-val (not tuned, for speed)
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train_scaled, y_train)
        preds = rf.predict(X_test_scaled)

        acc = (preds == y_test.values).mean()
        fold_scores.append(acc)
        all_preds.extend(preds)
        all_true.extend(y_test.values)

        train_start = X.index[train_idx[0]].date()
        test_end = X.index[test_idx[-1]].date()
        print(f"    Fold {fold+1}: train ends → test through {test_end} | acc = {acc:.3f}")

    print(f"\n  Mean accuracy: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")
    return {"preds": np.array(all_preds), "true": np.array(all_true), "fold_scores": fold_scores}


def tune_and_train(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Hyperparameter tuning with GridSearchCV on the training set,
    then final training with best params.
    """
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [15, 25],
        "max_features": ["sqrt", 0.5],
    }

    tscv = TimeSeriesSplit(n_splits=4)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    print("\n  Tuning hyperparameters (this may take ~30 seconds)...")
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=tscv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_scaled, y_train)

    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV F1: {gs.best_score_:.3f}")

    # Train final model with best params
    best_model = RandomForestClassifier(**gs.best_params_, random_state=42, n_jobs=-1)
    best_model.fit(X_scaled, y_train)

    return best_model, scaler


def evaluate_model(model, scaler, X_test: pd.DataFrame, y_test: pd.Series,
                   feature_names: list, save_dir: str = "results/figures"):
    """
    Full evaluation: classification report, confusion matrix, feature importances.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    X_scaled = scaler.transform(X_test)
    preds = model.predict(X_scaled)

    print("\n=== Model Evaluation ===")
    print(classification_report(y_test, preds,
                                 target_names=["Regime 0 (Calm)", "Regime 1 (Trending)"]))

    # --- Confusion Matrix ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Random Forest Regime Classifier — Evaluation", fontsize=14, fontweight="bold")

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Calm", "Trending"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix", fontsize=12)

    # --- Feature Importances ---
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.nlargest(12).sort_values()
    colors = ["#2a9d8f" if v < importances.mean() * 1.5 else "#e63946"
              for v in top_features.values]
    top_features.plot(kind="barh", ax=axes[1], color=colors)
    axes[1].set_title("Top 12 Feature Importances", fontsize=12)
    axes[1].set_xlabel("Mean Decrease in Impurity")
    axes[1].axvline(importances.mean(), color="gray", linestyle="--", alpha=0.7,
                    label="Mean importance")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    save_path = f"{save_dir}/model_evaluation.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Evaluation chart saved to {save_path}")
    plt.close()

    return preds


def save_model(model, scaler, path: str = "models/rf_regime_classifier.pkl"):
    """Save trained model and scaler for later use."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, path)
    print(f"  Model saved to {path}")


def load_model(path: str = "models/rf_regime_classifier.pkl"):
    """Load saved model."""
    obj = joblib.load(path)
    return obj["model"], obj["scaler"]
