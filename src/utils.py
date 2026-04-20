"""
utils.py
--------
Helper functions for plotting and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


def plot_feature_distributions(features: pd.DataFrame, regimes: pd.Series,
                                 save_path: str = None):
    """
    Plot distributions of key features, split by regime.
    Helps verify that our features are actually different across regimes.
    """
    key_features = ["rsi_14", "atr_pct", "bb_width", "adx_14",
                    "return_5d", "realized_vol_20"]
    key_features = [f for f in key_features if f in features.columns]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Feature Distributions by Regime", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    colors = {0: "#2a9d8f", 1: "#e63946"}
    labels = {0: "Regime 0 (Calm)", 1: "Regime 1 (Trending)"}

    for i, feat in enumerate(key_features[:6]):
        ax = axes[i]
        aligned = features[feat].reindex(regimes.index).dropna()
        reg_aligned = regimes.reindex(aligned.index)

        for regime in [0, 1]:
            vals = aligned[reg_aligned == regime]
            ax.hist(vals, bins=40, alpha=0.6, color=colors[regime],
                    label=labels[regime], density=True)

        ax.set_title(feat.replace("_", " ").title(), fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """Compute rolling Sharpe ratio."""
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    return (roll_mean / (roll_std + 1e-9)) * np.sqrt(252)


def correlation_heatmap(features: pd.DataFrame, save_path: str = None):
    """Plot correlation heatmap of features."""
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = features.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, fmt=".1f", cmap="RdBu_r",
                center=0, ax=ax, linewidths=0.3)
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_divider(title: str = ""):
    """Pretty console divider."""
    if title:
        print(f"\n{'─'*20} {title} {'─'*20}")
    else:
        print("─" * 55)
