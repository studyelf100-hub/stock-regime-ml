"""
regime_detection.py
-------------------
Uses a Hidden Markov Model (HMM) to detect market regimes from price data.

The idea: the market has hidden "states" (regimes) that we can't directly observe,
but we can infer them from observable quantities like returns and volatility.

State 0 → Calm / Mean-Reverting  (low vol, prices oscillate)
State 1 → Volatile / Trending    (high vol, prices move directionally)
"""

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from features import fetch_data

warnings.filterwarnings("ignore")


def fit_hmm(returns: pd.Series, n_states: int = 2, n_iter: int = 1000) -> tuple:
    """
    Fit a Gaussian Hidden Markov Model on daily log returns.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.
    n_states : int
        Number of hidden states (regimes). We use 2 by default.
    n_iter : int
        Max EM iterations for convergence.

    Returns
    -------
    model : GaussianHMM
        Fitted HMM model.
    regimes : np.ndarray
        Predicted state sequence (0 or 1 for each day).
    """
    # HMM expects shape (n_samples, n_features)
    X = returns.values.reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=42,
    )
    model.fit(X)
    regimes = model.predict(X)

    # --- Align state labels: State 0 = low vol, State 1 = high vol ---
    # Calculate mean absolute return per state
    state_vols = {}
    for s in range(n_states):
        mask = regimes == s
        state_vols[s] = np.abs(returns.values[mask]).mean()

    # Sort states so 0 = lowest vol, 1 = highest vol
    sorted_states = sorted(state_vols, key=lambda s: state_vols[s])
    remap = {sorted_states[i]: i for i in range(n_states)}
    regimes = np.array([remap[s] for s in regimes])

    # Print regime statistics
    print("\n=== HMM Regime Statistics ===")
    for s in range(n_states):
        mask = regimes == s
        state_returns = returns.values[mask]
        label = "Mean-Reverting (calm)" if s == 0 else "Trending (volatile)"
        print(f"  State {s} [{label}]:")
        print(f"    Days: {mask.sum()} ({100*mask.mean():.1f}%)")
        print(f"    Avg daily return: {state_returns.mean()*100:.3f}%")
        print(f"    Avg daily |return|: {np.abs(state_returns).mean()*100:.3f}%")
        print(f"    Annualized vol: {state_returns.std() * np.sqrt(252) * 100:.1f}%")

    return model, regimes


def plot_regimes(price: pd.Series, regimes: np.ndarray, ticker: str = "SPY", save_path: str = None):
    """
    Plot the price series with regime shading.
    Green = calm/mean-reverting, Red = volatile/trending.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    fig.suptitle(f"{ticker} — Market Regime Detection via HMM", fontsize=16, fontweight="bold")

    # Price plot with regime shading
    ax1.plot(price.index, price.values, color="#1a1a2e", linewidth=1.2, label="Price")
    ax1.set_ylabel("Price ($)", fontsize=11)
    ax1.set_title("Price with Regime Overlay", fontsize=12)

    # Shade regimes
    dates = price.index
    for i in range(1, len(regimes)):
        if regimes[i] == 1:  # trending / volatile
            ax1.axvspan(dates[i - 1], dates[i], alpha=0.15, color="#e63946", linewidth=0)
        else:  # calm / mean-reverting
            ax1.axvspan(dates[i - 1], dates[i], alpha=0.08, color="#2a9d8f", linewidth=0)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2a9d8f", alpha=0.5, label="Regime 0: Calm / Mean-Reverting"),
        Patch(facecolor="#e63946", alpha=0.5, label="Regime 1: Volatile / Trending"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    # Regime as a step function
    ax2.fill_between(dates, regimes, step="mid", alpha=0.7,
                     color=["#e63946" if r == 1 else "#2a9d8f" for r in regimes][0])
    ax2.step(dates, regimes, color="#444", linewidth=0.8, where="mid")
    ax2.set_ylabel("Regime", fontsize=11)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Calm", "Trending"])
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Chart saved to {save_path}")
    else:
        plt.show()

    plt.close()
    return fig


def get_regime_labels(df: pd.DataFrame, features_index: pd.Index) -> pd.Series:
    """
    Full pipeline: fit HMM, return aligned regime labels for the features index.
    This is called by the main strategy and model training code.
    """
    # Compute returns over the full price series
    returns = np.log(df["close"] / df["close"].shift(1)).dropna()

    # Fit HMM
    _, regimes = fit_hmm(returns)

    # Align to same index (returns has one fewer row than df due to .shift(1))
    regime_series = pd.Series(regimes, index=returns.index, name="regime")

    # Keep only dates that exist in features
    regime_aligned = regime_series.reindex(features_index).dropna().astype(int)

    return regime_aligned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize market regimes via HMM")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-01-01")
    args = parser.parse_args()

    df = fetch_data(args.ticker, args.start, args.end)
    returns = np.log(df["close"] / df["close"].shift(1)).dropna()

    model, regimes = fit_hmm(returns)

    # Align price with regimes (same length as returns)
    price_aligned = df["close"].loc[returns.index]
    plot_regimes(price_aligned, regimes, ticker=args.ticker,
                 save_path=f"results/figures/regimes_{args.ticker}.png")
