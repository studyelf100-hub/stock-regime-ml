"""
strategy.py
-----------
Regime-switching trading strategy with full backtesting engine.

Strategy logic:
    IF predicted_regime == 1 (Trending):
        → Momentum: buy if 5-day return > 0, sell if < 0
    
    IF predicted_regime == 0 (Calm / Mean-Reverting):
        → Mean-reversion: buy if RSI < 35 (oversold), sell if RSI > 65 (overbought)
    
    ELSE (low confidence):
        → Sit in cash

Compared against: Buy & Hold, Momentum Only, Mean-Reversion Only.
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from features import build_features, fetch_data
from model import evaluate_model, save_model, tune_and_train, prepare_dataset
from regime_detection import get_regime_labels

warnings.filterwarnings("ignore")

TRANSACTION_COST = 0.001  # 0.1% per trade (one-way)


def backtest_regime_switching(
    df: pd.DataFrame,
    features: pd.DataFrame,
    regime_predictions: pd.Series,
    transaction_cost: float = TRANSACTION_COST,
) -> pd.DataFrame:
    """
    Backtest the regime-switching strategy.

    Returns a DataFrame with daily positions, returns, and cumulative equity.
    """
    close = df["close"].reindex(features.index)
    daily_returns = np.log(close / close.shift(1))

    rsi = features["rsi_14"]
    ret_5d = features["return_5d"]

    position = pd.Series(0.0, index=features.index)  # 1 = long, 0 = cash
    trades = 0

    for i in range(1, len(features)):
        date = features.index[i]
        prev_date = features.index[i - 1]

        pred = regime_predictions.get(prev_date, np.nan)
        if pd.isna(pred):
            position[date] = 0
            continue

        if pred == 1:  # Trending regime → momentum
            signal = 1 if ret_5d[prev_date] > 0 else 0
        else:          # Calm regime → mean-reversion
            if rsi[prev_date] < 35:
                signal = 1
            elif rsi[prev_date] > 65:
                signal = 0
            else:
                signal = position[prev_date]  # hold current

        position[date] = signal
        if signal != position[prev_date]:
            trades += 1

    print(f"  Total trades: {trades}")

    # Apply transaction costs
    trade_costs = position.diff().abs() * transaction_cost
    strat_returns = (position.shift(1) * daily_returns - trade_costs).fillna(0)
    strat_equity = (1 + strat_returns).cumprod()

    result = pd.DataFrame({
        "close": close,
        "position": position,
        "daily_return": strat_returns,
        "equity": strat_equity,
    })
    return result


def backtest_buy_and_hold(df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    close = df["close"].reindex(features.index)
    daily_ret = np.log(close / close.shift(1)).fillna(0)
    return (1 + daily_ret).cumprod()


def backtest_momentum_only(df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    close = df["close"].reindex(features.index)
    daily_ret = np.log(close / close.shift(1))
    position = (features["return_5d"].shift(1) > 0).astype(float)
    costs = position.diff().abs() * TRANSACTION_COST
    ret = (position.shift(1) * daily_ret - costs).fillna(0)
    return (1 + ret).cumprod()


def backtest_meanrev_only(df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    close = df["close"].reindex(features.index)
    daily_ret = np.log(close / close.shift(1))

    position = pd.Series(0.0, index=features.index)
    rsi = features["rsi_14"]
    for i in range(1, len(features)):
        prev = features.index[i - 1]
        curr = features.index[i]
        if rsi[prev] < 35:
            position[curr] = 1
        elif rsi[prev] > 65:
            position[curr] = 0
        else:
            position[curr] = position[prev]

    costs = position.diff().abs() * TRANSACTION_COST
    ret = (position.shift(1) * daily_ret - costs).fillna(0)
    return (1 + ret).cumprod()


def compute_metrics(equity: pd.Series) -> dict:
    """Compute Sharpe ratio, max drawdown, CAGR, and win rate."""
    daily_ret = equity.pct_change().dropna()
    n_years = len(daily_ret) / 252

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0

    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = drawdown.min()
    win_rate = (daily_ret > 0).mean()

    return {
        "Total Return": f"{total_return*100:.1f}%",
        "CAGR": f"{cagr*100:.1f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd*100:.1f}%",
        "Win Rate (daily)": f"{win_rate*100:.1f}%",
    }


def plot_backtest(results: dict, ticker: str, save_path: str = None):
    """Plot equity curves and drawdowns for all strategies."""
    fig = plt.figure(figsize=(16, 10))
    gs_layout = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    colors = {
        "Regime-Switching": "#e63946",
        "Buy & Hold": "#457b9d",
        "Momentum Only": "#f4a261",
        "Mean-Reversion Only": "#2a9d8f",
    }

    # --- Equity Curves ---
    ax1 = fig.add_subplot(gs_layout[0, :])
    for name, equity in results.items():
        ax1.plot(equity.index, equity.values, label=name,
                 color=colors[name], linewidth=2 if name == "Regime-Switching" else 1.2,
                 linestyle="-" if name == "Regime-Switching" else "--")
    ax1.set_title(f"{ticker} — Strategy Comparison: Equity Curves", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Portfolio Value (starting at $1)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.axhline(1, color="gray", linestyle=":", alpha=0.5)

    # --- Drawdowns ---
    ax2 = fig.add_subplot(gs_layout[1, 0])
    for name, equity in results.items():
        roll_max = equity.cummax()
        dd = (equity - roll_max) / roll_max * 100
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.3, color=colors[name])
        ax2.plot(dd.index, dd.values, label=name, color=colors[name], linewidth=0.8)
    ax2.set_title("Drawdowns", fontsize=12)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # --- Metrics Table ---
    ax3 = fig.add_subplot(gs_layout[1, 1])
    ax3.axis("off")
    metrics_data = []
    columns = ["Strategy", "Total Return", "Sharpe", "Max DD"]
    for name, equity in results.items():
        m = compute_metrics(equity)
        metrics_data.append([name, m["Total Return"], m["Sharpe Ratio"], m["Max Drawdown"]])

    table = ax3.table(
        cellText=metrics_data,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    # Highlight our strategy row
    for j in range(len(columns)):
        table[1, j].set_facecolor("#ffeaa7")
        table[1, j].set_text_props(fontweight="bold")
    for j in range(len(columns)):
        table[0, j].set_facecolor("#2d3436")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax3.set_title("Performance Summary", fontsize=12, fontweight="bold", pad=15)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Backtest chart saved to {save_path}")
    else:
        plt.show()

    plt.close()


def run_full_pipeline(ticker: str, start: str, end: str):
    print(f"\n{'='*55}")
    print(f"  REGIME-SWITCHING STRATEGY — {ticker}")
    print(f"{'='*55}")

    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. Data & Features
    print("\n[1/5] Fetching data and engineering features...")
    df = fetch_data(ticker, start, end)
    features = build_features(df)

    # 2. Regime Labels via HMM
    print("\n[2/5] Detecting regimes via Hidden Markov Model...")
    regimes = get_regime_labels(df, features.index)
    common_idx = features.index.intersection(regimes.index)
    features = features.loc[common_idx]
    regimes = regimes.loc[common_idx]

    # 3. Train/Test Split (temporal — last 20% is test)
    print("\n[3/5] Training Random Forest regime classifier...")
    X, y = prepare_dataset(features, regimes)
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model, scaler = tune_and_train(X_train, y_train)
    preds = evaluate_model(model, scaler, X_test, y_test,
                            feature_names=X.columns.tolist())
    save_model(model, scaler)

    # 4. Generate predictions for the TEST period
    print("\n[4/5] Running backtest on test period...")
    import numpy as np_inner
    X_test_scaled = scaler.transform(X_test)
    regime_preds_test = pd.Series(
        model.predict(X_test_scaled),
        index=X_test.index,
        name="predicted_regime",
    )

    test_features = features.loc[X_test.index]
    result_df = backtest_regime_switching(df, test_features, regime_preds_test)
    bh = backtest_buy_and_hold(df, test_features)
    mom = backtest_momentum_only(df, test_features)
    mr = backtest_meanrev_only(df, test_features)

    all_strategies = {
        "Regime-Switching": result_df["equity"],
        "Buy & Hold": bh,
        "Momentum Only": mom,
        "Mean-Reversion Only": mr,
    }

    # 5. Report
    print("\n[5/5] Generating results...")
    print("\n=== BACKTEST RESULTS (Test Period) ===")
    for name, equity in all_strategies.items():
        m = compute_metrics(equity)
        tag = " ← OURS" if name == "Regime-Switching" else ""
        print(f"\n  {name}{tag}")
        for k, v in m.items():
            print(f"    {k}: {v}")

    plot_backtest(
        all_strategies,
        ticker=ticker,
        save_path=f"results/figures/backtest_{ticker}.png",
    )

    # Save trade log
    result_df.to_csv(f"results/backtest_report_{ticker}.csv")
    print(f"\n  Trade log saved to results/backtest_report_{ticker}.csv")
    print(f"\n{'='*55}")
    print("  Pipeline complete.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regime-switching ML strategy")
    parser.add_argument("--ticker", default="SPY", help="Stock ticker (e.g. SPY, QQQ, GLD)")
    parser.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-01-01", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    run_full_pipeline(args.ticker, args.start, args.end)
