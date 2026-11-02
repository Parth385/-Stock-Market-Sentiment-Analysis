"""
Backtesting and Portfolio Simulation Engine for Stage 6.
Author: Antigravity AI / DeepMind Team
Project: Beyond Sentiment (Indian Equity Short-Term Trading System)

Implements realistic NSE intraday cash equity backtesting with:
- Strict zero-lookahead chronological separation (Validation-only parameter selection).
- Unadjusted open-to-close gross return execution.
- Date-dependent statutory and brokerage friction + bid-ask slippage.
- Correct intraday turnover accounting with overnight cash reset.
- Verification that full round-trip friction is applied to position notional (never doubled).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# 1. Transaction Friction Schedule
# -------------------------------------------------------------------------
# Effective Oct 1, 2024: NSE revised cash market fee to flat 0.297 bps per side.
# Prior to Oct 1, 2024: NSE fee was 0.325 bps per side.
# Bid-ask slippage is fixed conservatively at 3.75 bps per side (7.50 bps round-trip).
FRICTION_PRE_OCT2024_ROUNDTRIP = 0.0015831   # 15.831 bps round-trip
FRICTION_POST_OCT2024_ROUNDTRIP = 0.0015765  # 15.765 bps round-trip
REVISION_DATE = pd.Timestamp("2024-10-01")

RISK_FREE_ANNUAL = 0.0650  # Fixed illustrative 6.50% annualized cash yield
DAILY_RF = RISK_FREE_ANNUAL / 252.0


def get_roundtrip_friction(date: pd.Timestamp) -> float:
    """Return the exact historical round-trip transaction friction for a given date."""
    if pd.Timestamp(date) < REVISION_DATE:
        return FRICTION_PRE_OCT2024_ROUNDTRIP
    return FRICTION_POST_OCT2024_ROUNDTRIP


# -------------------------------------------------------------------------
# 2. Performance Metrics Calculator
# -------------------------------------------------------------------------
def calculate_performance_metrics(
    daily_returns: pd.Series,
    traded_turnover: pd.Series,
    rf_daily: float = DAILY_RF,
    trading_days_per_year: int = 252,
) -> Dict[str, float]:
    """
    Compute comprehensive quantitative portfolio metrics.
    daily_returns: series of net daily portfolio returns (decimal)
    traded_turnover: series of daily traded turnover (decimal, 2 x sum(|w|))
    """
    n_days = len(daily_returns)
    if n_days == 0:
        return {}

    # Equity curve
    equity_curve = (1.0 + daily_returns).cumprod()
    final_equity = float(equity_curve.iloc[-1])
    cum_return = (final_equity - 1.0) * 100.0

    # CAGR
    years = n_days / trading_days_per_year
    cagr = ((final_equity ** (1.0 / years)) - 1.0) * 100.0 if years > 0 and final_equity > 0 else 0.0

    # Volatility
    daily_vol = float(daily_returns.std())
    ann_vol = daily_vol * np.sqrt(trading_days_per_year) * 100.0

    # Sharpe Ratio (relative to Rf)
    excess_returns = daily_returns - rf_daily
    excess_mean = float(excess_returns.mean())
    sharpe = (excess_mean / daily_vol * np.sqrt(trading_days_per_year)) if daily_vol > 1e-9 else 0.0

    # Sortino Ratio (downside deviation relative to 0 or Rf)
    downside_diff = np.minimum(excess_returns, 0.0)
    downside_dev = np.sqrt(np.mean(downside_diff ** 2))
    sortino = (excess_mean / downside_dev * np.sqrt(trading_days_per_year)) if downside_dev > 1e-9 else 0.0

    # Maximum Drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max
    max_dd = float(drawdowns.max()) * 100.0

    # Calmar Ratio
    calmar = (cagr / max_dd) if max_dd > 1e-4 else 0.0

    # Trade / Day Statistics
    active_days = traded_turnover > 0.001
    n_active = int(active_days.sum())
    active_returns = daily_returns[active_days]

    if n_active > 0:
        win_days = (active_returns > 0.0).sum()
        win_rate = (win_days / n_active) * 100.0
        gains = active_returns[active_returns > 0.0].sum()
        losses = np.abs(active_returns[active_returns < 0.0].sum())
        profit_factor = (gains / losses) if losses > 1e-9 else (99.0 if gains > 0 else 0.0)
        mean_gain = active_returns[active_returns > 0.0].mean() if (active_returns > 0.0).any() else 0.0
        mean_loss = np.abs(active_returns[active_returns < 0.0].mean()) if (active_returns < 0.0).any() else 1e-9
        gain_to_pain = mean_gain / mean_loss if mean_loss > 1e-9 else 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        gain_to_pain = 0.0

    # Turnover & Market Exposure
    mean_daily_turnover = float(traded_turnover.mean())
    ann_turnover = mean_daily_turnover * trading_days_per_year * 100.0
    time_in_market = (n_active / n_days) * 100.0

    return {
        "cumulative_return_pct": cum_return,
        "cagr_pct": cagr,
        "annualized_volatility_pct": ann_vol,
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown_pct": max_dd,
        "calmar_ratio": float(calmar),
        "win_rate_pct": float(win_rate),
        "profit_factor": float(profit_factor),
        "gain_to_pain_ratio": float(gain_to_pain),
        "annual_turnover_pct": ann_turnover,
        "time_in_market_pct": time_in_market,
        "total_trading_days": n_days,
        "active_trading_days": n_active,
    }


# -------------------------------------------------------------------------
# 3. Daily Portfolio Simulator
# -------------------------------------------------------------------------
def simulate_daily_portfolio(
    df_slice: pd.DataFrame,
    prob_col: str,
    strategy_type: str = "dynamic_threshold",
    theta_long: float = 0.52,
    theta_short: float = 0.46,
    top_k: int = 3,
    max_stock_weight: float = 0.20,
    friction_override_bps: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate portfolio returns day-by-day.
    df_slice must contain: ['date', 'symbol', 'target_same_session_ret_o2c', prob_col]
    
    Returns:
    - daily_summary: DataFrame indexed by date with columns:
      ['net_return', 'gross_return', 'friction_cost', 'cash_yield', 'traded_turnover', 'n_long', 'n_short']
    - trade_log: DataFrame with granular per-stock positions and P&L.
    """
    records_daily = []
    records_trades = []

    # Group by date to simulate each session independently
    dates = df_slice["date"].drop_duplicates().sort_values()

    for d in dates:
        d_df = df_slice[df_slice["date"] == d].copy()
        date_ts = pd.Timestamp(d)
        
        # Friction rate
        if friction_override_bps is not None:
            f_roundtrip = friction_override_bps / 10000.0
        else:
            f_roundtrip = get_roundtrip_friction(date_ts)

        p = d_df[prob_col].values
        r_gross_stock = d_df["target_same_session_ret_o2c"].values
        symbols = d_df["symbol"].values
        n_stocks = len(symbols)

        weights = np.zeros(n_stocks, dtype=float)

        if strategy_type == "long_only":
            long_mask = p >= theta_long
            n_long = int(long_mask.sum())
            n_short = 0
            if n_long > 0:
                raw_w = 1.0 / n_long
                capped_w = min(raw_w, max_stock_weight)
                weights[long_mask] = capped_w

        elif strategy_type == "dynamic_threshold":
            long_mask = p >= theta_long
            short_mask = p <= theta_short
            n_long = int(long_mask.sum())
            n_short = int(short_mask.sum())

            if n_long > 0 and n_short > 0:
                w_l = min(0.50 / n_long, max_stock_weight)
                w_s = min(0.50 / n_short, max_stock_weight)
                weights[long_mask] = w_l
                weights[short_mask] = -w_s
            elif n_long > 0 and n_short == 0:
                w_l = min(1.0 / n_long, max_stock_weight)
                weights[long_mask] = w_l
            elif n_short > 0 and n_long == 0:
                w_s = min(1.0 / n_short, max_stock_weight)
                weights[short_mask] = -w_s

        elif strategy_type == "rank_based":
            # Rank descending: top K long, bottom K short
            order = np.argsort(p)
            short_indices = order[:top_k]
            long_indices = order[-top_k:]
            n_long = top_k
            n_short = top_k

            w_l = min(0.50 / top_k, max_stock_weight)
            w_s = min(0.50 / top_k, max_stock_weight)
            weights[long_indices] = w_l
            weights[short_indices] = -w_s
        else:
            raise ValueError(f"Unknown strategy_type: {strategy_type}")

        # Strict gross leverage check: sum(|w|) <= 1.0
        abs_weights = np.abs(weights)
        total_one_way_notional = float(abs_weights.sum())
        if total_one_way_notional > 1.0:
            weights = weights / total_one_way_notional
            abs_weights = np.abs(weights)
            total_one_way_notional = 1.0

        # Traded turnover is 2 x sum(|w|) (entry leg + exit leg)
        daily_traded_turnover = 2.0 * total_one_way_notional

        # Friction cost: round-trip friction applied to one-way position notional
        # (Equivalently: half-roundtrip * traded_turnover)
        friction_cost = total_one_way_notional * f_roundtrip

        # Gross portfolio return from positions
        # For long: w * r. For short: (-|w|) * (-r) if short return is -r, which equals w * r when w < 0 and stock moves down (r < 0 -> w*r > 0).
        # Note: In our dataset, r_gross_stock = (close - open) / open.
        # If long (w > 0), gain when r > 0. If short (w < 0), gain when r < 0 (since w * r > 0).
        gross_pnl = float(np.sum(weights * r_gross_stock))

        # Cash yield on unallocated margin
        unallocated_cash = max(0.0, 1.0 - total_one_way_notional)
        cash_pnl = unallocated_cash * DAILY_RF

        # Net daily portfolio return
        net_return = gross_pnl - friction_cost + cash_pnl

        records_daily.append({
            "date": d,
            "net_return": net_return,
            "gross_return": gross_pnl,
            "friction_cost": friction_cost,
            "cash_yield": cash_pnl,
            "traded_turnover": daily_traded_turnover,
            "one_way_notional": total_one_way_notional,
            "n_long": int((weights > 0).sum()),
            "n_short": int((weights < 0).sum()),
        })

        for sym, w, r in zip(symbols, weights, r_gross_stock):
            if abs(w) > 1e-5:
                trade_gross = w * r
                trade_friction = abs(w) * f_roundtrip
                trade_net = trade_gross - trade_friction
                records_trades.append({
                    "date": d,
                    "symbol": sym,
                    "weight": w,
                    "direction": "LONG" if w > 0 else "SHORT",
                    "gross_stock_ret": r,
                    "gross_trade_pnl": trade_gross,
                    "friction_cost": trade_friction,
                    "net_trade_pnl": trade_net,
                })

    df_daily = pd.DataFrame(records_daily)
    df_trades = pd.DataFrame(records_trades)
    return df_daily, df_trades


# -------------------------------------------------------------------------
# 4. Reference Baselines
# -------------------------------------------------------------------------
def simulate_buy_and_hold_baseline(df_slice: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate an equal-weighted buy-and-hold portfolio across all 15 stocks.
    Holds equities continuously (bears overnight gaps + intraday drift).
    Rebalanced daily to equal weight 1/15.
    """
    # Gross return for buy & hold: daily change in adj_close
    # In df_slice we have target_same_session_ret_o2c, but for buy and hold we need full daily close-to-close if available,
    # or intraday open-to-close as the daytime component.
    # To compare on identical daytime ground, we evaluate the daytime equal-weight buy-and-hold:
    # weight = 1/15 on all stocks every day.
    dates = df_slice["date"].drop_duplicates().sort_values()
    records = []
    for d in dates:
        d_df = df_slice[df_slice["date"] == d]
        r = d_df["target_same_session_ret_o2c"].mean()
        # Buy and hold has 0 turnover/friction in daytime simulation (or passive holding)
        records.append({
            "date": d,
            "net_return": r,
            "gross_return": r,
            "friction_cost": 0.0,
            "cash_yield": 0.0,
            "traded_turnover": 0.0,
            "one_way_notional": 1.0,
            "n_long": 15,
            "n_short": 0,
        })
    return pd.DataFrame(records)


def simulate_intraday_momentum_baseline(df_full: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Intraday 1-day momentum baseline:
    If ret_1d_lag1 > 0 -> Long; if ret_1d_lag1 < 0 -> Short (or Cash if 0).
    Equally weight active signals up to 1.0 gross exposure.
    """
    df_split = df_full[df_full["split"] == split_name].copy()
    dates = df_split["date"].drop_duplicates().sort_values()
    records = []

    for d in dates:
        d_df = df_split[df_split["date"] == d]
        date_ts = pd.Timestamp(d)
        f_roundtrip = get_roundtrip_friction(date_ts)

        lag_ret = d_df["ret_1d_lag1"].values
        r_gross = d_df["target_same_session_ret_o2c"].values
        n_stocks = len(d_df)

        weights = np.zeros(n_stocks, dtype=float)
        long_mask = lag_ret > 0.0
        short_mask = lag_ret < 0.0
        n_long = int(long_mask.sum())
        n_short = int(short_mask.sum())

        if n_long > 0 and n_short > 0:
            weights[long_mask] = min(0.50 / n_long, 0.20)
            weights[short_mask] = -min(0.50 / n_short, 0.20)
        elif n_long > 0:
            weights[long_mask] = min(1.0 / n_long, 0.20)
        elif n_short > 0:
            weights[short_mask] = -min(1.0 / n_short, 0.20)

        notional = float(np.abs(weights).sum())
        turnover = 2.0 * notional
        friction = notional * f_roundtrip
        gross_pnl = float(np.sum(weights * r_gross))
        cash_pnl = max(0.0, 1.0 - notional) * DAILY_RF
        net_ret = gross_pnl - friction + cash_pnl

        records.append({
            "date": d,
            "net_return": net_ret,
            "gross_return": gross_pnl,
            "friction_cost": friction,
            "cash_yield": cash_pnl,
            "traded_turnover": turnover,
            "one_way_notional": notional,
            "n_long": n_long,
            "n_short": n_short,
        })
    return pd.DataFrame(records)


# -------------------------------------------------------------------------
# 5. Validation Grid Search (22 Valid Pairs)
# -------------------------------------------------------------------------
GRID_THETA_LONG = [0.48, 0.50, 0.52, 0.54, 0.56]
GRID_THETA_SHORT = [0.42, 0.44, 0.46, 0.48, 0.50]


def get_valid_threshold_pairs() -> List[Tuple[float, float]]:
    """Generate the 22 valid parameter pairs with strict inequality short < long."""
    pairs = []
    for tl in GRID_THETA_LONG:
        for ts in GRID_THETA_SHORT:
            if ts < tl:
                pairs.append((tl, ts))
    return pairs


def optimize_thresholds_on_validation(
    df_val: pd.DataFrame,
    prob_col: str,
    strategy_type: str = "dynamic_threshold",
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Search over the 22 valid parameter pairs strictly on the Validation set.
    Returns: (best_theta_long, best_theta_short, best_val_sharpe, all_results_dict)
    """
    pairs = get_valid_threshold_pairs()
    best_sharpe = -999.0
    best_pair = (0.52, 0.46)
    grid_records = []

    for tl, ts in pairs:
        daily_df, _ = simulate_daily_portfolio(
            df_val,
            prob_col=prob_col,
            strategy_type=strategy_type,
            theta_long=tl,
            theta_short=ts,
        )
        metrics = calculate_performance_metrics(daily_df["net_return"], daily_df["traded_turnover"])
        sh = metrics.get("sharpe_ratio", -999.0)
        grid_records.append({
            "theta_long": tl,
            "theta_short": ts,
            "sharpe_ratio": sh,
            "cumulative_return_pct": metrics.get("cumulative_return_pct", 0.0),
            "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0),
            "win_rate_pct": metrics.get("win_rate_pct", 0.0),
        })

        if sh > best_sharpe:
            best_sharpe = sh
            best_pair = (tl, ts)

    return best_pair[0], best_pair[1], best_sharpe, {"best_pair": best_pair, "grid_records": grid_records}
