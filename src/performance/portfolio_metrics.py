import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from src.optimisation.mvp.mvp_solver import mvp_weights
from src.optimisation.hrp.hrp_weights import hrp_weights
from src.optimisation.eqw.eq_weights import equal_weights

RF_ANNUAL = 0.0725          # South African repo rate ~ 7.25%, adjust to your period average
RF_DAILY  = RF_ANNUAL / 252
PERIODS   = 252


def portfolio_metrics(returns: pd.Series, rf_daily=RF_DAILY, periods=PERIODS) -> dict:
    
    excess        = returns - rf_daily
    ann_return    = returns.mean() * periods
    ann_vol       = returns.std() * np.sqrt(periods)
    sharpe        = (ann_return - RF_ANNUAL) / ann_vol
    
    # Drawdown
    cum           = (1 + returns).cumprod()
    rolling_max   = cum.cummax()
    drawdown      = (cum - rolling_max) / rolling_max
    max_dd        = drawdown.min()
    
    # Calmar
    calmar        = ann_return / abs(max_dd)
    
    # Sortino — downside deviation relative to rf
    downside      = returns[returns < rf_daily] - rf_daily
    downside_vol  = np.sqrt((downside**2).mean()) * np.sqrt(periods)
    sortino       = (ann_return - RF_ANNUAL) / downside_vol

    return {
        'Annualised Return (%)' : round(ann_return  * 100, 2),
        'Annualised Vol (%)':     round(ann_vol     * 100, 2),
        'Sharpe Ratio'       :    round(sharpe,           3),
        'Max Drawdown (%)'   :    round(max_dd      * 100, 2),
        'Calmar Ratio'       :    round(calmar,           3),
        'Sortino Ratio'      :    round(sortino,          3)
    }


def sub_period_metrics(returns_df):
    # Ensure a sorted DatetimeIndex so label-based slicing works
    returns_df = returns_df.sort_index()

    # Define sub-periods
    periods = {
        'Full sample'  : ('2019-03-04', '2025-12-30'),
        'Pre-COVID'    : ('2019-03-04', '2020-02-29'),
        'COVID crisis' : ('2020-03-01', '2020-04-30'),
        'Recovery'     : ('2020-05-01', '2021-12-31'),
        'Rate shock'   : ('2022-01-01', '2022-12-31'),
        'Post-shock'   : ('2023-01-01', '2025-12-30')
    }

    rows = []
    for period_name, (start, end) in periods.items():
        for strategy in returns_df.columns:
            slice_ = returns_df[strategy].loc[start:end]
            metrics = portfolio_metrics(slice_)
            metrics['Period']   = period_name
            metrics['Strategy'] = strategy
            rows.append(metrics)

    table2 = pd.DataFrame(rows).set_index(['Period', 'Strategy'])

    return table2


def calculate_portfolio_sharpe(
    weights_df: pd.DataFrame,
    test_returns: pd.DataFrame,
    risk_free_rate: float = RF_ANNUAL,
):
    """
    Calculate out-of-sample portfolio Sharpe ratio.

    Parameters
    ----------
    weights_df : pd.DataFrame
        Portfolio weights at each rebalance date.
        Index = rebalance dates
        Columns = asset tickers

    test_returns : pd.DataFrame
        Asset returns in the test period.
        Index = dates
        Columns = asset tickers

    risk_free_rate : float
        Annual risk-free rate.

    annualisation_factor : int
        252 for daily returns,
        52 for weekly returns,
        12 for monthly returns.

    Returns
    -------
    dict containing:
        sharpe_ratio
        annual_return
        annual_volatility
        portfolio_returns
    """

    portfolio_returns = test_returns @ weights_df

    excess_returns = portfolio_returns - risk_free_rate / 252

    sharpe = (
        excess_returns.mean()
        / excess_returns.std()
        * np.sqrt(252)
    )

    return sharpe


def diversification_ratio(weights: pd.Series,
                           cov:     pd.DataFrame) -> float:
    """
    Diversification Ratio = weighted average vol / portfolio vol.
    
    For portfolios with short positions, absolute weights are used
    in the numerator to avoid artificial inflation of DR caused by
    negative weights partially cancelling positive volatility terms.
    
    DR = 1   → no diversification benefit
    DR > 1   → genuine diversification, higher is better
    """
    weights = weights.reindex(cov.index).fillna(0)
    weights = weights / weights.sum()

    asset_vols = np.sqrt(np.diag(cov.values))

    # use absolute weights in numerator — correct for long/short portfolios
    weighted_avg_vol = np.abs(weights.values) @ asset_vols

    # portfolio volatility uses signed weights — reflects actual risk
    port_vol = np.sqrt(weights.values @ cov.values @ weights.values)

    if port_vol == 0:
        return np.nan

    return round(float(weighted_avg_vol / port_vol), 4)