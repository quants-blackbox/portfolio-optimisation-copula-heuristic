import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from src.optimisation.hmv.recursive_bisection import heuristic_optimisation
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

