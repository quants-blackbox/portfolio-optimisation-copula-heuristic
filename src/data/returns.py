import numpy as np
import pandas as pd
from scipy import stats
from ..data.wrangler import get_close_prices

def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate logarithmic returns.

    r_t = ln(P_t / P_{t-1})
    """

    returns = np.log(prices / prices.shift(1))

    return returns.dropna()

def calculate_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate arithmetic returns.

    r_t = (P_t - P_{t-1}) / P_{t-1}
    """

    returns = prices.pct_change()

    return returns.dropna()

def get_log_returns() -> pd.DataFrame:
    """
    Load cleaned prices and calculate log returns.
    """

    prices = get_close_prices()

    return calculate_log_returns(prices)

def get_simple_returns() -> pd.DataFrame:
    """
    Load cleaned prices and calculate simple returns.
    """

    prices = get_close_prices()

    return calculate_simple_returns(prices)

def split_returns():

    returns = get_log_returns()
    
    split_index = int(len(returns) * 0.65)

    train = returns.iloc[:split_index]
    test = returns.iloc[split_index:]

    return train, test


def descriptive_statistics(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-asset descriptive statistics for the results section.
    Covers moments, normality tests, and tail behaviour.
    """

    rows = []
    for col in returns_df.columns:
        r = returns_df[col].dropna()

        # annualised figures
        ann_return = r.mean()  * 252 * 100
        ann_vol    = r.std()   * np.sqrt(252) * 100

        # higher moments
        skew = round(stats.skew(r),   3)
        kurt = round(stats.kurtosis(r), 3)   # excess kurtosis

        # Jarque-Bera normality test
        jb_stat, jb_p = stats.jarque_bera(r)

        # tail behaviour
        var_95  = round(np.percentile(r, 5)  * 100, 3)   # historical VaR
        cvar_95 = round(r[r <= np.percentile(r, 5)].mean() * 100, 3)

        rows.append({
            'Asset'             : col,
            'Ann. Return (%)'   : round(ann_return, 2),
            'Ann. Vol (%)'      : round(ann_vol,    2),
            'Skewness'          : skew,
            'Excess Kurtosis'   : kurt,
            'JB p-value'        : round(jb_p, 4),
            'Normal?'           : 'No' if jb_p < 0.05 else 'Yes',
            'VaR 95% (%)'       : var_95,
            'CVaR 95% (%)'      : cvar_95,
        })

    return pd.DataFrame(rows).set_index('Asset')
