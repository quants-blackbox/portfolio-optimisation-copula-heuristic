import numpy as np
import pandas as pd

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