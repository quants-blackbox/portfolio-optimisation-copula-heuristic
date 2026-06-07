import pandas as pd

from src.optimisation.hmv.recursive_bisection import heuristic_optimisation
from src.optimisation.hrp.hrp_weights import hrp_weights
from src.optimisation.eqw.eq_weights import equal_weights
from src.optimisation.mvp.mvp_solver import mvp_weights


def portfolio_return_dataframe(train_returns: pd.DataFrame,
                               test_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Generate out-of-sample portfolio return series for all strategies.

    Parameters
    ----------
    train_returns : pd.DataFrame
        Training return matrix.

    test_returns : pd.DataFrame
        Test return matrix.

    Returns
    -------
    pd.DataFrame
        Portfolio return series indexed by date.
    """

    # weights estimated on training set
    hmv_weights = heuristic_optimisation(train_returns, gamma=0.86)

    hrp_w = hrp_weights(train_returns)

    eqw_w = equal_weights(train_returns)

    mvp_w = mvp_weights(train_returns.cov())

    # out-of-sample portfolio returns
    returns_df = pd.DataFrame({
        "HMV-Copula": test_returns @ hmv_weights,
        "HRP": test_returns @ hrp_w,
        "Equal Weight": test_returns @ eqw_w,
        "MVP": test_returns @ mvp_w
    })

    return returns_df