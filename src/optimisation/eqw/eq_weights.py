import pandas as pd
import numpy as np

def equal_weights(returns):
    n_assets = returns.shape[1]

    return pd.Series(
        np.repeat(1 / n_assets, n_assets),
        index=returns.columns
    )