import numpy as np
import pandas as pd

def mvp_weights(cov):
    """
    Global Minimum Variance Portfolio
    """

    ones = np.ones(len(cov))

    inv_cov = np.linalg.pinv(cov)

    w = inv_cov @ ones

    w = w / (ones.T @ inv_cov @ ones)

    return w
