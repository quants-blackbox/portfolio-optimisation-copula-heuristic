from RiskLabAI.optimization.hrp import hrp
import pandas as pd
import numpy as np

def hrp_weights(returns):

    cov = returns.cov().to_numpy()
    corr = returns.corr().to_numpy()
    
    return hrp(cov=cov, corr=corr)