from RiskLabAI.optimization.hrp import hrp
import pandas as pd

def hrp_weights(returns):
    
    tickers = returns.columns

    cov = returns.cov().to_numpy()
    corr = returns.corr().to_numpy()

    w = hrp(cov=cov, corr=corr, labels=tickers)
    
    return pd.Series(w, index=tickers)