from src.data.returns import get_log_returns, get_close_prices

def pseudo_observations():
    """
    Convert returns to pseudo-observations U in (0,1) using ranks.
    """
    returns = get_log_returns()

    ranks = returns.rank(method='average')
    u_hat = ranks / (len(returns) + 1.0)

    return u_hat