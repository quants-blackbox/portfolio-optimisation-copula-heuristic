from src.dependence.pseudo_observations import pseudo_observations

def create_ticker_map():
    pseudo_obs = pseudo_observations()
    ticker_map = {
        i: ticker
        for i, ticker in enumerate(pseudo_obs.columns)
    }
    return ticker_map

def get_ticker_map():
    return create_ticker_map()