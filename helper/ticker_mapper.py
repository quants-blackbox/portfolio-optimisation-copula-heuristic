from src.dependence.pseudo_observations import pseudo_observations

def create_ticker_map():
    pseudo_obs = pseudo_observations()
    # pyvinecopulib labels variables 1..d in structure.matrix, so the map
    # must be 1-based to match (0-based keys shift every label by one and
    # leave variable d unmapped).
    ticker_map = {
        i + 1: ticker
        for i, ticker in enumerate(pseudo_obs.columns)
    }
    return ticker_map

def get_ticker_map():
    return create_ticker_map()