import pandas as pd
import json
from pathlib import Path

def load_data():
    """
    Load data from local file or fetch from Yahoo Finance if not available
    """

    file_path = Path("data/raw/raw_data.csv")

    # Check if file exists and is not empty
    if file_path.exists() and file_path.stat().st_size > 0:

        print("Loading local data...")
        raw_data = pd.read_csv(
                    file_path,
                    header=[0, 1],  # multi-index columns
                    index_col=0,  # first column is date
                    parse_dates=True)
    else:
        print("Fetching data from Yahoo Finance...")
        # fetch logic here
        raw_data = None

    return raw_data

def load_asset_config():
    # project root
    BASE_DIR = Path(__file__).resolve().parents[2]

    # config/assets.json
    config_path = BASE_DIR / "config" / "assets.json"

    with open(config_path, "r") as f:
        return json.load(f)

def get_all_tickers():
    """Return all tickers"""

    config = load_asset_config()

    tickers_by_sector = config['tickers_by_sector']
    tickers = []

    for sector in tickers_by_sector.values():
        tickers.extend(sector)
    return sorted(list(set(tickers)))