import pandas as pd
import yfinance as yf
import json
from pathlib import Path

def load_data(start_date="2005-01-01", end_date="2025-12-31"):
    """
    Load data from local file or fetch from Yahoo Finance if not available
    """

    file_path = Path("data/raw/raw_data.csv")

    # Check if file exists and is not empty
    if file_path.exists() and file_path.stat().st_size > 0:

        print("Loading local data...")
        raw_data = pd.read_csv(
                    file_path, parse_dates=["Date"], index_col="Date")
    else:
        print("Fetching data from Yahoo Finance...")
        tickers = get_all_tickers()
        raw_data = yf.download(tickers, start=start_date, end=end_date)["Close"]
        raw_data.to_csv("data/raw/raw_data.csv")

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