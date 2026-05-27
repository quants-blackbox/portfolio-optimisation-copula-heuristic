import yfinance as yf
import pandas as pd
import numpy as np

def load_data(ticker, start_date, end_date):
    # Download historical data for the specified ticker and date range
    data = yf.download(ticker, start=start_date, end=end_date)

    # Check if data is empty
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker} between {start_date} and {end_date}.")

    # Keep only the 'Close' price
    data = data[['Close']]

    return data