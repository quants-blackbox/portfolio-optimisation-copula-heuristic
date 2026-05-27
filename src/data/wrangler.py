import pandas as pd

from ..data.loader import load_data, get_all_tickers

def data_wrangler():
    """
    Wrangle raw data to prepare for analysis and modeling
     - Ensure data availability during critical periods GCF (2007-2009) and COVID (2020)
     - Limit missing data to <10% for each ticker
    """
    raw_data = load_data()

    # Ensure date index (safe, idempotent)
    raw_data.index = pd.to_datetime(raw_data.index)

    # Define critical periods where data availability is crucial
    forbidden_periods = [
        ('2007-01-01', '2009-12-31'),
        ('2020-01-01', '2020-12-31')
    ]

    # Extract close prices correctly
    # Case 1: columns = (Feature, Ticker)
    close_prices = raw_data.xs('Close', level=0, axis=1).copy()

    # If instead columns = (Ticker, Feature), use:
    # close_prices = raw_data.xs('Close', level=1, axis=1).copy()

    max_count = close_prices.count().max()

    results = []
    keep_cols = []

    for col in close_prices.columns:
        series = close_prices[col]

        observed_count = series.notna().sum()
        missing_count = series.isna().sum()
        missing_ratio = missing_count / max_count if max_count > 0 else np.nan

        # Check forbidden windows
        missing_in_forbidden_period = False
        for start, end in forbidden_periods:
            s, e = pd.to_datetime(start), pd.to_datetime(end)
            mask = (series.index >= s) & (series.index <= e)
            if mask.any() and series.loc[mask].isna().any():
                missing_in_forbidden_period = True
                break

        keep = (missing_ratio < 0.1) and (not missing_in_forbidden_period)

        results.append({
            'ticker': col,
            'observed_count': int(observed_count),
            'missing_ratio': float(missing_ratio),
            'missing_in_forbidden_period': missing_in_forbidden_period,
            'keep': keep
        })

        if keep:
            keep_cols.append(col)

    summary = (
        pd.DataFrame(results)
        .set_index('ticker')
        .sort_values('missing_ratio', ascending=False)
    )

    return close_prices, keep_cols, summary

def get_close_prices():
    """
    Get cleaned close price data ready for analysis and modeling
    """
    close_prices, keep_cols, summary = data_wrangler()
    return close_prices[keep_cols]

def data_summary():
    """
    Get summary of data quality and availability for each ticker
    """
    _, _, summary = data_wrangler()
    return summary

def get_kept_tickers():
    _, kept_tickers, _ = data_wrangler()
    return kept_tickers

def retention_ratio():
    """
    Calculate retention ratio of tickers after wrangling
    """
    all_tickers = get_all_tickers()
    kept_tickers = get_kept_tickers()
    if len(all_tickers) == 0:
        return 0.0
    return len(kept_tickers) / len(all_tickers)