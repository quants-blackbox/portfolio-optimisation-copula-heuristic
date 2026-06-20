# src/performance/cumulative_returns.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.dependence.dependence import PROJECT_ROOT


def compute_cumulative_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily return series to cumulative growth of R1 invested.

    Parameters
    ----------
    returns_df : pd.DataFrame of daily returns, one column per strategy

    Returns
    -------
    pd.DataFrame of cumulative returns (starts at 1.0 on first date)
    """
    cumulative = (1 + returns_df).cumprod()

    # normalise so every strategy starts at exactly 1.0
    cumulative = cumulative / cumulative.iloc[0]

    return cumulative


def plot_cumulative_returns(returns_df: pd.DataFrame,
                             log_scale:  bool = True,
                             output_dir=None):
    """
    Plot cumulative returns for all strategies on a single axis.

    Features
    --------
    - Log scale y-axis (log_scale=True) so percentage moves are
      visually comparable across the full period
    - Crisis bands shaded: COVID-19 crash and 2022 rate shock
    - Drawdown subplot beneath the main chart
    - Each strategy labelled with its final cumulative return

    Parameters
    ----------
    returns_df : pd.DataFrame with DatetimeIndex, one column per strategy
    log_scale  : bool — use log scale on y-axis (recommended)
    output_dir : pathlib.Path — defaults to data/output/results/
    """

    returns_df = returns_df.sort_index()
    cumulative = compute_cumulative_returns(returns_df)

    # ── colour and style map ──────────────────────────────────────────
    style_map = {
        'HMV-Copula'  : {'color': '#1a6faf', 'lw': 2.5,  'ls': '-',  'zorder': 5},
        'HRP'         : {'color': '#e05c2a', 'lw': 1.8,  'ls': '--', 'zorder': 4},
        'MVP'         : {'color': '#2e9e56', 'lw': 1.8,  'ls': ':',  'zorder': 3},
        'Equal Weight': {'color': '#888888', 'lw': 1.4,  'ls': '-.', 'zorder': 2},
    }

    # ── crisis period definitions ─────────────────────────────────────
    crisis_bands = [
        {
            'start' : '2020-02-01',
            'end'   : '2020-04-30',
            'label' : 'COVID-19\ncrash',
            'color' : '#d73027',
            'alpha' : 0.10,
        },
        {
            'start' : '2022-01-01',
            'end'   : '2022-12-31',
            'label' : 'Rate\nshock',
            'color' : '#fc8d59',
            'alpha' : 0.10,
        },
    ]

    # ── layout: main chart + drawdown subplot ─────────────────────────
    fig, (ax_main, ax_dd) = plt.subplots(
        2, 1,
        figsize=(13, 8),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )

    # ─────────────────────────────────────────────────────────────────
    # MAIN CHART — cumulative returns
    # ─────────────────────────────────────────────────────────────────
    for strategy in cumulative.columns:
        style = style_map.get(strategy,
                              {'color': 'black', 'lw': 1.5,
                               'ls': '-', 'zorder': 1})
        ax_main.plot(
            cumulative.index,
            cumulative[strategy],
            color   = style['color'],
            lw      = style['lw'],
            ls      = style['ls'],
            zorder  = style['zorder'],
            label   = strategy
        )

        # end-of-line label showing final value
        final_val  = cumulative[strategy].iloc[-1]
        final_date = cumulative.index[-1]
        ax_main.annotate(
            f'{final_val:.2f}x',
            xy       = (final_date, final_val),
            xytext   = (8, 0),
            textcoords = 'offset points',
            fontsize = 8,
            color    = style['color'],
            va       = 'center'
        )

    # crisis bands on main chart
    for band in crisis_bands:
        ax_main.axvspan(
            pd.Timestamp(band['start']),
            pd.Timestamp(band['end']),
            alpha = band['alpha'],
            color = band['color'],
            zorder = 0
        )
        # label at top of band
        mid_date = pd.Timestamp(band['start']) + (
            pd.Timestamp(band['end']) - pd.Timestamp(band['start'])
        ) / 2
        ax_main.text(
            mid_date,
            ax_main.get_ylim()[1] if not log_scale else 1,
            band['label'],
            ha       = 'center',
            va       = 'bottom',
            fontsize = 7.5,
            color    = band['color'],
            alpha    = 0.85
        )

    # reference line at 1.0 (starting value)
    ax_main.axhline(y=1.0, color='black', lw=0.6,
                    ls=':', alpha=0.4, zorder=0)

    if log_scale:
        ax_main.set_yscale('log')
        ax_main.set_ylabel('Cumulative return (log scale, R1 = 1.0)',
                           fontsize=10)
        # clean log tick labels
        from matplotlib.ticker import FuncFormatter
        ax_main.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f'{y:.1f}x')
        )
    else:
        ax_main.set_ylabel('Cumulative return (R1 = 1.0)', fontsize=10)
        ax_main.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y:.1f}x')
        )

    ax_main.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax_main.set_title(
        'Figure 6: Cumulative Returns — Out-of-Sample Test Period (2019–2025)',
        fontsize=12, pad=10
    )
    ax_main.grid(axis='y', alpha=0.25, lw=0.5)
    ax_main.grid(axis='x', alpha=0.15, lw=0.5)

    # ─────────────────────────────────────────────────────────────────
    # DRAWDOWN SUBPLOT
    # ─────────────────────────────────────────────────────────────────
    for strategy in returns_df.columns:
        style   = style_map.get(strategy,
                                {'color': 'black', 'lw': 1.2,
                                 'ls': '-', 'zorder': 1})
        cum     = cumulative[strategy]
        dd      = (cum - cum.cummax()) / cum.cummax()

        ax_dd.plot(
            dd.index, dd * 100,
            color  = style['color'],
            lw     = style['lw'],
            ls     = style['ls'],
            zorder = style['zorder'],
            alpha  = 0.85
        )

    # crisis bands on drawdown chart
    for band in crisis_bands:
        ax_dd.axvspan(
            pd.Timestamp(band['start']),
            pd.Timestamp(band['end']),
            alpha = band['alpha'],
            color = band['color'],
            zorder = 0
        )

    ax_dd.axhline(y=0, color='black', lw=0.6, ls=':', alpha=0.4)
    ax_dd.set_ylabel('Drawdown (%)', fontsize=9)
    ax_dd.set_xlabel('Date', fontsize=10)
    ax_dd.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.0f}%')
    )
    ax_dd.grid(axis='y', alpha=0.2, lw=0.5)

    # ── final layout and save ─────────────────────────────────────────
    plt.tight_layout()

    if output_dir is None:
        output_dir = PROJECT_ROOT / 'data' / 'output' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / 'figure6_cumulative_returns.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.show()


def print_crisis_performance(returns_df: pd.DataFrame):
    """
    Print a summary table of performance during each crisis band.
    Useful for the thesis write-up discussion of Figure 6.

    Parameters
    ----------
    returns_df : pd.DataFrame with DatetimeIndex, one column per strategy
    """
    from src.performance.portfolio_metrics import portfolio_metrics

    returns_df = returns_df.sort_index()

    crisis_periods = {
        'COVID crash'  : ('2020-02-01', '2020-04-30'),
        'Rate shock'   : ('2022-01-01', '2022-12-31'),
        'Full test'    : (str(returns_df.index[0].date()),
                          str(returns_df.index[-1].date())),
    }

    rows = []
    for period_name, (start, end) in crisis_periods.items():
        for strategy in returns_df.columns:
            slice_  = returns_df[strategy].loc[start:end]
            metrics = portfolio_metrics(slice_)
            metrics['Period']   = period_name
            metrics['Strategy'] = strategy
            rows.append(metrics)

    summary = pd.DataFrame(rows).set_index(['Period', 'Strategy'])
    print("\nCrisis period performance summary:")
    print(summary.to_string())

    return summary