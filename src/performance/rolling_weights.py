# src/performance/rolling_weights.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.optimisation.hmv.recursive_bisection import heuristic_optimisation
from src.optimisation.hrp.hrp_weights         import hrp_weights
from src.optimisation.mvp.mvp_solver          import mvp_weights
from src.optimisation.eqw.eq_weights          import equal_weights
from src.dependence.dependence                import PROJECT_ROOT


def rolling_weights(returns_df: pd.DataFrame,
                    train_window: int = 504,
                    refit_freq:   int = 63,
                    gamma:      float = 0.0) -> dict:
    """
    Compute portfolio weights on a rolling chronological basis.

    At each refit date, weights are estimated on the preceding
    train_window days and recorded. This simulates real deployment
    where the model is updated periodically as new data arrives.

    Parameters
    ----------
    returns_df   : full DataFrame of daily log returns (DatetimeIndex)
    train_window : number of trading days in each training window (default 504 = 2 years)
    refit_freq   : how often to refit in trading days (default 63 = quarterly)
    gamma        : HMV gamma parameter — use your optimal gamma from Figure 3

    Returns
    -------
    dict of {strategy_name: pd.DataFrame}
    Each DataFrame has refit dates as index and assets as columns.
    """

    

    returns_df = returns_df.sort_index()
    
    n          = len(returns_df)
    dates      = returns_df.index
    assets     = returns_df.columns.tolist()

    weight_records = {
        'HMV-Copula'  : [],
        'HRP'         : [],
        'MVP'         : [],
        'Equal Weight': []
    }

    refit_positions = range(train_window, n, refit_freq)
    print(f"Total refit points : {len(list(refit_positions))}")

    for t in refit_positions:
        window     = returns_df.iloc[t - train_window : t]
        refit_date = dates[t]

        print(f"  Refitting {refit_date.date()} "
              f"(window: {dates[t - train_window].date()} "
              f"→ {dates[t - 1].date()})")

        # ── HMV-Copula ────────────────────────────────────────────────
        try:
            w = heuristic_optimisation(train_returns=window, gamma=gamma)
            w = w.reindex(assets).fillna(0)
            w = w / w.sum()
        except Exception as e:
            print(f"    HMV failed: {e} — using equal weight fallback")
            w = pd.Series(1 / len(assets), index=assets)
        weight_records['HMV-Copula'].append((refit_date, w))

        # ── HRP ───────────────────────────────────────────────────────
        try:
            w = hrp_weights(window)
            w = w.reindex(assets).fillna(0)
            w = w / w.sum()
        except Exception as e:
            print(f"    HRP failed: {e} — using equal weight fallback")
            w = pd.Series(1 / len(assets), index=assets)
        weight_records['HRP'].append((refit_date, w))

        # ── MVP ───────────────────────────────────────────────────────
        try:
            w_arr = mvp_weights(window.cov())
            w     = pd.Series(w_arr, index=assets)
            w     = w.reindex(assets).fillna(0)
            w     = w / w.sum()
        except Exception as e:
            print(f"    MVP failed: {e} — using equal weight fallback")
            w = pd.Series(1 / len(assets), index=assets)
        weight_records['MVP'].append((refit_date, w))

        # ── Equal Weight ──────────────────────────────────────────────
        w = equal_weights(returns=window)
        w = w.reindex(assets).fillna(0)
        weight_records['Equal Weight'].append((refit_date, w))

    # ── convert lists to DataFrames ───────────────────────────────────
    weight_dfs = {}
    for strategy, records in weight_records.items():
        idx          = [r[0] for r in records]
        weight_list  = [r[1] for r in records]
        weight_dfs[strategy] = pd.DataFrame(weight_list,
                                             index=idx,
                                             columns=assets)

    return weight_dfs


def portfolio_turnover(weight_df: pd.DataFrame) -> pd.Series:
    """
    Compute one-way turnover between consecutive refit dates.
    Turnover = sum of absolute weight changes across all assets.
    A value of 0.10 means 10% of the portfolio reshuffled that period.

    Parameters
    ----------
    weight_df : DataFrame (refit dates × assets)

    Returns
    -------
    pd.Series of per-period turnover values
    """
    diff     = weight_df.diff().abs()
    turnover = diff.sum(axis=1).dropna()

    print(f"  Mean turnover : {turnover.mean() * 100:.2f}% per period")
    print(f"  Max  turnover : {turnover.max()  * 100:.2f}%")
    print(f"  Std  turnover : {turnover.std()  * 100:.2f}%")

    return turnover


def plot_rolling_weights(weight_dfs: dict,
                         top_n: int = 10,
                         output_dir=None):
    """
    Stacked area chart of rolling weights for each strategy.
    Assets outside the top_n by average weight are grouped as 'Other'.
    Crisis periods are shaded in red.

    Parameters
    ----------
    weight_dfs : dict returned by rolling_weights()
    top_n      : number of individual assets to display separately
    output_dir : pathlib.Path — defaults to data/output/results/
    """

    strategies = list(weight_dfs.keys())
    n_strat    = len(strategies)

    fig, axes = plt.subplots(n_strat, 1,
                              figsize=(14, 4 * n_strat),
                              sharex=True)

    if n_strat == 1:
        axes = [axes]

    crisis_bands = [
        ('2020-02-01', '2020-04-30', 'COVID-19'),
        ('2022-01-01', '2022-12-31', 'Rate shock'),
    ]

    for ax, strategy in zip(axes, strategies):
        df = weight_dfs[strategy].copy()

        # identify top_n assets by mean absolute weight
        top_assets   = df.mean().sort_values(ascending=False).head(top_n).index.tolist()
        other_assets = [a for a in df.columns if a not in top_assets]

        plot_df = df[top_assets].copy()
        if other_assets:
            plot_df['Other'] = df[other_assets].sum(axis=1)

        # Weights can be long/short (e.g. unconstrained MVP), so a single
        # stacked area is invalid. Stack positive exposures upward and
        # negative exposures downward, sharing one colour per asset.
        cols   = plot_df.columns.tolist()
        cmap   = plt.get_cmap('tab20')
        colors = [cmap(i % 20) for i in range(len(cols))]

        pos = plot_df.clip(lower=0)
        neg = plot_df.clip(upper=0)

        ax.stackplot(plot_df.index,
                     *[pos[c].values for c in cols],
                     colors=colors, alpha=0.82, labels=cols)
        ax.stackplot(plot_df.index,
                     *[neg[c].values for c in cols],
                     colors=colors, alpha=0.82)
        ax.axhline(0, color='black', linewidth=0.6)

        ax.set_title(f'{strategy} — Rolling Portfolio Weights',
                     fontsize=11, pad=6)
        ax.set_ylabel('Weight', fontsize=9)

        ymax = pos.sum(axis=1).max()
        ymin = neg.sum(axis=1).min()
        ax.set_ylim(min(ymin * 1.05, 0.0), max(ymax * 1.05, 1.0))
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y:.0%}')
        )
        ax.legend(loc='upper left',
                  bbox_to_anchor=(1.01, 1),
                  fontsize=7, ncol=1)

        # shade crisis periods
        for start, end, label in crisis_bands:
            try:
                ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                           alpha=0.12, color='red')
                ax.text(pd.Timestamp(start), 0.97, label,
                        transform=ax.get_xaxis_transform(),
                        fontsize=7, color='darkred', va='top')
            except Exception:
                pass

    axes[-1].set_xlabel('Date', fontsize=10)

    plt.suptitle(
        'Figure 4: Rolling Portfolio Weight Stability (2019–2025)\n'
        'Train window = 2 years · Refit frequency = quarterly',
        fontsize=12, y=1.01
    )
    plt.tight_layout()

    if output_dir is None:
        output_dir = PROJECT_ROOT / 'data' / 'output' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / 'figure4_rolling_weights.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.show()


def plot_turnover_comparison(weight_dfs: dict, output_dir=None):
    """
    Bar chart of mean quarterly turnover across all strategies.

    Parameters
    ----------
    weight_dfs : dict returned by rolling_weights()
    output_dir : pathlib.Path — defaults to data/output/results/
    """

    print("Turnover summary:")
    turnovers = {}
    for strategy, df in weight_dfs.items():
        print(f"\n  {strategy}")
        to = portfolio_turnover(df)
        turnovers[strategy] = to.mean()

    turnover_series = pd.Series(turnovers).sort_values()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = turnover_series.plot.bar(ax=ax,
                                     color='steelblue',
                                     edgecolor='white',
                                     width=0.5)

    ax.set_ylabel('Mean one-way turnover per quarter', fontsize=10)
    ax.set_title('Portfolio Turnover by Strategy', fontsize=11)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.1%}')
    )
    ax.set_xticklabels(turnover_series.index, rotation=15, ha='right')

    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f'{bar.get_height():.1%}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if output_dir is None:
        output_dir = PROJECT_ROOT / 'data' / 'output' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / 'figure4b_turnover_comparison.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {filepath}")
    plt.show()