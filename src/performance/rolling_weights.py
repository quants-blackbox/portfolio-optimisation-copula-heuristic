# src/performance/rolling_weights.py

import json

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

    Rolling window operates on the in-sample period (2006–2019) only.
    No crisis bands are shown — crisis periods fall in the test set.

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

    for ax, strategy in zip(axes, strategies):
        df = weight_dfs[strategy].copy()

        # identify top_n assets by mean absolute weight
        top_assets   = df.mean().sort_values(ascending=False).head(top_n).index.tolist()
        other_assets = [a for a in df.columns if a not in top_assets]

        plot_df = df[top_assets].copy()
        if other_assets:
            plot_df['Other'] = df[other_assets].sum(axis=1)

        # split positive and negative weights so stacking is valid
        # for long-short portfolios like unconstrained MVP
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

    axes[-1].set_xlabel('Date', fontsize=10)

    plt.suptitle(
        'Figure 4: Rolling Portfolio Weight Stability — In-Sample Period (2006–2019)\n'
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


# def plot_gamma_weights(weights_dict: dict,
#                        top_n:        int  = 15,
#                        output_dir         = None):
#     """
#     Grouped bar chart of top_n asset weights for each gamma value.
#     Shows how weight concentration changes as gamma moves from 0 to 1.

#     Parameters
#     ----------
#     weights_dict : dict {gamma: pd.Series of weights}
#     top_n        : number of assets to display
#     output_dir   : pathlib.Path
#     """

#     # identify top_n assets by maximum weight across any gamma
#     all_weights = pd.DataFrame(weights_dict).T    # gamma × assets
#     top_assets  = all_weights.max(axis=0).nlargest(top_n).index.tolist()

#     plot_data = all_weights[top_assets].copy()
#     plot_data.index = [f'γ={g}' for g in plot_data.index]

#     fig, ax = plt.subplots(figsize=(14, 5))

#     x      = np.arange(len(top_assets))
#     n_bars = len(plot_data)
#     width  = 0.8 / n_bars
#     colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, n_bars))

#     for i, (label, row) in enumerate(plot_data.iterrows()):
#         offset = (i - n_bars / 2 + 0.5) * width
#         ax.bar(x + offset, row.values * 100,
#                width=width * 0.9,
#                label=label,
#                color=colors[i],
#                edgecolor='white',
#                linewidth=0.5)

#     ax.set_xticks(x)
#     ax.set_xticklabels(top_assets, rotation=45, ha='right', fontsize=8)
#     ax.set_ylabel('Weight (%)', fontsize=10)
#     ax.set_title(f'Portfolio Weights by Gamma — Top {top_n} Assets',
#                  fontsize=12)
#     ax.legend(title='Gamma', fontsize=8, title_fontsize=9,
#               bbox_to_anchor=(1.01, 1), loc='upper left')
#     ax.yaxis.set_major_formatter(
#         plt.FuncFormatter(lambda y, _: f'{y:.1f}%')
#     )
#     ax.grid(axis='y', alpha=0.25, lw=0.5)

#     plt.tight_layout()

#     if output_dir is None:
#         output_dir = PROJECT_ROOT / 'data' / 'output' / 'results'
#     output_dir.mkdir(parents=True, exist_ok=True)

#     filepath = output_dir / 'gamma_weights.png'
#     plt.savefig(filepath, dpi=300, bbox_inches='tight')
#     print(f"Saved: {filepath}")
#     plt.show()


def plot_gamma_metrics(results_df: pd.DataFrame,
                       output_dir=None):
    """
    Two-panel plot of Sharpe Ratio and Diversification Ratio vs gamma.
    Marks the gamma that maximises each metric.

    Parameters
    ----------
    results_df : pd.DataFrame indexed by gamma with columns
                 ['Sharpe Ratio', 'Diversification Ratio']
    output_dir : pathlib.Path
    """

    gamma_vals = results_df.index.tolist()
    sharpes    = results_df['Sharpe Ratio'].values
    drs        = results_df['Diversification Ratio'].values

    best_sharpe_gamma = gamma_vals[np.argmax(sharpes)]
    best_dr_gamma     = gamma_vals[np.argmax(drs)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Sharpe ratio panel ────────────────────────────────────────────
    ax1.plot(gamma_vals, sharpes,
             color='#1a6faf', lw=2.5, marker='o',
             markersize=8, label='Sharpe Ratio')
    ax1.axvline(best_sharpe_gamma, color='black',
                ls=':', lw=1.5,
                label=f'Optimal γ = {best_sharpe_gamma}')
    ax1.scatter(best_sharpe_gamma,
                sharpes[np.argmax(sharpes)],
                color='black', zorder=5, s=80)
    ax1.annotate(
        f'γ = {best_sharpe_gamma}\nSharpe = {max(sharpes):.3f}',
        xy     = (best_sharpe_gamma, max(sharpes)),
        xytext = (best_sharpe_gamma + 0.05, max(sharpes) - 0.04),
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                  ec='grey', alpha=0.8),
        arrowprops=dict(arrowstyle='->', color='black')
    )
    ax1.set_xlabel('γ  (0 = HRP  →  1 = MVP)', fontsize=11)
    ax1.set_ylabel('Sharpe Ratio', fontsize=11)
    ax1.set_title('Sharpe Ratio vs Gamma', fontsize=12)
    ax1.set_xticks(gamma_vals)
    ax1.grid(alpha=0.25, lw=0.5)
    ax1.legend(fontsize=9)

    # endpoint labels
    ax1.text(0,   ax1.get_ylim()[0], 'HRP', fontsize=8,
             color='grey', va='bottom', ha='center')
    ax1.text(1.0, ax1.get_ylim()[0], 'MVP', fontsize=8,
             color='grey', va='bottom', ha='center')

    # ── Diversification Ratio panel ───────────────────────────────────
    ax2.plot(gamma_vals, drs,
             color='#e05c2a', lw=2.5, marker='s',
             markersize=8, label='Diversification Ratio')
    ax2.axvline(best_dr_gamma, color='black',
                ls=':', lw=1.5,
                label=f'Max DR γ = {best_dr_gamma}')
    ax2.scatter(best_dr_gamma,
                drs[np.argmax(drs)],
                color='black', zorder=5, s=80)
    ax2.annotate(
        f'γ = {best_dr_gamma}\nDR = {max(drs):.3f}',
        xy     = (best_dr_gamma, max(drs)),
        xytext = (best_dr_gamma + 0.05, max(drs) - 0.03),
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                  ec='grey', alpha=0.8),
        arrowprops=dict(arrowstyle='->', color='black')
    )
    ax2.set_xlabel('γ  (0 = HRP  →  1 = MVP)', fontsize=11)
    ax2.set_ylabel('Diversification Ratio', fontsize=11)
    ax2.set_title('Diversification Ratio vs Gamma', fontsize=12)
    ax2.set_xticks(gamma_vals)
    ax2.grid(alpha=0.25, lw=0.5)
    ax2.legend(fontsize=9)

    ax2.text(0,   ax2.get_ylim()[0], 'HRP', fontsize=8,
             color='grey', va='bottom', ha='center')
    ax2.text(1.0, ax2.get_ylim()[0], 'MVP', fontsize=8,
             color='grey', va='bottom', ha='center')

    plt.suptitle(
        'HMV Gamma Sensitivity — Sharpe Ratio and Diversification Ratio\n'
        'Test period 2019–2025  ·  RF = 7.25%',
        fontsize=12, y=1.02
    )
    plt.tight_layout()

    if output_dir is None:
        output_dir = PROJECT_ROOT / 'data' / 'output' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / 'figure3_gamma_metrics.pdf'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.show()   

# colour palette — consistent across all three plots
_COLORS = {
    'HRP'          : '#e05c2a',
    'MVP'          : '#2e9e56',
    'Equal Weight' : '#9b59b6',
}

def _strategy_color(label: str, gamma=None) -> str:
    """Return a consistent colour for each strategy label."""
    if label in _COLORS:
        return _COLORS[label]
    # HMV — blue gradient by gamma value
    intensity = 0.25 + 0.65 * float(gamma if gamma is not None else 0)
    return plt.cm.Blues(intensity)


def _add_benchmark_separator(ax, n_total: int, n_benchmarks: int):
    """Draw a subtle separator line above the benchmark rows."""
    sep_y = n_benchmarks - 0.5
    ax.axhline(sep_y, color='grey', lw=0.8, ls=':', alpha=0.5)
    ax.text(ax.get_xlim()[0], sep_y + 0.05,
            'Benchmarks', fontsize=8, color='grey', va='bottom')


def plot_sharpe_comparison(comparison_df: pd.DataFrame,
                            output_dir=None):
    """
    Horizontal bar chart of Sharpe ratios.
    HMV gamma values shown as blue gradient.
    HRP, MVP, Equal Weight shown as distinct benchmark colours.
    """
    labels  = comparison_df.index.tolist()
    sharpes = comparison_df['sharpe'].values
    colors  = [_strategy_color(l, comparison_df.loc[l, 'gamma'])
               for l in labels]

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.barh(labels, sharpes,
                   color=colors, edgecolor='white', height=0.6)

    # value labels
    for bar, val in zip(bars, sharpes):
        x_pos = bar.get_width() + (0.003 if val >= 0 else -0.003)
        ha    = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha=ha, fontsize=9)

    ax.axvline(0, color='black', lw=0.8, ls='--', alpha=0.4)

    # separator above benchmarks (last 3 rows)
    n_benchmarks = comparison_df[comparison_df['type'] == 'benchmark'].shape[0]
    _add_benchmark_separator(ax, len(labels), n_benchmarks)

    ax.set_xlabel('Sharpe Ratio  (RF = 7.25%)', fontsize=10)
    ax.set_title('Sharpe Ratio: HMV Gamma Sweep vs Benchmarks\n'
                 'Test period 2019–2025',
                 fontsize=11, pad=10)
    ax.grid(axis='x', alpha=0.25, lw=0.5)

    # manual legend for benchmarks
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=_COLORS['HRP'],          label='HRP'),
        mpatches.Patch(color=_COLORS['MVP'],          label='MVP'),
        mpatches.Patch(color=_COLORS['Equal Weight'], label='Equal Weight'),
        mpatches.Patch(color=plt.cm.Blues(0.6),       label='HMV (blue gradient)'),
    ]
    ax.legend(handles=legend_patches, fontsize=8,
              loc='lower right', framealpha=0.9)

    plt.tight_layout()

    if output_dir is None:
        output_dir = PROJECT_ROOT / 'data' / 'output' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / 'gamma_sharpe_comparison.pdf'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.show()


def plot_dr_comparison(comparison_df: pd.DataFrame,
                        output_dir=None):
    """
    Horizontal bar chart of Diversification Ratios.
    DR = 1 reference line marks the no-diversification baseline.
    """
    labels = comparison_df.index.tolist()
    drs    = comparison_df['diversification_ratio'].values
    colors = [_strategy_color(l, comparison_df.loc[l, 'gamma'])
              for l in labels]

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.barh(labels, drs,
                   color=colors, edgecolor='white', height=0.6)

    for bar, val in zip(bars, drs):
        ax.text(bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', ha='left', fontsize=9)

    # DR = 1 reference
    ax.axvline(1.0, color='black', lw=0.8, ls='--', alpha=0.4)
    ax.text(1.002, 0.02, 'DR = 1\n(no benefit)',
            fontsize=7, color='grey', va='bottom',
            transform=ax.get_xaxis_transform())

    n_benchmarks = comparison_df[comparison_df['type'] == 'benchmark'].shape[0]
    _add_benchmark_separator(ax, len(labels), n_benchmarks)

    ax.set_xlabel('Diversification Ratio  (higher = better)', fontsize=10)
    ax.set_title('Diversification Ratio: HMV Gamma Sweep vs Benchmarks\n'
                 'Computed on training covariance matrix',
                 fontsize=11, pad=10)
    ax.grid(axis='x', alpha=0.25, lw=0.5)

    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=_COLORS['HRP'],          label='HRP'),
        mpatches.Patch(color=_COLORS['MVP'],          label='MVP'),
        mpatches.Patch(color=_COLORS['Equal Weight'], label='Equal Weight'),
        mpatches.Patch(color=plt.cm.Blues(0.6),       label='HMV (blue gradient)'),
    ]
    ax.legend(handles=legend_patches, fontsize=8,
              loc='lower right', framealpha=0.9)

    plt.tight_layout()

    if output_dir is None:
        output_dir = PROJECT_ROOT / 'data' / 'output' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / 'gamma_dr_comparison.pdf'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.show()


def plot_gamma_weights(comparison_df: pd.DataFrame,
                        top_n:         int = 15,
                        output_dir         = None):
    """
    Grouped bar chart of top_n asset weights for each strategy.
    Includes all HMV gamma values plus HRP, MVP, Equal Weight.
    """
    all_weights = pd.DataFrame(
        {label: row['weights'] for label, row in comparison_df.iterrows()}
    ).T

    # top_n assets by max weight across any strategy
    top_assets = all_weights.max(axis=0).nlargest(top_n).index.tolist()
    plot_data  = all_weights[top_assets].copy()

    labels   = comparison_df.index.tolist()
    colors   = [_strategy_color(l, comparison_df.loc[l, 'gamma'])
                for l in labels]

    n_strats = len(plot_data)
    n_assets = len(top_assets)
    x        = np.arange(n_assets)
    width    = 0.8 / n_strats

    fig, ax = plt.subplots(figsize=(16, 5))

    for i, (label, color) in enumerate(zip(plot_data.index, colors)):
        offset = (i - n_strats / 2 + 0.5) * width
        ax.bar(x + offset,
               plot_data.loc[label].values * 100,
               width     = width * 0.9,
               label     = label,
               color     = color,
               edgecolor = 'white',
               linewidth = 0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(top_assets, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Weight (%)', fontsize=10)
    ax.set_title(f'Portfolio Weights by Strategy — Top {top_n} Assets\n'
                 'HMV gamma sweep vs HRP, MVP, and Equal Weight',
                 fontsize=11)
    ax.legend(title='Strategy',
              fontsize=8, title_fontsize=9,
              bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.1f}%')
    )
    ax.grid(axis='y', alpha=0.25, lw=0.5)

    plt.tight_layout()

    if output_dir is None:
        output_dir = PROJECT_ROOT / 'data' / 'output' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / 'gamma_weights_comparison.pdf'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.show()


def plot_individual_weights(comparison_df: pd.DataFrame,
                             top_n:        int  = 30,
                             output_dir         = None):
    """
    For each strategy in comparison_df, plot a vertical bar chart
    of asset weights. Tickers come directly from the weights Series
    in comparison_df — no external config needed.

    Parameters
    ----------
    comparison_df : DataFrame from gamma_comparison() with 'weights' column
    top_n         : max assets to show by absolute weight
    output_dir    : pathlib.Path
    """

    if output_dir is None:
        output_dir = PROJECT_ROOT / 'data' / 'output' / 'results' / 'weights'
    output_dir.mkdir(parents=True, exist_ok=True)

    # build a consistent colour per ticker across all strategy charts
    # use the full ticker universe from the first strategy's weights
    all_tickers = comparison_df.iloc[0]['weights'].index.tolist()
    cmap        = plt.cm.get_cmap('tab20', len(all_tickers))
    ticker_color = {
        ticker: cmap(i) for i, ticker in enumerate(all_tickers)
    }

    for label, row in comparison_df.iterrows():

        weights = row['weights'].copy()

        # select top_n by absolute weight, sort descending
        weights = (weights
                   .reindex(weights.abs().nlargest(top_n).index)
                   .sort_values(ascending=False))

        tickers = weights.index.tolist()
        values  = weights.values * 100
        colors  = [ticker_color[t] for t in tickers]

        # ── plot ───────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(max(12, len(tickers) * 0.45), 5))

        x = np.arange(len(tickers))

        for i, (val, color) in enumerate(zip(values, colors)):
            if val >= 0:
                ax.bar(x[i], val,
                       color     = color,
                       edgecolor = 'white',
                       linewidth = 0.5,
                       width     = 0.7)
            else:
                ax.bar(x[i], val,
                       color     = color,
                       alpha     = 0.4,
                       edgecolor = color,
                       linewidth = 0.8,
                       hatch     = '//',
                       width     = 0.7)

        # zero reference line
        ax.axhline(0, color='black', lw=0.8, alpha=0.5)

        # value labels
        for i, val in enumerate(values):
            va     = 'bottom' if val >= 0 else 'top'
            offset = 0.1      if val >= 0 else -0.1
            ax.text(x[i], val + offset,
                    f'{val:.1f}%',
                    ha='center', va=va,
                    fontsize=6.5, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Weight (%)', fontsize=10)
        ax.set_title(f'{label}  —  Portfolio Weights', fontsize=12, pad=10)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y:.1f}%')
        )
        ax.grid(axis='y', alpha=0.25, lw=0.5)

        # summary annotation
        n_pos = (weights > 0).sum()
        n_neg = (weights < 0).sum()
        ax.text(0.01, 0.01,
                f'Assets: {len(tickers)}  |  '
                f'Positive: {n_pos}  |  '
                f'Negative: {n_neg}  |  '
                f'Total shown: {weights.sum()*100:.1f}%',
                transform=ax.transAxes,
                fontsize=7, color='grey', va='bottom')

        plt.tight_layout()

        safe_label = (label
                      .replace(' ', '_')
                      .replace('=', '')
                      .replace('.', 'p'))
        filepath = output_dir / f'weights_{safe_label}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.show()
        plt.close(fig)