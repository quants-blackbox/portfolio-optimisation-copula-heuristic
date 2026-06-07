from RiskLabAI.optimization import recursive_bisection
from RiskLabAI.optimization.hrp import *
import riskfolio as rp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.dependence.dependence import *
from src.data.returns import *
from src.optimisation.hmv.clustering import dependence_matrix
from src.optimisation.hmv.seriation import quasi_diagonalisation
from src.dependence.dependence import PROJECT_ROOT


def heuristic_optimisation(train_returns, gamma=0):
    """
    Orchestrate the heuristic allocation pipeline:

      1. Clustering + quasi-diagonalisation (handled in `seriation`/`clustering`).
      2. Reorder the dependence matrix into the quasi-diagonal asset order.
      3. Recursive bisection:
           - 'HRP' -> RiskLabAI inverse-variance recursive bisection.
           - 'HMV' -> Schur-complement recursive HMV (`recursive_hmv`).

    opt_type : {'HRP', 'HMV'}
    gamma    : Schur shrinkage in [0, 1]; only used for 'HMV'
               (gamma=0 -> HRP-like, gamma=1 -> full Schur/MV).
    returns  : pandas Series of weights indexed by asset, summing to 1.
    """

    # 1. + 2. clustering / quasi-diagonalisation phase
    _, ordered_assets = quasi_diagonalisation()

    cov_matrix = train_returns.cov()
    reordered_cov = cov_matrix.loc[ordered_assets, ordered_assets]

    weights = pd.Series(_recursive_hmv(reordered_cov, gamma, ordered_assets))
    
    # plot_weights(weights=weights, opt_method='HMV')

    return weights

def _recursive_hmv(sigma, gamma, orderded_assets, eps=1e-8):
    """
    Recursive Hierarchical Minimum Variance (HMV) allocation.

    Parameters
    ----------
    sigma : pd.DataFrame
        Covariance matrix in quasi-diagonal order.

    gamma : float
        Schur complement intensity.
        gamma = 0 -> HRP-like
        gamma = 1 -> full HMV

    orderded_assets : list
        Asset labels.

    Returns
    -------
    dict
        Asset weights summing to one.
    """

    n = len(orderded_assets)

    # ---------------- base case ----------------
    if n == 1:
        return {orderded_assets[0]: 1.0}

    mid = n // 2

    left = orderded_assets[:mid]
    right = orderded_assets[mid:]

    # ---------------- block partition ----------------
    A = sigma.loc[left, left].values
    D = sigma.loc[right, right].values
    B = sigma.loc[left, right].values
    C = sigma.loc[right, left].values

    A_inv = _safe_inv(A)
    D_inv = _safe_inv(D)

    # ---------------- Schur complements ----------------
    A_c = A - gamma * (B @ D_inv @ C)
    D_c = D - gamma * (C @ A_inv @ B)

    bA = np.ones(A.shape[0]) - gamma * (
        B @ D_inv @ np.ones(D.shape[0])
    )

    bD = np.ones(D.shape[0]) - gamma * (
        C @ A_inv @ np.ones(A.shape[0])
    )

    # ---------------- Cotton A' matrices ----------------
    bbA = np.outer(bA, bA)
    bbD = np.outer(bD, bD)

    A_prime = _safe_inv(
        _safe_inv(A_c) * bbA
    )

    D_prime = _safe_inv(
        _safe_inv(D_c) * bbD
    )

    A_prime = nearest_pd(A_prime)
    D_prime = nearest_pd(D_prime)

    eigA_fixed = np.linalg.eigvalsh(A_prime)
    eigD_fixed = np.linalg.eigvalsh(D_prime)

    if eigA_fixed.min() <= 0:
        print("A_prime repair failed:", eigA_fixed.min())

    if eigD_fixed.min() <= 0:
        print("D_prime repair failed:", eigD_fixed.min())

    # ---------------- cluster fitness ----------------
    onesA = np.ones(A_prime.shape[0])
    onesD = np.ones(D_prime.shape[0])

    ga = onesA @ _safe_inv(A_prime) @ onesA
    gd = onesD @ _safe_inv(D_prime) @ onesD

    if (ga + gd) <= eps or not np.isfinite(ga + gd):
        raise ValueError(
            f"Invalid fitness: ga={ga}, gd={gd}"
        )

    # ---------------- Cotton A'' matrices ----------------
    A_pp = A_c / np.maximum(bbA, eps)
    D_pp = D_c / np.maximum(bbD, eps)

    # A_pp = nearest_pd(A_pp)
    # D_pp = nearest_pd(D_pp) 

    # eigA_pp = np.linalg.eigvalsh(0.5 * (A_pp + A_pp.T))
    # eigD_pp = np.linalg.eigvalsh(0.5 * (D_pp + D_pp.T))

    # if eigA_pp.min() <= 0:
    #     print("A_pp NOT PD:", eigA_pp.min())

    # if eigD_pp.min() <= 0:
    #     print("D_pp NOT PD:", eigD_pp.min())

    A_pp = pd.DataFrame(
        A_pp,
        index=left,
        columns=left
    )

    D_pp = pd.DataFrame(
        D_pp,
        index=right,
        columns=right
    )

    # ---------------- recurse ----------------
    left_inner = _recursive_hmv(
        A_pp,
        gamma,
        left,
        eps
    )

    right_inner = _recursive_hmv(
        D_pp,
        gamma,
        right,
        eps
    )

    denom = ga + gd

    # ---------------- inter-cluster allocation ----------------
    left_scale = ga / denom
    right_scale = gd / denom

    out = {}

    for k, v in left_inner.items():
        out[k] = v * left_scale

    for k, v in right_inner.items():
        out[k] = v * right_scale

    return out  

def _safe_inv(sigma,
              eps=1e-6,
              cond_threshold=1e8):

    cond = np.linalg.cond(sigma)

    if not np.isfinite(cond):
        return np.linalg.pinv(
            sigma + eps*np.eye(sigma.shape[0])
        )

    if cond > cond_threshold:
        return np.linalg.pinv(
            sigma + eps*np.eye(sigma.shape[0])
        )

    return np.linalg.inv(
        sigma + eps*np.eye(sigma.shape[0])
    )
    
def nearest_pd(M, eps=1e-6):

    M = 0.5 * (M + M.T)

    eigvals, eigvecs = np.linalg.eigh(M)

    eigvals = np.maximum(eigvals, eps)

    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def plot_weights(weights, opt_method='HRP', corr_type='Kendall tau'):
    """
    Render a donut chart of the capital allocation.

    weights : pandas Series of weights indexed by asset.
    """
    output_dir = PROJECT_ROOT / "data" / "output" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, max(4, 0.4 * len(weights))))

    sorted_w = weights.sort_values()
    colors = ["#d7191c" if w < 0 else "#2c7bb6" for w in sorted_w]

    ax.barh(sorted_w.index.astype(str), sorted_w.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Weight")

    for y, w in enumerate(sorted_w.values):
        ax.text(w, y, f" {w:.1%}", va="center",
                ha="left" if w >= 0 else "right", fontsize=8)

    plt.title(f"Capital Allocation {opt_method} - {corr_type}")

    filename = f"capital_allocation_{opt_method}_{corr_type}.png"
    filepath = output_dir / filename

    plt.savefig(filepath, dpi=300, bbox_inches="tight")

    print(f"Saved chart to: {filepath}")

    # plt.show()

def gamma_sensitivity(train_returns, test_returns, 
                      gamma_grid=None,
                      rf_annual=0.0725, 
                      periods=252):
    """
    Sweep gamma from 0 to 1, compute out-of-sample portfolio metrics
    at each point, and plot variance + Sharpe against gamma.

    train_returns : pd.DataFrame  (assets as columns, training period rows)
    test_returns  : pd.DataFrame  (assets as columns, test period rows)
    gamma_grid    : array-like of gamma values; defaults to 51 points in [0,1]
    rf_annual     : annual risk-free rate (default 7.25% SA repo)
    periods       : trading days per year (default 252)

    returns : pd.DataFrame with columns ['gamma','variance','sharpe','weights']
    """

    if gamma_grid is None:
        gamma_grid = np.linspace(0, 1, 51)

    rf_daily = rf_annual / periods

    records  = []
    
    for gamma in gamma_grid:

        # ── 1. fit weights on training data at this gamma ─────────────────
        weights = heuristic_optimisation(
            train_returns=train_returns,
            gamma=gamma
        )

        # align weights to test columns (safety check)
        weights = weights.reindex(test_returns.columns).fillna(0)
        weights = weights / weights.sum()           # re-normalise after fill

        # ── 2. evaluate on test data ──────────────────────────────────────
        port_returns = test_returns @ weights

        ann_return = port_returns.mean() * periods
        ann_vol    = port_returns.std()  * np.sqrt(periods)
        ann_var    = ann_vol ** 2
        sharpe     = (ann_return - rf_annual) / ann_vol

        records.append({
            'gamma'    : round(float(gamma), 4),
            'variance' : ann_var,
            'sharpe'   : sharpe,
            'weights'  : weights                    # stored for later inspection
        })

        print(f"γ = {gamma:.2f}  |  Sharpe = {sharpe:.3f}  |  Var = {ann_var:.4f}")

    results = pd.DataFrame(records)
    return results

def plot_gamma_sensitivity(results, output_dir=None):
    """
    Plot annualised variance and Sharpe ratio against gamma.
    Marks the optimal gamma (highest Sharpe) with a vertical dashed line.

    results    : DataFrame returned by gamma_sensitivity()
    output_dir : pathlib.Path or None; if provided, saves PDF to that directory
    """

    gamma_grid     = results['gamma'].values
    variances      = results['variance'].values
    sharpes        = results['sharpe'].values

    # ── identify optimal gamma ────────────────────────────────────────────
    optimal_idx    = int(np.argmax(sharpes))
    optimal_gamma  = gamma_grid[optimal_idx]
    optimal_sharpe = sharpes[optimal_idx]
    optimal_var    = variances[optimal_idx]

    print(f"\nOptimal γ        : {optimal_gamma:.2f}")
    print(f"Sharpe at opt γ  : {optimal_sharpe:.3f}")
    print(f"Variance at opt γ: {optimal_var:.4f}")
    print(f"\nEndpoint check:")
    print(f"  γ=0 (HRP)  Sharpe : {sharpes[0]:.3f}")
    print(f"  γ=1 (MVP)  Sharpe : {sharpes[-1]:.3f}")

    # ── plot ──────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # primary axis — variance
    color_var = '#2c7bb6'
    ax1.plot(gamma_grid, variances * 100,
             color=color_var, linewidth=2,
             label='Annualised Variance (%)')
    ax1.set_xlabel('γ   (0 = HRP  →  1 = MVP)', fontsize=12)
    ax1.set_ylabel('Annualised Variance (%)', color=color_var, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=color_var)

    # secondary axis — Sharpe
    ax2 = ax1.twinx()
    color_sharpe = '#d7191c'
    ax2.plot(gamma_grid, sharpes,
             color=color_sharpe, linewidth=2,
             linestyle='--', label='Sharpe Ratio')
    ax2.set_ylabel('Sharpe Ratio', color=color_sharpe, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color_sharpe)

    # optimal gamma vertical line
    ax1.axvline(x=optimal_gamma, color='black',
                linestyle=':', linewidth=1.5,
                label=f'Optimal γ = {optimal_gamma:.2f}')

    # dots at optimal point on each curve
    ax1.scatter(optimal_gamma, optimal_var * 100,
                color='black', zorder=5, s=70)
    ax2.scatter(optimal_gamma, optimal_sharpe,
                color='black', zorder=5, s=70, marker='D')

    # annotation box
    ax2.annotate(
        f'Optimal γ = {optimal_gamma:.2f}\nSharpe = {optimal_sharpe:.3f}',
        xy=(optimal_gamma, optimal_sharpe),
        xytext=(min(optimal_gamma + 0.06, 0.88), optimal_sharpe - 0.06),
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', alpha=0.8),
        arrowprops=dict(arrowstyle='->', color='black')
    )

    # endpoint regime labels
    y_bottom = ax1.get_ylim()[0]
    ax1.text(0.01,  y_bottom, 'HRP', fontsize=9, color='grey', va='bottom')
    ax1.text(0.975, y_bottom, 'MVP', fontsize=9, color='grey',
             va='bottom', ha='right')

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper right', fontsize=9, framealpha=0.9)

    plt.title(
        'Figure 3: Gamma Sensitivity — HMV Portfolio Variance and Sharpe Ratio\n'
        '(Test period 2013–2025, RF = 7.25%)',
        fontsize=12, pad=10
    )
    plt.tight_layout()

    # ── save ──────────────────────────────────────────────────────────────
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "output" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / "figure3_gamma_sensitivity.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to: {filepath}")
    plt.show()

    return optimal_gamma, optimal_sharpe