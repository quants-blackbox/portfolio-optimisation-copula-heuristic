from RiskLabAI.optimization import recursive_bisection
from RiskLabAI.optimization.hrp import *
import riskfolio as rp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.dependence.dependence import cov_matrix
from src.optimisation.hmv.clustering import dependence_matrix
from src.optimisation.hmv.seriation import quasi_diagonalisation


def heuristic_optimisation(opt_type='HRP', gamma=0):
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
    ordered_assets = quasi_diagonalisation()

    # 3. recursive bisection phase
    if opt_type == 'HRP':
        # HRP bisects on the dependence (Kendall-tau) matrix.
        reordered = dependence_matrix(1).loc[ordered_assets, ordered_assets]
        return recursive_bisection(reordered, ordered_assets)
    elif opt_type == 'HMV':
        # HMV needs a genuine covariance matrix (Schur complements are scale-aware).
        reordered_cov = cov_matrix().loc[ordered_assets, ordered_assets]
        return pd.Series(recursive_hmv(reordered_cov, gamma, ordered_assets))

    raise ValueError(f"Unknown opt_type {opt_type!r}; expected 'HRP' or 'HMV'.")


def recursive_hmv(sigma, gamma, orderded_assets, eps=1e-8):
    """
    Recursive HMV weight builder using Schur complements.
    sigma (covariance matrix, rows/cols == ordered_assets)
    ordered_assets : list of asset labels in quasi-diagonal order
    gamma : parameter in [0,1]; gamma=0 -> HRP-like, gamma=1 -> full Schur/MV
    returns : dict {asset: weight} summing to 1
    """

    # base case
    n = len(orderded_assets)
    if n == 1:
        return {orderded_assets[0]: 1.0}

    mid = n // 2
    left = orderded_assets[:mid]
    right = orderded_assets[mid:]

    # extract blocks (use .loc to keep index/column alignment)
    A = sigma.loc[left, left].values       # size m x m
    D = sigma.loc[right, right].values     # size k x k
    B = sigma.loc[left, right].values      # m x k
    C = sigma.loc[right, left].values      # k x m  (C = B.T for symmetric cov)

    
    # safe inverses
    A_inv = _safe_inv(A)
    D_inv = _safe_inv(D)

    # Schur complements and b-vectors
    # A_c(gamma) = A - gamma * B D^{-1} C
    # b_A(gamma) = 1_m - gamma * B D^{-1} 1_k
    A_c = A - gamma * (B @ D_inv @ C)
    bA = np.ones((A.shape[0],)) - gamma * (B @ D_inv @ np.ones((D.shape[0],)))

    # D_c(gamma) = D - gamma * C A^{-1} B
    # b_D(gamma) = 1_k - gamma * C A^{-1} 1_m
    D_c = D - gamma * (C @ A_inv @ B)
    bD = np.ones((D.shape[0],)) - gamma * (C @ A_inv @ np.ones((A.shape[0],)))

    # invert Schur blocks robustly
    A_c_inv = _safe_inv(A_c)
    D_c_inv = _safe_inv(D_c)

    # compute group-level unnormalised allocations
    # vector contributions for each group:
    wa_unn = A_c_inv @ bA        # length m
    wd_unn = D_c_inv @ bD        # length k

    # collapse to group-level scalars (how much mass goes to left vs right)
    ga = wa_unn.sum()
    gd = wd_unn.sum()

    # If both zero (numerical degenerate), fall back to equal split
    if (ga + gd) == 0 or not np.isfinite(ga + gd):
        ga = 1.0
        gd = 1.0

    # recursively compute inside-group relative allocations (these return dicts summing to 1 per group)
    left_inner = recursive_hmv(sigma.loc[left, left], gamma, left)
    right_inner = recursive_hmv(sigma.loc[right, right], gamma, right)

    left_scale = ga / (ga + gd)
    right_scale = gd / (ga + gd)

    out = {}
    for k, v in left_inner.items():
        out[k] = v * left_scale
    for k, v in right_inner.items():
        out[k] = v * right_scale

    return out      




def _safe_inv(sigma, eps=1e-8):
    """
    Invert M robustly: try np.linalg.inv, else use ridge regularization,
    else use pseudo-inverse. Returns matrix of same shape.       
    """
    try:
        return np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        try:
            return np.linalg.inv(sigma + eps * np.eye(sigma.shape[0]))
        except np.linalg.LinAlgError:
            return np.linalg.pinv(sigma)


def plot_weights(weights, opt_method='HRP', corr_type='Kendall tau'):
    """
    Render a donut chart of the capital allocation.

    weights : pandas Series of weights indexed by asset.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts = ax.pie(weights, startangle=90, wedgeprops={'width': 0.4})
    percentages = [f'{p:.1%}' for p in weights]  # convert to percentage string
    legend_labels = [f'{label}: {pct}' for label, pct in zip(weights.index, percentages)]

    ax.legend(wedges, legend_labels, title="Assets", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title(f'Capital Allocation {opt_method} - {corr_type}')
    plt.show()