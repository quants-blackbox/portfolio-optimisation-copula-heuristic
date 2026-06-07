import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.dependence.dependence import PROJECT_ROOT

def mvp_weights(cov):
    """
    Global Minimum Variance Portfolio
    """

    ones = np.ones(len(cov))

    inv_cov = np.linalg.pinv(cov)

    w = inv_cov @ ones

    w = w / (ones.T @ inv_cov @ ones)

    # label weights by asset so plotting/inspection works on a Series
    assets = cov.columns if isinstance(cov, pd.DataFrame) else range(len(w))
    w = pd.Series(w, index=assets)

    # _plot_weights(w)

    return w

def _plot_weights(weights, opt_method='MVP', corr_type='Sample covariance'):
    """
    Render a horizontal bar chart of the capital allocation.

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

