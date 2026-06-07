from RiskLabAI.optimization import *

from src.optimisation.hmv.clustering import clustering_matrix, dependence_matrix
from src.data.returns import split_returns
from src.dependence.dependence import PROJECT_ROOT

def quasi_diagonalisation():
    """
    
    """

    link = clustering_matrix()

    dep_matrix = dependence_matrix(1)

    train_returns, _ = split_returns()
    cov_matrix = train_returns.cov()

    sorted_items_idx = quasi_diagonal(link)
    sorted_items_names = dep_matrix.index[sorted_items_idx].tolist()

    dep_matrix_reordered = dep_matrix.loc[sorted_items_names, sorted_items_names]  # reorder
    # plot_corr_matrix(PROJECT_ROOT / "data" / "output" / "results"/ "quasi_diagonalisation.png", cov_matrix_reordered, labels=cov_matrix_reordered.columns)

    return dep_matrix_reordered, sorted_items_names

def plot_covariance_matrix(cov_matrix,
                           title=r"Reordered Kendall's $\tau$ Matrix",
                           save=True):

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        cov_matrix,
        aspect='auto',
        cmap='YlOrRd'
    )

    ax.set_xticks(np.arange(len(cov_matrix.columns)))
    ax.set_yticks(np.arange(len(cov_matrix.index)))

    ax.set_xticklabels(
        cov_matrix.columns,
        rotation=90,
        fontsize=8
    )

    ax.set_yticklabels(
        cov_matrix.index,
        fontsize=8
    )

    ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"Kendall's $\tau$", rotation=270, labelpad=15)

    plt.tight_layout()

    if save:
        print("Save dendrogram")

        out_path = PROJECT_ROOT / "data" / "output" / "results" / "quasi_diag_kendalls_tau.png"
        plt.savefig(out_path, dpi=150)

    # plt.show()