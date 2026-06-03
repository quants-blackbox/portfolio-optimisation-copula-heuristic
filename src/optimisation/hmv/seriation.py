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

    cov_matrix_reordered = cov_matrix.loc[sorted_items_names, sorted_items_names]  # reorder
    plot_corr_matrix(PROJECT_ROOT / "data" / "output" / "results"/ "quasi_diagonalisation.png", cov_matrix_reordered, labels=cov_matrix_reordered.columns)

    return sorted_items_names
