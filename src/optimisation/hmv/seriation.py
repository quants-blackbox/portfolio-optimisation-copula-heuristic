from RiskLabAI.optimization import *

from src.optimisation.hmv.clustering import clustering_matrix, dependence_matrix
from src.dependence.dependence import PROJECT_ROOT

def quasi_diagonalisation():
    """
    
    """

    link = clustering_matrix()
    
    dep_matrix = dependence_matrix(1)

    sortIx = quasi_diagonal(link)
    sortIx_index = sortIx.copy()
    sortIx = dep_matrix.index[sortIx].tolist()
    cov_matrix_reordered = dep_matrix.loc[sortIx, sortIx]  # reorder
    plot_corr_matrix(PROJECT_ROOT / "data" / "output" / "results"/ "quasi_diagonalisation.png", cov_matrix_reordered, labels=cov_matrix_reordered.columns)

    sorted_items_names = dep_matrix.index[sortIx_index].tolist()
    
    return sorted_items_names
