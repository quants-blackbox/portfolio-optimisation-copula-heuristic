from RiskLabAI.optimization import *
import pandas as pd


from src.dependence.dependence import PROJECT_ROOT

output_dir = PROJECT_ROOT / "data" / "output" / "results" / "vine_copula_table.csv"

def dependence_matrix (tree_index = 1):

    df = pd.read_csv(output_dir)
    tree = df[df["Tree"] == int(tree_index)]

    print('Shape of tree', tree.shape)

    assets = sorted(set(tree["Asset 1"]).union(set(tree["Asset 2"])))
    tau_matrix = pd.DataFrame(0.0, index=assets, columns=assets)

    for _, row in tree.iterrows():
        a, b, tau = row["Asset 1"], row["Asset 2"], row["Kendall's Tau"]
        tau_matrix.loc[a, b] = tau
        tau_matrix.loc[b, a] = tau

    for a in assets:
        tau_matrix.loc[a, a] = 1.0

    return tau_matrix

def clustering_matrix():

    # Dependence matrix
    dep_matrix = dependence_matrix(1)

    distance_matrix = distance_corr(dep_matrix)

    link = sch.linkage(distance_matrix, 'single')

    return link


def quasi_diagonalisation():

    link = clustering_matrix()
    
    dep_matrix = dependence_matrix(1)

    # 2. quasi-diagonalisation phase
    sortIx = quasi_diagonal(link)
    sortIx_index = sortIx.copy()
    sortIx = dep_matrix.index[sortIx].tolist()
    cov_matrix_reordered = dep_matrix.loc[sortIx, sortIx]  # reorder
    plot_corr_matrix(PROJECT_ROOT / "data" / "output" / "results"/ "quasi_diagonalisation.png", cov_matrix_reordered, labels=cov_matrix_reordered.columns)

    sorted_items_names = dep_matrix.index[sortIx_index].tolist()
    
    return sorted_items_names