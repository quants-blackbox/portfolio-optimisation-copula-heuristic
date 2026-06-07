from RiskLabAI.optimization import *
from RiskLabAI.optimization.hrp import distance_corr
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as scd

from src.data.wrangler import get_kept_tickers
from src.dependence.dependence import PROJECT_ROOT
from src.dependence.copula_modelling import load_vine_model
from src.dependence.reporting import vine_copula_table

output_dir = PROJECT_ROOT / "data" / "output" / "results" / "vine_copula_table.csv"

def dependence_matrix (tree_index = 1):

    if not output_dir.exists():
        vine_copula_table(load_vine_model())

    df = pd.read_csv(output_dir)
    tree = df[df["Tree"] == int(tree_index)]


    assets = get_kept_tickers() # sorted(set(tree["Asset 1"]).union(set(tree["Asset 2"])))

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

    distance_matrix = distance_corr(dep_matrix.values)

    # Condense the square distance matrix into the 1D vector scipy expects.
    dist_condensed = scd.squareform(distance_matrix, force='tovector', checks=False)

    link = sch.linkage(dist_condensed, 'single')

    # pass the ticker names so the dendrogram leaves are labelled by asset
    plot_dendrogram(link=link, labels=dep_matrix.columns.tolist())

    return link


def plot_dendrogram(link=None, labels=None, save=True):

    fig, ax = plt.subplots(figsize=(12, 6))
    sch.dendrogram(link, labels=labels, leaf_rotation=90, ax=ax)
    ax.set_title('Hierarchical Clustering Dendrogram (single linkage)')
    ax.set_xlabel('Assets')
    ax.set_ylabel('Distance')
    plt.tight_layout()

    if save:
        print("Save dendrogram")

        out_path = PROJECT_ROOT / "data" / "output" / "results" / "dendrogram.png"
        plt.savefig(out_path, dpi=150)

    return fig
