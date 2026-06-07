import logging

from src.data.returns import split_returns
from src.dependence.pseudo_observations import pseudo_observations
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import pyvinecopulib as pv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

def cov_matrix():
    returns,_ = split_returns()
    return returns.cov()
    

def person_correlation():
    """
    Calculate Pearson correlation matrix of log returns.
    """
    returns,_ = split_returns()

    return returns.corr()

def emp_kendall_tau():
    """
    Calculate empirical Kendall's tau matrix of rank based scores (pseudo-observations).
    """
    returns = pseudo_observations()

    return returns.corr(method='kendall')

def build_network():

    G = nx.Graph()
    tau_matrix = emp_kendall_tau()
    tickers = tau_matrix.columns

    for i in range(tau_matrix.shape[0]):
        for j in range(i + 1, tau_matrix.shape[0]):
            tau = abs(tau_matrix.iloc[i, j])
            G.add_edge(tickers[i], tickers[j], weight=tau)
    return G

def max_spanning_tree(G, algorithm="prim"):
    """
    Compute the maximum spanning tree of the graph G.
    """
    return nx.maximum_spanning_tree(G, algorithm=algorithm)

def plot_network(title, T):
    """
    Plot the graph G with edge weights.
    """
    plt.figure(figsize=(8, 6))

    pos = nx.spring_layout(T)

    nx.draw(
        T,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=300,
        font_size=6
    )

    plt.title(title)
    save_plot(plt, "dependence_network.png")

def save_plot(plt, plot_name):
    """
    Save the plot to the output directory.
    """
    output_dir = PROJECT_ROOT / "data" / "output" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / plot_name
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    plt.close()
    logging.log(logging.INFO, "Plot saved to %s", plot_path)