import pandas as pd

from src.dependence.copula_modelling import load_vine_model
from src.dependence.dependence import PROJECT_ROOT
from src.dependence.pseudo_observations import pseudo_observations
from helper.ticker_mapper import get_ticker_map

PROJECT_ROOT = PROJECT_ROOT / "data" / "output"/ "results"

def vine_copula_table(model):
    """
    Generate a table summarizing the fitted r-vine copula model, including the tree structure, copula families, Kendall's tau values, and degrees of freedom for Student's t copulas.
    """
    rows = []

    matrix = model.structure.matrix
    d = model.dim

    ticker_map = get_ticker_map()

    for tree in range(d - 1):

        for edge in range(d - tree - 1):

            bicop = model.get_pair_copula(tree, edge)

            family = bicop.family.name
            tau = bicop.tau

            df = None
            if family.lower() == "student":
                params = bicop.parameters.flatten()
                if len(params) > 1:
                    df = float(params[1])

            var1 = matrix[tree, edge]
            var2 = matrix[tree + 1, edge]

            rows.append({
                "Tree": tree + 1,
                "Asset 1": ticker_map.get(int(var1), var1),
                "Asset 2": ticker_map.get(int(var2), var2),
                "Family": family,
                "Kendall's Tau": round(float(tau), 4),
                "DF": df
            })

    _save_vine_copula_table(pd.DataFrame(rows))

def _save_vine_copula_table(table):

    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    table.to_csv(PROJECT_ROOT / "vine_copula_table.csv", index=False)