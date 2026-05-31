from src.dependence.pseudo_observations import pseudo_observations
from pathlib import Path

import logging
import numpy as np
import pyvinecopulib as pv
import pickle

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def fit_vine_model ():
    """
    Fit an r-vine copula model to the pseudo-observations of the log returns.
    """

    pseudo_obs = pseudo_observations()
    data = np.asfortranarray(pseudo_obs.to_numpy(dtype=np.float64))

    copula_families = [pv.BicopFamily.gaussian, pv.BicopFamily.student]
    controls = pv.FitControlsVinecop(family_set=copula_families)
    model = pv.Vinecop.from_data(data=data, controls=controls)

    logger.info(f"Fitted r-vine model: {model}")

    # Save model
    output_dir = PROJECT_ROOT / "data" / "output"/ "models"
    with open(output_dir / 'copula_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    logger.info(f"Saved copula model: {model}")

    return model

def load_vine_model():
    """
    Load the fitted r-vine copula model from disk.
    """

    input_dir = PROJECT_ROOT / "data" / "output"/ "models"
    if input_dir.exists() and (input_dir / 'copula_model.pkl').exists():
        with open(input_dir / 'copula_model.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        logger.warning("No saved copula model found. Fitting a new model.")
        model = fit_vine_model()
    return model

