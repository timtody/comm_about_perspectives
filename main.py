from experiments.shared_ref_mnist import Experiment
from functions import run
from typing import NamedTuple


# TODO: implement command line overrides and use type hints to provide autocomplete
class Config(NamedTuple):
    # experiement params
    seed: int = 123
    nprocs: int = 5
    gpu: bool = True
    logfreq: int = 1000
    nsteps: int = 5001
    nagents: int = 3

    # nets
    latent_dim: int = 30
    lr: float = 0.001
    bsize: int = 64

    # bnorm
    bnorm: bool = True
    affine: bool = True

    # channel noise
    sigma: float = 0.0

    # hyperparameters
    eta_ae: float = 1
    eta_lsa: float = 0
    eta_msa: float = 0
    eta_dsa: float = 0

    # assessment of abstraction
    nsteps_pred_latent: int = 100
    bsize_pred_latent: int = 64


if __name__ == "__main__":
    run(Experiment, Config())