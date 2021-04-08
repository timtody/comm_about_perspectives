from experiments.shared_ref_mnist import Experiment
from functions import run, merge_cfg_with_cli
from typing import NamedTuple
import numpy as np


# TODO: implement command line overrides and use type hints to provide autocomplete
class Config(NamedTuple):
    # experiement params
    seed: int = 123
    nprocs: int = 5
    gpu: bool = True
    logfreq: int = 10000
    nsteps: int = 50001
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
    eta_ae: float = 0
    eta_lsa: float = 0.0
    eta_msa: float = 0.0
    eta_dsa: float = 0.0

    # assessment of abstraction
    nsteps_pred_latent: int = 5000
    bsize_pred_latent: int = 64


if __name__ == "__main__":
    # TODO: create the experiment path here probably
    cfg = Config()
    args = merge_cfg_with_cli(cfg)
    gridsize = 5
    # TODO: fix this and make a real hyp search
    tb_tracking_vars = "sigma", "eta_lsa", "eta_ae"
    args.tb_tracking_vars = tb_tracking_vars
    for sigma in np.linspace(0, 1, gridsize):
        for eta in np.linspace(0, 1, gridsize):
            args.sigma = round(sigma, 2)
            args.eta_lsa = round(eta, 2)
            args.eta_ae = round(1 - eta, 2)
            print("Running with args", args)
            run(Experiment, args)