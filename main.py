from experiments.shared_ref_mnist import Experiment
from functions import run
from typing import NamedTuple
import argparse


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
    eta_ae: float = 1.0
    eta_lsa: float = 0.0
    eta_msa: float = 0.0
    eta_dsa: float = 0.0

    # assessment of abstraction
    nsteps_pred_latent: int = 100
    bsize_pred_latent: int = 64


if __name__ == "__main__":
    cfg = Config()
    parser = argparse.ArgumentParser()
    for field in cfg._fields:
        if isinstance(cfg.__getattribute__(field), bool):
            parser.add_argument(f"--{field}", type=eval)
        else:
            parser.add_argument(
                f"--{field}",
                type=type(cfg.__getattribute__(field)),
                default=cfg.__getattribute__(field),
            )
    args = parser.parse_args()
    print("Running with args", args)
    run(Experiment, Config())