from experiments.shared_ref_mnist import Experiment
from functions import run
from typing import NamedTuple


# TODO: implement command line overrides and use type hints to provide autocomplete
class Config(NamedTuple):
    seed: int = 123
    nprocs: int = 5
    gpu: bool = True
    nsteps: int = 5000
    latent_dim: int = 30
    bnorm: bool = True
    affine: bool = True
    lr: float = 0.001
    nagents: int = 3
    logfreq: int = 1000
    bsize: int = 129

    eta_ae: float = 1
    eta_lsa: float = 1
    eta_msa: float = 1
    eta_dsa: float = 1


if __name__ == "__main__":
    run(Experiment, Config())