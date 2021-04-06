from experiments.marl_ae import Experiment
from functions import run
from typing import NamedTuple


class Config(NamedTuple):
    seed: int = 0
    nprocs: int = 2
    gpu: bool = True


if __name__ == "__main__":
    run(Experiment, Config())