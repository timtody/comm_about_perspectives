import multiprocessing
import os
from typing import NamedTuple

import numpy as np

from experiments.shared_ref_mnist import Experiment
from functions import (
    create_exp_name_and_datetime_path,
    merge_cfg_with_cli,
    run,
    run_single_from_sweep,
)

# TODO: make this config modular!
class Config(NamedTuple):
    # experiement params
    seed: int = 123
    nprocs: int = 1
    gpu: bool = True
    logfreq: int = 1000
    nsteps: int = 50001
    nagents: int = 3
    ngpus: int = 4

    # hypsearch
    grid_size: int = 2
    nsamples: int = 10

    # nets
    latent_dim: int = 30
    lr: float = 0.001
    bsize: int = 128

    # bnorm
    bnorm: bool = False
    affine: bool = False

    # channel noise
    sigma: float = 0.0

    # hyperparameters
    eta_ae: float = 0.0
    eta_lsa: float = 0.0
    eta_msa: float = 0.0
    eta_dsa: float = 0.0

    # assessment of abstraction
    nsteps_pred_latent: int = 2000
    bsize_pred_latent: int = 128


def generate_exp_path(exp, args, tracking_vars):
    name_and_datetime: str = create_exp_name_and_datetime_path(exp)
    path_root = os.path.join("results", "sweeps", name_and_datetime)
    tracking = ""
    for varname in tracking_vars:
        tracking += f"{varname}:{args.__getattribute__(varname)}_"
    path = os.path.join(path_root, tracking)
    return path


def generate_sweep_path(experiment):
    """Generate the path for a sweep which has the form

    results/sweeps/EXP_NAME/DATE/TIME/

    It is then passed to generate_run_path as a root path
    """
    exp_path = create_exp_name_and_datetime_path(experiment)
    return os.path.join("results", "sweeps", exp_path)


def generate_run_path(root_path, args, tracking_vars):
    """Generates the path for the individual run. Will be inferred
    from tracking_vars and root_path. The result will be passed to the experiment.
    """
    tracking = ""
    for varname in tracking_vars:
        tracking += f"{varname}:{args.__getattribute__(varname)}_"
    path = os.path.join(root_path, tracking)
    return path


if __name__ == "__main__":
    # TODO: generate sbatch script on the fly probably
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("fork")
    cfg = Config()
    args = merge_cfg_with_cli(cfg)

    tracking_vars = "sigma", "eta_lsa", "eta_ae"
    args.tracking_vars = tracking_vars

    sweep_root_path = generate_sweep_path(Experiment)

    processes = []
    for _ in range(args.nsamples):
        eta = np.random.rand()
        sigma = np.random.rand()

        args.sigma = round(sigma, 2)
        args.eta_lsa = round(eta, 2)
        args.eta_ae = round(1 - eta, 2)

        path: str = generate_run_path(sweep_root_path, args, args.tracking_vars)
        print("Starting experiment on path", path)
        procs = run_single_from_sweep(Experiment, args, path)
        processes += procs

    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()
