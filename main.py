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


class Config(NamedTuple):
    # experiement params
    seed: int = 123
    nprocs: int = 1
    gpu: bool = True
    logfreq: int = 10000
    nsteps: int = 50001
    nagents: int = 3

    # hypsearch
    grid_size: int = 2

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
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("fork")
    # TODO: create the experiment path here probably
    cfg = Config()
    args = merge_cfg_with_cli(cfg)
    # TODO: fix this and make a real hyp search
    tacking_vars = "sigma", "eta_lsa", "eta_ae"
    args.tacking_vars = tacking_vars

    sweep_root_path = generate_sweep_path(Experiment)

    processes = []
    for sigma in np.linspace(0, 1, args.grid_size):
        for eta in np.linspace(0, 1, args.grid_size):
            args.sigma = round(sigma, 2)
            args.eta_lsa = round(eta, 2)
            args.eta_ae = round(1 - eta, 2)
            print("Running with args", args)

            path: str = generate_run_path(sweep_root_path, args, args.tacking_vars)
            procs = run_single_from_sweep(Experiment, args, path)
            processes += procs

    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()
