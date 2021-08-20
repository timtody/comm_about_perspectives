import multiprocessing
import os
import time
from argparse import ArgumentParser
from typing import NamedTuple


from experiments.shared_ref_mnist import Experiment
from functions import (
    create_exp_name_and_datetime_path,
    merge_cfg_with_cli,
    run_single_from_sweep_mp,
)
from slurm_runner import run_single_from_sweep_slurm
from sweeper import Sweeper
from copy import copy


# TODO: make this config modular!
class Config(NamedTuple):
    # experiement params
    seed: int = 123
    nprocs: int = 3
    nogpu: bool = False
    logfreq: int = 5000
    nsteps: int = 50001
    nagents: int = 3
    ngpus: int = 1
    mp_method: str = "slurm"
    log_every: int = 50  # how often we write to readers / tb

    nodetach: bool = False

    # agents use same digit as input
    samedigit: bool = False

    # choose the dataset
    dataset: str = "MNIST"  # MNIST, CLUTTER, CIFAR10, CIFAR100

    # hypsearch
    sweeper_mode: str = "grid"  # 'grid' or 'sample'
    gridsteps: int = 1
    nsamples: int = 1

    # nets
    latent_dim: int = 30
    lr: float = 0.001
    bsize: int = 4096

    # bnorm
    bnorm: bool = False
    affine: bool = False

    # channel noise
    sigma: float = 0.25

    # hyperparameters
    eta_ae: float = 1.0
    eta_lsa: float = 0.0
    eta_msa: float = 0.0
    eta_dsa: float = 0.0

    # assessment of abstraction
    nsteps_pred_latent: int = 5000
    bsize_pred_latent: int = 4096


class RunnerCfg(NamedTuple):
    jobname: str = "job"
    gpu_or_cpu: str = "gpu"
    gb: int = 16  # 16 or 32
    nnodes: int = 1
    ntasks: int = 1
    time: str = "20:00:00"
    cpus_per_task: int = 2
    nrpocs: int = 3
    qos: str = "qos_gpu-t3"


def generate_exp_path(exp, args, tracking_vars):
    name_and_datetime: str = create_exp_name_and_datetime_path(exp)
    path_root = os.path.join("results", "sweeps", name_and_datetime)
    tracking = ""
    for varname in tracking_vars:
        tracking += f"{varname}:{args.__getattribute__(varname)}-"
    path = os.path.join(path_root, tracking)
    return path


def generate_sweep_path(experiment):
    """Generate the path for a sweep which has the form

    results/sweeps/EXP_NAME/DATE/TIME/

    It is then passed to generate_run_path as a root path
    """
    exp_path = create_exp_name_and_datetime_path(experiment)
    return os.path.join("results", "sweeps", exp_path)


def generate_tracking_tag(tracking_vars):
    tracking_tag = ""
    for varname in tracking_vars:
        tracking_tag += f"{varname}:{args.__getattribute__(varname)}-"
    return tracking_tag


def generate_run_path(root_path, args, tracking_vars):
    """Generates the path for the individual run. Will be inferred
    from tracking_vars and root_path. The result will be passed to the experiment.
    """
    path = os.path.join(root_path, generate_tracking_tag(tracking_vars))
    return path


class InvalidConfigurationException(BaseException):
    """Exception"""

    pass


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    cfg = Config()
    parser = ArgumentParser()
    merge_cfg_with_cli(cfg, parser)
    runner_args = RunnerCfg()
    args = parser.parse_args()

    hparams = ["eta_ae", "eta_lsa", "eta_msa", "eta_dsa", "sigma", "nagents"]

    sweep_root_path = generate_sweep_path(Experiment)

    # this is specific to the jean-zay cluster
    if args.mp_method == "slurm":
        sweep_root_path = os.path.join(os.path.expandvars("$SCRATCH"), sweep_root_path)

    processes = []

    sweeper = Sweeper(
        sample_vars=hparams[:-1],
        grid_vars=hparams[-1:],
        nsamples=args.nsamples,
        gridsteps=args.gridsteps,
    )
    param_list = [
        [
            ("eta_ae", 0.0),
            ("eta_lsa", 0.0),
            ("eta_msa", 1.0),
            ("eta_dsa", 0.0),
        ],
        [
            ("eta_ae", 0.53),
            ("eta_lsa", 0.01),
            ("eta_msa", 0.74),
            ("eta_dsa", 0.84),
        ],
        [
            ("eta_ae", 0.81),
            ("eta_lsa", 0.14),
            ("eta_msa", 0.95),
            ("eta_dsa", 0.01),
        ],
        [
            ("eta_ae", 1.0),
            ("eta_lsa", 0.1),
            ("eta_msa", 0.0),
            ("eta_dsa", 0.0),
        ],
    ]

    noise_levels = [0.67]
    n_agents = [3]
    fixed_sweep = []
    for params in param_list:
        for nl in noise_levels:
            for nag in n_agents:
                new_params = copy(params)
                new_params.append(("sigma", nl))
                new_params.append(("nagents", nag))
                fixed_sweep.append(new_params)

    print("[SWEEPER]: Starting experiment at path:", sweep_root_path)
    for vars in fixed_sweep:
        for var, value in vars:
            args.__setattr__(var, value)

        path: str = generate_run_path(sweep_root_path, args, hparams)
        print("Starting experiment on path", path)
        if args.mp_method == "mp":
            procs = run_single_from_sweep_mp(Experiment, args, path)
            processes += procs
        elif args.mp_method == "slurm":
            for rank in range(args.nprocs):
                jobname = generate_tracking_tag(hparams) + str(rank)
                print("[SWEEPER]: Starting SLURM job:", jobname)
                run_single_from_sweep_slurm(args, runner_args, path, rank, jobname)
            # this is required by the IDRIS administration to keep the throughput of jobs lower
            time.sleep(2)
        else:
            raise InvalidConfigurationException("[SWEEPER]: Invalid mp method name.")

    if args.mp_method == "mp":
        for proc in processes:
            proc.start()
        for proc in processes:
            proc.join()
