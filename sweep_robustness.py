import multiprocessing
import os
from argparse import ArgumentParser
from typing import NamedTuple


from experiments.robustness import Experiment
from functions import (
    create_exp_name_and_datetime_path,
    merge_cfg_with_cli,
    run_single_from_sweep_mp,
)
from sweeper import Sweeper

from experiments.robustness import Config


class RunnerCfg(NamedTuple):
    jobname: str = "job"
    gpu_or_cpu: str = "gpu"
    qos: str = "qos_gpu-t3"
    gb: int = 16  # 16 or 32
    nnodes: int = 1
    ntasks: int = 1
    time: str = "20:00:00"
    cpus_per_task: int = 2


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
    cfg: Config = Config()
    parser = ArgumentParser()
    merge_cfg_with_cli(cfg, parser)
    runner_args = RunnerCfg()
    args = parser.parse_args()

    hparams = ["sigma"]

    sweep_root_path = generate_sweep_path(Experiment)

    # this is specific to the jean-zay cluster
    if args.mp_method == "SLURM":
        sweep_root_path = os.path.join(os.path.expandvars("$SCRATCH"), sweep_root_path)

    processes = []

    sweeper = Sweeper(
        grid_vars=hparams,
        gridsteps=args.gridsteps,
        warmup=False,
        grid_range=2
    )

    print("[SWEEPER]: Starting experiment at path:", sweep_root_path)
    for vars in sweeper.sweep_grid():
        for var, value in vars:
            args.__setattr__(var, value)

        path: str = generate_run_path(sweep_root_path, args, hparams)
        print("Starting experiment on path", path)
        if args.mp_method == "MP":
            procs = run_single_from_sweep_mp(Experiment, args, path)
            processes += procs
        elif args.mp_method == "SLURM":
            for rank in range(args.nprocs):
                jobname = generate_tracking_tag(hparams) + str(rank)
                print("[SWEEPER]: Starting SLURM job:", jobname)
                #run_single_from_sweep_slurm(args, runner_args, path, rank, jobname)
            # this is required by the IDRIS administration to keep the throughput of jobs lower
            #time.sleep(5)
        else:
            raise InvalidConfigurationException("[SWEEPER]: Invalid mp method name.")

    if args.mp_method == "mp":
        for proc in processes:
            proc.start()
        for proc in processes:
            proc.join()
