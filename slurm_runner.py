from argparse import ArgumentParser
from typing import NamedTuple
from functions import merge_cfg_with_cli
import subprocess


def unpack_args(**kwargs):
    return "".join(map(lambda kv: f"--{kv[0]} {kv[1]} ", kwargs.items()))


def run_single_from_sweep_slurm(cfg, runner_args, path, rank, jobname):
    # TODO: make experiment configurable from command line s.t. we can pass it on
    # as an argument here
    sbatch_file = (
        f"#!/bin/bash\n"
        f"#SBATCH --job-name={jobname}\n"
        f"#SBATCH -C v100-{runner_args.gb}g\n"
        f"#(use '-C v100-32g' for 32 GB GPUs only)\n"
        f"#SBATCH -A imi@{runner_args.gpu_or_cpu}\n"
        f"#SBATCH --output=out/%j\n"
        f"#SBATCH --error=err/%j\n"
        f"#SBATCH --time={runner_args.time}\n"
        f"#SBATCH --nodes={runner_args.nnodes}\n"
        f"#SBATCH --ntasks={runner_args.ntasks}\n"
        f"#SBATCH --gres=gpu:1\n"
        f"#SBATCH --cpus-per-task={runner_args.cpus_per_task}\n"
        f"module purge\n"
        f"module load python/3.8.2\n"
        f"set -x\n"
        f"srun python run_single_slurm.py --rank {rank} --path {path} {unpack_args(**vars(cfg))}\n"
    )
    subprocess.run(["sbatch", sbatch_file])