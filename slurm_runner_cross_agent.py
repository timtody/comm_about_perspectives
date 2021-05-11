import os
import subprocess
from typing import NamedTuple
import random


def unpack_args(**kwargs):
    flags = ""
    for key, value in kwargs.items():
        if isinstance(value, bool):
            flags += f"--{key} " if value else ""
        else:
            flags += f"--{key} {value} "
    return flags


def run_single_from_sweep_slurm(cfg: NamedTuple, runner_args, path, rank, jobname):
    # TODO: make experiment configurable from command line s.t. we can pass it on
    # as an argument here
    print("Running SLURM experiment at path:", path)
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print("Path exists, skipping file creation...")
    sbatch_file = (
        f"#!/bin/bash\n"
        f"#SBATCH --job-name={jobname + str(random.randint(0, 99999))}\n"
        f"#SBATCH -C v100-{runner_args.gb}g\n"
        f"#(use '-C v100-32g' for 32 GB GPUs only)\n"
        f"#SBATCH -A imi@{runner_args.gpu_or_cpu}\n"
        f"#SBATCH --output={path}/out\n"
        f"#SBATCH --error={path}/err\n"
        f"#SBATCH --time={runner_args.time}\n"
        f"#SBATCH --nodes={runner_args.nnodes}\n"
        f"#SBATCH --ntasks={runner_args.ntasks}\n"
        f"#SBATCH --gres=gpu:1\n"
        f"#SBATCH --cpus-per-task={runner_args.cpus_per_task}\n"
        f"module purge\n"
        f"module load python/3.8.2\n"
        f"set -x\n"
        f"srun python run_exp.py --rank {rank} --path {path} {unpack_args(**cfg._asdict())}\n"
    )
    os.chdir(os.path.expandvars("$SCRATCH"))
    print(os.getcwd())
    print(sbatch_file)
    with open("tmp", "w") as f:
        f.writelines(sbatch_file)
    print("Sending it off..")
    subprocess.run(["sbatch", "tmp", "&"])
    print("Cleanup...")
    subprocess.run(["rm", "tmp"])
