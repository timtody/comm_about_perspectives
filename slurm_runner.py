import os
import subprocess


def unpack_args(**kwargs):
    flags = ""
    for key, value in kwargs.items():
        if isinstance(value, bool):
            flags += f"--{key} " if value else ""
        else:
            flags += f"--{key} {value} "
    return flags


def run_single_from_sweep_slurm(cfg, runner_args, path, rank, jobname):
    # TODO: make experiment configurable from command line s.t. we can pass it on
    # as an argument here
    if not os.path.exists(path):
        os.makedirs(path)
    sbatch_file = (
        f"#!/bin/bash\n"
        f"#SBATCH --job-name={jobname}\n"
        f"#SBATCH -C v100-{runner_args.gb}g\n"
        f"#(use '-C v100-32g' for 32 GB GPUs only)\n"
        f"#SBATCH -A imi@{runner_args.gpu_or_cpu}\n"
        f"#SBATCH --qos={runner_args.qos}"
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
        f"srun python run_single_slurm --rank {rank} --path {path} {unpack_args(**vars(cfg))}\n"
    )
    with open("tmp", "w") as f:
        f.writelines(sbatch_file)
    subprocess.run(["sbatch", "tmp", "&"])
    subprocess.run(["rm", "tmp"])
