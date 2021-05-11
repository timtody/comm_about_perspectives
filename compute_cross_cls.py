import os
import pathlib
from typing import NamedTuple

from slurm_runner_cross_agent import run_single_from_sweep_slurm

PATH = "results/sweeps/shared_ref_mnist/2021-04-20/14-58-18"
PATH = os.path.join(os.path.expandvars("$SCRATCH"), PATH)


class Config(NamedTuple):
    nogpu: bool = False
    nprocs: int = 0
    seed: int = 123
    ngpus: int = 1
    nsteps: int = 0
    bsize: int = 0
    eval_bsize: int = 0


class RunnerCfg(NamedTuple):
    jobname: str = "job"
    gpu_or_cpu: str = "gpu"
    gb: int = 16  # 16 or 32
    nnodes: int = 1
    ntasks: int = 1
    time: str = "5:00:00"
    cpus_per_task: int = 2
    nprocs: int = 5


if __name__ == "__main__":
    exp_paths = pathlib.Path(PATH).glob("*")
    runner_cfg = RunnerCfg()

    for path in list(exp_paths)[:1]:
        cfg = Config(nsteps=2500, bsize=2048, eval_bsize=8192, nogpu=False, nprocs=5)
        for rank in range(runner_cfg.nprocs):
            run_single_from_sweep_slurm(cfg, runner_cfg, path, rank, "cooljob")
