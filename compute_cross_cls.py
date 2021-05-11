from experiments.experiment import BaseConfig
from chunked_writer import TidyWriter
import pathlib
from typing import NamedTuple
from experiments.plot_cross_agent_cls import Experiment
from slurm_runner_cross_agent import run_single_from_sweep_slurm


PATH = "results/jeanzay/results/sweeps/shared_ref_mnist/2021-04-20/14-58-18"


class Config(BaseConfig):
    pass


class RunnerCfg(NamedTuple):
    jobname: str = "job"
    gpu_or_cpu: str = "gpu"
    gb: int = 16  # 16 or 32
    nnodes: int = 1
    ntasks: int = 1
    time: str = "20:00:00"
    cpus_per_task: int = 2
    nprocs: int = 5


exp_paths = pathlib.Path(PATH).glob("*")
cfg = Config(nogpu=False, nprocs=5)
runner_cfg = RunnerCfg()

for path in exp_paths:
    for rank in range(runner_cfg.nprocs):
        run_single_from_sweep_slurm(
            cfg, runner_cfg, path, rank, str(path) + f"-rank_{rank}"
        )
