import multiprocessing as mp
import os
import time
from multiprocessing import Barrier
from typing import Callable, Dict, NamedTuple

import c_types
from chunked_writer import MultiProcessingWriter, TidyReader


def create_exp_path(experiment):
    exp_path = os.path.join("results", experiment.__module__.split(".")[1])
    ymd_path = os.path.join(exp_path, time.strftime("%Y-%m-%d"))
    hms_path = os.path.join(ymd_path, time.strftime("%H-%M-%S"))
    return hms_path


def start_procs(
    fn: Callable,
    cfg: Dict = {},
    experiment: c_types.BaseExperiment = None,
    path: str = "",
    barrier: Barrier = None,
):
    processes = []
    for rank in range(cfg.nprocs):
        writer = c_types.MultiProcessingWriter(path, rank=rank)
        reader = c_types.TidyReader(path)
        proc = mp.Process(
            target=fn, args=(experiment, cfg, rank, writer, reader, path, barrier)
        )
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()


def start_exp(
    experiment: c_types.BaseExperiment,
    cfg: Dict,
    rank: int,
    writer: MultiProcessingWriter,
    reader: TidyReader,
    path: str,
    barrier: Barrier,
):
    exp = experiment(cfg, rank, writer, reader, path, barrier)
    exp.run()


def run(experiment: c_types.BaseExperiment, cfg: NamedTuple):
    path: str = create_exp_path(experiment)
    os.makedirs(path)
    barrier = Barrier(cfg.nprocs)
    start_procs(start_exp, cfg, experiment, path, barrier)
