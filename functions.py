import argparse
import multiprocessing as mp
import os
import time
from multiprocessing import Barrier
from typing import Callable, Dict, List, NamedTuple
from argparse import Namespace

import c_types
from chunked_writer import MultiProcessingWriter, TidyReader


def parse_args(cfg: NamedTuple):
    parser = argparse.ArgumentParser()
    for field in cfg._fields:
        if isinstance(cfg.__getattribute__(field), bool):
            parser.add_argument(
                f"--{field}", type=eval, default=cfg.__getattribute__(field)
            )
        else:
            parser.add_argument(
                f"--{field}",
                type=type(cfg.__getattribute__(field)),
                default=cfg.__getattribute__(field),
            )
    args = parser.parse_args()
    return args


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
    data_path = os.path.join(path, "data")
    os.makedirs(data_path)
    for rank in range(cfg.nprocs):
        writer = c_types.MultiProcessingWriter(data_path, rank=rank)
        reader = c_types.TidyReader(data_path)
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
    exp.run(cfg)


def run(experiment: c_types.BaseExperiment, cfg: Namespace):
    path: str = create_exp_path(experiment)
    os.makedirs(path)
    barrier = Barrier(cfg.nprocs)
    start_procs(start_exp, cfg, experiment, path, barrier)


def run_sweep(experiment: c_types.BaseExperiment, cfgs: List[Namespace]):
    path: str = create_exp_path(experiment)
    os.makedirs(path)
    barrier = Barrier(cfgs[0].nprocs)

    # TODO: maybe just not plot?! or alternatively pass the plotting functions to constructor?
    # also: finish this correctly
    for cfg in cfgs:
        start_procs(start_exp, cfg, experiment, path, barrier)