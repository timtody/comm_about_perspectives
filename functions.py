import argparse
from experiments.shared_ref_mnist import Experiment
from experiments.experiment import BaseExperiment
import multiprocessing as mp
import os
import time
from argparse import ArgumentParser, Namespace
from multiprocessing import Barrier
from multiprocessing.context import Process
from typing import Callable, Dict, List, NamedTuple

import copy

import c_types
from chunked_writer import MultiProcessingWriter, TidyReader


def merge_cfg_with_cli(cfg: NamedTuple, parser: ArgumentParser = None):
    if parser is None:
        parser = argparse.ArgumentParser()
    for field in cfg._fields:
        if isinstance(cfg.__getattribute__(field), bool):
            parser.add_argument(f"--{field}", action="store_true")
        else:
            parser.add_argument(
                f"--{field}",
                type=type(cfg.__getattribute__(field)),
                default=cfg.__getattribute__(field),
            )
    return parser


def create_exp_name_and_datetime_path(experiment):
    exp_path = os.path.join(experiment.__module__.split(".")[1])
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
            target=fn,
            args=(experiment, copy.deepcopy(cfg), rank, writer, reader, path, barrier),
        )
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()


def start_procs_without_join(
    cfg: Dict = {},
    experiment: c_types.BaseExperiment = None,
    path: str = "",
    barrier: Barrier = None,
) -> List[Process]:
    processes = []
    data_path = os.path.join(path, "data")
    os.makedirs(data_path)
    for rank in range(cfg.nprocs):
        writer = c_types.MultiProcessingWriter(data_path, rank=rank)
        reader = c_types.TidyReader(data_path)
        proc = mp.Process(
            target=start_exp,
            args=(experiment, copy.deepcopy(cfg), rank, writer, reader, path, barrier),
        )
        # proc.start()
        processes.append(proc)
    return processes


def start_proc(
    experiment: BaseExperiment, cfg: Namespace, path: str, rank: int, barrier: Barrier
) -> None:
    data_path = os.path.join(path, "data")
    os.makedirs(data_path)
    writer = c_types.MultiProcessingWriter(data_path, rank=rank)
    reader = c_types.TidyReader(data_path)
    exp = experiment(cfg, rank, writer, reader, path, barrier)
    exp.start()


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


def run(experiment: c_types.BaseExperiment, cfg: Namespace, path: str):
    os.makedirs(path)
    barrier = Barrier(cfg.nprocs)
    start_procs(start_exp, cfg, experiment, path, barrier)


def run_single_from_sweep_mp(
    experiment: c_types.BaseExperiment, cfg: Namespace, path: str
) -> List[Process]:
    os.makedirs(path)
    barrier = Barrier(cfg.nprocs)
    return start_procs_without_join(cfg, experiment, path, barrier)