import argparse
import copy
import importlib
import multiprocessing as mp
import os
import time
from argparse import ArgumentParser, Namespace
from multiprocessing import Barrier
from multiprocessing.context import Process
from typing import Callable, Dict, List, NamedTuple, get_type_hints

import c_types
from reader.chunked_writer import MultiProcessingWriter, TidyReader
from experiments.experiment import BaseExperiment


def import_experiment_class(filename: str):
    module = importlib.import_module("." + filename, package="experiments")
    Experiment = getattr(module, "Experiment")
    return Experiment


def import_config_class(filename: str):
    module = importlib.import_module("." + filename, package="experiments")
    Config = getattr(module, "Config")
    return Config


def merge_cfg_with_cli(cfg: NamedTuple, parser: ArgumentParser = None):
    parser = parser if parser is not None else argparse.ArgumentParser()
    for field, dtype in get_type_hints(cfg).items():
        if dtype is bool:
            parser.add_argument(f"--{field}", action="store_true")
        else:
            if field in cfg._field_defaults.keys():
                parser.add_argument(
                    f"--{field}",
                    type=dtype,
                    default=cfg._field_defaults[field],
                )
            else:
                parser.add_argument(
                    f"--{field}",
                    type=dtype,
                    required=True,
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
    experiment: BaseExperiment = None,
    path: str = "",
    barrier: Barrier = None,
):
    processes = []
    data_path = os.path.join(path, "data")
    os.makedirs(data_path)
    for rank in range(cfg.nprocs):
        writer = MultiProcessingWriter(data_path, rank=rank)
        reader = TidyReader(data_path)
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
    experiment: BaseExperiment = None,
    path: str = "",
    barrier: Barrier = None,
) -> List[Process]:
    processes = []
    data_path = os.path.join(path, "data")
    os.makedirs(data_path)
    for rank in range(cfg.nprocs):
        writer = MultiProcessingWriter(data_path, rank=rank)
        reader = TidyReader(data_path)
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
    writer = MultiProcessingWriter(data_path, rank=rank)
    reader = TidyReader(data_path)
    exp = experiment(cfg, rank, writer, reader, path, barrier)
    exp.start()


def start_exp(
    experiment: BaseExperiment,
    cfg: Dict,
    rank: int,
    writer: MultiProcessingWriter,
    reader: TidyReader,
    path: str,
    barrier: Barrier,
):
    exp = experiment(cfg, rank, writer, reader, path, barrier)
    exp._run(cfg)


def run(experiment: BaseExperiment, cfg: Namespace, path: str):
    os.makedirs(path)
    start_procs(start_exp, cfg, experiment, path)


def run_single_from_sweep_mp(
    experiment: BaseExperiment, cfg: Namespace, path: str
) -> List[Process]:
    os.makedirs(path)
    barrier = Barrier(cfg.nprocs)
    return start_procs_without_join(cfg, experiment, path, barrier)
