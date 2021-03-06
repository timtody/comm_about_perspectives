import json
import multiprocessing as mp
import os
from abc import ABC, abstractmethod, abstractstaticmethod
from multiprocessing import Barrier
from typing import Any, Dict, Literal, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from reader.chunked_writer import MultiProcessingWriter, TidyReader
from pandas import DataFrame


def set_seeds(rank):
    torch.manual_seed(100 + rank)
    np.random.seed(100 + rank)


def get_device(gpu=False, rank: int = 0, ngpus: int = 1):
    return (
        torch.device(f"cuda:{int(rank) % int(ngpus)}")
        if gpu and torch.cuda.is_available()
        else torch.device("cpu")
    )


class BaseConfig(NamedTuple):
    nogpu: bool
    nprocs: int
    seed: int = 123
    ngpus: int = 1


class BaseExperiment(ABC):
    def __init__(
        self,
        cfg: NamedTuple,
        rank: int,
        writer: MultiProcessingWriter,
        reader: TidyReader,
        path: str,
        barrier: Barrier = None,
        handin: Dict = {},
    ) -> None:
        self.cfg = cfg
        self.step: Literal = 0
        self.rank = rank
        self.path = path
        self._set_seeds()
        self.dev: torch.Device = get_device(not self.cfg.nogpu, rank, cfg.ngpus)
        self.writer = writer
        self.reader = reader
        self.handin = handin
        self.barrier = barrier
        if rank == 0:
            self._dump_cfg(cfg)

    def _dump_cfg(self, cfg):
        with open(os.path.join(self.path, "cfg.json"), "w") as f:
            try:
                json.dump(vars(cfg), f)
            except:
                json.dump(cfg._asdict(), f)

    def _run_experiment(self):
        self.pre_forward_hook()
        results = self.run()
        self.post_forward_hook()
        return results

    def pre_forward_hook(self):
        ...

    def _run(self, cfg):
        self.run(cfg)
        self.writer._write()

    @abstractmethod
    def run(self, cfg):
        ...

    def log(self, step: int = 0):
        self.writer._write()
        # self.barrier.wait()
        if self.rank == 0:
            self._plot(step)

    def _plot(self, step):
        data = self.load_data(self.reader)
        log_path = os.path.join(self.path, "plots")
        cwd = os.getcwd()
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        os.chdir(log_path)
        plt.figure()
        self.plot(data, step)
        plt.close()
        os.chdir(cwd)

    @abstractstaticmethod
    def plot(dataframes, plot_path) -> None:
        raise NotImplementedError

    @abstractstaticmethod
    def load_data(reader: TidyReader) -> Any:
        ...

    @staticmethod
    def define_data(manager) -> Dict:
        return manager.list()

    def _set_seeds(self) -> None:
        set_seeds(int(self.cfg.seed) + int(self.rank))

    @classmethod
    def experiment_will_mount(cls, manager: mp.Manager):
        shared_data_types = cls.define_data(manager)
        return shared_data_types

    @classmethod
    def experiment_will_unmount(cls, data: DataFrame, cfg: Dict) -> None:
        with open("config.json", "w") as f:
            json.dump(cfg, f)
        dataframe = cls.data_transform(data)
        dataframe.to_csv("results.csv")
        cls.plot(dataframe)
