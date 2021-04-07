import json
import multiprocessing as mp
import os
from abc import ABC, abstractmethod, abstractstaticmethod
from multiprocessing import Barrier
from typing import Any, Dict, Literal, NamedTuple

import c_types
import matplotlib.pyplot as plt
import numpy as np
import torch
from chunked_writer import MultiProcessingWriter, TidyReader


def set_seeds(rank):
    torch.manual_seed(100 + rank)
    np.random.seed(100 + rank)


def get_device(gpu=False):
    return (
        torch.device("cuda")
        if gpu and torch.cuda.is_available()
        else torch.device("cpu")
    )


class BaseExperiment(ABC):
    def __init__(
        self,
        cfg: NamedTuple,
        rank: int,
        writer: MultiProcessingWriter,
        reader: TidyReader,
        path: str,
        barrier: Barrier,
        handin: Dict = {},
    ) -> None:
        self.cfg = cfg
        self.step: Literal = 0
        self.rank = rank
        self.path = path
        self._set_seeds()
        self.dev: torch.Device = get_device(self.cfg.gpu)
        self.writer = writer
        self.reader = reader
        self.handin = handin
        self.barrier = barrier
        if rank == 0:
            self._dump_cfg()

    def _dump_cfg(self):
        with open(os.path.join(self.path, "cfg.json"), "w") as f:
            json.dump(vars(self.cfg), f)

    def _run_experiment(self):
        self.pre_forward_hook()
        results = self.run()
        self.post_forward_hook()
        return results

    def pre_forward_hook(self):
        ...

    @abstractmethod
    def run(self):
        ...

    def log(self, step: int = 0):
        self.writer._write()
        self.barrier.wait()
        if self.rank == 0:
            self._plot(step)

    def _plot(self, step):
        data = self.load_data(self.reader)
        # TODO: let's not do that. Let's instead overwrite the plots and results
        # Since it makes no sense to store the old data multiple times?!
        log_path = os.path.join(self.path, "plots", f"step_{step}")
        cwd = os.getcwd()
        os.makedirs(log_path)
        os.chdir(log_path)
        plt.figure()
        self.plot(data, step)
        plt.close()
        os.chdir(cwd)

    @abstractstaticmethod
    def plot(*args) -> None:
        raise NotImplementedError

    @abstractstaticmethod
    def load_data(reader) -> Any:
        ...

    @staticmethod
    def define_data(manager) -> Dict:
        return manager.list()

    def _set_seeds(self) -> None:
        set_seeds(self.cfg.seed + self.rank)

    @classmethod
    def experiment_will_mount(cls, manager: mp.Manager):
        shared_data_types = cls.define_data(manager)
        return shared_data_types

    @classmethod
    def experiment_will_unmount(cls, data: c_types.DataFrame, cfg: Dict) -> None:
        with open("config.json", "w") as f:
            json.dump(cfg, f)
        dataframe = cls.data_transform(data)
        dataframe.to_csv("results.csv")
        cls.plot(dataframe)
