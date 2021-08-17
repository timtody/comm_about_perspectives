from experiments.experiment import BaseExperiment, BaseConfig
from reader.chunked_writer import TidyReader
from typing import Any

from torchvision.datasets import CIFAR10

class Config(BaseConfig):
    pass

class Experiment(BaseExperiment):
    def run(self, cfg):
        train_set = CIFAR10("data", train=True, download=True)
        test_set = CIFAR10("data", train=False, download=True)
        print("Success!")
        print(train_set)
        print(test_set)

    def plot(self):
        pass

    def load_data(reader: TidyReader) -> Any:
        pass

