from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size
from numpy.lib.type_check import imag
import scipy.io
import torch


class DigitClutter:
    def __init__(self, split=0.9) -> None:
        data = scipy.io.loadmat("data/light_debris/light_debris_with_debris.mat")
        # moveaxis transforms data to pytorch format (NxCxHxW)
        images = torch.tensor(data["images"], dtype=torch.float32).moveaxis(-1, 1)
        labels = torch.tensor(data["targets"])

        split_index = int(len(labels) * split)
        self.train = Subset(images[:split_index], labels[:split_index])
        self.eval = Subset(images[split_index:], labels[split_index:])

        # clutter data seems to be in the range 0 - 85
        self.transform = lambda x: x / 85.0

    def _get_indices(self, bsize: int, eval: bool = False) -> np.ndarray:
        if eval:
            return np.random.randint(len(self.eval), size=bsize) % len(self.eval)
        else:
            return np.random.randint(len(self.train), size=bsize) % len(self.train)

    def sample_with_label(self, bsize: int) -> torch.Tensor:
        sampling_indices = self._get_indices(bsize)
        images = self.train.images[sampling_indices]
        labels = self.train.labels[sampling_indices]
        return images, labels

    def sample_digit(self, digit: int, bsize: int = 100) -> torch.Tensor:
        # TODO: Implement
        raise NotImplementedError

    def sample_all_digits_once(self) -> torch.Tensor:
        # TODO: Implement
        raise NotImplementedError

    def sample(self, bsize: int, eval: bool = False) -> torch.Tensor:
        sampling_indices = self._get_indices(bsize, eval)
        if eval:
            return self.transform(self.eval.images[sampling_indices])
        else:
            return self.transform(self.train.images[sampling_indices])

    def sample_with_indices(self, indices: list) -> torch.Tensor:
        return self.train[indices]

    def length(self) -> int:
        return len(self.train)


class Subset:
    def __init__(self, images: list, labels: list) -> None:
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    dataset = DigitClutter()
    data = dataset.sample(3)
    print(data.size())