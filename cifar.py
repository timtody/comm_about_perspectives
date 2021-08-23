from typing import Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import CIFAR10, CIFAR100


class CifarMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = np.array(self.targets)
        self.data = np.array(self.data)
        self.dict = self._gen_dict(self.data, self.targets)

    def __getitem__(self, indices: np.ndarray) -> Tuple[Any, Any]:
        img, target = self.data[indices], self.targets[indices]
        return img, target

    @staticmethod
    def _gen_dict(images, labels):
        return {digit: images[(labels == digit)] for digit in range(10)}


class Cifar100Wrapper(CifarMixin, CIFAR100):
    pass


class Cifar10Wrapper(CifarMixin, CIFAR10):
    pass


class CifarDataset:
    def __init__(self, dataset="CIFAR10", colour=False) -> None:
        # CIFAR start here
        if dataset == "CIFAR10":
            self.train = Cifar10Wrapper("data", train=True, download=True)
            self.eval = Cifar10Wrapper("data", train=False, download=True)
        elif dataset == "CIFAR100":
            self.train = Cifar100Wrapper("data", train=True, download=True)
            self.eval = Cifar100Wrapper("data", train=False, download=True)
        # cifar data seems to be in the range 0 - 255
        self.transform = lambda x: torch.tensor(x / 255.0).float()
        self.colour = colour

    def _get_indices(self, bsize: int, eval: bool = False) -> np.ndarray:
        if eval:
            return np.random.randint(len(self.eval), size=bsize) % len(self.eval)
        else:
            return np.random.randint(len(self.train), size=bsize) % len(self.train)

    def sample_with_label(
        self, bsize: int, eval=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sampling_indices = self._get_indices(bsize, eval)
        if eval:
            images, labels = self.eval[sampling_indices]
        else:
            images, labels = self.train[sampling_indices]

        images = self.transform(images).permute([0, 3, 1, 2]).mean(dim=1, keepdim=True)
        return (
            images,
            torch.tensor(labels),
        )

    def sample_digit(self, digit: int, bsize: int = 100, eval=False) -> torch.Tensor:
        subset_len = len(self.train.dict[digit])

        indices = np.random.randint(subset_len, size=bsize) % subset_len
        return self.transform(self.train.dict[digit][indices]).permute([0, 3, 1, 2]).\
            mean(dim=1, keepdim=True).float()

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


if __name__ == "__main__":
    dataset = CifarDataset(dataset="CIFAR100")
    dig = dataset.sample_digit(1, 2)

    ims, labels = dataset.sample_with_label(10)
    print(ims.shape)
    for im, label in zip(ims, labels):
        print(im, label, im.size)
        plt.imshow(im.reshape(32, 32))
        print(label)
        plt.show()
