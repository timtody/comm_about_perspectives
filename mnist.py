import random
from collections import defaultdict

import numpy as np
import torch
import torchvision
from six.moves import urllib

opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)


class MNISTDataset:
    def __init__(self, test_train_split=0.9) -> None:
        dataset = torchvision.datasets.MNIST(
            "data/mnist",
            download=True,
        )
        self.dataset = dataset
        self.transform = lambda x: x / 255.0
        self._train, self._test = self.build_train_and_test_set(
            dataset, test_train_split
        )
        self._data_dict_train = self.build_dict_and_maybe_dataset(dataset)
        # self._data_dict_test = self.build_dict_and_maybe_dataset(dataset, train=False)

    def sample_with_label(self, bsize):
        indices = np.random.randint(len(self.images), size=bsize)
        images = self.transform(torch.stack([self.images[i] for i in indices])).reshape(
            -1, 1, 28, 28
        )
        labels = torch.tensor([self.labels[i] for i in indices])
        return images, labels

    def build_dict_and_maybe_dataset(self, dataset, train=True) -> dict:
        """Builds the dictionary of MNIST data and sorts all
        the images into sublists organised by image identity.
        For example, dict[2] contains all images of the digit "2".

        Args:
            dataset (torch.utils.data.Dataset): The mnist dataset
            from pytoch.

        """
        self.images = []
        self.labels = []

        dct = defaultdict(list)
        # TODO: CAREFUL! test dataset is absolutely WRONG because it gets zipped
        # with the training data (DUH!?)
        for (_, label), im in zip(dataset, self._train if train else self._test):
            dct[label].append(im.data)
            if train:
                self.images.append(im)
                self.labels.append(label)
        return dct

    @staticmethod
    def build_train_and_test_set(dataset, test_ratio=0.9) -> tuple:
        """Splits the dataset in train and val

        Args:
            dataset (torch.utils.data.Dataset): The dataset

        Returns:
            tuple: train set, test set
        """
        split_indices = int(len(dataset) * test_ratio)
        train_set = dataset.data[:split_indices]
        val_set = dataset.data[split_indices:]
        return train_set, val_set

    def sample_digit(self, digit: int, bsize: int = 100) -> torch.Tensor:
        """Samples a minibatch of data from the TRAINING set. Allows to
        sample only from a subset of images of a specific digit.

        Args:
            digit (int): The identity of the digit we want to sample.
            bsize (int, optional): Minibatch size. Defaults to 100.

        Returns:
            tuple: Batch of images randomly sampled from the subgroup
            specified by :digit:.
        """
        return self.transform(
            torch.stack(random.sample(self._data_dict_train[digit], k=bsize))
        ).reshape(-1, 1, 28, 28)

    def sample_all_digits_once(self) -> torch.Tensor:
        digit_tensors = []
        for i in range(10):
            digit_tensors.append(self.sample_digit(i, bsize=1))
        return digit_tensors

    def sample(self, bsize, eval=False) -> tuple:
        """Samples at random from the train set.

        Args:
            bsize (int): Size of minibatch to be sampled.

        Returns:
            torch.Tensor: The minibatch of MNIST images
        """
        set = self._train if not eval else self._test
        sampling_indices = np.random.randint(len(set), size=bsize)
        return self.transform(set.data[sampling_indices]).reshape(-1, 1, 28, 28)

    def sample_with_indices(self, indices):
        """Samples from the train set not at random but at indices given
        by :indices:.

        Args:
            indices (list-like): For each entry in :indices: we retrieve
            one sample from the train set.

        Returns:
            torch.Tensor: Minibatch of MNIST images
        """
        return self.transform(self._train.data[indices]).reshape(-1, 1, 28, 28)

    def length(self):
        """Returns the length of the training set

        Returns:
            int: The amount of elements in self._train
        """
        return len(self._train.data)
