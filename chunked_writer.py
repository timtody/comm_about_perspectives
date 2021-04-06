"""
Writer class for chunked writing of csv files. Data needs to be in tidy format.
"""
import csv
import os
import random
import string
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Callable, List, AnyStr

from pandas.core.frame import DataFrame
import c_types
import pandas as pd


class TidyWriter:
    def __init__(self, path: str = "", max_queue_len: int = 1000) -> None:
        """Writer class with is designed to handle tidy form csv's. Similar
        to the tensorflow summary writers, this writer has a queue and writes to
        disk when the queue is full. This allows to maintain a small memory footprint
        during experiments.

        Args:
            path (str, optional): Path to folder where data is written. Defaults to "".
            max_queue_len (int, optional): Max len of queue. When this is exceeded,
            writer writes to disk automaticaly. Defaults to 1000.
        """
        assert path != "", "Cannot initialise unnamed writer."
        self.path = path
        self.max_queue_len = max_queue_len
        self._init_queue()
        self.blob_name = self._gen_blob_name()

    def _init_queue(self):
        self.queue = defaultdict(list)

    def _gen_blob_name(self) -> AnyStr:
        """Generates a unique id for a fragment of data.

        Returns:
            str: The unique id.
        """
        return (
            "".join(random.choices(string.ascii_letters, k=10))
            + "."
            + datetime.now().strftime("%H%M%S")
        )

    def _write(self) -> None:
        """Writes the current queue to disk, resets the queue and
        generates a new unique blob id for naming the next fragment.
        """
        for tag in self.queue.keys():
            write_path = os.path.join(self.path, f"{self.blob_name}.{tag}.csv")
            with open(write_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.queue[tag])
        self._init_queue()
        self.blob_name = self._gen_blob_name()

    def add(self, data: tuple, tag: str = "default") -> None:
        """Adds data to the summary and then checks if the queue length
        is exceeded. If it is exceeded, calls the write function.

        Args:
            data (tuple): The data entry. *NEEDS* to be in tidy format!
            This means that data row contains all necessary information to identify the tuple.

            For example: data = (rank, step, loss, type, agent)

            Note how all the metadata can be inferred from the actual data tuple. This property
            allows the asynchronous and unstructured collection of data.
        """
        self.queue[tag].append(data)
        if len(self.queue) >= self.max_queue_len:
            self._write()

    def close(self) -> None:
        """Closes the writer by writing the remaining data to disk."""
        if len(self.queue) > 0:
            self._write()


class MultiProcessingWriter(TidyWriter):
    def __init__(self, path: str, max_queue_len: int = 100, rank: int = 0) -> None:
        super().__init__(path=path, max_queue_len=max_queue_len)
        self.rank = rank

    def add(self, data: tuple, tag: str = "default") -> None:
        data = (self.rank, *data)
        super().add(data, tag=tag)


class TidyReader:
    def __init__(self, path: str = "") -> None:
        """Tidy reader reads the data from a directory and is able to
        construct a dataframe from it.

        Args:
            path (str, optional): The location of the data. Defaults to "".
        """
        assert path != "", "Cannot init nameless reader."
        self.path = path

    def read(self, tag: str = "default", columns: List[str] = None) -> DataFrame:
        return self._read(self.path, tag, columns)

    def _read(self, path: str, tag: str, columns: List[str]) -> pd.DataFrame:
        """Reads the data from :path:. This method currently tries to sort
        the ist of posix paths by using the timestamp in the name. This is currently
        buggy but not required for functionality.

        Args:
            path (str): The data location.

        Returns:
            pd.DataFrame: The resulting dataframe.
        """
        csv_paths: List = list(Path(path).glob(f"*.{tag}.csv"))
        sort_key: Callable[[PosixPath], str] = lambda x: str(x).split(".")[0]
        sorted_paths: List = sorted(csv_paths, key=sort_key, reverse=True)

        dataframes: List[pd.DataFrame] = []
        for path in sorted_paths:
            df: pd.DataFrame = pd.read_csv(path, header=None, names=columns)
            dataframes.append(df)
        dataframe = pd.concat(dataframes, ignore_index=True)
        return dataframe