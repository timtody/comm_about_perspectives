import os
from pathlib import PosixPath
from chunked_writer import TidyReader
from typing import List


def stem_to_params(stem: str):
    """
    Helper function which takes in a path stem of the form "param:value-param2:value2..."
    and returns a dictionary of parameters, e.g. "{"param":value,...}
    Args:
        stem: String. The name of the folder.
    """
    params = {k: v for d in map(eval_d, stem.split("-")[:-1]) for k, v in d.items()}
    return params


def eval_d(string: str):
    """
    Helper function which parses a string of the form 'x:y' into
    a dictionary {"x": "y"}. This is used for inferring the hyperparameters
    from the folder names.
    Args:
        string: String. The input string to evaluate.
    """
    k, v = string.split(":")
    return {k: v}


def load_df_and_params(
    posixpath: PosixPath, tag: str, columns: List[str], datafolder="data"
):
    """
    Args:
        posixpath: Posixpath. Path to one specific epxeriment
        tag: String. The name of the metric which we want to retrieve
        columns: List[String]. The column headers of the resulting dataframe
    Returns:
        df: DataFrame
    """
    reader = TidyReader(os.path.join(posixpath, datafolder))
    df = reader.read(tag=tag, columns=columns)
    return df