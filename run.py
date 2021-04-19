import os
import argparse
import importlib
from functions import create_exp_name_and_datetime_path, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp")
    args = parser.parse_args()
    exp = args.exp.split("/")[-1].split(".")[0]

    module = importlib.import_module("." + exp, package="experiments")
    Experiment = getattr(module, "Experiment")
    cfg = getattr(module, "Config")()
    path = create_exp_name_and_datetime_path(Experiment)
    path = os.path.join("results", path)
    run(Experiment, cfg, path)