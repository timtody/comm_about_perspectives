import argparse
import importlib
import os
import sys

from functions import create_exp_name_and_datetime_path, merge_cfg_with_cli, run

from human_id import generate_id

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    exp = sys.argv[1].split("/")[-1].split(".")[0]

    module = importlib.import_module("." + exp, package="experiments")
    Experiment = getattr(module, "Experiment")
    cfg = getattr(module, "Config")
    parser = merge_cfg_with_cli(cfg)
    parser.add_argument("exp")
    args = parser.parse_args()
    path = create_exp_name_and_datetime_path(Experiment)
    path = os.path.join("results", path)
    args.run_id = generate_id()
    run(Experiment, args, path)
