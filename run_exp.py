import argparse

from chunked_writer import MultiProcessingWriter
from compute_cross_cls import Config
from experiments.plot_cross_agent_cls import Experiment
from functions import merge_cfg_with_cli

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True)
parser.add_argument("--rank", required=True)
merge_cfg_with_cli(Config, parser)
args = parser.parse_args()

cfg = Config()
writer = MultiProcessingWriter(args.path, rank=args.rank)
setattr(cfg, "path", args.path)
experiment = Experiment(cfg, args.rank, writer, None, args.path)
experiment.run(cfg)
experiment.writer.close()