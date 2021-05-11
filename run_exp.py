import argparse

from numpy import exp
from chunked_writer import MultiProcessingWriter
from compute_cross_cls import Config
from experiments.plot_cross_agent_cls import Experiment

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True)
parser.add_argument("--rank", required=True)
args = parser.parse_args()

cfg = Config()
writer = MultiProcessingWriter(args.path, rank=args.rank)
experiment = Experiment(cfg, args.rank, writer, None, args.path)
experiment.run(cfg)
experiment.writer.close()