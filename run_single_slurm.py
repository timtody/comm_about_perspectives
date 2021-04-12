import os
from functions import merge_cfg_with_cli
from main import Experiment, Config
from chunked_writer import MultiProcessingWriter, TidyReader
from argparse import ArgumentParser

cfg: Config = Config()
parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--rank", type=int, required=True)
merge_cfg_with_cli(cfg, parser)
args = parser.parse_args()
if not os.path.exists(args.path):
    os.makedirs(args.path)
writer = MultiProcessingWriter(args.path, rank=args.rank)
reader = TidyReader(args.path)
exp = Experiment(args, args.rank, writer, reader, args.path)
exp.run(args)
