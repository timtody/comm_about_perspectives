import os
from functions import merge_cfg_with_cli
from sweep_robustness import Experiment, Config
from reader.chunked_writer import MultiProcessingWriter, TidyReader
from argparse import ArgumentParser

cfg: Config = Config()
parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--rank", type=int, required=True)
merge_cfg_with_cli(cfg, parser)
args = parser.parse_args()
data_path = os.path.join(args.path, "data")
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
writer = MultiProcessingWriter(data_path, rank=args.rank)
reader = TidyReader(data_path)
exp = Experiment(args, args.rank, writer, reader, args.path)
exp.run(args)
