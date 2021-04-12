from functions import merge_cfg_with_cli
from main import Experiment, Config
from slurm_runner import RunnerCfg
from chunked_writer import MultiProcessingWriter, TidyReader
from multiprocessing import Barrier
from argparse import ArgumentParser

cfg: Config = Config()
parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--rank", type=int, required=True)
merge_cfg_with_cli(cfg, parser)
args = parser.parse_args()

writer = MultiProcessingWriter(args.path, rank=args.rank)
reader = TidyReader(cfg.path)
exp = Experiment(cfg, args.rank, writer, reader, args.path)
exp.run()
