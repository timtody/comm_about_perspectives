from functions import import_experiment_class
import os
import argparse
from reader.chunked_writer import TidyReader

parser = argparse.ArgumentParser()
parser.add_argument("exp")
parser.add_argument("path")
args = parser.parse_args()

exp_filename = args.exp.split("/")[-1].split(".")[0]
Experiment = import_experiment_class(exp_filename)

reader_path = os.path.join(args.path, "data")
plot_path = os.path.join(args.path, "plotting/plots")
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
reader = TidyReader(os.path.join(args.path, "data"))
dataframes = Experiment.load_data(reader)
Experiment.plot(dataframes, plot_path)
