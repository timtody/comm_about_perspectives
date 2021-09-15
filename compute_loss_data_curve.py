import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import argparse
from cifar import CifarDataset
from autoencoder import _AutoEncoder
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing as mp
import glob
from functools import partial
from utils import stem_to_params, path_to_stem
from functions import create_timestap_path


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(input_dim, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, hidden_size)
        self.fc_3 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = f.elu(self.fc_1(x.flatten(start_dim=1)))
        x = f.elu(self.fc_2(x))
        x = f.elu(self.fc_3(x))
        return x


class CifarAutoEncoder(_AutoEncoder, nn.Module):
    def __init__(self, lr=0.001, name=None):
        super().__init__()
        self._encoder = cifar_encoder()
        self._decoder = cifar_decoder()
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.name = name


def cifar_encoder():
    return nn.Sequential(
        nn.Conv2d(1, 16, 4),
        nn.ELU(),
        nn.Conv2d(16, 16, 4),
    )


def cifar_decoder():
    return nn.Sequential(
        nn.ConvTranspose2d(16, 16, 4),
        nn.ELU(),
        nn.ConvTranspose2d(16, 1, 4),
    )


def create_run_folder(func):
    # TODO: make this compatible with JZ by allowing to write to $SCRATCH
    def wrapper(args):
        args.owd = os.getcwd()
        path = create_timestap_path()
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        print("Starting experiment at path:", path)
        with open("cfg.json", "w") as f:
            sorted_args = dict(sorted(vars(args).items(), key=lambda item: item[0]))
            json.dump(sorted_args, f, indent=2)
        return func(args)

    return wrapper


def plot_curves(df, metric):
    df = df[df.Metric == metric]
    print(df)
    sns.set_palette(sns.color_palette("Set1"))
    sns.lineplot(
        data=df,
        x="Size",
        y="Value",
        hue="Run",
        legend="brief",
        style="Run",
        markers=True,
        dashes=False,
    )
    if metric == "Loss":
        plt.yscale("log")
    plt.xscale("log")
    plt.ylabel(metric)
    plt.xlabel("Dataset size")
    plt.savefig(f"reprieve_curves_{metric}.pdf")
    plt.clf()


def map_params_to_name(params: dict):
    if params["eta_msa"] == "1.0":
        return "DTI"
    if params["eta_msa"] == "0.74":
        return "All"
    if params["eta_msa"] == "0.95":
        return "AE+MTM"
    if params["eta_ae"] == "1.0" and params["eta_lsa"] == "0.1":
        return "AE"
    if params["eta_lsa"] == "0.1":
        return "AE+MTM_pure"


def transform(x) -> torch.Tensor:
    """
    Converts to greyscale and float representation
    :param x: Tensor input
    :return: Tensor
    """
    return x.mean(dim=1, keepdim=True).float()


def train_fn(
    mlp: MLP, batch, opt, train_steps, dataset_size, dev, repr_fn=lambda x: x
) -> MLP:
    X, y = map(lambda x: x.to(dev), batch)
    dataset_blowup_factor = int(dataset_size * 0.8 / len(X))
    print("Increasing dataset by factor", dataset_blowup_factor)
    if dataset_blowup_factor > 1:
        X = torch.cat(dataset_blowup_factor * [X], dim=0)
        y = torch.cat(dataset_blowup_factor * [y], dim=0)
    for _ in range(train_steps):
        indices = np.random.randint(len(X), size=1024)
        predictions = mlp(repr_fn(transform(X[indices])))
        error = f.cross_entropy(predictions, y[indices].long())
        opt.zero_grad()
        error.backward()
        opt.step()
    return mlp


def eval_fn(mlp, batch, dev, repr_fn=lambda x: x) -> tuple:
    X, y = map(lambda x: x.to(dev), batch)
    with torch.no_grad():
        prediction = mlp(repr_fn(transform(X)))
        loss = f.cross_entropy(prediction, y.long())
        accuracy = (prediction.argmax(dim=1) == y).float().mean()
    return loss.item(), accuracy.item()


def split(x: np.ndarray, train_size=0.8):
    split_index = int(len(x) * train_size)
    return torch.tensor(x[:split_index]).float(), torch.tensor(x[split_index:]).float()


def get_dev(rank, use_gpu=True, ngpus=1):
    return (
        torch.device(f"cuda:{rank % ngpus}")
        if torch.cuda.is_available() and use_gpu
        else torch.device("cpu")
    )


def load_encoder(path, rank, use_gpu, exp_step, agent="A"):
    dev = (
        torch.device("cuda")
        if torch.cuda.is_available() and use_gpu
        else torch.device("cpu")
    )
    cifar_ae = CifarAutoEncoder()
    cifar_ae.load_state_dict(
        torch.load(
            os.path.join(path, "params", f"step_{exp_step}", f"rank_{rank}", f"{agent}.pt"),
            map_location=dev,
        )
    )
    cifar_ae.to(dev)
    return cifar_ae


def compute_curve(
    X: np.ndarray,
    y: np.ndarray,
    sizes: np.ndarray,
    train_steps: int,
    path: str,
    use_gpu: bool,
    exp_step: int,
    rank: int,
    queue: mp.Queue,
) -> None:
    np.random.seed(123 + rank)

    data = []
    dev = get_dev(rank)
    for agent in ["A", "B", "C"]:
        ae = load_encoder(path, rank, use_gpu, exp_step, agent)
        for size in sizes:
            print("Rank", rank, "working on size", size, "and agent", agent, ".")
            indices = np.random.randint(len(y), size=int(size))
            X_sub, y_sub = X[indices], y[indices]
            X_train, X_test = split(X_sub)
            y_train, y_test = split(y_sub)
            latent_size = reduce(
                lambda a, b: a * b, ae.encode(transform(X_train).to(dev)).size()[1:]
            )
            print(latent_size)
            exit(1)
            mlp = MLP(latent_size, 10).to(dev)
            opt = optim.Adam(mlp.parameters())
            mlp = train_fn(
                mlp, (X_train, y_train), opt, train_steps, sizes[-1], dev, repr_fn=ae.encode
            )
            loss, acc = eval_fn(mlp, (X_test, y_test), dev, repr_fn=ae.encode)
            data.append((rank, size, agent, "Loss", loss))
            data.append((rank, size, agent, "Accuracy", acc))

    df = pd.DataFrame(data, columns=["Rank", "Size", "Agent", "Metric", "Value"])
    queue.put(df)


def gather_results(
    X: np.ndarray,
    y: np.ndarray,
    sizes: np.ndarray,
    train_steps: int,
    seeds: int,
    weights_path: str,
    use_gpu: bool,
    exp_step: int,
) -> pd.DataFrame:
    result_q = mp.Queue()
    processes = []
    for rank in range(seeds):
        p = mp.Process(
            target=partial(
                compute_curve, X, y, sizes, train_steps, weights_path, use_gpu, exp_step
            ),
            args=(rank, result_q),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    results = []
    while not result_q.empty():
        results.append(result_q.get())
    return pd.concat(results)


@create_run_folder
def main(args: argparse.Namespace):
    print("Cuda available:", torch.cuda.is_available())
    print("Working on", os.cpu_count(), "cores and", args.seeds, "seeds.")
    sizes = np.logspace(np.log10(args.min_size), np.log10(args.max_size), num=args.nsizes)
    dataset = CifarDataset(f"CIFAR{args.n_classes}", path=args.owd + "/data")
    X, y = dataset.eval.data.transpose([0, 3, 1, 2]) / 255.0, dataset.eval.targets
    results = []
    for path in glob.glob(args.weights_path + "/*"):
        params = stem_to_params(path_to_stem(path))
        df = gather_results(
            X, y, sizes, args.train_steps, args.seeds, path, args.use_gpu, args.exp_step
        )
        df["Run"] = map_params_to_name(params)
        results.append(df)
    df = pd.concat(results)
    df.to_csv("loss_acc_data.csv")
    df = df[df.Run != "All"]
    plot_curves(df, "Loss")
    plot_curves(df, "Accuracy")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_size", type=int, default=10)
    parser.add_argument("--max_size", type=int, default=10000)
    parser.add_argument("--nsizes", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train_steps", type=int, default=int(1e5))
    parser.add_argument("--n_classes", type=int, default=10, choices=(10, 100))
    parser.add_argument("--no_gpu", action="store_false", dest="use_gpu")
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--only_plot", action="store_true", dest="only_plot")
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--exp_step", type=int, default=49999)
    main(parser.parse_args())
