import os
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


def plot_curves(results, ns, path):
    sns.set_palette(sns.color_palette("Set1"))
    ax = sns.lineplot(
        data=results,
        x="Size",
        y="Value",
        # hue="Metric",
        legend="brief",
        # style="name",
        markers=True,
        dashes=False,
    )
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[1:], labels=labels[1:])
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Loss")
    plt.xlabel("Dataset size")
    plt.savefig(f"{path}_reprieve_curves.pdf")


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


def transform(x):
    dev = get_dev()
    return torch.tensor(x).mean(dim=1, keepdim=True).float().to(dev)


def train_fn(mlp: MLP, batch, opt, train_steps, dataset_size, repr_fn=lambda x: x) -> MLP:
    dev = get_dev()
    X, y = batch
    dataset_blowup_factor = int(dataset_size * 0.8 / len(X))
    print("Increasing dataset by factor", dataset_blowup_factor)
    if dataset_blowup_factor > 1:
        X = np.concatenate(dataset_blowup_factor * [X], axis=0)
        y = np.concatenate(dataset_blowup_factor * [y], axis=0)
    for _ in range(train_steps):
        indices = np.random.randint(len(X), size=1024)
        predictions = mlp(repr_fn(transform(X[indices])))
        error = f.cross_entropy(predictions, torch.tensor(y[indices]).to(dev))
        opt.zero_grad()
        error.backward()
        opt.step()
    return mlp


def eval_fn(mlp, batch, repr_fn=lambda x: x) -> tuple:
    dev = get_dev()
    X, y = batch
    with torch.no_grad():
        prediction = mlp(repr_fn(transform(X)))
        loss = f.cross_entropy(prediction, torch.tensor(y).to(dev))
        accuracy = (prediction.argmax(dim=1) == torch.tensor(y).to(dev)).float().mean()
    return loss.item(), accuracy.item()


def split(x: np.ndarray, train_size=0.8):
    split_index = int(len(x) * train_size)
    return x[:split_index], x[split_index:]


def get_dev(use_gpu=True):
    return (
        torch.device("cuda")
        if torch.cuda.is_available() and use_gpu
        else torch.device("cpu")
    )


def compute_curve(
    X: np.ndarray,
    y: np.ndarray,
    sizes: np.ndarray,
    train_steps: int,
    path: str,
    use_gpu: bool,
    rank: int,
) -> pd.DataFrame:
    ae = load_encoder(path, rank, use_gpu)
    data = []
    dev = get_dev()
    for size in sizes:
        print("Rank", rank, "working on size", size)
        indices = np.random.randint(len(y), size=int(size))
        X_sub, y_sub = X[indices], y[indices]
        X_train, X_test = split(X_sub)
        y_train, y_test = split(y_sub)
        latent_size = reduce(lambda a, b: a * b, ae.encode(transform(X)).size()[1:])
        mlp = MLP(latent_size, 10).to(dev)
        opt = optim.Adam(mlp.parameters())
        mlp = train_fn(
            mlp, (X_train, y_train), opt, train_steps, sizes[-1], repr_fn=ae.encode
        )
        loss, acc = eval_fn(mlp, (X_test, y_test), repr_fn=ae.encode)
        data.append((rank, size, "Loss", loss))
        data.append((rank, size, "Accuracy", acc))

    df = pd.DataFrame(data, columns=["Rank", "Size", "Metric", "Value"])
    return df


def load_encoder(path, rank, use_gpu):
    dev = (
        torch.device("cuda")
        if torch.cuda.is_available() and use_gpu
        else torch.device("cpu")
    )
    cifar_ae = CifarAutoEncoder()
    cifar_ae.load_state_dict(
        torch.load(
            os.path.join(path, "params", "step_49999", f"rank_{rank}", "A.pt"),
            map_location=dev,
        )
    )
    cifar_ae.to(dev)
    return cifar_ae


def gather_results(
    X: np.ndarray,
    y: np.ndarray,
    sizes: np.ndarray,
    train_steps: int,
    seeds: int,
    weights_path: str,
    use_gpu: bool,
) -> pd.DataFrame:
    with mp.Pool(seeds) as pool:
        results = pool.map(
            partial(compute_curve, X, y, sizes, train_steps, weights_path, use_gpu),
            range(seeds),
        )
    return pd.concat(results)


def main(args: argparse.Namespace):
    print("Cuda available:", torch.cuda.is_available())
    print("Working on", os.cpu_count(), "cores.")
    sizes = np.logspace(np.log10(args.min_size), np.log10(args.max_size), num=args.steps)
    if not args.only_plot:
        dataset = CifarDataset(f"CIFAR{args.n_classes}")
        X, y = dataset.eval.data.transpose([0, 3, 1, 2]) / 255.0, dataset.eval.targets
        # results = []
        # for path in glob.glob(args.weights_path + "/*"):
        #     print(path)
        #     print("---")
        # exit(1)
        results = gather_results(
            X, y, sizes, args.train_steps, args.seeds, args.weights_path, args.use_gpu
        )
        results.to_csv("results/cifar_curves.csv")
    df = pd.read_csv("results/cifar_curves.csv")
    df = df[df.Metric == "Loss"]
    plot_curves(df, sizes, "results/testtest")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_size", type=int, default=10)
    parser.add_argument("--max_size", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--interpolation_steps", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train_steps", type=int, default=int(1e5))
    parser.add_argument("--n_classes", type=int, default=10, choices=(10, 100))
    parser.add_argument("--no_gpu", action="store_false", dest="use_gpu")
    parser.add_argument(
        "--metric", type=str, default="accuracy", choices=("accuracy", "loss")
    )
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--only_plot", action="store_true", dest="only_plot")
    main(parser.parse_args())
