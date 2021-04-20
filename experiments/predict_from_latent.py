import random
import string
from typing import Any, List, NamedTuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from autoencoder import AutoEncoder
from mnist import MNISTDataset

from experiments.experiment import BaseExperiment

sns.set(style="whitegrid")


class Config(NamedTuple):
    nprocs: int = 3
    seed: int = 123
    nogpu: bool = False
    ngpus: int = 1


class Experiment(BaseExperiment):
    def run(self, cfg: Config):
        path = (
            f"results/jeanzay/results/sweeps/shared_ref_mnist/2021-04-16/13-15-58/"
            f"sigma:0-eta_lsa:0-eta_msa:1-eta_dsa:0-eta_ae:0-/params/step_49999/rank_{self.rank % 3}"
        )
        dataset = MNISTDataset()
        all_agents: List[AutoEncoder] = self._load_aes(path)

        mlps: List[MLP] = [MLP(30) for _ in all_agents]
        bsize = 512
        nsteps = 5000

        for mlp, agent in zip(mlps, all_agents):
            for i in range(nsteps):
                ims, targets = dataset.sample_with_label(bsize)
                encoding = agent.encode(ims)
                mlp.train(encoding, targets)
                acc = mlp.compute_acc(encoding, targets)
                self.writer.add(
                    (
                        self.rank,
                        i,
                        "MA" if agent.name != "baseline" else "Baseline",
                        acc,
                    ),
                    step=i,
                )

    def _load_aes(self, path):
        autoencoders = [
            AutoEncoder(30, bnorm=False, affine=False, name=name, lr=0.001)
            for name in string.ascii_uppercase[:3]
        ]
        baseline = AutoEncoder(30, bnorm=False, affine=False, name="baseline", lr=0.001)

        all_agents: List[AutoEncoder] = autoencoders + [baseline]
        [
            agent.load_state_dict(
                torch.load(f"{path}/{agent.name}.pt", map_location=self.dev)
            )
            for agent in all_agents
        ]
        return all_agents

    @staticmethod
    def plot(df, plot_path) -> None:
        df.loc[df.Agent == "MARL", "Agent"] = "MA"
        df.to_csv(plot_path + "/results.csv")
        sns.lineplot(
            data=df,
            x="Step",
            y="Accuracy",
            style="Agent",
            dashes=False,
            markers=True,
            hue="Agent",
        )
        plt.savefig(plot_path + "/accuracy.pdf")
        plt.savefig(plot_path + "/accuracy.svg")

    @staticmethod
    def load_data(reader) -> Any:
        df = reader.read(columns=["Rank", "Step", "Agent", "Accuracy"])
        df = df.groupby(["Agent"], as_index=False).apply(lambda x: x[::125])
        return df


class MLP(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.l = nn.Linear(input_size, 10)
        self.opt = optim.Adam(self.parameters())

    def forward(self, x):
        return self.l(x)

    def compute_acc(self, ims, labels):
        pred = self.l(ims).argmax(dim=1)
        acc = (pred == labels).float().mean()
        return acc.item()

    def train(self, inputs, targets):
        x = self.l(inputs)
        loss = F.cross_entropy(x, targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
