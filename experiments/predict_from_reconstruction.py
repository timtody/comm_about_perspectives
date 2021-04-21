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
            f"results/"
            f"sigma:0.936-eta_lsa:0.003-eta_msa:0.525-eta_dsa:0.707-eta_ae:0.631-/params/step_49999/rank_{self.rank % 3}"
        )
        dataset = MNISTDataset()
        all_agents: List[AutoEncoder] = self._load_aes(path)

        cnns: List[CNN] = [CNN() for _ in all_agents]
        bsize = 512
        nsteps = 1000

        for cnn, agent in zip(cnns, all_agents):
            for i in range(nsteps):
                ims, targets = dataset.sample_with_label(bsize)
                reconstruction = agent(ims)
                cnn.train(reconstruction, targets)
                acc = cnn.compute_acc(reconstruction, targets)
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
        df = reader.read(
            columns=["Step", "Rank", "Rank_", "Step_", "Agent", "Accuracy"]
        )
        df = df.groupby(["Agent"], as_index=False).apply(lambda x: x[::50])
        return df


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)
        self.l1 = nn.Linear(160, 10)
        self.opt = optim.Adam(self.parameters())

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = self.l1(xb.reshape(-1, 160))
        return xb

    def compute_acc(self, ims, labels):
        pred = self(ims).argmax(dim=1)
        acc = (pred == labels).float().mean()
        return acc.item()

    def train(self, inputs, targets):
        x = self(inputs)
        loss = F.cross_entropy(x, targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
