import os
import string
from typing import Any, List

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from autoencoder import AutoEncoder
from reader.chunked_writer import TidyReader
from mnist import MNISTDataset
from pandas.core.frame import DataFrame
from torch.utils.tensorboard import SummaryWriter

from experiments.experiment import BaseConfig, BaseExperiment


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


class Config(BaseConfig):
    path: str
    nsteps: int
    bsize: int


class Experiment(BaseExperiment):
    def run(self, cfg: Config):

        paths = []

        tb_path = f"{self.path}/tb/{self.rank}"
        self.tb = SummaryWriter(tb_path)

        self.dataset = MNISTDataset()
        agents = self.load_aes(
            os.path.join(
                self.path,
                "params",
                "step_39999",
            )
        )
        mlps = []
        for agent in agents:
            mlp = self.train_classifier(agent)
            mlps.append(mlp)
        self.compute_cross_agent_cls(agents, mlps)

    def load_aes(self, path: str) -> List[AutoEncoder]:
        autoencoders = [
            AutoEncoder(30, False, False, 0.001, name)
            for name in string.ascii_uppercase[:3]
        ]
        base1 = AutoEncoder(30, False, False, 0.001, "baseline1").to(self.dev)
        base2 = AutoEncoder(30, False, False, 0.001, "baseline2").to(self.dev)
        baselines = [base1, base2]

        for agent in autoencoders:
            agent.load_state_dict(
                torch.load(
                    f"{path}/rank_{int(self.rank) % 5}/{agent.name}.pt",
                    map_location=self.dev,
                )
            )
        for i, agent in enumerate(baselines):
            agent.load_state_dict(
                torch.load(
                    f"{path}/rank_{(int(self.rank) + i) % 5}/{agent.name}.pt",
                    map_location=self.dev,
                )
            )
        return autoencoders + baselines

    def train_classifier(self, agent: AutoEncoder):
        mlp = MLP(30).to(self.dev)
        agent.to(self.dev)

        for i in range(int(self.cfg.nsteps)):
            X, y = map(
                lambda x: x.to(self.dev),
                self.dataset.sample_with_label(int(self.cfg.bsize)),
            )
            latent = agent.encode(X)
            mlp.train(latent, y)
            acc = mlp.compute_acc(latent, y)
            self.tb.add_scalar("Accuracy-Post", acc, global_step=i)
            # self.writer.add((acc.item(), agent.name), step=i)
        return mlp

    def compute_cross_agent_cls(self, agents: List[AutoEncoder], mlps: List[MLP]):
        ma_aes, ma_mlps = agents[:3], mlps[:3]
        sa_aes, sa_mlps = agents[3:], mlps[3:]

        X, y = map(
            lambda x: x.to(self.dev),
            self.dataset.sample_with_label(int(self.cfg.bsize)),
        )
        self._compute_cross_acc(X, y, ma_aes, ma_mlps, "MA")
        self._compute_cross_acc(X, y, sa_aes, sa_mlps, "Base")

    def _compute_cross_acc(self, X, y, aes, mlps, tag, rot=1):
        for i, (ae, mlp) in enumerate(zip(aes, mlps[rot:] + mlps[:rot])):
            latent = ae.encode(X)
            acc = mlp.compute_acc(latent, y)
            self.writer.add((tag, acc), step=i, tag="cross_agent_accuracy_override")
            self.tb.add_scalar(f"cross_agent_acc_{tag}", acc)

    def load_data(reader: TidyReader) -> Any:
        return reader.read(columns=["Rank", "Step", "Agent", "Accuracy"])

    def plot(df: DataFrame, plot_path: str) -> None:
        df.to_csv(plot_path + "/data.csv")
        sns.barplot(data=df, x="Agent", y="Accuracy")
        plt_name = f"{plot_path}/cross_agent_pred_acc_latent"
        plt.savefig(plt_name + ".svg")
        plt.savefig(plt_name + ".pdf")
        plt.savefig(plt_name + ".png")
