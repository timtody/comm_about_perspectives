import itertools
import os
import random
from typing import AnyStr, List, NamedTuple, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from autoencoder import AutoEncoder
from c_types import DataFrame, TidyReader
from mnist import MNISTDataset
from torch.tensor import Tensor
from torch.utils.tensorboard import SummaryWriter

from experiments.experiment import BaseExperiment

# TODO: add the sweep parameters to the writers


class Experiment(BaseExperiment):
    @staticmethod
    def load_data(reader: TidyReader) -> Tuple[DataFrame]:
        df1 = filter_df(
            reader.read("loss", ["Rank", "Step", "Loss", "Type", "Agent_X", "Agent_Y"]),
            group_keys=["Rank", "Type", "Agent_X", "Agent_Y"],
            sort=True,
            datapoints=25,
        )
        df2 = filter_df(
            reader.read(
                "pred_from_latent",
                ["Rank", "Epoch", "Step", "Value", "Metric", "Type", "Agent"],
            ),
            group_keys=["Rank", "Epoch", "Type", "Agent", "Metric"],
            sort=True,
            datapoints=25,
        )
        return (df1, df2)

    @staticmethod
    def plot(dataframes: Tuple[DataFrame], step: int) -> None:
        pass
        # (df_losses, df_prediction) = dataframes

        # df_losses.to_csv("res_losses.csv")
        # plot_lineplot(df_losses, "AE")
        # plot_relplot(df_losses, "LSA")
        # plot_relplot(df_losses, "LSA-MBVAR")
        # plot_relplot(df_losses, "DSA")
        # plot_relplot(df_losses, "MSA")

        # df_prediction.to_csv("res_prediction.csv")
        # plot_prediction_errors(df_prediction, step)

    def run(self, cfg: NamedTuple):
        self.dataset = MNISTDataset()

        tb_path = f"{self.path}/tb/{self.rank}"
        self.tb = SummaryWriter(tb_path)

        base = self.generate_autoencoder("baseline")
        # TODO: change ABCDEFG to something more general, this would fail with more than 7 agents
        agents = [self.generate_autoencoder(f"{i}") for i in "ABCDEFG"[: cfg.nagents]]
        agents_and_base = [base] + agents

        agent_index_pairs = list(itertools.combinations(range(len(agents)), r=2))
        for i in range(cfg.nsteps):
            for agent_index_pair in agent_index_pairs:
                # maybe shuffle order
                shuffle = random.choice([0, 1])
                agent_indices = (
                    reversed(agent_index_pair) if shuffle else agent_index_pair
                )
                agent_a, agent_b = [agents[i] for i in (agent_indices)]

                # execute marl-ae step
                self.sync_ae_step(i, agent_a, agent_b)
                if agent_a.name == "A" or agent_b.name == "A":
                    self.control_step(i, base)

            if i % cfg.logfreq == cfg.logfreq - 1:
                self.predict_from_latent_and_reconstruction(agents_and_base, i)
                self.save_params(i, agents_and_base)
        self.writer.close()

    def save_params(self, step: int, agents: List[AutoEncoder]):
        path = os.path.join(self.path, "params", f"step_{step}", f"rank_{self.rank}")
        os.makedirs(path)
        for agent in agents:
            torch.save(agent.state_dict(), os.path.join(path, f"{agent.name}.pt"))

    def generate_autoencoder(self, name: AnyStr):
        return AutoEncoder(
            self.cfg.latent_dim, self.cfg.bnorm, self.cfg.affine, self.cfg.lr, name
        ).to(self.dev)

    def sync_ae_step(self, step: int, agent_a: AutoEncoder, agent_b: AutoEncoder):
        digit = random.choice(range(10))
        batch_a = self.dataset.sample_digit(digit, bsize=self.cfg.bsize).to(self.dev)
        batch_b = self.dataset.sample_digit(digit, bsize=self.cfg.bsize).to(self.dev)

        # compute message and reconstructions
        msg_a = agent_a.encode(batch_a)
        msg_b = agent_b.encode(batch_b)

        # add channel noise to messages
        msg_a += torch.randn_like(msg_a) * self.cfg.sigma
        msg_b += torch.randn_like(msg_b) * self.cfg.sigma

        ## rec_aa is a's reconstruction of a's message
        ## rec_ab is a's reconstruction of b's message and so forth..
        rec_aa = agent_a.decode(msg_a)
        rec_bb = agent_b.decode(msg_b)
        rec_ab = agent_a.decode(msg_b)
        rec_ba = agent_b.decode(msg_a)

        # autoencoding
        ae_loss_a = F.mse_loss(rec_aa, batch_a)
        ae_loss_b = F.mse_loss(rec_bb, batch_b)

        # latent space adaptation
        lsa_loss_a = F.mse_loss(msg_a, msg_b.detach())
        lsa_loss_b = F.mse_loss(msg_b, msg_a.detach())

        # message space adaptation
        msa_loss_a = F.mse_loss(rec_ab, batch_a)
        msa_loss_b = F.mse_loss(rec_ba, batch_b)

        # decoding space adaptation
        # NOTE: we might NOT want to detach here, discuss with Clem
        dsa_loss_a = F.mse_loss(rec_ab, rec_aa.detach())
        dsa_loss_b = F.mse_loss(rec_ba, rec_bb.detach())

        with torch.no_grad():
            # compute mean batch var per feature
            # TODO: probably compute running average here.
            mbvar_a = msg_a.var(dim=0).mean()
            mbvar_b = msg_b.var(dim=0).mean()

        ab_name = agent_a.name + agent_b.name
        ba_name = agent_b.name + agent_a.name

        self.tb.add_scalar(f"AEloss{ab_name}", ae_loss_a, step)
        self.tb.add_scalar(f"AEloss{ba_name}", ae_loss_b, step)

        self.tb.add_scalar(f"mbvar{ab_name}", mbvar_a, step)
        self.tb.add_scalar(f"mbvar{ba_name}", mbvar_b, step)

        self.tb.add_scalar(f"lsa{ab_name}", lsa_loss_a, step)
        self.tb.add_scalar(f"lsa{ba_name}", lsa_loss_b, step)

        self.tb.add_scalar(f"lsa-mbvar{ab_name}", lsa_loss_a / mbvar_a, step)
        self.tb.add_scalar(f"lsa-mbvar{ba_name}", lsa_loss_b / mbvar_b, step)

        total_loss_a: Tensor = (
            self.cfg.eta_ae * ae_loss_a
            + self.cfg.eta_lsa * lsa_loss_a
            + self.cfg.eta_msa * msa_loss_a
            + self.cfg.eta_dsa * dsa_loss_a
        )
        total_loss_b: Tensor = (
            self.cfg.eta_ae * ae_loss_b
            + self.cfg.eta_lsa * lsa_loss_b
            + self.cfg.eta_msa * msa_loss_b
            + self.cfg.eta_dsa * dsa_loss_b
        )

        agent_a.opt.zero_grad()
        agent_b.opt.zero_grad()
        total_loss_a.backward(retain_graph=True)
        total_loss_b.backward()
        agent_a.opt.step()
        agent_b.opt.step()

        # TODO: add step as kwarg to add and add_multiple
        self.writer.add_multiple(
            [
                (step, ae_loss_a.item(), "AE", agent_a.name, agent_b.name),
                (step, lsa_loss_a.item(), "LSA", agent_a.name, agent_b.name),
                (step, msa_loss_a.item(), "MSA", agent_a.name, agent_b.name),
                (step, dsa_loss_a.item(), "DSA", agent_a.name, agent_b.name),
                (step, mbvar_a.item(), "MBVAR", agent_a.name, agent_b.name),
                (
                    step,
                    lsa_loss_a.item() / mbvar_a.item(),
                    "LSA-MBVAR",
                    agent_a.name,
                    agent_b.name,
                ),
                (step, ae_loss_b.item(), "AE", agent_b.name, agent_a.name),
                (step, lsa_loss_b.item(), "LSA", agent_b.name, agent_a.name),
                (step, msa_loss_b.item(), "MSA", agent_b.name, agent_a.name),
                (step, dsa_loss_b.item(), "DSA", agent_b.name, agent_a.name),
                (step, mbvar_b.item(), "MBVAR", agent_b.name, agent_a.name),
                (
                    step,
                    lsa_loss_b.item() / mbvar_b.item(),
                    "LSA-MBVAR",
                    agent_b.name,
                    agent_a.name,
                ),
            ],
            tag="loss",
        )

    def control_step(self, step: int, agent: AutoEncoder):
        digit = random.choice(range(10))
        batch = self.dataset.sample_digit(digit, bsize=self.cfg.bsize).to(self.dev)
        loss = F.mse_loss(agent(batch), batch)
        agent.opt.zero_grad()
        loss.backward()
        agent.opt.step()
        self.writer.add((step, loss.item(), "AE", agent.name, ""), tag="loss")

    def predict_from_latent_and_reconstruction(
        self, agents: List[AutoEncoder], step: int
    ) -> None:
        # TODO: Implement n-best metric
        for agent in agents:
            mlp: MLP = MLP(self.cfg.latent_dim).to(self.dev)
            mlp_rec: CNN = CNN().to(self.dev)
            for i in range(self.cfg.nsteps_pred_latent):
                ims, labels = map(
                    lambda x: x.to(self.dev),
                    self.dataset.sample_with_label(self.cfg.bsize_pred_latent),
                )
                latent = agent.encode(ims)
                reconstruction = agent(ims)

                loss_latent = mlp.train(latent, labels)
                acc_latent = mlp.compute_acc(latent, labels)

                loss_rec = mlp_rec.train(reconstruction, labels)
                acc_rec = mlp_rec.compute_acc(reconstruction, labels)

                self.tb.add_scalar(
                    f"acc_from_latent_{agent.name}_epoch_{step}", acc_latent, i
                )
                self.tb.add_scalar(
                    f"acc_from_rec_{agent.name}_epoch_{step}", acc_rec, i
                )

                self.writer.add_multiple(
                    [
                        (step, i, loss_latent, "Loss", "Latent", agent.name),
                        (step, i, acc_latent, "Accuracy", "Latent", agent.name),
                        (step, i, loss_rec, "Loss", "Reconstruction", agent.name),
                        (step, i, acc_rec, "Accuracy", "Reconstruction", agent.name),
                    ],
                    tag="pred_from_latent",
                )


def filter_df(
    df: DataFrame,
    group_keys: List[AnyStr],
    datapoints: int = 50,
    sort: bool = False,
    filter: bool = True,
):
    if sort:
        df = df.sort_values(by=["Step"])
    if filter:
        nsteps = len(list(df.groupby(group_keys).groups.values())[0])
        df = df.groupby(group_keys).apply(lambda x: x[:: nsteps // datapoints])
    return df


def plot_prediction_errors(df: DataFrame, step: int):
    sns.relplot(
        data=df[df["Epoch"] == step],
        x="Step",
        y="Value",
        col="Type",
        row="Metric",
        hue="Agent",
        style="Agent",
        markers=True,
        dashes=False,
        kind="line",
        facet_kws=dict(sharey=False, margin_titles=True),
    )
    plt.savefig("pred.pdf")
    plt.savefig("pred.svg")


def plot_relplot(df: DataFrame, type: str):
    sns.relplot(
        data=df[df["Type"] == type],
        x="Step",
        y="Loss",
        col="Agent_X",
        row="Agent_Y",
        style="Type",
        hue="Type",
        dashes=False,
        markers=True,
        kind="line",
        row_order=["A", "B", "C"],
        col_order=["A", "B", "C"],
        facet_kws=dict(margin_titles=True),
    )
    plt.savefig(f"catplot_{type}.pdf")
    plt.savefig(f"catplot_{type}.svg")


def plot_lineplot(df: DataFrame, type: str):
    sns.lineplot(
        data=df[df["Type"] == type],
        x="Step",
        y="Loss",
        style="Agent_X",
        hue="Agent_X",
        dashes=False,
        markers=True,
    )
    plt.savefig(f"lineplot_{type}.pdf")
    plt.savefig(f"lineplot_{type}.svg")


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


if __name__ == "__main__":
    ...
