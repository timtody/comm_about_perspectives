import itertools
import random
from typing import Any, AnyStr, Tuple
import typing

from torch.tensor import Tensor

from c_types import DataFrame, TidyReader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn.functional as F
from autoencoder import AutoEncoder
from mnist import MNISTDataset

from experiments.experiment import BaseExperiment


def filter_df(
    df: DataFrame,
    datapoints: int = 50,
    sort: bool = False,
    filter: bool = True,
):
    if sort:
        df = df.sort_values(by=["Step"])
    if filter:
        filters = ["Rank", "Type", "Agent_X", "Agent_Y"]
        nsteps = len(list(df.groupby(filters).groups.values())[0])
        df = df.groupby(filters).apply(lambda x: x[:: nsteps // datapoints])
    return df


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


class Experiment(BaseExperiment):
    @staticmethod
    def load_data(
        reader: TidyReader,
    ) -> Tuple[DataFrame, DataFrame]:
        df1 = filter_df(
            reader.read("loss", ["Rank", "Step", "Loss", "Type", "Agent_X", "Agent_Y"]),
            sort=True,
            datapoints=10,
        )
        return df1

    @staticmethod
    def plot(args) -> None:
        df: DataFrame = args
        df.to_csv("res.csv")
        plot_lineplot(df, "AE")
        plot_relplot(df, "LSA")
        plot_relplot(df, "DSA")
        plot_relplot(df, "MSA")

    def run(self, cfg):
        self.dataset = MNISTDataset()
        base = self.generate_autoencoder("baseline")
        agents = [self.generate_autoencoder(f"{i}") for i in "ABCDEFG"[: cfg.nagents]]

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
                # TODO: check if this will lead to duplicate loss entries?
                # this actually will lead to duplicates on a step basis.
                # is this bad? What does this mean? What is the alternative?
                if agent_a.name == "agent_0" or agent_b.name == "agent_0":
                    self.control_step(i, base)

            if i % cfg.logfreq == cfg.logfreq - 1:
                self.log(i)
        self.writer.close()

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
                (step, ae_loss_b.item(), "AE", agent_b.name, agent_a.name),
                (step, lsa_loss_b.item(), "LSA", agent_b.name, agent_a.name),
                (step, msa_loss_b.item(), "MSA", agent_b.name, agent_a.name),
                (step, dsa_loss_b.item(), "DSA", agent_b.name, agent_a.name),
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
        self.writer.add((step, loss.item(), "AE", agent.name), tag="loss")
