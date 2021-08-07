import itertools
import os
import random
import string
from typing import List, NamedTuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from autoencoder import AutoEncoder
from chunked_writer import TidyReader
from mnist import MNISTDataset
from clutter import ClutterDataset
from torch.tensor import Tensor
from torch.utils.tensorboard import SummaryWriter

from experiments.experiment import BaseExperiment

eps = 0.0001



class Config(NamedTuple):
    # experiement params
    seed: int = 123
    nprocs: int = 1
    nogpu: bool = False
    logfreq: int = 10
    nsteps: int = 20
    nagents: int = 3
    ngpus: int = 1
    log_every: int = 50  # how often we write to readers / tb

    # message boundary
    nodetach: bool = False

    # choose the dataset
    dataset: str = "MNIST"  # MNIST or CLUTTER

    # agents use same digit as input
    samedigit: bool = False

    # nets
    latent_dim: int = 30
    lr: float = 0.001
    bsize: int = 32

    # bnorm
    bnorm: bool = False
    affine: bool = False

    # channel noise
    sigma: float = 0.0

    # hyperparameters
    eta_ae: float = 0.0
    eta_lsa: float = 0.0
    eta_msa: float = 0.0
    eta_dsa: float = 0.0

    # assessment of abstraction
    nsteps_pred_latent: int = 10
    bsize_pred_latent: int = 32


class Experiment(BaseExperiment):
    def log(self, step: int, agents: "list[AutoEncoder]"):
        mlps = self.predict_from_latent_and_reconstruction(agents + [self.base_2], step)
        # add base_2 here to have 2 agents for swapping in the base case
        self.compute_cross_acc(agents + [self.base_2], mlps, step)
        self.save_params(step, agents + [self.base_2])
        self.writer._write()

    def compute_cross_acc(
        self, agents: "list[AutoEncoder]", mlps: "list[MLP]", step: int
    ):
        # TODO: DANGEROUS! will NOT scale with more agents. FIX!!!
        ma_aes, ma_mlps = agents[:3], mlps[:3]
        sa_aes, sa_mlps = agents[3:], mlps[3:]
        X, y = map(
            lambda x: x.to(self.dev),
            self.dataset.sample_with_label(int(self.cfg.bsize)),
        )
        self._compute_cross_acc(X, y, ma_aes, ma_mlps, "MA", step)
        self._compute_cross_acc(X, y, sa_aes, sa_mlps, "Base", step)

    def _compute_cross_acc(self, X, y, aes, mlps, tag, step, rot=1):
        for i, (ae, mlp) in enumerate(zip(aes, mlps[rot:] + mlps[:rot])):
            latent = ae.encode(X)
            acc = mlp.compute_acc(latent, y)
            self.writer.add((step, tag, acc), step=i, tag="cross_agent_acc")
            self.tb.add_scalar(f"cross_agent_acc_{tag}_step{step}", acc)

    def run(self, cfg: NamedTuple):
        if self.cfg.dataset == "MNIST":
            self.dataset = MNISTDataset()
        elif self.cfg.dataset == "CLUTTER":
            self.dataset = ClutterDataset()
        else:
            raise Exception("Wrong dataset name specified")

        tb_path = f"{self.path}/tb/{self.rank}"
        self.tb = SummaryWriter(tb_path)

        base = self.generate_autoencoder("baseline")
        self.base_2 = self.generate_autoencoder("baseline_2")
        agents = [
            self.generate_autoencoder(f"{i}")
            for i in string.ascii_uppercase[: cfg.nagents]
        ]
        agents_and_base = agents + [base]

        agent_index_pairs = list(itertools.combinations(range(len(agents)), r=2))
        for i in range(cfg.nsteps):
            for agent_index_pair in agent_index_pairs:
                # maybe shuffle order to break symmetry
                shuffle = random.choice([0, 1])
                agent_indices = (
                    reversed(agent_index_pair) if shuffle else agent_index_pair
                )
                agent_a, agent_b = [agents[i] for i in (agent_indices)]

                # execute marl-ae step
                self.sync_ae_step(i, agent_a, agent_b)
                if agent_a.name == "A" or agent_b.name == "A":
                    self.control_step(i, base)
                    self.control_step(i, self.base_2)

            if i % cfg.logfreq == cfg.logfreq - 1:
                self.log(i, agents_and_base)
                self.writer._write()
        self.writer.close()

    def save_params(self, step: int, agents: List[AutoEncoder]):
        path = os.path.join(self.path, "params", f"step_{step}", f"rank_{self.rank}")
        os.makedirs(path)
        for agent in agents:
            torch.save(agent.state_dict(), os.path.join(path, f"{agent.name}.pt"))

    def generate_autoencoder(self, name: str,):
        return AutoEncoder(
            self.cfg.latent_dim,
            self.cfg.bnorm,
            self.cfg.affine,
            self.cfg.lr,
            name,
            pre_latent_dim=36 if self.cfg.dataset == "MNIST" else 64,
        ).to(self.dev)

    def sync_ae_step(self, step: int, agent_a: AutoEncoder, agent_b: AutoEncoder):
        digit = random.choice(range(10))
        batch_a = self.dataset.sample_digit(digit, bsize=self.cfg.bsize).to(self.dev)
        batch_b = (
            batch_a
            if self.cfg.samedigit
            else self.dataset.sample_digit(digit, bsize=self.cfg.bsize).to(self.dev)
        )

        # compute message and reconstructions
        msg_a = agent_a.encode(batch_a)
        msg_b = agent_b.encode(batch_b)
        msg_a_maybe_detached = msg_a if self.cfg.nodetach else msg_a.detach()
        msg_b_maybe_detached = msg_b if self.cfg.nodetach else msg_b.detach()

        # add channel noise to messages
        msg_a += torch.randn_like(msg_a) * self.cfg.sigma
        msg_b += torch.randn_like(msg_b) * self.cfg.sigma

        ## rec_aa is a's reconstruction of a's message
        ## rec_ab is a's reconstruction of b's message and so forth..
        rec_aa = agent_a.decode(msg_a)
        rec_bb = agent_b.decode(msg_b)
        # needs to
        rec_ab = agent_a.decode(msg_b_maybe_detached)
        rec_ba = agent_b.decode(msg_a_maybe_detached)

        # autoencoding
        ae_loss_a = F.mse_loss(rec_aa, batch_a)
        ae_loss_b = F.mse_loss(rec_bb, batch_b)

        # latent space adaptation
        lsa_loss_a = F.mse_loss(msg_a, msg_b_maybe_detached)
        lsa_loss_b = F.mse_loss(msg_b, msg_a_maybe_detached)

        # message space adaptation
        msa_loss_a = F.mse_loss(rec_ab, batch_a)
        msa_loss_b = F.mse_loss(rec_ba, batch_b)

        # decoding space adaptation
        # NOTE: we might or might not want to detach rec_aa/rec_bb, discuss with Clem
        dsa_loss_a = F.mse_loss(rec_ab, rec_aa)
        dsa_loss_b = F.mse_loss(rec_ba, rec_bb)

        with torch.no_grad():
            # compute mean batch var per feature
            # TODO: probably compute running average here.
            mbvar_a = msg_a.var(dim=0).mean().item()
            mbvar_b = msg_b.var(dim=0).mean().item()

            # compute difference between the decodings. Should decrease if agents abstract
            dec_diff_a = dec_diff_b = F.mse_loss(rec_aa, rec_bb)

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

        ab_name = agent_a.name + agent_b.name
        ba_name = agent_b.name + agent_a.name

        if step % self.cfg.log_every == 0:
            self.tb.add_scalar(f"AEloss{ab_name}", ae_loss_a, step)
            self.tb.add_scalar(f"AEloss{ba_name}", ae_loss_b, step)

            self.tb.add_scalar(f"mbvar{ab_name}", mbvar_a, step)
            self.tb.add_scalar(f"mbvar{ba_name}", mbvar_b, step)

            self.tb.add_scalar(f"LSAloss{ab_name}", lsa_loss_a, step)
            self.tb.add_scalar(f"LSAloss{ba_name}", lsa_loss_b, step)

            self.tb.add_scalar(
                f"LSA-mbvar{ab_name}", lsa_loss_a / (mbvar_a + eps), step
            )
            self.tb.add_scalar(
                f"LSA-mbvar{ba_name}", lsa_loss_b / (mbvar_b + eps), step
            )

            self.writer.add_multiple(
                [
                    (ae_loss_a.item(), "AE", agent_a.name, agent_b.name),
                    (lsa_loss_a.item(), "LSA", agent_a.name, agent_b.name),
                    (msa_loss_a.item(), "MSA", agent_a.name, agent_b.name),
                    (dsa_loss_a.item(), "DSA", agent_a.name, agent_b.name),
                    (mbvar_a, "MBVAR", agent_a.name, agent_b.name),
                    (
                        lsa_loss_a.item() / (mbvar_a + eps),
                        "LSA-MBVAR",
                        agent_a.name,
                        agent_b.name,
                    ),
                    (ae_loss_b.item(), "AE", agent_b.name, agent_a.name),
                    (lsa_loss_b.item(), "LSA", agent_b.name, agent_a.name),
                    (msa_loss_b.item(), "MSA", agent_b.name, agent_a.name),
                    (dsa_loss_b.item(), "DSA", agent_b.name, agent_a.name),
                    (mbvar_b, "MBVAR", agent_b.name, agent_a.name),
                    (
                        lsa_loss_b.item() / (mbvar_b + eps),
                        "LSA-MBVAR",
                        agent_b.name,
                        agent_a.name,
                    ),
                    (dec_diff_a.item(), "DECDIFF", agent_a.name, agent_b.name),
                    (dec_diff_b.item(), "DECDIFF", agent_b.name, agent_a.name),
                ],
                step=step,
                tag="loss",
            )

    def control_step(self, step: int, agent: AutoEncoder):
        digit = random.choice(range(10))
        batch = self.dataset.sample_digit(digit, bsize=self.cfg.bsize).to(self.dev)

        latent = agent.encode(batch)
        latent += torch.randn_like(latent) * self.cfg.sigma
        rec = agent.decode(latent)

        loss = F.mse_loss(rec, batch) * (
            self.cfg.eta_ae
            + 0.5 * self.cfg.eta_lsa
            + 0.5 * self.cfg.eta_dsa
            + 0.5 * self.cfg.eta_msa
        )
        agent.opt.zero_grad()
        loss.backward()
        agent.opt.step()
        if step % self.cfg.log_every == 0:
            self.tb.add_scalar(f"ae_loss_control_{agent.name}", loss.item(), step)
            self.writer.add((loss.item(), "AE", agent.name, ""), step=step, tag="loss")

    def predict_from_latent_and_reconstruction(
        self, agents: List[AutoEncoder], step: int
    ) -> None:
        mlps = []
        test_ims, test_targets = map(
            lambda x: x.to(self.dev),
            self.dataset.sample_with_label(self.cfg.bsize_pred_latent, eval=True),
        )
        for agent in agents:
            mlp: MLP = MLP(self.cfg.latent_dim).to(self.dev)
            # mlp_rec: CNN = CNN().to(self.dev)

            for i in range(self.cfg.nsteps_pred_latent):
                ims, labels = map(
                    lambda x: x.to(self.dev),
                    self.dataset.sample_with_label(self.cfg.bsize_pred_latent),
                )
                latent = agent.encode(ims)
                # reconstruction = agent(ims)

                loss_latent = mlp.train(latent, labels)
                acc_latent = mlp.compute_acc(latent, labels)
                test_acc_latent = mlp.compute_acc(agent.encode(test_ims), test_targets)

                # loss_rec = mlp_rec.train(reconstruction, labels)
                # acc_rec = mlp_rec.compute_acc(reconstruction, labels)

                if i % 50 == 0:
                    self.tb.add_scalar(
                        f"acc_from_latent_{agent.name}_epoch_{step}", acc_latent, i
                    )
                    # self.tb.add_scalar(
                    #    f"acc_from_rec_{agent.name}_epoch_{step}", acc_rec, i
                    # )

                    self.writer.add_multiple(
                        [
                            (i, loss_latent, "Loss", "Latent", agent.name),
                            (i, acc_latent, "Accuracy", "Latent", agent.name),
                            (i, test_acc_latent, "Test accuracy", "Latent", agent.name)
                            # (i, loss_rec, "Loss", "Reconstruction", agent.name),
                            # (i, acc_rec, "Accuracy", "Reconstruction", agent.name),
                        ],
                        step=step,
                        tag="pred_from_latent",
                    )
            mlps.append(mlp)
        return mlps

    @staticmethod
    def load_data(reader: TidyReader):
        df = reader.read(
            tag="loss", columns=["Step", "Rank", "Loss", "Type", "Agent_A", "Agent_B"]
        )
        df.loc[df["Agent_A"] == "baseline", "Agent_B"] = "baseline"
        groups = df.groupby(
            ["Rank", "Type", "Agent_A", "Agent_B"], as_index=False
        ).apply(lambda x: x.sort_values(by="Step")[::50])
        return groups

    @staticmethod
    def plot(df, path):
        sns.relplot(
            data=df,
            x="Step",
            y="Loss",
            col="Type",
            hue="Agent_A",
            kind="line",
            # ci=None,
            col_wrap=3,
            facet_kws=dict(sharey=False),
        )
        plot_path = f"{path}/loss"
        df.to_csv(plot_path + ".csv")
        plt.savefig(plot_path + ".pdf")
        plt.savefig(plot_path + ".svg")


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
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = self.l1(xb.flatten(start_dim=1))
        return xb

    def compute_acc(self, ims, labels):
        pred = self(ims).argmax(dim=1)
        acc = (pred == labels).float().mean()
        return acc.item()

    def train(self, inputs: list, targets):
        x = self(inputs)
        loss = F.cross_entropy(x, targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
