import itertools
import os
import random
import string
from typing import AnyStr, List, NamedTuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from autoencoder import AutoEncoder
from chunked_writer import TidyReader
from mnist import MNISTDataset
from torch.tensor import Tensor
from torch.utils.tensorboard import SummaryWriter

from experiments.experiment import BaseExperiment


class Experiment(BaseExperiment):
    def log(self, step: int, agents):
        self.predict_from_latent_and_reconstruction(agents, step)
        self.save_params(step, agents)
        self.writer._write()

    def run(self, cfg: NamedTuple):
        self.dataset = MNISTDataset()

        tb_path = f"{self.path}/tb/{self.rank}"
        self.tb = SummaryWriter(tb_path)

        base = self.generate_autoencoder("baseline")
        agents = [
            self.generate_autoencoder(f"{i}")
            for i in string.ascii_uppercase[: cfg.nagents]
        ]
        agents_and_base = [base] + agents

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

            if i % cfg.logfreq == cfg.logfreq - 1:
                self.log(i, agents_and_base)
                self.writer._write()
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
        rec_ab = agent_a.decode(msg_b.detach())
        rec_ba = agent_b.decode(msg_a.detach())

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

            # compute difference between the decodings. Should decrease if agents abstract
            dec_diff_a = dec_diff_b = F.mse_loss(rec_aa, rec_bb)

        ab_name = agent_a.name + agent_b.name
        ba_name = agent_b.name + agent_a.name

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

        if step % 50 == 0:
            self.tb.add_scalar(f"AEloss{ab_name}", ae_loss_a, step)
            self.tb.add_scalar(f"AEloss{ba_name}", ae_loss_b, step)

            self.tb.add_scalar(f"mbvar{ab_name}", mbvar_a, step)
            self.tb.add_scalar(f"mbvar{ba_name}", mbvar_b, step)

            self.tb.add_scalar(f"LSAloss{ab_name}", lsa_loss_a, step)
            self.tb.add_scalar(f"LSAloss{ba_name}", lsa_loss_b, step)

            self.tb.add_scalar(
                f"LSA-mbvar{ab_name}", lsa_loss_a / (mbvar_a + 0.0001), step
            )
            self.tb.add_scalar(
                f"LSA-mbvar{ba_name}", lsa_loss_b / (mbvar_b + 0.0001), step
            )

            self.writer.add_multiple(
                [
                    (ae_loss_a.item(), "AE", agent_a.name, agent_b.name),
                    (lsa_loss_a.item(), "LSA", agent_a.name, agent_b.name),
                    (msa_loss_a.item(), "MSA", agent_a.name, agent_b.name),
                    (dsa_loss_a.item(), "DSA", agent_a.name, agent_b.name),
                    (mbvar_a.item(), "MBVAR", agent_a.name, agent_b.name),
                    (
                        lsa_loss_a.item() / (mbvar_a.item() + 0.0001),
                        "LSA-MBVAR",
                        agent_a.name,
                        agent_b.name,
                    ),
                    (ae_loss_b.item(), "AE", agent_b.name, agent_a.name),
                    (lsa_loss_b.item(), "LSA", agent_b.name, agent_a.name),
                    (msa_loss_b.item(), "MSA", agent_b.name, agent_a.name),
                    (dsa_loss_b.item(), "DSA", agent_b.name, agent_a.name),
                    (mbvar_b.item(), "MBVAR", agent_b.name, agent_a.name),
                    (
                        lsa_loss_b.item() / (mbvar_b.item() + 0.0001),
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
        if step % 100 == 0:
            self.tb.add_scalar("ae_loss_control", loss.item(), step)
            self.writer.add((loss.item(), "AE", agent.name, ""), step=step, tag="loss")

    def predict_from_latent_and_reconstruction(
        self, agents: List[AutoEncoder], step: int
    ) -> None:
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

                if i % 50 == 0:
                    self.tb.add_scalar(
                        f"acc_from_latent_{agent.name}_epoch_{step}", acc_latent, i
                    )
                    self.tb.add_scalar(
                        f"acc_from_rec_{agent.name}_epoch_{step}", acc_rec, i
                    )

                    self.writer.add_multiple(
                        [
                            (i, loss_latent, "Loss", "Latent", agent.name),
                            (i, acc_latent, "Accuracy", "Latent", agent.name),
                            (i, loss_rec, "Loss", "Reconstruction", agent.name),
                            (i, acc_rec, "Accuracy", "Reconstruction", agent.name),
                        ],
                        step=step,
                        tag="pred_from_latent",
                    )

    @staticmethod
    def load_data(reader: TidyReader):
        df = reader.read(
            tag="loss", columns=["Step", "Rank", "Loss", "Type", "Agent_A", "Agent_B"]
        )
        groups = df.groupby(["Rank", "Type", "Agent_A"], as_index=False).apply(
            lambda x: x[::10]
        )
        return df

    @staticmethod
    def plot(df, path):
        # df = df[df.Type == "AE"]
        sns.relplot(
            data=df,
            x="Step",
            y="Loss",
            col="Type",
            hue="Agent_A",
            kind="line",
            ci=None,
            col_wrap=3,
            facet_kws=dict(sharey=False),
        )
        plot_path = f"{path}/loss"
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
