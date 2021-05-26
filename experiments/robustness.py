import os
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


class Config(NamedTuple):
    nprocs: int = 10
    seed: int = 123
    nogpu: bool = False
    ngpus: int = 1
    sigma: float = 0.0
    mp_method: str = "SLURM"
    gridsteps: int = 10
    nsteps: int = 5000
    bsize: int = 4092


class Experiment(BaseExperiment):
    def run(self, cfg: Config):
        centralised = True

        if centralised:
            base_path = os.path.join(os.path.expandvars("$SCRATCH"),
                                     "results/sweeps/shared_ref_mnist/2021-05-20/21-26-45/")
            # base_path = "results/jeanzay/results/sweeps/shared_ref_mnist/2021-05-20/21-26-45/"

            ae_path = os.path.join(base_path, "eta_ae:1-eta_lsa:0.0-eta_msa:0.0-eta_dsa:0.0-sigma:0.67-")
            msa_path = os.path.join(base_path, "eta_ae:0.0-eta_lsa:0.0-eta_msa:1-eta_dsa:0.0-sigma:0.67-")
            lsa_path = os.path.join(base_path, "eta_ae:0.53-eta_lsa:0.01-eta_msa:0.74-eta_dsa:0.84-sigma:0.33-")
        else:
            base_path = "results/gridsweep"

            ae_path = os.path.join(base_path, "sigma:0.67-eta_ae:1.0-eta_msa:0.0-eta_lsa:0.0-eta_dsa:0.0-")
            msa_path = os.path.join(base_path, "sigma:0.67-eta_ae:0.0-eta_msa:1.0-eta_lsa:0.0-eta_dsa:0.0-")
            lsa_path = os.path.join(base_path, "sigma:0.33-eta_ae:0.67-eta_msa:0.67-eta_lsa:0.0-eta_dsa:0.0-")

        paths = {"AE": ae_path, "MTI": msa_path, "AE-MTM": lsa_path}

        for exp_name, path in paths.items():

            path = path + f"/params/step_39999/rank_{self.rank % 3}"
            dataset = MNISTDataset()
            all_agents: List[AutoEncoder] = self._load_aes(path)

            mlps: List[MLP] = [MLP(30) for _ in all_agents]

            for mlp, agent in zip(mlps, all_agents):
                for i in range(cfg.nsteps):
                    ims, targets = dataset.sample_with_label(cfg.bsize)
                    encoding = agent.encode(ims)
                    encoding = encoding + torch.randn_like(encoding) * cfg.sigma
                    mlp.train(encoding, targets)
                    acc = mlp.compute_acc(encoding, targets)
                    self.writer.add(
                        (
                            centralised,
                            exp_name,
                            self.rank,
                            i,
                            "MA" if agent.name != "baseline" else "Baseline",
                            acc,
                        ),
                        step=i,
                    )

    def _load_aes(self, path):
        autoencoders = [
            AutoEncoder(30, False, False, 0.001, name)
            for name in string.ascii_uppercase[:3]
        ]
        baseline = AutoEncoder(30, False, False, 0.001, "baseline")

        all_agents: List[AutoEncoder] = autoencoders + [baseline]
        [
            agent.load_state_dict(
                torch.load(f"{path}/{agent.name}.pt", map_location=self.dev)
            )
            for agent in all_agents
        ]
        return all_agents

    @staticmethod
    def load_data(reader) -> Any:
        pass

    @staticmethod
    def plot(dataframes, plot_path) -> None:
        pass


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
