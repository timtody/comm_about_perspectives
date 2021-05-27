import os
import string
from typing import Any, List, NamedTuple

import itertools
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
    gridsteps: int = 1
    nsteps: int = 2000
    bsize: int = 128
    centralised: bool = False
    home: bool = False


class Experiment(BaseExperiment):
    def run(self, cfg: Config):

        if cfg.home:
            base_path = "results/jeanzay/results/sweeps"
        else:
            base_path = os.path.join(os.path.expandvars("$SCRATCH"), "results/sweeps")

        if cfg.centralised:
            ae_path = os.path.join(base_path, "shared_ref_mnist/2021-05-20/21-26-45/"
                                              "eta_ae:1-eta_lsa:0.0-eta_msa:0.0-eta_dsa:0.0-sigma:0.67-")
            msa_path = os.path.join(base_path, "shared_ref_mnist/2021-05-20/21-26-45/"
                                               "eta_ae:0.0-eta_lsa:0.0-eta_msa:1-eta_dsa:0.0-sigma:0.67-")
            lsa_path = os.path.join(base_path, "shared_ref_mnist/2021-05-20/21-26-45/"
                                               "eta_ae:0.53-eta_lsa:0.01-eta_msa:0.74-eta_dsa:0.84-sigma:0.33-")
        else:
            if cfg.home:
                ae_path = "results/gridsweep/sigma:0.67-eta_ae:1.0-eta_msa:0.0-eta_lsa:0.0-eta_dsa:0.0-"
                msa_path = "results/gridsweep/sigma:0.67-eta_ae:0.0-eta_msa:1.0-eta_lsa:0.0-eta_dsa:0.0-"
                lsa_path = "results/gridsweep/sigma:0.33-eta_ae:0.67-eta_msa:0.0-eta_lsa:0.67-eta_dsa:0.0-"
                msa_ae_path = "results/gridsweep/sigma:0.67-eta_ae:1.0-eta_msa:1.0-eta_lsa:0.0-eta_dsa:0.0-"

            else:
                ae_path = os.path.join(base_path, "shared_ref_mnist/2021-05-16/"
                                                  "13-09-07/"
                                                  "sigma:0.67-eta_ae:1.0-eta_msa:0.0-eta_lsa:0.0-eta_dsa:0.0-")
                msa_path = os.path.join(base_path, "shared_ref_mnist/2021-05-16/"
                                                   "13-09-07/"
                                                   "sigma:0.67-eta_ae:0.0-eta_msa:1.0-eta_lsa:0.0-eta_dsa:0.0-")
                lsa_path = os.path.join(base_path, "shared_ref_mnist/2021-05-15/"
                                                   "13-21-57/"
                                                   "sigma:0.33-eta_ae:0.67-eta_msa:0.67-eta_lsa:0.0-eta_dsa:0.0-")

        paths = {"AE-MTI": msa_ae_path, "AE": ae_path, "MTI": msa_path, "AE-MTM": lsa_path}

        dataset = MNISTDataset()
        self.dataset = dataset
        for exp_name, path in paths.items():
            path = path + f"/params/step_39999"
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
                            cfg.centralised,
                            exp_name,
                            self.rank,
                            i,
                            "MA" if agent.name != "baseline" else "Baseline",
                            acc,
                        ),
                        step=i,
                        tag="training"
                    )
                self.writer._write()
            self.compute_cross_agent_cls(all_agents, mlps, cfg.centralised, exp_name)
            self.writer._write()

    def compute_cross_agent_cls(self, agents: List[AutoEncoder], mlps: "List[MLP]", centralised: bool, exp_name: str):
        ma_aes, ma_mlps = agents[:3], mlps[:3]
        sa_aes, sa_mlps = agents[3:], mlps[3:]

        X, y = map(
            lambda x: x.to(self.dev),
            self.dataset.sample_with_label(int(self.cfg.bsize)),
        )
        self._compute_cross_acc(X, y, ma_aes, ma_mlps, "MA", centralised, exp_name)
        self._compute_cross_acc(X, y, sa_aes, sa_mlps, "Base", centralised, exp_name)

    def _compute_cross_acc(self, X, y, aes, mlps, tag, centralised, exp_name, rot=1):
        for i, ((ae1, mlp1), (ae2, mlp2)) in enumerate(itertools.combinations(zip(aes, mlps), r=2)):
            latent = ae1.encode(X)
            pred_1 = mlp1.predict(latent)

            pred_2 = mlp2.predict(latent)
            score = (pred_1 == pred_2).float().mean()
            result = (i, tag, score.item(), centralised, exp_name)
            print(result)

            self.writer.add(result, step=i, tag="agreement")

        for i, (ae, mlp) in enumerate(zip(aes, mlps[rot:] + mlps[:rot])):
            latent = ae.encode(X)
            acc = mlp.compute_acc(latent, y)
            self.writer.add((centralised, exp_name, tag, acc), step=i, tag="cross_acc")

    def _load_aes(self, path):
        autoencoders = [
            AutoEncoder(30, False, False, 0.001, name)
            for name in string.ascii_uppercase[:3]
        ]
        baseline1 = AutoEncoder(30, False, False, 0.001, "baseline1").to(self.dev)
        baseline2 = AutoEncoder(30, False, False, 0.001, "baseline2").to(self.dev)
        baselines = [baseline1, baseline2]

        for agent in autoencoders:
            agent.load_state_dict(
                torch.load(
                    f"{path}/rank_{int(self.rank) % 3}/{agent.name}.pt",
                    map_location=self.dev,
                )
            )
        for i, agent in enumerate(baselines):
            agent.load_state_dict(
                torch.load(
                    f"{path}/rank_{(int(self.rank) + i) % 3}/{'baseline'}.pt",
                    map_location=self.dev,
                )
            )
        return autoencoders + baselines

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

    def predict(self, x):
        return self.l(x).argmax(dim=1)

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
