import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mnist import MNISTDataset
from abc import ABC


class AEInterface(ABC):
    def forward(self, x):
        x = self._encoder(x)
        x = self._decoder(x)
        return x

    def encode(self, x) -> torch.Tensor:
        return self._encoder(x)

    def decode(self, x) -> torch.Tensor:
        return self._decoder(x)


class BigEncoder(nn.Module):
    def __init__(self, latent_dim, bnorm=True, affine=True, /, pre_latent_dim=49):
        super().__init__()
        self.pre_latent_dim = pre_latent_dim
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 1, 4, 1, 1)
        self.l1 = nn.Linear(self.pre_latent_dim, latent_dim)
        self.bnorm_l = nn.BatchNorm1d(latent_dim, affine=affine) if bnorm else nn.Identity()

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = self.l1(x.flatten(start_dim=1))
        x = self.bnorm_l(x)
        return x


class BigDecoder(nn.Module):
    def __init__(self, latent_dim, /, pre_latent_dim=49):
        super().__init__()
        self.l1 = nn.Linear(latent_dim, pre_latent_dim)
        self.conv1 = nn.ConvTranspose2d(1, 32, 4, 1, 1)
        self.conv2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.conv3 = nn.ConvTranspose2d(32, 1, 4, 2, 1)
        self.pre_latent_dim = pre_latent_dim

    def forward(self, x):
        x = self.l1(x)
        side_len = int(math.sqrt(self.pre_latent_dim))
        x = x.reshape(-1, side_len, side_len).unsqueeze(1)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        return x


class BigAutoEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        bnorm: bool,
        affine: bool,
        lr: float,
        name: str,
        /,
        pre_latent_dim=49,
    ):
        super().__init__()
        self._encoder = BigEncoder(latent_dim, bnorm, affine, pre_latent_dim=pre_latent_dim)
        self._decoder = BigDecoder(latent_dim, pre_latent_dim=pre_latent_dim)
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.name = name

    def forward(self, x):
        x = self._encoder(x)
        x = self._decoder(x)
        return x

    def encode(self, x) -> torch.Tensor:
        return self._encoder(x)

    def decode(self, x) -> torch.Tensor:
        return self._decoder(x)


class Encoder(nn.Module):
    def __init__(self, latent_dim, bnorm=True, affine=True, /, pre_latent_dim=49):
        super().__init__()
        self.pre_latent_dim = pre_latent_dim
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 1, 4, 2, 1)
        self.l1 = nn.Linear(self.pre_latent_dim, latent_dim)
        self.bnorm_l = nn.BatchNorm1d(latent_dim, affine=affine) if bnorm else nn.Identity()

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.l1(x.reshape(-1, self.pre_latent_dim))
        x = self.bnorm_l(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, /, pre_latent_dim=49):
        super().__init__()
        self.l1 = nn.Linear(latent_dim, pre_latent_dim)
        self.conv1 = nn.ConvTranspose2d(1, 32, 4, 2, 1)
        self.conv2 = nn.ConvTranspose2d(32, 1, 4, 2, 1)
        self.pre_latent_dim = pre_latent_dim

    def forward(self, x):
        x = self.l1(x)
        side_len = int(math.sqrt(self.pre_latent_dim))
        x = x.reshape(-1, side_len, side_len).unsqueeze(1)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        return x


class AutoEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        bnorm: bool,
        affine: bool,
        lr: float,
        name: str,
        /,
        pre_latent_dim=49,
    ):
        super().__init__()
        self._encoder = Encoder(latent_dim, bnorm, affine, pre_latent_dim=pre_latent_dim)
        self._decoder = Decoder(latent_dim, pre_latent_dim=pre_latent_dim)
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.name = name

    def forward(self, x):
        x = self._encoder(x)
        x = self._decoder(x)
        return x

    def encode(self, x) -> torch.Tensor:
        return self._encoder(x)

    def decode(self, x) -> torch.Tensor:
        return self._decoder(x)


class CifarAutoEncoder(AEInterface, nn.Module):
    def __init__(self, lr=0.001, name=None):
        super().__init__()
        self._encoder = cifar_encoder()
        self._decoder = cifar_decoder()
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.name = name


def cifar_encoder():
    return nn.Sequential(
        nn.Conv2d(3, 32, 5),
        nn.ELU(),
        nn.Conv2d(32, 64, 5),
        nn.ELU(),
        nn.Conv2d(64, 64, 5),
        nn.ELU(),
        nn.Conv2d(64, 64, 5),
        nn.ELU(),
        nn.Conv2d(64, 4, 5),
    )


def cifar_decoder():
    return nn.Sequential(
        nn.ConvTranspose2d(4, 64, 5),
        nn.ELU(),
        nn.ConvTranspose2d(64, 64, 5),
        nn.ELU(),
        nn.ConvTranspose2d(64, 64, 5),
        nn.ELU(),
        nn.ConvTranspose2d(64, 32, 5),
        nn.ELU(),
        nn.ConvTranspose2d(32, 3, 5),
    )
