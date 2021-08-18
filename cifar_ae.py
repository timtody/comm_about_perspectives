import torch
import torch.nn as nn
import torch.nn.functional as F
from autoencoder import CifarAutoEncoder
import matplotlib.pyplot as plt

from cifar import CifarDataset

dataset = CifarDataset("CIFAR10", colour=True)
ae = CifarAutoEncoder()


def test(ae, dataset):
    _, axes = plt.subplots(nrows=2, ncols=10)
    ims, _ = dataset.sample_with_label(10)
    reconstructions = ae(ims)

    print(ims.size(), reconstructions.size())

    for ax, im in zip(axes.flatten(), torch.cat([reconstructions, ims], dim=0)):
        ax.imshow(im.detach().permute(1, 2, 0))


losses = []

for i in range(500):
    batch, _ = dataset.sample_with_label(128)
    ae.opt.zero_grad()
    reconstruction = ae(batch)
    loss = F.mse_loss(reconstruction, batch)
    loss.backward()
    ae.opt.step()

    losses.append(losses)

test(ae, dataset)
plt.show()
plt.plot(losses)
plt.show()
print(losses)
