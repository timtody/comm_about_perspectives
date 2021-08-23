import wandb
import torch
import torch.nn.functional as F
from autoencoder import CifarAutoEncoder
import matplotlib.pyplot as plt
from typing import NamedTuple
from cifar import CifarDataset
from experiments.shared_ref_mnist import MLP


class Config(NamedTuple):
    lr: float = 0.0001
    bsize: int = 1024


def log_reconstructions(ae, dataset, dev):
    _, axes = plt.subplots(nrows=2, ncols=10)
    ims, _ = dataset.sample_with_label(10)
    reconstructions = ae(ims.to(dev))

    for ax, im in zip(axes.flatten(), torch.cat([reconstructions.cpu(), ims], dim=0)):
        ax.imshow(im.detach().permute(1, 2, 0))

    wandb.log({"reconstructions": plt})


def predict_classes(cfg, ae, dataset, dev, step):
    mlp = MLP(432, 100).to(dev)
    test_ims, test_targets = map(
        lambda x: x.to(dev),
        dataset.sample_with_label(cfg.bsize, eval=True),
    )
    ims, labels = map(
        lambda x: x.to(dev),
        dataset.sample_with_label(cfg.bsize),
    )
    latent = ae.encode(ims).flatten(start_dim=1)
    loss_latent = mlp.train(latent, labels)
    acc_latent = mlp.compute_acc(latent, labels)
    test_acc_latent = mlp.compute_acc(
        ae.encode(test_ims).flatten(start_dim=1), test_targets
    )
    wandb.log({
                f"step_{step}_loss": loss_latent,
                f"step_{step}_acc": acc_latent,
                f"step_{step}_test_acc": test_acc_latent
                })


def main():
    cfg = Config()
    dataset = CifarDataset("CIFAR100")
    ae = CifarAutoEncoder(lr=cfg.lr)

    wandb.init(project='cifar-100-autoencoder', entity='timtody', config=cfg._asdict())
    wandb.watch(ae)

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ae.to(dev)

    for i in range(10000):
        batch, _ = dataset.sample_with_label(2048)
        ae.opt.zero_grad()
        reconstruction = ae(batch.to(dev))
        loss = F.mse_loss(reconstruction, batch.to(dev))
        loss.backward()
        ae.opt.step()
        wandb.log({"Reconstruction loss": loss.item()})
        if i % 100 == 0:
            log_reconstructions(ae, dataset, dev)

        if i % 250 == 0:
            predict_classes(cfg, ae, dataset, dev, i)


if __name__ == "__main__":
    main()
