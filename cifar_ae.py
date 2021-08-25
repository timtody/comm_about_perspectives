import wandb
import torch
import torch.nn.functional as F
from autoencoder import CifarAutoEncoder
import matplotlib.pyplot as plt
from typing import NamedTuple
from cifar import CifarDataset
from experiments.shared_ref_mnist import MLP


class Config(NamedTuple):
    lr: float = 0.0005
    bsize: int = 1024
    eval_steps: int = 5000
    n_classes: int = 10
    n_latent_channels: int = 1


def log_reconstructions(ae, dataset, dev):
    _, axes = plt.subplots(nrows=2, ncols=10)
    ims, _ = dataset.sample_with_label(10)
    reconstructions = ae(ims.to(dev))

    for ax, im in zip(axes.flatten(), torch.cat([reconstructions.cpu(), ims], dim=0)):
        ax.imshow(im.detach().squeeze())

    wandb.log({"reconstructions": plt})


def predict_classes(cfg, ae, dataset, dev, step):
    test_ims, test_targets = map(
        lambda x: x.to(dev),
        dataset.sample_with_label(cfg.bsize, eval=True),
    )
    input_size = ae.encode(test_ims[0].unsqueeze(0)).flatten().size(0)
    mlp = MLP(input_size, cfg.n_classes).to(dev)

    for i in range(cfg.eval_steps):
        ims, labels = map(
            lambda x: x.to(dev),
            dataset.sample_with_label(cfg.bsize),
        )
        latent = ae.encode(ims).flatten(start_dim=1)
        loss_latent = mlp.train(latent, labels)
        acc_latent = mlp.compute_acc(latent, labels, topk=5)
        test_acc_latent = mlp.compute_acc(
            ae.encode(test_ims).flatten(start_dim=1), test_targets, topk=5
        )
        wandb.log(
            {
                f"mlp loss": loss_latent,
                f"mlp acc train": acc_latent,
                f"mlp acc test": test_acc_latent,
            }
        )


def main():
    cfg = Config()
    assert cfg.n_classes == 10 or cfg.n_classes == 100, "10 or 100 classes only"
    wandb.init(
        project="cifar-100-autoencoder", entity="origin-flowers", config=cfg._asdict()
    )
    ae = CifarAutoEncoder(lr=cfg.lr, n_latent_channels=cfg.n_latent_channels)
    wandb.watch(ae)
    dataset = CifarDataset(f"CIFAR{cfg.n_classes}")

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ae.to(dev)

    for i in range(50000):
        batch, _ = dataset.sample_with_label(2048)
        ae.opt.zero_grad()
        latent = ae.encode(batch.to(dev))
        reconstruction = ae.decode(latent)
        loss = F.mse_loss(reconstruction, batch.to(dev)) + latent.mean()
        loss.backward()
        ae.opt.step()
        wandb.log({"Reconstruction loss": loss.item()})
        if i % 1000 == 0:
            log_reconstructions(ae, dataset, dev)

        if i % 5000 == 0:
            predict_classes(cfg, ae, dataset, dev, i)


if __name__ == "__main__":
    main()
