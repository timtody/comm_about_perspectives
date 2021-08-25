import os
import pandas as pd
import jax
import torchvision
import reprieve
from reprieve.algorithms import mlp as alg
from mnist import MNISTDataset
from autoencoder import CifarAutoEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from torch import Tensor
import argparse
import time
import torch
import glob
import string


def _closest_valid_ns(df, ns):
    closest_ns = []
    available_ns = sorted(list(df.samples.unique()))
    last_match = 0
    for desired_n in sorted(ns):
        i = last_match
        while desired_n > available_ns[i] and i < len(available_ns) - 1:
            i += 1
        last_match = i
        closest_ns.append(available_ns[i])
    return closest_ns


def evaluate_representations(
    representations: Tensor,
    labels: Tensor,
    nclasses: int,
    repr_shape: tuple,
    args: argparse.Namespace,
    name: str,
) -> pd.DataFrame:
    init_fn, train_step_fn, eval_fn = alg.make_algorithm(repr_shape, nclasses)

    raw_loss_data_estimator = reprieve.LossDataEstimator(
        init_fn,
        train_step_fn,
        eval_fn,
        (representations, labels),
        train_steps=args.train_steps,
        n_seeds=args.seeds,
        use_vmap=args.use_vmap,
        cache_data=args.cache_data,
        verbose=True,
        batch_size=args.bsize,
    )
    results = raw_loss_data_estimator.compute_curve(n_points=args.points)
    results["name"] = name
    return results


def evaluate_experiment(
    path_dti: str, path_mtm: str, data_x: Tensor, data_y: Tensor
) -> pd.DataFrame:
    result_df_container = []

    for path_dti in glob.glob(f"{path_dti}/*")[:3]:
        for i in range(2):
            ae = CifarAutoEncoder()
            repres = ae.encode(data_x).detach()
            results = evaluate_representations(
                repres, data_y, 10, (7744,), args, "Random features"
            )
            results["Agent"] = i
            result_df_container.append(results)

        for i in range(2):
            ae = CifarAutoEncoder()
            ae.load_state_dict(
                torch.load(path_dti + ("/baseline.pt" if i == 0 else "/baseline_2.pt"))
            )
            repres = ae.encode(data_x).detach()
            results = evaluate_representations(repres, data_y, 10, (7744,), args, "AE")
            results["Agent"] = i
            result_df_container.append(results)

        for i in range(3):
            ae = CifarAutoEncoder()
            ae.load_state_dict(torch.load(path_dti + f"/{string.ascii_uppercase[i]}.pt"))
            repres = ae.encode(data_x).detach()
            results = evaluate_representations(repres, data_y, 10, (7744,), args, "DTI")
            results["Agent"] = i
            result_df_container.append(results)

    for path_mtm in glob.glob(f"{path_mtm}/*")[:3]:
        for i in range(3):
            ae = CifarAutoEncoder()
            ae.load_state_dict(torch.load(path_mtm + f"/{string.ascii_uppercase[i]}.pt"))
            repres = ae.encode(data_x).detach()
            results = evaluate_representations(repres, data_y, 10, (7744,), args, "AE+MTM")
            results["Agent"] = i
            result_df_container.append(results)

    results = pd.concat(result_df_container)
    return results


def plot_curves(results, ns, path):
    sns.set_palette(sns.color_palette("Set1"))
    ax = sns.lineplot(
        data=results,
        x="samples",
        y="val_loss",
        hue="name",
        legend="brief",
        style="name",
        markers=True,
        dashes=False,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Validation loss")
    plt.xlabel("Dataset size")

    ns = _closest_valid_ns(results, ns)
    for n in ns:
        plt.vlines(n, 0, 10, linestyles="dashed")

    plt.hlines(0.2, 10, 10000, linestyles="dashed")
    plt.hlines(1, 10, 10000, linestyles="dashed")

    plt.savefig(f"{path}_reprieve_curves.pdf")


def main(args):
    ns = [
        10,
        100,
        1000,
        10000,
    ]
    epsilons = [0.2, 1]

    ds = MNISTDataset()
    data_x, data_y = ds.test_set.data.unsqueeze(1) / 255.0, ds.test_set.targets
    results_path = "results/full_res_cifar.csv"

    if not os.path.exists(results_path):
        results = evaluate_experiment(
            args.dti_path,
            args.mtm_path,
            data_x,
            data_y,
        )
        results.to_csv("results/full_res_cifar.csv")
    else:
        results = pd.read_csv(results_path)

    save_path = (
        "results/"
        f"{args.name}"
        f"_train{args.train_steps}"
        f"_seed{args.seeds}"
        f"_point{args.points}_cifar"
    )

    plot_curves(results, ns, save_path)
    metrics_df = reprieve.compute_metrics(results, ns, epsilons)
    reprieve.render_latex(metrics_df, save_path=f"{save_path}metrics.tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="jax")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_vmap", dest="use_vmap", action="store_false")
    parser.add_argument("--no_cache", dest="cache_data", action="store_false")
    parser.add_argument("--train_steps", type=float, default=4e3)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--points", type=int, default=10)
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--mtm_path", type=str, required=True)
    parser.add_argument("--dti_path", type=str, required=True)
    args = parser.parse_args()

    start = time.time()
    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)
    end = time.time()
    print(f"Time: {end - start :.3f} seconds")
