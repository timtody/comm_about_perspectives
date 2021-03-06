import os
import pandas as pd
import jax
import torchvision
import reprieve
from reprieve.algorithms import mlp as alg
from mnist import MNISTDataset
from autoencoder import AutoEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from torch import Tensor
import argparse
import time
import torch
import glob
import string
from utils import remove_legend_titles
from plotting_functions import prepare_plot, get_size


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
    )
    results = raw_loss_data_estimator.compute_curve(n_points=args.points)
    results["name"] = name
    return results


def evaluate_experiment(
    path_dti: str, path_mtm: str, data_x: Tensor, data_y: Tensor
) -> pd.DataFrame:
    result_df_container = []

    for path_dti in glob.glob(f"{path_dti}/*"):
        for i in range(2):
            ae = AutoEncoder(30, False, False, 0.001, "bruh", pre_latent_dim=49)
            repres = ae.encode(data_x).detach()
            results = evaluate_representations(
                repres, data_y, 10, (30,), args, "Random features"
            )
            results["Agent"] = i
            result_df_container.append(results)

        for i in range(2):
            ae = AutoEncoder(30, False, False, 0.001, "bruh", pre_latent_dim=49)
            ae.load_state_dict(
                torch.load(path_dti + ("/baseline.pt" if i == 0 else "/baseline_2.pt"))
            )
            repres = ae.encode(data_x).detach()
            results = evaluate_representations(repres, data_y, 10, (30,), args, "AE")
            results["Agent"] = i
            result_df_container.append(results)

        for i in range(3):
            ae = AutoEncoder(30, False, False, 0.001, "bruh", pre_latent_dim=49)
            ae.load_state_dict(torch.load(path_dti + f"/{string.ascii_uppercase[i]}.pt"))
            repres = ae.encode(data_x).detach()
            results = evaluate_representations(repres, data_y, 10, (30,), args, "DTI")
            results["Agent"] = i
            result_df_container.append(results)

    for path_mtm in glob.glob(f"{path_mtm}/*"):
        for i in range(3):
            ae = AutoEncoder(30, False, False, 0.001, "bruh", pre_latent_dim=49)
            ae.load_state_dict(torch.load(path_mtm + f"/{string.ascii_uppercase[i]}.pt"))
            repres = ae.encode(data_x).detach()
            results = evaluate_representations(repres, data_y, 10, (30,), args, "AE+MTM")
            results["Agent"] = i
            result_df_container.append(results)

        results = pd.concat(result_df_container)
        return results


def plot_external(ax):
    results = get_results()
    results = results[results.name != "Random features"]
    sns.lineplot(
        data=results,
        ax=ax,
        x="samples",
        y="val_loss",
        hue="name",
        legend="brief",
        style="name",
        markers=True,
        dashes=False,
        hue_order=["AE", "AE+MTM", "DTI"],
    )
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("Validation loss")
    ax.set_xlabel("Dataset size")


def plot_curves(results, ns, path):
    prepare_plot()
    fig_w, fig_h = get_size("neurips")
    fig = plt.figure(constrained_layout=True, figsize=(fig_w, fig_h))
    results = results[results.name != "Random features"]
    ax = sns.lineplot(
        data=results,
        x="samples",
        y="val_loss",
        hue="name",
        legend="brief",
        style="name",
        markers=True,
        dashes=False,
        hue_order=["AE", "AE+MTM", "DTI"],
    )
    sns.despine(ax=ax)
    remove_legend_titles(ax)
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Validation loss")
    plt.xlabel("Dataset size")

    # ns = _closest_valid_ns(results, ns)
    # for n in ns:
    #     plt.vlines(n, 0, 10, linestyles="dashed")

    # plt.hlines(0.2, 10, 10000, linestyles="dashed")
    # plt.hlines(1, 10, 10000, linestyles="dashed")

    fig.savefig(
        f"{path}_reprieve_curves_mnist.pdf",
        format="pdf",
        bbox_inches="tight",
    )


def get_results():
    results_path = "results/full_res.csv"
    if not os.path.exists(results_path):
        ds = MNISTDataset()
        data_x, data_y = ds.test_set.data.unsqueeze(1) / 255.0, ds.test_set.targets
        results = evaluate_experiment(
            args.dti_path,
            args.mtm_path,
            data_x,
            data_y,
        )
        return results
    else:
        return pd.read_csv(results_path)


def main(args):
    ns = [10, 100, 1000, 10000]
    epsilons = [0.2, 1]

    results = get_results()
    results.to_csv("results/full_res.csv")

    save_path = (
        "results/"
        f"{args.name}"
        f"_train{args.train_steps}"
        f"_seed{args.seeds}"
        f"_point{args.points}"
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
    parser.add_argument("--mtm_path", type=str, required=False)
    parser.add_argument("--dti_path", type=str, required=False)
    args = parser.parse_args()

    start = time.time()
    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)
    end = time.time()
    print(f"Time: {end - start :.3f} seconds")
