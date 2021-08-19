import os
import pandas as pd
import jax
import torchvision
import reprieve
from reprieve.algorithms import mlp as alg
from mnist import MNISTDataset
from autoencoder import AutoEncoder


def main(args):
    # load dataset
    ds = MNISTDataset()
    data_x, data_y = ds.sample_with_label(60000)

    # define the classification algorithm for the raw representation
    init_fn, train_step_fn, eval_fn = alg.make_algorithm((1, 28, 28), 10)

    # classify with raw representation (baseline)
    raw_loss_data_estimator = reprieve.LossDataEstimator(
        init_fn,
        train_step_fn,
        eval_fn,
        (data_x, data_y),
        train_steps=args.train_steps,
        n_seeds=args.seeds,
        use_vmap=args.use_vmap,
        cache_data=args.cache_data,
        verbose=True,
    )
    raw_results = raw_loss_data_estimator.compute_curve(n_points=args.points)

    # define our autoencoder
    ae = AutoEncoder(30, False, False, 0.001, "bruh", pre_latent_dim=49)
    vae_repr = ae.encode(data_x)

    # define the classification algorithm for our representation
    init_fn, train_step_fn, eval_fn = alg.make_algorithm((30,), 10)
    vae_loss_data_estimator = reprieve.LossDataEstimator(
        init_fn,
        train_step_fn,
        eval_fn,
        (vae_repr, data_y),
        train_steps=args.train_steps,
        n_seeds=args.seeds,
        use_vmap=args.use_vmap,
        cache_data=args.cache_data,
        verbose=True,
    )
    vae_results = vae_loss_data_estimator.compute_curve(n_points=args.points)

    raw_results["name"] = "Raw"
    vae_results["name"] = "Our-VAE"

    outcome_df = pd.concat(
        [
            raw_results,
            vae_results,
        ]
    )

    os.makedirs("results", exist_ok=True)
    save_path = (
        "results/"
        f"{args.name}"
        f"_train{args.train_steps}"
        f"_seed{args.seeds}"
        f"_point{args.points}"
    )

    ns = [
        10,
        100,
        1000,
        10000,
    ]  # the list of training set sizes to use for computing metrics
    epsilons = [1, 0.2]  # the settings of epsilon used for computing SDL and eSC
    reprieve.render_curve(outcome_df, ns, epsilons, save_path=save_path + ".pdf")
    metrics_df = reprieve.compute_metrics(outcome_df, ns, epsilons)
    reprieve.render_latex(metrics_df, save_path=save_path + ".tex")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="jax")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_vmap", dest="use_vmap", action="store_false")
    parser.add_argument("--no_cache", dest="cache_data", action="store_false")
    parser.add_argument("--train_steps", type=float, default=4e3)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--points", type=int, default=10)
    args = parser.parse_args()

    import time

    start = time.time()
    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)
    end = time.time()
    print(f"Time: {end - start :.3f} seconds")
