import random
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
from vcsmc_module import VcsmcModule
import tensorflow as tf
import keras
from typing import Any


def batch_data(data, batch_size):
    sites = data.shape[2]
    sites_list = list(range(sites))
    num_batches = sites // batch_size
    slices = []

    for _ in range(num_batches):
        sampled_indices = random.sample(sites_list, batch_size)
        slices.append(sampled_indices)
        sites_list = list(set(sites_list) - set(sampled_indices))

    if len(sites_list) != 0:
        slices.append(sites_list)

    batches = [np.take(data, slice, axis=2) for slice in slices]
    return batches


def create_save_dir(args):
    # Create local directory and save experiment results
    tm = str(datetime.now())
    local_rlt_root = (
        "./results/"
        + str(args.dataset)
        + "/"
        + str(args.nested)
        + "/"
        + str(args.n_particles)
        + "/"
    )
    save_dir = (
        local_rlt_root + (tm[:10] + "-" + tm[11:13] + tm[14:16] + tm[17:19]) + "/"
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir


def write_run_parameters(args, initial_elbo, optimizer, save_dir):
    with open(save_dir + "run_parameters.txt", "w") as rp:
        rp.write("Initial evaluation of ELBO : " + str(initial_elbo))
        rp.write("\n")
        for k, v in vars(args).items():
            rp.write(str(k) + " : " + str(v))
            rp.write("\n")
        rp.write("Optimizer : " + str(optimizer))
        rp.close()


def get_optimizer(optimizer: str, lr: float):
    # legacy optimizers are faster on Apple M1/M2/M3
    optimizers = {
        "Adadelta": keras.optimizers.legacy.Adadelta(learning_rate=lr),
        "Adagrad": keras.optimizers.legacy.Adagrad(learning_rate=lr),
        "Adamax": keras.optimizers.legacy.Adamax(learning_rate=lr),
        "Adam": keras.optimizers.legacy.Adam(learning_rate=lr),
        "Ftrl": keras.optimizers.legacy.Ftrl(learning_rate=lr),
        "Nadam": keras.optimizers.legacy.Nadam(learning_rate=lr),
        "RMSprop": keras.optimizers.legacy.RMSprop(learning_rate=lr),
        "SGD": keras.optimizers.legacy.SGD(learning_rate=lr),
    }

    return optimizers[optimizer]


@tf.function
def train_step(vcsmc, optimizer, data):
    with tf.GradientTape() as tape:
        result = vcsmc(data)

    grads = tape.gradient(result["cost"], vcsmc.trainable_variables)
    optimizer.apply_gradients(zip(grads, vcsmc.trainable_variables))  # type: ignore

    return result


def train(
    genome_NxSxA: np.ndarray,
    *,
    K: int,
    branch_prior: float,
    epochs: int,
    batch_size: int,
    lr: float,
    optimizer: str,
    args,
):
    N = len(genome_NxSxA)

    data = np.array([genome_NxSxA] * K, dtype=np.float32)
    batches = batch_data(data, batch_size)

    print("Dataset shape (KxNxSxA):", data.shape)

    vcsmc = VcsmcModule(N=N, K=K, branch_prior=branch_prior)

    initial_result: dict[str, Any] = vcsmc(data)  # type: ignore

    print("Initial ELBO:", initial_result["elbo"].numpy())
    print("Trainable variables:", vcsmc.trainable_variables)

    save_dir = create_save_dir(args)
    write_run_parameters(args, initial_result["elbo"], optimizer, save_dir)

    print("Training begins...")

    summary_writer = tf.summary.create_file_writer(save_dir)  # type: ignore

    elbo_list = []
    Q_list = []
    left_branches_list = []
    right_branches_list = []
    jump_chains_list = []
    log_weights_list = []
    log_likelihood_list = []
    log_likelihood_tilde_list = []
    log_likelihood_R_list = []
    stat_probs_list = []
    nucleotide_exchanges_list = []
    mean_branch_lengths_list = []

    opt = get_optimizer(optimizer, lr)

    for epoch in tqdm(range(epochs)):
        # track averages for relevant metrics across batches
        cost_avg = keras.metrics.Mean()
        elbo_avg = keras.metrics.Mean()
        regularization_avg = keras.metrics.Mean()

        result: dict[str, Any] = {}

        for batch in tqdm(batches):
            result = train_step(vcsmc, opt, batch)  # type: ignore

            cost_avg.update_state(result["cost"])
            elbo_avg.update_state(result["elbo"])
            regularization_avg.update_state(result["regularization"])

        cost = cost_avg.result().numpy()
        elbo = elbo_avg.result().numpy()
        regularization = regularization_avg.result().numpy()

        mean_branch_length = (
            np.mean(result["lb_params"].numpy()) + np.mean(result["rb_params"].numpy())
        ) / 2

        elbo_list.append(elbo)
        Q_list.append(result["Q"].numpy())
        left_branches_list.append(result["left_branches"].numpy())
        right_branches_list.append(result["right_branches"].numpy())
        log_likelihood_list.append(result["log_likelihood"].numpy())
        log_likelihood_tilde_list.append(result["log_likelihood_tilde"].numpy())
        log_likelihood_R_list.append(result["log_likelihood_R"].numpy())
        log_weights_list.append(result["log_weights"].numpy())
        jump_chains_list.append(result["jump_chains"].numpy())
        stat_probs_list.append(result["stat_probs"].numpy())
        nucleotide_exchanges_list.append(result["nucleotide_exchanges"].numpy())
        mean_branch_lengths_list.append(mean_branch_length)

        with summary_writer.as_default(step=epoch):
            tf.summary.scalar("Cost", cost)
            tf.summary.scalar("ELBO", elbo)
            tf.summary.scalar("Regularization", regularization)
            tf.summary.scalar("Mean branch length", mean_branch_length)
            tf.summary.histogram("Stationary probabilities", result["stat_probs"])
            tf.summary.histogram("Exchangeability", result["nucleotide_exchanges"])
            tf.summary.histogram("LB param", result["lb_params"])
            tf.summary.histogram("RB param", result["rb_params"])
            summary_writer.flush()

        print("Epoch", epoch)
        print("ELBO:", round(elbo, 3))
        print("Regularization:", round(regularization, 3))
        print("Mean branch length:", round(mean_branch_length, 3))
        print("Stationary probabilities:", result["stat_probs"].numpy())
        print("Exchangeability:", result["nucleotide_exchanges"].numpy())
        print("LB param:", result["lb_params"].numpy())
        print("RB param:", result["rb_params"].numpy())
        print()

    print("Training finished!")

    plt.imshow(Q_list[-1])
    plt.title("Trained Q matrix")
    plt.savefig(save_dir + "Qmatrix.png")

    plt.figure(figsize=(10, 10))
    plt.plot(elbo_list)
    plt.ylabel("log $Z_{SMC}$")
    plt.xlabel("Epochs")
    plt.title("Elbo convergence across epochs")
    plt.savefig(save_dir + "ELBO.png")

    plt.figure(figsize=(10, 10))
    myll = np.asarray(log_likelihood_R_list)
    plt.plot(myll[:, :], c="black", alpha=0.2)
    plt.plot(np.average(myll[:, :], axis=1), c="yellow")
    plt.ylabel("log likelihood")
    plt.xlabel("Epochs")
    plt.title("Log likelihood convergence across epochs")
    plt.savefig(save_dir + "ll.png")

    plt.figure(figsize=(10, 10))
    plt.plot(mean_branch_lengths_list)
    plt.ylabel("Mean branch length")
    plt.xlabel("Epochs")
    plt.title("Mean branch length across epochs")
    plt.savefig(save_dir + "branch_lengths.png")

    # Save best log-likelihood value and jump chain
    best_log_lik = np.asarray(log_likelihood_R_list)[np.argmax(elbo_list)]
    print("Best log likelihood values:\n", best_log_lik)
    best_jump_chain = jump_chains_list[np.argmax(elbo_list)]

    resultDict = {
        "elbos": np.asarray(elbo_list),
        "nParticles": K,
        "nTaxa": N,
        "lr": lr,
        "log_weights": np.asarray(log_weights_list),
        "Qmatrices": np.asarray(Q_list),
        "left_branches": left_branches_list,
        "right_branches": right_branches_list,
        "log_lik": np.asarray(log_likelihood_list),
        "ll_tilde": np.asarray(log_likelihood_tilde_list),
        "log_lik_R": np.asarray(log_likelihood_R_list),
        "jump_chain_evolution": jump_chains_list,
        "best_epoch": np.argmax(elbo_list),
        "best_log_lik": best_log_lik,
        "best_jump_chain": best_jump_chain,
        "stationary_probs": np.asarray(stat_probs_list),
        "exchangeabilities": np.asarray(nucleotide_exchanges_list),
    }

    with open(save_dir + "results.p", "wb") as f:
        # pdb.set_trace()
        pickle.dump(resultDict, f)
