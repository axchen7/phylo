import random
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
from vcsmc_module import VcsmcModule
import tensorflow as tf
import keras


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

    initial_list = vcsmc(data)
    initial_elbo = initial_list[1]
    initial_jc = initial_list[2][0]

    print("Initial ELBO:", initial_elbo)
    print("Initial jump chain:", initial_jc)
    print("Trainable variables:", vcsmc.trainable_variables)

    save_dir = create_save_dir(args)
    write_run_parameters(args, initial_elbo, optimizer, save_dir)

    print("Training begins...")

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

    if optimizer == "Adam":
        opt = keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = keras.optimizers.legacy.SGD(learning_rate=lr)  # TODO

    for epoch in tqdm(range(epochs)):
        # SKIP BATCHING!!!

        # for batch in batches:
        #     with tf.GradientTape() as tape:
        #         cost = vcsmc(batch)[0]
        #         elbo = vcsmc(batch)[1]

        #     grads = tape.gradient(cost, vcsmc.trainable_variables)
        #     opt.apply_gradients(zip(grads, vcsmc.trainable_variables))

        with tf.GradientTape() as tape:
            (
                cost,
                elbo,
                jump_chains,
                regularization,
                stat_probs,
                Q,
                left_branches,
                right_branches,
                log_weights,
                log_likelihood,
                log_likelihood_tilde,
                log_likelihood_R,
                v_minus,
                lb_params,
                rb_params,
                nucleotide_exchanges,
            ) = vcsmc(data)

        grads = tape.gradient(cost, vcsmc.trainable_variables)
        opt.apply_gradients(zip(grads, vcsmc.trainable_variables))

        mean_branch_length = np.mean(lb_params.numpy()) + np.mean(rb_params.numpy())
        mean_branch_length /= 2

        elbo_list.append(elbo.numpy())
        Q_list.append(Q.numpy())
        left_branches_list.append(left_branches.numpy())
        right_branches_list.append(right_branches.numpy())
        log_likelihood_list.append(log_likelihood.numpy())
        log_likelihood_tilde_list.append(log_likelihood_tilde.numpy())
        log_likelihood_R_list.append(log_likelihood_R.numpy())
        log_weights_list.append(log_weights.numpy())
        jump_chains_list.append(jump_chains.numpy())
        stat_probs_list.append(stat_probs.numpy())
        nucleotide_exchanges_list.append(nucleotide_exchanges.numpy())
        mean_branch_lengths_list.append(mean_branch_length)

        print("Epoch", epoch + 1)
        print("ELBO:", round(elbo.numpy(), 3))
        print("Regularization:", round(regularization.numpy(), 3))
        print("Stationary probabilities:", stat_probs.numpy())
        print("Exchangeability:", nucleotide_exchanges.numpy())
        print("LB param:", lb_params.numpy())
        print("RB param:", rb_params.numpy())
        print("Mean branch length:", round(mean_branch_length, 3))
        print()
