import tensorflow as tf
import numpy as np
from vcsmc_module import VcsmcModule


def train(
    genome_NxSxA: np.ndarray,
    *,
    K: int,
    branch_prior: float,
    epochs: int,
    batch_size: int,
    lr: float
):
    N = len(genome_NxSxA)

    data = tf.constant([genome_NxSxA] * K)

    vcsmc = VcsmcModule(N=N, K=K, branch_prior=branch_prior)
    vcsmc(data)
