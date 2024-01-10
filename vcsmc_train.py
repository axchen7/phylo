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

    data = np.array([genome_NxSxA] * K, dtype=np.float32)

    vcsmc = VcsmcModule(N=N, K=K, branch_prior=branch_prior)
    vcsmc(data)
