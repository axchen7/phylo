import tensorflow as tf
import numpy as np

DTYPE_INT = tf.int32
DTYPE_FLOAT = tf.float32


class VcsmcModule(tf.Module):
    """For now, only CellPhy GT16 with no error is implemented."""

    def __init__(
        self,
        *,
        N: int,
        K: int,
        branch_prior: float,
        reg_lambda_stat_probs: float = 1e4,
        reg_lambda_branch_params: float = 1e2,
    ):
        super().__init__()

        # constants

        self.N = N
        self.K = K
        self.A = 16

        self.taxa = [str(i) for i in range(self.N)]

        self.reg_lambda_stat_probs = reg_lambda_stat_probs
        self.reg_lambda_branch_params = reg_lambda_branch_params

        # variables

        initial_branches_log = np.zeros(self.N - 1) + np.log(branch_prior)

        self._lb_params = tf.Variable(
            initial_branches_log, dtype=DTYPE_FLOAT, name="left branch parameters"
        )
        self._rb_params = tf.Variable(
            initial_branches_log, dtype=DTYPE_FLOAT, name="right branch parameters"
        )

        self._nucleotide_exchanges = tf.Variable(
            np.zeros(5), dtype=DTYPE_FLOAT, name="nucleotide exchangeabilities"
        )
        self._stat_probs = tf.Variable(
            np.zeros(15), dtype=DTYPE_FLOAT, name="stationary probabilities"
        )

    @tf.function
    def get_variables(self):
        """
        Transforms model variables into values that are used in the actual computations.
        """

        # branch lengths
        lb_params = tf.exp(self._lb_params)
        rb_params = tf.exp(self._rb_params)

        # nucleotide exchangeabilities

        # 6th exchangeability can be fixed to match degrees of freedom
        nucleotide_exchanges = tf.concat(
            [self._nucleotide_exchanges, [tf.constant(0, dtype=DTYPE_FLOAT)]],
            axis=0,
        )
        # use exp to ensure all entries are positive
        nucleotide_exchanges = tf.exp(nucleotide_exchanges)
        # normalize to ensure mean is 1
        nucleotide_exchanges /= tf.reduce_mean(nucleotide_exchanges)

        # stationary probabilities

        # 16th stationary probability can be fixed to match degrees of freedom
        stat_probs = tf.concat(
            [self._stat_probs, [tf.constant(0, dtype=DTYPE_FLOAT)]], axis=0
        )
        # use softmax to ensure all entries are positive
        stat_probs = tf.exp(stat_probs)
        # normalize to ensure sum is 1
        stat_probs /= tf.reduce_sum(stat_probs)

        # Q matrix
        Q = self.get_Q_GT16(nucleotide_exchanges, stat_probs)

        # regularization
        sum_square_stat_probs = tf.reduce_sum(tf.square(stat_probs))
        square_mean_branch_params = tf.square(
            tf.reduce_mean(tf.concat([lb_params, rb_params], axis=0))
        )
        regularization = (
            self.reg_lambda_stat_probs * sum_square_stat_probs
            + self.reg_lambda_branch_params * square_mean_branch_params
        )

        # return everything
        return lb_params, rb_params, nucleotide_exchanges, stat_probs, Q, regularization

    @tf.function
    def get_Q_GT16(self, nucleotide_exchanges, stat_probs):
        """
        Forms the transition matrix using the CellPhy GT16 model. Assumes A=16.
        """

        pi = nucleotide_exchanges  # length 6
        pi8 = tf.repeat(pi, 8)

        # index helpers for Q matrix
        AA, CC, GG, TT, AC, AG, AT, CG, CT, GT, CA, GA, TA, GC, TC, TG = range(16)

        # fmt: off
        updates = [
          # | first base changes                    | second base changes
            [AA, CA], [AC, CC], [AG, CG], [AT, CT], [AA, AC], [CA, CC], [GA, GC], [TA, TC], # A->C
            [AA, GA], [AC, GC], [AG, GG], [AT, GT], [AA, AG], [CA, CG], [GA, GG], [TA, TG], # A->G
            [AA, TA], [AC, TC], [AG, TG], [AT, TT], [AA, AT], [CA, CT], [GA, GT], [TA, TT], # A->T
            [CA, GA], [CC, GC], [CG, GG], [CT, GT], [AC, AG], [CC, CG], [GC, GG], [TC, TG], # C->G
            [CA, TA], [CC, TC], [CG, TG], [CT, TT], [AC, AT], [CC, CT], [GC, GT], [TC, TT], # C->T
            [GA, TA], [GC, TC], [GG, TG], [GT, TT], [AG, AT], [CG, CT], [GG, GT], [TG, TT], # G->T
        ]
        # fmt: on

        R = tf.scatter_nd(updates, pi8, [16, 16])
        R = R + tf.transpose(R)

        y_q = tf.matmul(R, tf.linalg.diag(stat_probs))
        hyphens = tf.reduce_sum(y_q, axis=1)

        Q = tf.linalg.set_diag(y_q, -hyphens)
        return Q

    @tf.function
    def __call__(self, data_NxSxA):
        """
        Main sampling routine that performs combinatorial SMC by calling the rank update subroutine
        """

        (
            lb_params,
            rb_params,
            nucleotide_exchanges,
            stat_probs,
            Q,
            regularization,
        ) = self.get_variables()

        # accumulated values used by subsequent iteration

        leafnode_num_record_KxN = tf.Variable(
            tf.constant(1, shape=[self.K, self.N]), dtype=DTYPE_INT
        )
        v_minus_K = tf.Variable(tf.constant(1, shape=[self.K]), dtype=DTYPE_INT)

        log_weights_NxK = tf.TensorArray(
            DTYPE_FLOAT, size=self.N, element_shape=[self.K]
        )
        # TODO init log_weights_NxK with .write() ?

        jump_chains_NxKxV = tf.TensorArray(  # V: variable dimension
            dtype=tf.dtypes.string, size=self.N, element_shape=[self.K, None]
        )

        # accumulated values not used by subsequent iteration
        # (0th entry is unused)

        left_branches_NxK = tf.TensorArray(
            DTYPE_FLOAT, size=self.N, element_shape=[self.K]
        )
        right_branches_NxK = tf.TensorArray(
            DTYPE_FLOAT, size=self.N, element_shape=[self.K]
        )

        log_likelihood_NxK = tf.TensorArray(
            DTYPE_FLOAT, size=self.N, element_shape=[self.K]
        )
        log_likelihood_tilde_NxK = tf.TensorArray(
            DTYPE_FLOAT, size=self.N, element_shape=[self.K]
        ).unstack([tf.constant(np.zeros(self.K) + np.log(1 / self.K))])

        # main loop (N-1 iterations)
        for r in tf.range(1, self.N):
            # resample?
            if r == 1:
                # the first time, just initialize
                initial_jc_K_V = tf.constant([self.taxa] * self.K)
                jump_chains_NxKxV.write(r, initial_jc_K_V)
            else:
                # resample partial states by drawing from a categorical
                # distribution whose parameters are normalized importance
                # weights
                log_weights_K = log_weights_NxK.read(r - 1)
                log_normalized_weights_K = log_weights_K - tf.reduce_logsumexp(
                    log_weights_K
                )
                indices_K = tf.random.categorical(
                    tf.expand_dims(log_normalized_weights_K, axis=0), self.K
                )[0]

        return Q


if __name__ == "__main__":
    test = VcsmcModule(N=10, K=10, branch_prior=1)
    test(None)
