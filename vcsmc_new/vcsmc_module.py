import tensorflow as tf

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

        self.N = tf.constant(N)
        self.K = tf.constant(K)
        self.A = tf.constant(16)

        self.reg_lambda_stat_probs = reg_lambda_stat_probs
        self.reg_lambda_branch_params = reg_lambda_branch_params

        # variables

        initial_branches_log = tf.constant(
            tf.math.log(float(branch_prior)), shape=[self.N - 1], dtype=DTYPE_FLOAT
        )

        self._lb_params = tf.Variable(
            initial_branches_log, name="left branch parameters"
        )
        self._rb_params = tf.Variable(
            initial_branches_log, name="right branch parameters"
        )

        self._nucleotide_exchanges = tf.Variable(
            tf.constant(0, shape=[5], dtype=DTYPE_FLOAT),
            name="nucleotide exchangeabilities",
        )
        self._stat_probs = tf.Variable(
            tf.constant(0, shape=[15], dtype=DTYPE_FLOAT),
            name="stationary probabilities",
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
        (
            lb_params,
            rb_params,
            nucleotide_exchanges,
            stat_probs,
            Q,
            regularization,
        ) = self.get_variables()

        return Q


if __name__ == "__main__":
    test = VcsmcModule(N=10, K=10, branch_prior=1)
    test(None)
    print(test.trainable_variables)
