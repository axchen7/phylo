import tensorflow as tf
import tensorflow_probability as tfp
import math

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
        """
        Args:
            N: Number of taxa.
            K: Number of particles.
            branch_prior: Prior on branch lengths.
            reg_lambda_stat_probs: Regularization parameter for stationary probabilities.
            reg_lambda_branch_params: Regularization parameter for branch lengths.
        """

        super().__init__()

        # constants

        self.N = int(N)
        self.K = int(K)
        self.A = 16

        self.reg_lambda_stat_probs = float(reg_lambda_stat_probs)
        self.reg_lambda_branch_params = float(reg_lambda_branch_params)

        self.taxa = ["S" + str(i) for i in range(N)]

        # variables

        initial_branches_log = tf.constant(
            math.log(branch_prior), shape=[self.N - 1], dtype=DTYPE_FLOAT
        )

        self._lb_params = tf.Variable(
            initial_branches_log, dtype=DTYPE_FLOAT, name="left_branch_parameters"
        )
        self._rb_params = tf.Variable(
            initial_branches_log, dtype=DTYPE_FLOAT, name="right_branch_parameters"
        )

        self._nucleotide_exchanges = tf.Variable(
            tf.constant(0, shape=[5], dtype=DTYPE_FLOAT),
            name="nucleotide_exchangeabilities",
        )
        self._stat_probs = tf.Variable(
            tf.constant(0, shape=[15], dtype=DTYPE_FLOAT),
            name="stationary_probabilities",
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
    def ncr(self, n, r):
        """Compute combinatorial term n choose r."""
        numer = tf.reduce_prod(tf.range(n - r + 1, n + 1))
        denom = tf.reduce_prod(tf.range(1, r + 1))
        return numer // denom

    @tf.function
    def _double_factorial_loop_body(self, n, result, two):
        result = tf.where(
            tf.greater_equal(n, two),
            result + tf.math.log(tf.cast(n, DTYPE_FLOAT)),
            result,
        )
        return n - two, result, two

    @tf.function
    def _double_factorial_loop_condition(self, n, result, two):
        del result  # Unused
        return tf.cast(tf.math.count_nonzero(tf.greater_equal(n, two)), tf.bool)

    @tf.function
    def log_double_factorial(self, n):
        """
        Computes the double factorial of `n`. Note:
            In the following, A1 to An are optional batch dimensions.

        Args:
            n: A tensor of shape `[A1, ..., An]` containing positive integer
            values.

        Returns:
            A tensor of shape `[A1, ..., An]` containing the double factorial of
            `n`.
        """
        two = tf.ones_like(n) * 2
        result = tf.math.log(tf.ones_like(n, dtype=DTYPE_FLOAT))
        _, result, _ = tf.while_loop(  # type: ignore
            cond=self._double_factorial_loop_condition,
            body=self._double_factorial_loop_body,
            loop_vars=[n, result, two],
        )
        return result

    @tf.function
    def gather_across_2d(self, a, idx, a_shape_1, idx_shape_1):
        """
        Gathers as such:
            if a is K-by-N, idx is K-by-M, then it returns a Tensor with
            structure like [tf.gather(a[k], idx[k]) for k in range(K)]. But it
            broadcasts and doesn't actually use for-loop.
        """

        K = a.shape[0]
        a_reshaped = tf.reshape(a, [K * a_shape_1, -1])
        add_to_idx = a_shape_1 * tf.transpose(
            tf.tile(tf.expand_dims(tf.range(K), axis=0), [idx_shape_1, 1])
        )
        a_gathered = tf.gather(a_reshaped, idx + add_to_idx)
        a_gathered = tf.reshape(a_gathered, [K, -1])
        return a_gathered

    @tf.function
    def gather_across_core(self, a, idx, a_shape_1, idx_shape_1, A):
        """
        Gathers from the core as such:
            if a is K-by-N-by-S-by-A, idx is K-by-M, then it returns a Tensor
            with structure like [tf.gather(a[k], idx[k]) for k in range(K)].
            But it broadcasts and doesn't actually use for-loop.
        """

        K = a.shape[0]
        a_reshaped = tf.reshape(a, [K * a_shape_1, -1, A])
        add_to_idx = a_shape_1 * tf.transpose(
            tf.tile(tf.expand_dims(tf.range(K), axis=0), [idx_shape_1, 1])
        )
        a_gathered = tf.gather(a_reshaped, idx + add_to_idx)
        a_gathered = tf.reshape(a_gathered, [K, idx_shape_1, -1, A])
        return a_gathered

    @tf.function
    def broadcast_conditional_likelihood_K(
        self, Q, l_data_KxSxA, r_data_KxSxA, l_branch_samples_K, r_branch_samples_K
    ):
        left_message_KxAxA = tf.tensordot(l_branch_samples_K, Q, axes=0)
        right_message_KxAxA = tf.tensordot(r_branch_samples_K, Q, axes=0)
        left_Pmat_KxAxA = tf.linalg.expm(left_message_KxAxA)
        right_Pmat_KxAxA = tf.linalg.expm(right_message_KxAxA)
        left_prob_KxSxA = tf.matmul(l_data_KxSxA, left_Pmat_KxAxA)
        right_prob_KxSxA = tf.matmul(r_data_KxSxA, right_Pmat_KxAxA)
        likelihood_KxSxA = left_prob_KxSxA * right_prob_KxSxA
        return likelihood_KxSxA

    @tf.function
    def compute_forest_posterior(
        self, stat_probs, data_KxXxSxA, leafnode_num_record, r
    ):
        """
        Forms a log probability measure by dotting the stationary probs with
        tree likelihood and add that to log-prior of tree topology.

        Note:
            We add log-prior of branch-lengths in body_update_weights.
        """

        data_reshaped = tf.reshape(
            data_KxXxSxA, (self.K * (self.N - r - 1), -1, self.A)
        )
        stat_probs_transposed = tf.transpose(tf.expand_dims(stat_probs, axis=0))
        stationary_probs = tf.tile(
            tf.expand_dims(stat_probs_transposed, axis=0),
            [self.K * (self.N - r - 1), 1, 1],
        )
        forest_lik = tf.squeeze(tf.matmul(data_reshaped, stationary_probs))
        forest_lik = tf.reshape(forest_lik, (self.K, self.N - r - 1, -1))
        forest_loglik = tf.reduce_sum(tf.math.log(forest_lik), axis=(1, 2))
        forest_logprior = tf.reduce_sum(
            -self.log_double_factorial(2 * tf.maximum(leafnode_num_record, 2) - 3),
            axis=1,
        )

        return forest_loglik + forest_logprior

    @tf.function
    def overcounting_correct(self, leafnode_num_record):
        """
        Computes overcounting correction term to the proposal distribution.
        """
        v_minus = tf.reduce_sum(
            leafnode_num_record
            - tf.cast(tf.equal(leafnode_num_record, 1), leafnode_num_record.dtype),
            axis=1,
        )
        return v_minus

    @tf.function
    def get_log_likelihood(
        self, log_likelihood, left_branches, right_branches, lb_params, rb_params
    ):
        """
        Computes last rank-event's log_likelihood P(Y|t, theta) by removing prior from
        the already computed log_likelihood, which includes prior.
        """
        l_exponent = tf.multiply(
            tf.transpose(left_branches), tf.expand_dims(lb_params, axis=0)
        )
        r_exponent = tf.multiply(
            tf.transpose(right_branches), tf.expand_dims(rb_params, axis=0)
        )
        l_multiplier = tf.expand_dims(tf.math.log(lb_params), axis=0)
        r_multiplier = tf.expand_dims(tf.math.log(rb_params), axis=0)
        left_branches_logprior = tf.reduce_sum(l_multiplier - l_exponent, axis=1)
        right_branches_logprior = tf.reduce_sum(r_multiplier - r_exponent, axis=1)
        log_likelihood_R = (
            tf.gather(log_likelihood, self.N - 2)
            + self.log_double_factorial(2 * self.N - 3)
            - left_branches_logprior
            - right_branches_logprior
        )
        return log_likelihood_R

    @tf.function
    def compute_log_ZSMC(self, log_weights):
        """
        Forms the estimator log_ZSMC, a multi sample lower bound to the
        likelihood. Z_SMC is formed by averaging over weights and multiplying
        over coalescent events.
        """
        log_Z_SMC = tf.reduce_sum(
            tf.reduce_logsumexp(
                log_weights - tf.math.log(tf.cast(self.K, DTYPE_FLOAT)), axis=1
            )
        )
        return log_Z_SMC

    @tf.function
    def resample(self, core, leafnode_num_record, JC_K, log_weights):
        """
        Resample partial states by drawing from a categorical distribution whose
        parameters are normalized importance weights. JumpChain (JC_K) is a
        tensor formed from a numpy array of lists of strings, returns a
        resampled JumpChain tensor.
        """
        log_normalized_weights = log_weights - tf.reduce_logsumexp(log_weights)
        indices = tf.squeeze(tf.random.categorical([log_normalized_weights], self.K))
        resampled_core = tf.gather(core, indices)
        resampled_record = tf.gather(leafnode_num_record, indices)
        resampled_JC_K = tf.gather(JC_K, indices)
        return resampled_core, resampled_record, resampled_JC_K, indices

    @tf.function
    def extend_partial_state(self, JCK, r):
        """
        Extends partial state by sampling two states to coalesce (Gumbel-max
        trick to sample without replacement). JumpChain (JC_K) is a tensor
        formed from a numpy array of lists of strings, returns a new JumpChain
        tensor.
        """

        # Compute combinatorial term
        ncr = self.ncr(self.N - r, 2)
        q = tf.constant(1, DTYPE_FLOAT) / tf.cast(ncr, DTYPE_FLOAT)
        # Gumbel-max trick to sample without replacement
        z = -tf.math.log(-tf.math.log(tf.random.uniform([self.K, self.N - r], 0, 1)))
        top_values, coalesced_indices = tf.nn.top_k(z, 2)
        bottom_values, remaining_indices = tf.nn.top_k(tf.negative(z), self.N - r - 2)
        JC_keep = tf.gather(tf.reshape(JCK, [self.K * (self.N - r)]), remaining_indices)
        particles = tf.gather(
            tf.reshape(JCK, [self.K * (self.N - r)]), coalesced_indices
        )
        particle1 = particles[:, 0]
        particle2 = particles[:, 1]
        # Form new state
        particle_coalesced = particle1 + "+" + particle2
        # Form new Jump Chain
        JCK = tf.concat([JC_keep, tf.expand_dims(particle_coalesced, axis=1)], axis=1)

        # return particle1, particle2, particle_coalesced, coalesced_indices, remaining_indices, q, JCK
        return coalesced_indices, remaining_indices, q, JCK

    @tf.function
    def cond_true_resample(
        self,
        log_likelihood_tilde,
        core,
        leafnode_num_record,
        log_weights,
        log_likelihood,
        jump_chains,
        jump_chain_tensor,
        r,
    ):
        core, leafnode_num_record, jump_chain_tensor, indices = self.resample(  # type: ignore
            core, leafnode_num_record, jump_chain_tensor, tf.gather(log_weights, r)
        )
        log_likelihood_tilde = tf.gather_nd(
            tf.gather(tf.transpose(log_likelihood), indices),
            [[k, r] for k in range(self.K)],
        )
        jump_chains = tf.concat([jump_chains, jump_chain_tensor], axis=1)
        return (
            log_likelihood_tilde,
            core,
            leafnode_num_record,
            jump_chains,
            jump_chain_tensor,
        )

    @tf.function
    def cond_false_resample(
        self,
        log_likelihood_tilde,
        core,
        leafnode_num_record,
        log_weights,
        log_likelihood,
        jump_chains,
        jump_chain_tensor,
        r,
    ):
        jump_chains = tf.concat([jump_chains, jump_chain_tensor], axis=1)
        return (
            log_likelihood_tilde,
            core,
            leafnode_num_record,
            jump_chains,
            jump_chain_tensor,
        )

    @tf.function
    def body_rank_update(
        self,
        lb_params,
        rb_params,
        stat_probs,
        Q,
        log_weights,
        log_likelihood,
        log_likelihood_tilde,
        jump_chains,
        jump_chain_tensor,
        core,
        leafnode_num_record,
        left_branches,
        right_branches,
        v_minus,
        r,
    ):
        """
        Define tensors for log_weights, log_likelihood, jump_chain_tensor and core (state data for distribution over characters for ancestral taxa)
        by iterating over rank events.
        """

        # Resample
        (
            log_likelihood_tilde,
            core,
            leafnode_num_record,
            jump_chains,
            jump_chain_tensor,
        ) = tf.cond(  # type: ignore
            r > 0,
            # TODO lambda???
            lambda: self.cond_true_resample(
                log_likelihood_tilde,
                core,
                leafnode_num_record,
                log_weights,
                log_likelihood,
                jump_chains,
                jump_chain_tensor,
                r,
            ),
            lambda: self.cond_false_resample(
                log_likelihood_tilde,
                core,
                leafnode_num_record,
                log_weights,
                log_likelihood,
                jump_chains,
                jump_chain_tensor,
                r,
            ),
        )

        # Extend partial states
        (
            coalesced_indices,
            remaining_indices,
            q_log_proposal,
            jump_chain_tensor,
        ) = self.extend_partial_state(jump_chain_tensor, r)

        # Branch lengths
        left_branches_param_r = tf.gather(lb_params, r)
        right_branches_param_r = tf.gather(rb_params, r)
        q_l_branch_dist = tfp.distributions.Exponential(rate=left_branches_param_r)
        q_r_branch_dist = tfp.distributions.Exponential(rate=right_branches_param_r)
        q_l_branch_samples = q_l_branch_dist.sample(self.K)
        q_r_branch_samples = q_r_branch_dist.sample(self.K)
        left_branches = tf.concat([left_branches, [q_l_branch_samples]], axis=0)
        right_branches = tf.concat([right_branches, [q_r_branch_samples]], axis=0)

        # Update partial set data
        remaining_core = self.gather_across_core(
            core, remaining_indices, self.N - r, self.N - r - 2, self.A
        )  # Kx(N-r-2)xSxA
        l_coalesced_indices = tf.reshape(
            tf.gather(tf.transpose(coalesced_indices), 0), (self.K, 1)
        )
        r_coalesced_indices = tf.reshape(
            tf.gather(tf.transpose(coalesced_indices), 1), (self.K, 1)
        )
        l_data_KxSxA = tf.squeeze(
            self.gather_across_core(core, l_coalesced_indices, self.N - r, 1, self.A)
        )
        r_data_KxSxA = tf.squeeze(
            self.gather_across_core(core, r_coalesced_indices, self.N - r, 1, self.A)
        )
        new_mtx_KxSxA = self.broadcast_conditional_likelihood_K(
            Q, l_data_KxSxA, r_data_KxSxA, q_l_branch_samples, q_r_branch_samples
        )
        new_mtx_Kx1xSxA = tf.expand_dims(new_mtx_KxSxA, axis=1)
        core = tf.concat([remaining_core, new_mtx_Kx1xSxA], axis=1)  # Kx(N-r-1)xSxA

        remaining_leafnode_num_record = self.gather_across_2d(
            leafnode_num_record, remaining_indices, self.N - r, self.N - r - 2
        )
        new_leafnode_num = tf.expand_dims(
            tf.reduce_sum(
                self.gather_across_2d(
                    leafnode_num_record, coalesced_indices, self.N - r, 2
                ),
                axis=1,
            ),
            axis=1,
        )
        leafnode_num_record = tf.concat(
            [remaining_leafnode_num_record, new_leafnode_num], axis=1
        )

        # Compute weights
        log_likelihood_r = self.compute_forest_posterior(
            stat_probs, core, leafnode_num_record, r
        )

        left_branches_select = tf.gather(left_branches, tf.range(1, r + 2))  # (r+1)xK
        right_branches_select = tf.gather(right_branches, tf.range(1, r + 2))  # (r+1)xK
        left_branches_logprior = tf.reduce_sum(
            -left_branches_param_r * left_branches_select
            + tf.math.log(left_branches_param_r),
            axis=0,
        )
        right_branches_logprior = tf.reduce_sum(
            -right_branches_param_r * right_branches_select
            + tf.math.log(right_branches_param_r),
            axis=0,
        )
        log_likelihood_r = (
            log_likelihood_r + left_branches_logprior + right_branches_logprior
        )

        v_minus = self.overcounting_correct(leafnode_num_record)
        l_branch = tf.gather(left_branches, r + 1)
        r_branch = tf.gather(right_branches, r + 1)

        log_weights_r = (
            log_likelihood_r
            - log_likelihood_tilde
            - (
                tf.math.log(left_branches_param_r)
                - left_branches_param_r * l_branch
                + tf.math.log(right_branches_param_r)
                - right_branches_param_r * r_branch
            )
            + tf.math.log(tf.cast(v_minus, DTYPE_FLOAT))
            - q_log_proposal
        )

        log_weights = tf.concat([log_weights, [log_weights_r]], axis=0)
        log_likelihood = tf.concat(
            [log_likelihood, [log_likelihood_r]], axis=0
        )  # pi(t) = pi(Y|t, b, theta) * pi(t, b|theta) / pi(Y)

        r = r + 1

        return (
            lb_params,
            rb_params,
            stat_probs,
            Q,
            log_weights,
            log_likelihood,
            log_likelihood_tilde,
            jump_chains,
            jump_chain_tensor,
            core,
            leafnode_num_record,
            left_branches,
            right_branches,
            v_minus,
            r,
        )

    @tf.function
    def cond_rank_update(
        self,
        lb_params,
        rb_params,
        stat_probs,
        Q,
        log_weights,
        log_likelihood,
        log_likelihood_tilde,
        jump_chains,
        jump_chain_tensor,
        core,
        leafnode_num_record,
        left_branches,
        right_branches,
        v_minus,
        r,
    ):
        return r < self.N - 1

    @tf.function
    def __call__(self, core):
        """
        Main sampling routine that performs combinatorial SMC by calling the
        rank update subroutine.
        """

        N = self.N
        A = self.A
        K = self.K

        core = tf.cast(core, DTYPE_FLOAT)

        (
            lb_params,
            rb_params,
            nucleotide_exchanges,
            stat_probs,
            Q,
            regularization,
        ) = self.get_variables()

        leafnode_num_record = tf.constant(1, shape=(K, N))  # Keeps track of core

        left_branches = tf.constant(0, shape=(1, K), dtype=DTYPE_FLOAT)
        right_branches = tf.constant(0, shape=(1, K), dtype=DTYPE_FLOAT)

        log_weights = tf.constant(0, shape=(1, K), dtype=DTYPE_FLOAT)
        log_likelihood = tf.constant(0, shape=(1, K), dtype=DTYPE_FLOAT)
        log_likelihood_tilde = tf.constant(
            math.log(1 / K), shape=[K], dtype=DTYPE_FLOAT
        )

        jump_chains = tf.constant("", shape=(K, 1), dtype=tf.string)
        jump_chain_tensor = tf.constant([self.taxa] * K, dtype=tf.string)
        v_minus = tf.constant(1, shape=[K])  # to be used in overcounting_correct

        # --- MAIN LOOP ----+
        (
            lb_params,
            rb_params,
            stat_probs,
            Q,
            log_weights,
            log_likelihood,
            log_likelihood_tilde,
            jump_chains,
            jump_chain_tensor,
            core_final,
            record_final,
            left_branches,
            right_branches,
            v_minus,
            r,
        ) = tf.while_loop(  # type: ignore
            self.cond_rank_update,
            self.body_rank_update,
            loop_vars=[
                lb_params,
                rb_params,
                stat_probs,
                Q,
                log_weights,
                log_likelihood,
                log_likelihood_tilde,
                jump_chains,
                jump_chain_tensor,
                core,
                leafnode_num_record,
                left_branches,
                right_branches,
                v_minus,
                tf.constant(0),
            ],
            shape_invariants=[
                lb_params.get_shape(),
                rb_params.get_shape(),
                stat_probs.get_shape(),
                Q.get_shape(),
                tf.TensorShape([None, K]),
                tf.TensorShape([None, K]),
                log_likelihood_tilde.get_shape(),
                tf.TensorShape([K, None]),
                tf.TensorShape([K, None]),
                tf.TensorShape([K, None, None, A]),
                tf.TensorShape([K, None]),
                tf.TensorShape([None, K]),
                tf.TensorShape([None, K]),
                v_minus.get_shape(),
                tf.TensorShape([]),
            ],
        )
        # ------------------+

        # prevent branch lengths from being too large
        Lambda = tf.constant(5e3, DTYPE_FLOAT)
        mean_branches = tf.reduce_mean(
            tf.concat([left_branches, right_branches], axis=0)
        )
        regularization += tf.square(mean_branches) * Lambda

        log_weights = tf.gather(
            log_weights, list(range(1, N))
        )  # remove the trivial index 0
        log_likelihood = tf.gather(
            log_likelihood, list(range(1, N))
        )  # remove the trivial index 0
        left_branches = tf.gather(
            left_branches, list(range(1, N))
        )  # remove the trivial index 0
        right_branches = tf.gather(
            right_branches, list(range(1, N))
        )  # remove the trivial index 0
        elbo = self.compute_log_ZSMC(log_weights)
        log_likelihood_R = self.get_log_likelihood(
            log_likelihood, left_branches, right_branches, lb_params, rb_params
        )
        cost = -elbo + regularization
        log_likelihood_tilde = log_likelihood_tilde
        v_minus = v_minus

        return {
            "cost": cost,
            "elbo": elbo,
            "jump_chains": jump_chains,
            "regularization": regularization,
            "stat_probs": stat_probs,
            "Q": Q,
            "left_branches": left_branches,
            "right_branches": right_branches,
            "log_weights": log_weights,
            "log_likelihood": log_likelihood,
            "log_likelihood_tilde": log_likelihood_tilde,
            "log_likelihood_R": log_likelihood_R,
            "v_minus": v_minus,
            "lb_params": lb_params,
            "rb_params": rb_params,
            "nucleotide_exchanges": nucleotide_exchanges,
        }
