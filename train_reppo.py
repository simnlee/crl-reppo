"""
Section order (top-to-bottom):
  imports -> Args -> Normalizer -> AutoResetWrapper/wrap_for_training -> hl_gauss
  -> prefix_dict/log_metrics -> networks (residual_block, Actor, Critic,
  action_dist) -> make_env -> Transition/RolloutTransition -> HER
  helpers -> TD-lambda helpers -> stagger helpers -> losses/epoch runners ->
  ReppoTrainingState -> init_train_state -> collect_rollout / train_step /
  training_epoch -> eval wiring -> checkpointing -> capture_vis -> main.
"""

import functools
import os
import pickle
import random
import sys
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import distrax
import flax
import flax.linen as nn
import flax.struct as struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb
import wandb_osh
from brax import envs
from brax.envs.base import PipelineEnv, State, Wrapper
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper
from brax.io import html
from etils import epath
from flax.linen.initializers import variance_scaling
from flax.training.train_state import TrainState
from wandb_osh.hooks import TriggerWandbSyncHook

from evaluator import CrlEvaluator


# =============================================================================
# Args
# =============================================================================


@dataclass
class Args:
    # --- Run / logging ergonomics ---
    exp_name: str = "reppo"
    seed: int = 42
    track: bool = True
    wandb_project_name: str = "reppo-crl-depth"
    wandb_entity: str = "simlee-upenn"
    wandb_mode: str = "offline"        # 'offline'|'online'|'disabled'
    wandb_dir: str = "."
    wandb_group: str = "."             # '.' sentinel => None
    checkpoint: bool = False
    capture_vis: bool = True
    vis_length: int = 1000
    num_render: int = 10

    # --- env ---
    env_id: str = "ant"
    eval_env_id: str = ""              # empty => falls back to env_id
    episode_length: int = 1000
    action_repeat: int = 1
    goal_reach_thresh: float = 0.5
    # filled at runtime by make_env:
    obs_dim: int = 0
    goal_start_idx: int = 0
    goal_end_idx: int = 0

    # --- REPPO training sizes ---
    num_envs: int = 128
    num_eval_envs: int = 256
    num_steps: int = 1000
    num_mini_batches: int = 64         # minibatch size = num_steps*num_envs / num_mini_batches
    num_epochs: int = 8
    num_eval: int = 200
    total_time_steps: int = 50_000_000

    # --- REPPO LR + schedule ---
    actor_lr: float = 6e-4
    critic_lr: float = 3e-4
    anneal_lr: bool = False
    max_grad_norm: float = 0.5

    # --- REPPO algorithmic hparams ---
    gamma: float = 0.99
    lmbda: float = 0.95
    kl_start: float = 0.01
    kl_bound: float = 0.1
    ent_start: float = 0.01
    ent_target_mult: float = 0.5
    num_bins: int = 151
    vmin: float = 0.0
    vmax: float = 150.0
    num_action_samples: int = 16
    aux_loss_coeff: float = 0.0

    # --- HER ---
    use_her_critic: bool = False
    use_her_actor: bool = False
    use_her_td_lambda: bool = False
    her_k: int = 4
    her_goal_sampling: str = "uniform"   # 'uniform'|'geometric'
    normalize_hindsight_loss: bool = False
    sample_new_action_for_tdL: bool = False
    her_per_epoch: bool = False          # v7b: resample HER + recompute hindsight TD-lambda every inner epoch

    # --- Stagger ---
    stagger_envs: bool = False
    stagger_step_size: int = 0           # 0 sentinel => num_steps
    stagger_mode: str = "grouped"
    stagger_warmup_policy: str = "random"
    stagger_debug: bool = False

    # --- Networks (depth-scaling axis follows scaling-crl) ---
    actor_network_width: int = 256
    actor_depth: int = 32
    critic_network_width: int = 256
    critic_depth: int = 32
    actor_skip_connections: int = 0      # reserved
    critic_skip_connections: int = 0     # reserved
    use_relu: int = 0                    # 0 => swish, 1 => relu (matches scaling-crl)


# =============================================================================
# Normalizer
# =============================================================================


class NormalizationState(struct.PyTreeNode):
    mean: struct.PyTreeNode
    var: struct.PyTreeNode
    count: int


class Normalizer:
    """Welford online estimator of per-feature mean and variance.

    ``update`` folds a fresh batch into the running moments; ``update_masked``
    does the same but only counts entries where ``mask=True`` (used so that
    frozen ticks under staggered resets do not perturb the statistics).
    ``normalize`` returns ``(x - mean) / sqrt(var + 1e-8)``.
    """

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, tree: struct.PyTreeNode) -> NormalizationState:
        return NormalizationState(
            mean=jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), tree),
            var=jax.tree_util.tree_map(lambda x: jnp.ones_like(x), tree),
            count=jax.tree_util.tree_map(lambda x: jnp.array(0, dtype=jnp.int32), tree),
        )

    def _compute_stats(self, state_mean, state_var, state_count, obs: jax.Array):
        var = jnp.var(obs, axis=0)
        mean = jnp.mean(obs, axis=0)
        batch_size = obs.shape[0]
        delta = mean - state_mean
        count = state_count + batch_size
        new_mean = state_mean + delta * batch_size / count
        m_a = state_var * state_count
        m_b = var * batch_size
        M2 = m_a + m_b + jnp.square(delta) * state_count * batch_size / count
        new_var = M2 / count
        return new_mean, new_var, count

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: NormalizationState, tree: struct.PyTreeNode) -> NormalizationState:
        tree = jax.tree_util.tree_map(lambda x, m: x.reshape(-1, *m.shape), tree, state.mean)
        stats = jax.tree_util.tree_map(
            lambda m, v, c, x: self._compute_stats(m, v, c, x),
            state.mean, state.var, state.count, tree,
        )
        mean, var, count = jax.tree_util.tree_transpose(
            jax.tree_util.tree_structure(tree), jax.tree_util.tree_structure(("*", "*", "*")), stats,
        )
        return state.replace(mean=mean, var=var, count=count)

    def _compute_stats_masked(
        self, state_mean, state_var, state_count, obs: jax.Array, mask: jax.Array
    ):
        # obs: (N, *D), mask: (N,) with 0/1 float weights.
        w = mask.astype(jnp.float32)
        batch_size = jnp.sum(w)
        safe_n = jnp.maximum(batch_size, 1.0)
        w_exp = w.reshape((-1,) + (1,) * (obs.ndim - 1))
        batch_mean = jnp.sum(w_exp * obs, axis=0) / safe_n
        batch_var = jnp.sum(w_exp * jnp.square(obs - batch_mean), axis=0) / safe_n

        delta = batch_mean - state_mean
        count = state_count + batch_size
        safe_count = jnp.maximum(count, 1.0)
        new_mean = state_mean + delta * batch_size / safe_count
        m_a = state_var * state_count
        m_b = batch_var * batch_size
        M2 = m_a + m_b + jnp.square(delta) * state_count * batch_size / safe_count
        new_var = M2 / safe_count

        nonempty = batch_size > 0
        new_mean = jnp.where(nonempty, new_mean, state_mean)
        new_var = jnp.where(nonempty, new_var, state_var)
        new_count = jnp.where(nonempty, count, state_count).astype(state_count.dtype)
        return new_mean, new_var, new_count

    @functools.partial(jax.jit, static_argnums=0)
    def update_masked(
        self,
        state: NormalizationState,
        tree: struct.PyTreeNode,
        mask: jax.Array,
    ) -> NormalizationState:
        tree = jax.tree_util.tree_map(lambda x, m: x.reshape(-1, *m.shape), tree, state.mean)
        flat_mask = mask.reshape(-1)
        stats = jax.tree_util.tree_map(
            lambda m, v, c, x: self._compute_stats_masked(m, v, c, x, flat_mask),
            state.mean, state.var, state.count, tree,
        )
        mean, var, count = jax.tree_util.tree_transpose(
            jax.tree_util.tree_structure(tree), jax.tree_util.tree_structure(("*", "*", "*")), stats,
        )
        return state.replace(mean=mean, var=var, count=count)

    @functools.partial(jax.jit, static_argnums=0)
    def normalize(self, state: NormalizationState, tree: struct.PyTreeNode) -> struct.PyTreeNode:
        return jax.tree_util.tree_map(
            lambda x, m, v: (x - m) / jnp.sqrt(v + 1e-8),
            tree, state.mean, state.var,
        )


# =============================================================================
# AutoResetWrapper + StaggeredResetWrapper + wrap_for_training
# =============================================================================


class AutoResetWrapper(Wrapper):
    """Auto-resets env on done, storing pre-reset obs in info['raw_obs']."""

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["first_pipeline_state"] = state.pipeline_state
        state.info["first_obs"] = state.obs
        state.info["raw_obs"] = state.obs
        return state

    def step(self, state, action):
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)
        state.info["raw_obs"] = state.obs

        def where_done(x, y):
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(done, x, y)

        pipeline_state = jax.tree_util.tree_map(
            where_done, state.info["first_pipeline_state"], state.pipeline_state,
        )
        obs = jax.tree_util.tree_map(where_done, state.info["first_obs"], state.obs)
        return state.replace(pipeline_state=pipeline_state, obs=obs)


class StaggeredResetWrapper(Wrapper):
    """Reset wrapper implementing reset gate mechanism.

    When ``state.info['active']`` is all-False (the pre-stagger phase),
    behaves identically to :class:`AutoResetWrapper`: any env whose new
    ``done=True`` is reset to the cached initial state on the same tick.

    When active, the wrapper emits a real terminal transition on the tick
    of an intrinsic termination (``done=True`` with ``truncation=0``):

    * ``reward`` and ``done=1`` and ``truncation=0`` are preserved — the
      transition is like a MDP terminal step (``valid_mask=True``).
    * The carry ``obs`` is reset to ``first_obs`` (via ``reset_mask``) so
      the *next* tick's env.step starts from a clean initial state.
    * ``is_waiting`` flips True so the next tick freezes cleanly.

    On subsequent ticks while ``is_waiting=True`` the wrapper freezes:

    * ``pipeline_state`` and ``obs`` revert to the pre-step carry.
    * ``reward``, ``done``, and ``truncation`` are zeroed.
    * The EpisodeWrapper step counter is held frozen.
    * The transition emits ``valid_mask=False``.

    Every group's ``group_step`` counter advances in lockstep each tick.
    When it reaches the episode horizon ``H``, every env with that
    ``group_idx`` resets together — a 'group reset gate'. The gate tick
    emits ``truncation=1`` unconditionally (even when the env was waiting)
    so HER's ``boundaries = max(done, truncated)`` detects the gate as an
    episode boundary. The gate tick itself is emitted with
    ``valid_mask=False`` because the
    ``(obs_before_reset, action, 0, first_obs, ...)`` transition is not a
    real MDP step.
    """

    def __init__(self, env, group_idx: jax.Array, episode_length: int):
        super().__init__(env)
        self._group_idx = group_idx
        self._num_envs = int(group_idx.shape[0])
        self._H = int(episode_length)

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["first_pipeline_state"] = state.pipeline_state
        state.info["first_obs"] = state.obs
        state.info["raw_obs"] = state.obs
        state.info["group_idx"] = self._group_idx
        state.info["group_step"] = jnp.zeros((self._num_envs,), dtype=jnp.int32)
        state.info["is_waiting"] = jnp.zeros((self._num_envs,), dtype=jnp.bool_)
        state.info["valid_mask"] = jnp.ones((self._num_envs,), dtype=jnp.bool_)
        state.info["active"] = jnp.zeros((self._num_envs,), dtype=jnp.bool_)
        return state

    def step(self, state, action):
        active = state.info["active"]
        was_waiting = state.info["is_waiting"]
        prev_pipeline = state.pipeline_state
        prev_obs = state.obs
        prev_raw_obs = state.info["raw_obs"]
        prev_steps = state.info["steps"]
        prev_group_step = state.info["group_step"]

        # Zero EpisodeWrapper's step counter where the incoming state is done=True, then clear done before stepping.
        state.info["steps"] = jnp.where(
            state.done, jnp.zeros_like(prev_steps), prev_steps
        )
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)

        intrinsic_done = (state.done > 0) & (state.info["truncation"] == 0)
        freeze_now = active & was_waiting
        enters_waiting = active & intrinsic_done & (~was_waiting)
        group_step_new = prev_group_step + 1
        gate_now = active & (group_step_new >= self._H)

        def _where(mask, on_true, on_false):
            """Broadcast mask over leading axis and select on_true where True."""
            if not hasattr(on_true, "shape"):
                return on_false
            if on_true.shape and on_true.shape[0] == mask.shape[0]:
                shaped = jnp.reshape(
                    mask, (mask.shape[0],) + (1,) * (on_true.ndim - 1)
                )
                return jnp.where(shaped, on_true, on_false)
            return on_false

        # Freeze path: revert stepped leaves to their pre-step counterparts.
        pipeline_after_freeze = jax.tree_util.tree_map(
            lambda new_leaf, old_leaf: _where(freeze_now, old_leaf, new_leaf),
            state.pipeline_state, prev_pipeline,
        )
        obs_after_freeze = _where(freeze_now, prev_obs, state.obs)
        reward_after_freeze = jnp.where(
            freeze_now, jnp.zeros_like(state.reward), state.reward
        )
        done_after_freeze = jnp.where(
            freeze_now, jnp.zeros_like(state.done), state.done
        )
        steps_after_freeze = jnp.where(freeze_now, prev_steps, state.info["steps"])
        trunc_after_freeze = jnp.where(
            gate_now,
            jnp.ones_like(state.info["truncation"]),
            jnp.where(
                freeze_now,
                jnp.zeros_like(state.info["truncation"]),
                state.info["truncation"],
            ),
        )
        raw_obs_after_freeze = _where(freeze_now, prev_raw_obs, state.obs)

        # Reset path: staggered gate, terminal-this-tick, OR pre-stagger autoreset.
        passthrough_reset = (~active) & done_after_freeze.astype(jnp.bool_)
        reset_mask = gate_now | passthrough_reset | enters_waiting

        pipeline_final = jax.tree_util.tree_map(
            lambda cur, first: _where(reset_mask, first, cur),
            pipeline_after_freeze, state.info["first_pipeline_state"],
        )
        obs_final = _where(reset_mask, state.info["first_obs"], obs_after_freeze)

        # Bookkeeping
        group_step_final = jnp.where(
            gate_now, jnp.zeros_like(group_step_new), group_step_new
        )
        is_waiting_final = jnp.where(
            gate_now,
            jnp.zeros_like(was_waiting),
            freeze_now | enters_waiting,
        )
        valid_mask_final = jnp.where(
            active, (~freeze_now) & (~gate_now), jnp.ones_like(active)
        )
        steps_final = jnp.where(
            reset_mask, jnp.zeros_like(steps_after_freeze), steps_after_freeze
        )

        state.info["steps"] = steps_final
        state.info["truncation"] = trunc_after_freeze
        state.info["raw_obs"] = raw_obs_after_freeze
        state.info["group_step"] = group_step_final
        state.info["is_waiting"] = is_waiting_final
        state.info["valid_mask"] = valid_mask_final

        return state.replace(
            pipeline_state=pipeline_final,
            obs=obs_final,
            reward=reward_after_freeze,
            done=done_after_freeze,
        )


def wrap_for_training(
    env: PipelineEnv,
    episode_length: int,
    action_repeat: int = 1,
    *,
    stagger_envs: bool = False,
    group_idx: "jax.Array | None" = None,
):
    """VmapWrapper -> EpisodeWrapper -> (AutoReset|StaggeredReset) with raw_obs.

    When ``stagger_envs`` is True and ``group_idx`` is provided, the outer
    reset wrapper is :class:`StaggeredResetWrapper`. Otherwise the default
    path is byte-identical to the original: :class:`AutoResetWrapper`.
    """
    env = VmapWrapper(env)
    env = EpisodeWrapper(env, episode_length=episode_length, action_repeat=action_repeat)
    if stagger_envs and group_idx is not None:
        env = StaggeredResetWrapper(env, group_idx, episode_length)
    else:
        env = AutoResetWrapper(env)
    return env


# =============================================================================
# hl_gauss
# =============================================================================


def hl_gauss(
    values: jax.Array, num_bins: int, vmin: float, vmax: float, epsilon: float = 0.0,
) -> jax.Array:
    """HL-Gauss soft two-hot encoding.

    Projects a scalar target into a categorical distribution over ``num_bins``
    bins spanning ``[vmin, vmax]``. The target probability for each bin is the
    mass a Gaussian (centered at the clipped scalar, width ~0.75 * bin_width)
    places on that bin. This turns scalar TD targets into a categorical
    target for cross-entropy regression of the critic logits.
    """
    # Clip the scalar into the support, optionally contracted by (1 - epsilon).
    x = jnp.clip(values, vmin, vmax).squeeze() / (1 - epsilon)
    bin_width = (vmax - vmin) / (num_bins - 1)
    sigma_to_final_sigma_ratio = 0.75
    # Bin edges (one more edge than bins).
    support = jnp.linspace(
        vmin - bin_width / 2, vmax + bin_width / 2, num_bins + 1, dtype=jnp.float32,
    )
    sigma = bin_width * sigma_to_final_sigma_ratio
    # Per-edge Gaussian CDF; per-bin mass = CDF(right_edge) - CDF(left_edge).
    cdf_evals = jax.scipy.special.erf((support - x) / (jnp.sqrt(2) * sigma))
    z = cdf_evals[-1] - cdf_evals[0]
    target_probs = cdf_evals[1:] - cdf_evals[:-1]
    target_probs = (target_probs / z).reshape(*values.shape[:-1], num_bins)
    # Optional uniform smoothing for numerical stability (epsilon=0 by default).
    uniform = jnp.ones_like(target_probs) / num_bins
    return (1 - epsilon) * target_probs + epsilon * uniform


# =============================================================================
# Logging utilities
# =============================================================================


def prefix_dict(prefix: str, metrics: dict, sep: str = "/") -> dict:
    return {f"{prefix}{sep}{key}": value for key, value in metrics.items()}


def _to_scalar(v):
    return v.item() if hasattr(v, "item") else float(v)


def log_metrics(time_steps: int, eval_epoch: int, metrics: dict):
    training_metrics = {k: v for k, v in metrics.items() if k.startswith("training/")}
    eval_metrics = {k: v for k, v in metrics.items() if k.startswith("eval/")}
    sps = _to_scalar(training_metrics.get("training/sps", 0))

    print("\n" + "=" * 100)
    print(f"Eval {eval_epoch:>4} | Timesteps {time_steps:>10,} | SPS {sps:>10,.0f}")
    print("=" * 100)

    if training_metrics:
        print("\nTRAINING METRICS:")
        print("-" * 100)
        for k in sorted(training_metrics):
            print(f"  {k:<45} {_to_scalar(training_metrics[k]):>12.4f}")
        print()

    if eval_metrics:
        print("EVALUATION METRICS:")
        print("-" * 100)
        for k in sorted(eval_metrics):
            print(f"  {k:<45} {_to_scalar(eval_metrics[k]):>12.4f}")
        print()

    print("=" * 100)

    if wandb.run is not None:
        log_data = {k: _to_scalar(v) for k, v in metrics.items()}
        wandb.log(log_data, step=time_steps)


# =============================================================================
# Networks
#
# The residual-block stack (residual_block) and the Actor residual tower are
# copied from scaling-crl so this experiment shares the same depth-scaling
# axis as the CRL paper for a fair comparison. The Critic uses a single
# residual trunk over concat(state, goal, action) and emits HL-Gauss
# categorical return logits plus auxiliary next-embedding / next-reward
# predictions.
#
#   residual_block : Dense -> Norm -> Act, four times, with an identity
#                    shortcut added at the end.
#   Critic         : single residual trunk on concat(obs, action); a
#                    Dense+Norm+Act projection feeds HL-Gauss categorical
#                    logits over return bins, plus auxiliary predictions of
#                    the next embedding and the next reward.
#   Actor          : Dense stem + residual blocks, two Dense heads for the
#                    mean and log_std of a tanh-squashed diagonal Gaussian.
#                    Also carries two scalar log-parameters:
#                      * log_lagrangian  = log(beta), the KL multiplier
#                      * log_temperature = log(alpha), the entropy coef
#   action_dist    : distrax tanh-Normal used for sampling and log-probs.
# =============================================================================


lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
bias_init = nn.initializers.zeros


def residual_block(x, width, normalize, activation):
    identity = x
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = x + identity
    return x


class Actor(nn.Module):
    """
    Returns `(mean, log_std)` by default. With `deterministic=True`, returns
    `jnp.tanh(mean)`. `log_std` is unclamped.
    """

    action_size: int
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0
    kl_start: float = 0.01
    ent_start: float = 0.01

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        # Lagrange multiplier beta on the KL constraint (stored as log for
        # positivity). Updated by the actor's lagrangian_loss.
        _ = self.param(
            "log_lagrangian",
            nn.initializers.constant(jnp.log(self.kl_start)),
            (1,),
        )
        # Entropy coefficient alpha for the max-entropy objective (stored as
        # log for positivity). Updated by the actor's target_entropy_loss.
        _ = self.param(
            "log_temperature",
            nn.initializers.constant(jnp.log(self.ent_start)),
            (1,),
        )

        if self.norm_type == "layer_norm":
            normalize = lambda y: nn.LayerNorm()(y)
        else:
            normalize = lambda y: y

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        for i in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)
        # Two heads: mean and log_std of a diagonal Gaussian. Samples are
        # squashed through tanh by action_dist(mean, log_std).
        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)

        if deterministic:
            # Evaluation path: pick the tanh-squashed mean as a point action.
            return jnp.tanh(mean)
        return mean, log_std


class Critic(nn.Module):
    """Q(state, action, goal) with an HL-Gauss categorical return head.

    Single residual trunk on concat(obs, action) = concat(state, goal, action).
    The trunk output feeds a Dense+Norm+Act projection, which in turn feeds:

      value          : expected return under softmax(logits) over the
                       HL-Gauss bin support
      logits         : raw categorical logits (num_bins)
      probs          : softmax(logits)
      embed          : trunk representation, also used as an auxiliary
                       self-supervised target
      pred_features  : auxiliary prediction of the next-step trunk embed
                       (trained by MSE against a stop-grad target)
      pred_rew       : auxiliary scalar reward prediction
    """

    action_size: int
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0
    num_bins: int = 151
    vmin: float = 0.0
    vmax: float = 150.0

    @nn.compact
    def __call__(self, obs, action):
        normalize = lambda y: nn.LayerNorm()(y)
        activation = nn.relu if self.use_relu else nn.swish

        # Single residual trunk on concat(state, goal, action). The observation
        # already carries the goal spliced into its trailing slots, so
        # concat(obs, action) == concat(state, goal, action).
        x = jnp.concatenate([obs, action], axis=-1)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        for _ in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)
        fusion = x
        fusion_dim = self.network_width

        # Projection head: Dense -> Norm -> Act -> Dense to num_bins logits.
        q = nn.Dense(fusion_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(fusion)
        q = normalize(q)
        q = activation(q)
        # Categorical logits over HL-Gauss return bins.
        logits = nn.Dense(self.num_bins, kernel_init=lecun_unfirom, bias_init=bias_init)(q)

        # Learned bias of shape (num_bins,) initialized to HL-Gauss(0). Scaled
        # by 40.0 so the untrained critic predicts value ~= 0 (the HL-Gauss
        # bin concentrated at zero dominates the softmax at init).
        zero_dist = self.param(
            "zero_dist",
            lambda _rng, shape: hl_gauss(
                jnp.zeros((1,)), self.num_bins, self.vmin, self.vmax
            ),
            (self.num_bins,),
        )
        logits = logits + zero_dist * 40.0

        # Expected value under the categorical distribution over return bins:
        # value = sum_b softmax(logits)_b * support_b.
        probs = jax.nn.softmax(logits, axis=-1)
        support = jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True)
        value = jnp.sum(probs * support, axis=-1)

        # Auxiliary head off the same fused embedding. The final Dense emits
        # (fusion_dim + 1) outputs: the first fusion_dim predict the next-step
        # fused embedding, the last element predicts the next-step reward.
        pred = nn.Dense(fusion_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(fusion)
        pred = normalize(pred)
        pred = activation(pred)
        pred_raw = nn.Dense(
            fusion_dim + 1, kernel_init=lecun_unfirom, bias_init=bias_init
        )(pred)

        return {
            "value": value,
            "logits": logits,
            "probs": probs,
            "embed": fusion,
            "pred_features": pred_raw[..., :-1],
            "pred_rew": pred_raw[..., -1:],
        }


def action_dist(mean: jax.Array, log_std: jax.Array) -> distrax.Distribution:
    """Tanh-squashed diagonal Gaussian action distribution."""
    std = jnp.exp(log_std) + 1e-6
    return distrax.Independent(
        distrax.Transformed(distrax.Normal(mean, std), distrax.Tanh()),
        reinterpreted_batch_ndims=1,
    )


# =============================================================================
# Env registry (copied verbatim from scaling-crl for fair comparison)
#
# Each branch constructs a goal-conditioned Brax env and returns the
# (obs_dim, goal_start_idx, goal_end_idx) slicing triple so downstream code
# knows where the goal lives inside the concatenated observation.
# =============================================================================


def make_env(env_id: str):
    print(f"making env with env_id: {env_id}", flush=True)
    if env_id == "reacher":
        from envs.reacher import Reacher
        env = Reacher(backend="spring")
        obs_dim, goal_start_idx, goal_end_idx = 10, 4, 7
    elif env_id == "pusher":
        from envs.pusher import Pusher
        env = Pusher(backend="spring")
        obs_dim, goal_start_idx, goal_end_idx = 20, 10, 13
    elif env_id == "ant":
        from envs.ant import Ant
        env = Ant(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )
        obs_dim, goal_start_idx, goal_end_idx = 29, 0, 2
    elif "ant" in env_id and "maze" in env_id:
        if "gen" not in env_id:
            from envs.ant_maze import AntMaze
            env = AntMaze(
                backend="spring",
                exclude_current_positions_from_observation=False,
                terminate_when_unhealthy=True,
                maze_layout_name=env_id[4:],
            )
            obs_dim, goal_start_idx, goal_end_idx = 29, 0, 2
        else:
            from envs.ant_maze_generalization import AntMazeGeneralization
            gen_idx = env_id.find("gen")
            maze_layout_name = env_id[4 : gen_idx - 1]
            generalization_config = env_id[gen_idx + 4 :]
            print(
                f"maze_layout_name: {maze_layout_name}, "
                f"generalization_config: {generalization_config}",
                flush=True,
            )
            env = AntMazeGeneralization(
                backend="spring",
                exclude_current_positions_from_observation=False,
                terminate_when_unhealthy=True,
                maze_layout_name=maze_layout_name,
                generalization_config=generalization_config,
            )
            obs_dim, goal_start_idx, goal_end_idx = 29, 0, 2
    elif env_id == "ant_ball":
        from envs.ant_ball import AntBall
        env = AntBall(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )
        obs_dim, goal_start_idx, goal_end_idx = 31, 28, 30
    elif env_id == "ant_push":
        from envs.ant_push import AntPush
        env = AntPush(backend="mjx")
        obs_dim, goal_start_idx, goal_end_idx = 31, 0, 2
    elif env_id == "humanoid":
        from envs.humanoid import Humanoid
        env = Humanoid(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )
        obs_dim, goal_start_idx, goal_end_idx = 268, 0, 3
    elif "humanoid" in env_id and "maze" in env_id:
        from envs.humanoid_maze import HumanoidMaze
        env = HumanoidMaze(backend="spring", maze_layout_name=env_id[9:])
        obs_dim, goal_start_idx, goal_end_idx = 268, 0, 3
    elif env_id == "arm_reach":
        from envs.manipulation.arm_reach import ArmReach
        env = ArmReach(backend="mjx")
        obs_dim, goal_start_idx, goal_end_idx = 13, 7, 10
    elif env_id == "arm_binpick_easy":
        from envs.manipulation.arm_binpick_easy import ArmBinpickEasy
        env = ArmBinpickEasy(backend="mjx")
        obs_dim, goal_start_idx, goal_end_idx = 17, 0, 3
    elif env_id == "arm_binpick_hard":
        from envs.manipulation.arm_binpick_hard import ArmBinpickHard
        env = ArmBinpickHard(backend="mjx")
        obs_dim, goal_start_idx, goal_end_idx = 17, 0, 3
    elif env_id == "arm_binpick_easy_EEF":
        from envs.manipulation.arm_binpick_easy_EEF import ArmBinpickEasyEEF
        env = ArmBinpickEasyEEF(backend="mjx")
        obs_dim, goal_start_idx, goal_end_idx = 11, 0, 3
    elif "arm_grasp" in env_id:
        from envs.manipulation.arm_grasp import ArmGrasp
        cube_noise_scale = float(env_id[10:]) if len(env_id) > 9 else 0.3
        env = ArmGrasp(cube_noise_scale=cube_noise_scale, backend="mjx")
        obs_dim, goal_start_idx, goal_end_idx = 23, 16, 23
    elif env_id == "arm_push_easy":
        from envs.manipulation.arm_push_easy import ArmPushEasy
        env = ArmPushEasy(backend="mjx")
        obs_dim, goal_start_idx, goal_end_idx = 17, 0, 3
    elif env_id == "arm_push_hard":
        from envs.manipulation.arm_push_hard import ArmPushHard
        env = ArmPushHard(backend="mjx")
        obs_dim, goal_start_idx, goal_end_idx = 17, 0, 3
    else:
        raise NotImplementedError(f"Unknown env_id: {env_id}")

    return env, obs_dim, goal_start_idx, goal_end_idx


# =============================================================================
# Transition containers
#
# Transition        : NamedTuple consumed by the evaluator; carries
#                     (observation, action, reward, discount, extras).
# RolloutTransition : rollout-time PyTree with extra fields for HER goal
#                     relabeling (`next_obs`) and for staggered resets
#                     (`valid_mask`, `truncated`).
# =============================================================================


class Transition(NamedTuple):
    """Evaluator-shaped transition tuple (shape matches the scaling-crl evaluator)."""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: Any = ()


class RolloutTransition(struct.PyTreeNode):
    """Rollout transition struct with extra fields for HER and staggered resets.

    ``next_obs`` is the pre-auto-reset next observation so that HER goal
    relabeling can read the achieved goal from the true continuation, even
    when the env has just been auto-reset.

    ``valid_mask`` marks transitions that should contribute to learning
    losses. False for frozen steps (when an env is idling for its group's
    reset gate under staggered resets) and for the gate-reset tick itself.
    Always True when staggered resets are disabled.
    """
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    done: jax.Array
    truncated: jax.Array
    valid_mask: jax.Array
    extras: dict


# =============================================================================
# HER helpers (closures over goal_indices, goal_dim, goal_reach_thresh, args).
#
# Achieved goals are read from `obs[..., goal_indices]` (a contiguous range
# inside the state block, set per-env by make_env).
#
# replace_goal_in_obs : splice a new goal into the trailing goal_dim slots.
# compute_goal_reward : sparse 0/1 reward = 1{ ||achieved - goal|| < thresh }.
# her_augment         : "future" strategy - for each step t, sample k future
#                       achieved states from the same episode segment and
#                       return the augmented tensors.
# td0_targets         : one-step TD(0) target r + gamma * (1 - done) * V.
# =============================================================================


def make_her_helpers(args: "Args", goal_indices, goal_dim: int, goal_reach_thresh: float):
    """Factory returning (replace_goal_in_obs, compute_goal_reward, her_augment, td0_targets)."""

    def replace_goal_in_obs(obs, new_goal):
        """Replace last goal_dim dims of obs with new_goal."""
        obs = jnp.broadcast_to(obs, new_goal.shape[:-1] + (obs.shape[-1],))
        return obs.at[..., -goal_dim:].set(new_goal)

    def compute_goal_reward(next_obs_raw, goal):
        """Sparse reward: 1 if achieved state is within threshold of goal."""
        achieved = next_obs_raw[..., goal_indices]
        dist = jnp.linalg.norm(achieved - goal, axis=-1)
        return (dist < goal_reach_thresh).astype(jnp.float32)

    def her_augment(key, batch: RolloutTransition):
        """HER 'future' strategy: sample k hindsight goals per step.

        For each step t in each env, sample k indices from the remaining
        segment [t, end_idx[t]] (up to episode boundary), and take the
        achieved goal at each sampled step as a hindsight goal. Supports
        uniform or geometric sampling over future offsets. Slots with fewer
        than k available future steps are marked with weight=0 and
        truncated=1 so they are zeroed out of the critic loss.

        Returns (obs_alt, next_obs_alt, reward_alt, done_alt, trunc_alt,
                 weight_alt, g_alt, end_idx).
        """
        S, E, k = args.num_steps, args.num_envs, args.her_k

        # An episode boundary occurs when done=1 (terminal) or truncated=1.
        boundaries = jnp.maximum(batch.done, batch.truncated)
        step_idx = jnp.broadcast_to(jnp.arange(S)[:, None], (S, E))

        # Reverse scan: end_idx[t, e] = smallest t' >= t with boundaries=1
        # (or S-1 if no boundary before the rollout ends).
        def scan_end(carry, x):
            t, b = x
            end = jnp.where(b, t, carry)
            return end, end

        _, end_idx = jax.lax.scan(
            scan_end,
            jnp.full((E,), S - 1, dtype=jnp.int32),
            (step_idx, boundaries),
            reverse=True,
        )

        # Gumbel top-k over offsets in [0, S): samples k distinct offsets
        # without replacement, weighted by log_weights + Gumbel noise.
        range_size = end_idx - step_idx + 1
        offsets_range = jnp.arange(S)
        gumbel = -jnp.log(
            -jnp.log(jnp.clip(jax.random.uniform(key, (S, E, S)), 1e-20, 1.0 - 1e-7))
        )
        if args.her_goal_sampling == "geometric":
            # Geometric weighting log_w = offset * log(gamma) biases toward
            # near-term future steps (more weight on small offsets).
            log_weights = offsets_range * jnp.log(args.gamma)
        else:
            # Uniform sampling: every future offset in the segment is equally likely.
            log_weights = jnp.zeros(S)
        scores = log_weights[None, None, :] + gumbel
        # Mask out offsets that reach past the episode boundary.
        valid_offsets = offsets_range[None, None, :] < range_size[:, :, None]
        scores = jnp.where(valid_offsets, scores, -1e9)
        _, top_offsets = jax.lax.top_k(scores, k)
        future_idx = step_idx[:, :, None] + top_offsets

        # A hindsight slot is valid if the segment had at least (j+1) future
        # steps. Invalid slots are zeroed out of the loss below.
        valid = jnp.arange(k)[None, None, :] < jnp.minimum(k, range_size)[:, :, None]

        # Gather the achieved goal at each sampled future step; this is the
        # hindsight goal for the (t, j) slot.
        env_idx = jnp.broadcast_to(jnp.arange(E)[None, :, None], future_idx.shape)
        g_alt = batch.next_obs[future_idx, env_idx][..., goal_indices]

        # Splice g_alt into the trailing goal slots of obs and next_obs.
        obs_alt = replace_goal_in_obs(batch.obs[:, :, None, :], g_alt)
        next_obs_alt = replace_goal_in_obs(batch.next_obs[:, :, None, :], g_alt)

        # Hindsight reward is the sparse indicator for the new goal.
        reward_alt = compute_goal_reward(batch.next_obs[:, :, None, :], g_alt)

        # term = true terminal (done & not truncated). Truncations carry
        # through as the truncated flag below; this keeps the TD bootstrap
        # well-defined at time-limit boundaries.
        term = batch.done * (1.0 - batch.truncated)
        done_alt = jnp.broadcast_to(term[:, :, None], (S, E, k))

        trunc_alt = jnp.broadcast_to(batch.truncated[:, :, None], (S, E, k))
        # Invalid slots are marked truncated=1 so (1 - truncated) zeros them in L_Q.
        trunc_alt = jnp.where(valid, trunc_alt, 1.0)

        if args.normalize_hindsight_loss:
            # Weight each valid slot by 1/m_t so each step contributes a
            # total hindsight mass of 1 regardless of how many valid slots
            # it has.
            m_t = jnp.minimum(k, range_size)
            weight_alt = jnp.where(valid, 1.0 / m_t[:, :, None], 0.0)
        else:
            # Every valid slot contributes with weight 1.
            weight_alt = jnp.where(valid, 1.0, 0.0)
        return (
            obs_alt,
            next_obs_alt,
            reward_alt,
            done_alt,
            trunc_alt,
            weight_alt,
            g_alt,
            end_idx,
        )

    def td0_targets(soft_reward, next_value, done):
        """One-step TD(0) target: r_tilde + gamma * (1 - done) * V(next)."""
        # TODO: if use_her_td_lambda=False is ever used, the gate here and the
        # entropy mask in compute_extras_alt (L1172) both need to become
        # (1 - done * (1 - truncated)) to correctly bootstrap at brax time-limits
        # and include the entropy correction there. Both would need truncated_alt
        # threaded through compute_extras_alt's signature. Currently unreachable
        # under the slurm config (use_her_td_lambda=True) so deferred.
        return soft_reward + args.gamma * (1.0 - done) * next_value

    return replace_goal_in_obs, compute_goal_reward, her_augment, td0_targets


# =============================================================================
# TD-lambda target helpers (closures over args, actor, critic, normalizer,
# goal_indices, goal_reach_thresh, replace_goal_in_obs, fusion_dim).
#
# nstep_lambda                    : reverse-scan TD-lambda return on the
#                                   original trajectory. Skips freeze ticks
#                                   but keeps boundary (truncated / done)
#                                   ticks so they bootstrap earlier ticks.
# compute_extras                  : for the original trajectory, compute the
#                                   max-entropy soft reward
#                                      r - gamma * alpha * log pi(a_{t+1})
#                                   and V(next) averaged over
#                                   num_action_samples samples from
#                                   pi(. | next_obs).
# compute_extras_alt              : same as above but for a hindsight
#                                   trajectory with fresh actions sampled
#                                   under the alt goal.
# compute_alt_td_lambda_targets_k : IS-corrected TD-lambda target for one
#                                   k-slot of hindsight goals, built in a
#                                   single reverse scan (see inline comments
#                                   for the per-decision ratio rho and the
#                                   interior / boundary recursions).
# =============================================================================


def make_td_lambda_helpers(
    args: "Args",
    actor: "Actor",
    critic: "Critic",
    normalizer: Normalizer,
    goal_indices,
    goal_reach_thresh: float,
    replace_goal_in_obs,
    fusion_dim: int = 256,
):
    """Factory returning (nstep_lambda, compute_extras, compute_extras_alt,
    compute_alt_td_lambda_targets_k)."""

    def _actor_dist(params, obs):
        mean, log_std = actor.apply(params, obs)
        return action_dist(mean, log_std)

    def _alpha(actor_params):
        return jnp.exp(actor_params["params"]["log_temperature"])

    def nstep_lambda(batch):
        """Reverse-scan TD-lambda return using the max-entropy soft reward."""
        def loop(returns, transition):
            done = transition.done
            reward = transition.extras["soft_reward"]
            next_value = transition.extras["next_value"]
            truncated = transition.truncated
            valid = transition.valid_mask
            # Lambda mixes the upstream return (lambda) with the bootstrap
            # V(next) (1 - lambda). lambda=1 recovers the Monte-Carlo
            # return; lambda=0 recovers TD(0).
            lambda_sum = args.lmbda * returns + (1 - args.lmbda) * next_value
            # At a truncation (time-limit): bootstrap only.
            # Otherwise: r_tilde + gamma * (1 - done) * lambda_sum.
            lambda_return = reward + args.gamma * jnp.where(
                truncated, next_value, (1.0 - done) * lambda_sum
            )
            # Skip pure freeze ticks; keep gate ticks so their truncation
            # bootstrap propagates to earlier ticks of the old episode.
            is_boundary = (truncated > 0) | (done > 0)
            skip = (~valid.astype(jnp.bool_)) & (~is_boundary)
            lambda_return = jnp.where(skip, returns, lambda_return)
            return lambda_return, lambda_return

        # Reverse scan across time. init = V(next) at the last rollout step,
        # so the final step's target is r_tilde + gamma * V(next).
        _, lambda_return = jax.lax.scan(
            f=loop,
            init=batch.extras["next_value"][-1],
            xs=batch,
            reverse=True,
        )
        return lambda_return

    def compute_extras(key, train_state, batch):
        """Compute max-entropy soft reward and V(next) for the original trajectory."""
        actor_params = train_state.actor_state.params
        critic_params = train_state.critic_state.params
        alpha = _alpha(actor_params)

        # Next-action distribution evaluated at next_obs. Used twice below.
        key, next_act_key = jax.random.split(key)
        next_pi = _actor_dist(actor_params, batch.next_obs)
        next_action = next_pi.sample(seed=next_act_key)
        # At episode boundaries (done or truncated), batch.action[t+1] is
        # from a new episode, so fall back to a fresh sample. Otherwise
        # use the action actually executed at t+1 in the rollout.
        boundary_mask = jnp.maximum(batch.truncated, batch.done)
        true_next_action = jnp.where(
            boundary_mask[..., None],
            next_action,
            jnp.concatenate([batch.action[1:], next_action[-1:]], axis=0),
        )
        true_next_action = jnp.clip(true_next_action, -0.999, 0.999)
        true_next_log_prob = next_pi.log_prob(true_next_action)
        # Max-entropy soft reward: r_t - gamma * (1 - intrinsic_done) * alpha * log pi(a_{t+1}).
        # Under brax's EpisodeWrapper convention, done=1 fires at BOTH intrinsic
        # terminations and time-limits; truncated=1 disambiguates the time-limit
        # case. The "truly absorbing state" flag is thus done * (1 - truncated).
        # Matches the nstep_lambda bootstrap gate `jnp.where(truncated, V, (1-done)*V)`.
        intrinsic_done = batch.done * (1.0 - batch.truncated)
        soft_reward = batch.reward - args.gamma * (1.0 - intrinsic_done) * true_next_log_prob * alpha

        # V(next) = E_{a' ~ pi(.|next_obs)} Q(next_obs, a'), Monte-Carlo
        # estimated with num_action_samples draws to reduce variance.
        key, next_act_key = jax.random.split(key)
        next_sample_actions = next_pi.sample(
            seed=next_act_key, sample_shape=(args.num_action_samples,)
        )
        next_critic_output = critic.apply(
            critic_params,
            jnp.repeat(batch.next_obs[None, ...], args.num_action_samples, axis=0),
            next_sample_actions,
        )
        next_values = next_critic_output["value"].mean(0)
        # next_emb is used as the stop-grad target for the critic's
        # auxiliary next-embedding prediction head.
        next_emb = next_critic_output["embed"][0]

        return {
            "soft_reward": soft_reward,
            "next_value": next_values,
            "next_emb": jax.lax.stop_gradient(next_emb),
        }

    def compute_extras_alt(key, train_state, obs_alt, next_obs_alt, reward_alt, done_alt):
        """Soft reward and V(next) for a hindsight trajectory under g_alt.

        The rollout action at t+1 was executed under the original goal, not
        under g_alt, so we always sample a fresh action from pi(. | x_{t+1},
        g_alt) for both the max-entropy correction and the V(next) estimate.
        """
        actor_params = train_state.actor_state.params
        critic_params = train_state.critic_state.params
        alpha = _alpha(actor_params)

        # Fresh action under the alt goal. Used for both the log-prob
        # correction and the V(next) estimate below.
        key, next_act_key = jax.random.split(key)
        next_pi = _actor_dist(actor_params, next_obs_alt)
        next_action = next_pi.sample(seed=next_act_key)
        next_action = jnp.clip(next_action, -0.999, 0.999)
        next_log_prob = next_pi.log_prob(next_action)
        # Max-entropy soft reward using the fresh action's log-prob.
        # Gated by (1 - done_alt) to match td0_targets (which also uses just
        # (1 - done) on the bootstrap). See td0_targets TODO re: brax time-limits.
        soft_reward = reward_alt - args.gamma * (1.0 - done_alt) * next_log_prob * alpha

        # V(next) averaged over num_action_samples draws from pi(.|x, g_alt).
        key, next_act_key = jax.random.split(key)
        next_sample_actions = next_pi.sample(
            seed=next_act_key, sample_shape=(args.num_action_samples,)
        )
        next_critic_output = critic.apply(
            critic_params,
            jnp.repeat(next_obs_alt[None, ...], args.num_action_samples, axis=0),
            next_sample_actions,
        )
        next_values = next_critic_output["value"].mean(0)
        next_emb = next_critic_output["embed"][0]

        return {
            "soft_reward": soft_reward,
            "next_value": next_values,
            "next_emb": jax.lax.stop_gradient(next_emb),
        }

    def compute_alt_td_lambda_targets_k(
        key, train_state, batch_raw, norm_state, g_alt_k, end_idx,
    ):
        """IS-corrected TD-lambda targets for one slot of hindsight goals.

        For each starting time t and each env, the target G_t^{alt} is built
        over the remainder of the same episode segment [t, end_idx[t]] with:
          * reward_u = 1{ ||achieved(next_obs_u) - g_alt|| < thresh }
          * soft reward subtracts gamma * alpha * log pi under g_alt
          * per-decision IS ratio
              rho_{u+1} = min(1, pi(a_{u+1} | x_{u+1}, g_alt)
                              / pi(a_{u+1} | x_{u+1}, g_orig))
            corrects for the action mismatch between g_alt and g_orig
          * at u = end_idx[t]: bootstrap only (truncation case uses V,
            otherwise (1-term)*V)
          * at u < end_idx[t] (interior):
              G_u = r_tilde_u + gamma * (1 - term_u)
                    * ((1 - lambda * rho) * V_u + lambda * rho * G_{u+1})

        Implemented as a single reverse scan over u so peak memory is O(S*E)
        rather than the O(S^2*E) of a naive per-t inner loop.

        Returns (stop_grad(G_final), mean_rho, stop_grad(next_emb_final)).
        """
        S, E = args.num_steps, args.num_envs
        actor_params = train_state.actor_state.params
        critic_params = train_state.critic_state.params
        alpha = jax.lax.stop_gradient(_alpha(actor_params))

        # Log-prob of each rolled-out action under the frozen behavior
        # policy evaluated at the original goal. Denominator of the IS ratio.
        norm_obs = normalizer.normalize(norm_state, batch_raw.obs)
        orig_pi = _actor_dist(actor_params, norm_obs)
        orig_log_probs = orig_pi.log_prob(
            jnp.clip(batch_raw.action, -1 + 1e-4, 1 - 1e-4)
        )

        t_idx = jnp.arange(S)[:, None]

        def scan_step(carry, u):
            """Process step u once; update G at every t that covers u."""
            G, rng, rho_sum, rho_count, next_emb_acc = carry
            rng, act_key = jax.random.split(rng)

            # Build next_obs with g_alt spliced in, then normalize it so
            # the networks receive properly-scaled inputs.
            raw_next = batch_raw.next_obs[u]
            raw_next_exp = jnp.broadcast_to(
                raw_next[None, :, :], (S, E, raw_next.shape[-1])
            )
            obs_alt_u = replace_goal_in_obs(raw_next_exp, g_alt_k)
            obs_alt_u = normalizer.normalize(norm_state, obs_alt_u)

            # Fresh action a' ~ pi(. | x_{u+1}, g_alt). Used for V_u and
            # for the boundary max-entropy correction.
            pi_alt = _actor_dist(actor_params, obs_alt_u)
            fresh_action = pi_alt.sample(seed=act_key)
            fresh_action = jnp.clip(fresh_action, -0.999, 0.999)
            fresh_log_prob = pi_alt.log_prob(fresh_action)

            # Rolled-out action at step u+1 (clamped to the open interval to
            # keep the tanh-Gaussian log-prob finite). Numerator of the IS
            # ratio is evaluated at this action.
            a_next = batch_raw.action[jnp.minimum(u + 1, S - 1)]
            a_next_clipped = jnp.clip(a_next, -1 + 1e-4, 1 - 1e-4)
            a_next_exp = jnp.broadcast_to(
                a_next_clipped[None, :, :], (S, E, a_next.shape[-1])
            )
            is_num_log = pi_alt.log_prob(a_next_exp)

            # Q(next_obs_alt, a') and the fused embedding at step u.
            critic_out = critic.apply(critic_params, obs_alt_u, fresh_action)
            V_u = critic_out["value"]
            embed_u = critic_out["embed"]

            # Hindsight reward at step u: 1 if the achieved state at u+1
            # is within threshold of g_alt, else 0.
            achieved = raw_next[:, goal_indices]
            dist = jnp.linalg.norm(achieved[None, :, :] - g_alt_k, axis=-1)
            reward_u = (dist < goal_reach_thresh).astype(jnp.float32)

            # Per-decision IS ratio, clipped at 1 for variance reduction.
            #   rho = min(1, pi(a_{u+1}|x_{u+1}, g_alt)
            #              / pi(a_{u+1}|x_{u+1}, g_orig))
            is_denom = orig_log_probs[jnp.minimum(u + 1, S - 1)]
            log_rho = is_num_log - is_denom[None, :]
            rho = jnp.minimum(1.0, jnp.exp(log_rho))

            # Terminal flags for step u. Under brax's EpisodeWrapper, done=1
            # fires at both intrinsic terminations and time-limits; truncated=1
            # disambiguates the time-limit case. term_mask zeros the entropy
            # correction ONLY at intrinsic terminations (done=1, truncated=0),
            # matching the jnp.where(trunc, V, (1-term)*V) bootstrap gate below.
            term_u = batch_raw.done[u]
            trunc_u = batch_raw.truncated[u]
            term_mask = 1.0 - term_u[None, :] * (1.0 - trunc_u[None, :])

            if args.sample_new_action_for_tdL:
                # Use the fresh-action log-prob for both boundary and interior.
                soft_r_bnd = reward_u - args.gamma * alpha * term_mask * fresh_log_prob
                soft_r_int = soft_r_bnd
            else:
                # Boundary: fresh-action log-prob. Interior: per-decision IS
                # estimator of -H[pi_{theta'}(.|x_{u+1}, g_alt)] applied to
                # Lemma 1 with f(a) = log pi_{theta'}(a | x_{u+1}, g_alt), i.e.
                # rho * log pi_{theta'}(a_obs | x_{u+1}, g_alt) = rho * is_num_log.
                soft_r_bnd = reward_u - args.gamma * alpha * term_mask * fresh_log_prob
                soft_r_int = reward_u - args.gamma * alpha * term_mask * rho * is_num_log

            # For each t, u is the segment boundary if u == end_idx[t, e],
            # and is an interior step if t <= u < end_idx[t, e].
            end = end_idx
            is_boundary = (u == end)
            is_interior = (t_idx <= u) & (u < end)

            valid_u = batch_raw.valid_mask[u]
            valid_b = valid_u[None, :]
            valid_f = valid_b.astype(rho.dtype)

            # Track the mean clipped IS ratio over interior ticks (logged).
            rho_sum = rho_sum + jnp.sum(rho * is_interior * valid_f)
            rho_count = rho_count + jnp.sum(is_interior * valid_f)

            # Boundary target: r_tilde + gamma * (trunc ? V : (1 - term) * V).
            G_bnd = soft_r_bnd + args.gamma * jnp.where(
                trunc_u[None, :], V_u, (1.0 - term_u[None, :]) * V_u
            )
            # Interior target: r_tilde + gamma * (1 - term)
            #                           * ((1 - lambda*rho)*V + lambda*rho*G).
            G_int = soft_r_int + args.gamma * jnp.where(
                trunc_u[None, :],
                V_u,
                (1.0 - term_u[None, :])
                * ((1.0 - args.lmbda * rho) * V_u + args.lmbda * rho * G),
            )

            # Allow the scan to write at episode-boundary ticks (gate ticks
            # carry truncated=1 with valid_mask=0) so G_bnd still flows
            # backward into the interior's lambda * rho * G term at earlier ticks.
            boundary_u = (term_u[None, :] > 0) | (trunc_u[None, :] > 0)
            write_allowed = valid_b | boundary_u
            G_new = jnp.where(
                is_boundary, G_bnd, jnp.where(is_interior, G_int, G)
            )
            G_new = jnp.where(write_allowed, G_new, G)

            # Save the fused embedding for the critic's auxiliary target,
            # but only at the original step u == t and only on valid ticks.
            write_emb = ((u == t_idx) & valid_b)[..., None]
            next_emb_acc = jnp.where(write_emb, embed_u, next_emb_acc)

            return (G_new, rng, rho_sum, rho_count, next_emb_acc), None

        # Reverse scan across u from S-1 down to 0 so G at the interior
        # recursion always sees the already-updated G_{u+1}.
        (G_final, _, rho_sum, rho_count, next_emb_final), _ = jax.lax.scan(
            scan_step,
            (
                jnp.zeros((S, E)),
                key,
                jnp.float32(0.0),
                jnp.float32(0.0),
                jnp.zeros((S, E, fusion_dim)),
            ),
            jnp.arange(S),
            reverse=True,
        )
        mean_rho = rho_sum / jnp.maximum(rho_count, 1.0)
        return (
            jax.lax.stop_gradient(G_final),
            mean_rho,
            jax.lax.stop_gradient(next_emb_final),
        )

    return (
        nstep_lambda,
        compute_extras,
        compute_extras_alt,
        compute_alt_td_lambda_targets_k,
    )


# =============================================================================
# Stagger helpers
# =============================================================================


def compute_stagger_schedule(args: "Args"):
    """Return (stagger_step_size, num_groups, group_idx) for the given args.

    Shared by :func:`wrap_for_training` (needs ``group_idx`` at wrap time so
    :class:`StaggeredResetWrapper` can stash it in ``state.info``) and by
    :func:`make_stagger_helpers` (warmup uses the same partitioning).
    """
    step_size = int(args.stagger_step_size) if args.stagger_step_size else int(args.num_steps)
    if step_size <= 0:
        raise ValueError(f"stagger_step_size must be positive, got {step_size}.")
    episode_length = int(args.episode_length)
    num_groups = max(-(-episode_length // step_size), 1)
    group_idx = jnp.arange(int(args.num_envs), dtype=jnp.int32) % num_groups
    return step_size, num_groups, group_idx


def make_stagger_helpers(args: "Args", env):
    """Factory returning stagger helpers (select_active_envs, stagger_env_state,
    compute_stagger_debug_metrics, maybe_print_stagger_debug_summary)."""

    def select_active_envs(mask, new_tree, old_tree):
        """Select updated leaves for active envs; keep frozen envs unchanged."""

        def select_leaf(new_leaf, old_leaf):
            if not hasattr(new_leaf, "shape"):
                return new_leaf
            if new_leaf.shape and new_leaf.shape[0] == mask.shape[0]:
                leaf_mask = jnp.reshape(
                    mask, (mask.shape[0],) + (1,) * (new_leaf.ndim - 1)
                )
                return jnp.where(leaf_mask, new_leaf, old_leaf)
            return new_leaf

        return jax.tree_util.tree_map(select_leaf, new_tree, old_tree)

    def _stagger_step_size() -> int:
        step_size, _, _ = compute_stagger_schedule(args)
        return step_size

    def maybe_print_stagger_debug_summary(env_state, stagger_info: dict) -> None:
        """Print a concise summary of initial stagger offsets for debugging."""
        if not stagger_info.get("enabled") or not args.stagger_debug:
            return
        steps = env_state.info.get("steps")
        if steps is None:
            print("Stagger debug init: env_state.info['steps'] is unavailable.")
            return

        step_size = int(stagger_info["step_size"])
        num_groups = int(stagger_info["num_groups"])
        actual_steps = np.asarray(steps, dtype=np.int32)
        expected_offsets = np.asarray(stagger_info["offset_steps"], dtype=np.int32)
        expected_groups = np.asarray(stagger_info["group_idx"], dtype=np.int32)
        actual_groups = np.clip(actual_steps // step_size, 0, num_groups - 1)

        preview_count = min(16, args.num_envs)
        preview = ", ".join(
            f"{env_idx}:{actual_steps[env_idx]}/{expected_offsets[env_idx]}"
            for env_idx in range(preview_count)
        )
        print(f"Stagger debug init: env_idx actual_step/expected_offset -> {preview}")
        print(
            "Stagger debug init: "
            f"exact_offset_match={np.mean(actual_steps == expected_offsets):.3f}, "
            f"phase_match={np.mean(actual_groups == expected_groups):.3f}"
        )

        expected_counts = np.bincount(expected_groups, minlength=num_groups)
        actual_counts = np.bincount(actual_groups, minlength=num_groups)
        warmup_done_counts = np.asarray(stagger_info.get("warmup_done_counts", []))
        if warmup_done_counts.size:
            print(
                "Stagger debug init: "
                f"warmup_done_total={int(warmup_done_counts.sum())}, "
                f"warmup_done_env_frac={np.mean(warmup_done_counts > 0):.3f}"
            )
        if num_groups <= 16:
            print(
                "Stagger debug init: "
                f"expected_group_counts={expected_counts.tolist()}, "
                f"actual_group_counts={actual_counts.tolist()}"
            )
        else:
            print(
                "Stagger debug init: "
                f"expected_group_count_range=[{expected_counts.min()}, "
                f"{expected_counts.max()}], "
                f"actual_group_count_range=[{actual_counts.min()}, {actual_counts.max()}]"
            )

    def compute_stagger_debug_metrics(transitions: RolloutTransition) -> dict:
        """Summarize rollout phase coverage via EpisodeWrapper's per-env step counter."""
        if not args.stagger_debug or not args.stagger_envs:
            return {}
        if "steps" not in transitions.extras:
            return {"stagger/missing_steps": jnp.array(1.0)}

        stagger_step_size = _stagger_step_size()
        num_groups = max(-(-int(args.episode_length) // stagger_step_size), 1)

        steps = transitions.extras["steps"].astype(jnp.int32)
        valid_mask = transitions.valid_mask.astype(jnp.bool_)
        valid_flat = valid_mask.reshape(-1)

        phase_idx = jnp.clip(
            jnp.maximum(steps - 1, 0) // stagger_step_size, 0, num_groups - 1
        )
        phase_weights = valid_flat.astype(jnp.float32)
        phase_counts = jnp.bincount(
            phase_idx.reshape(-1), weights=phase_weights, length=num_groups
        )
        phase_fracs = phase_counts / jnp.maximum(jnp.sum(phase_counts), 1.0)

        steps_for_minmax = jnp.where(valid_mask, steps, jnp.iinfo(jnp.int32).max)
        steps_for_maxmax = jnp.where(valid_mask, steps, jnp.iinfo(jnp.int32).min)
        any_valid = jnp.any(valid_mask)
        step_min = jnp.where(any_valid, jnp.min(steps_for_minmax), 0)
        step_max = jnp.where(any_valid, jnp.max(steps_for_maxmax), 0)

        is_waiting = transitions.extras.get("is_waiting")
        waiting_frac = (
            jnp.mean(is_waiting.astype(jnp.float32))
            if is_waiting is not None
            else jnp.array(0.0)
        )
        gate_frac = jnp.mean((~valid_mask).astype(jnp.float32)) - waiting_frac
        gate_frac = jnp.maximum(gate_frac, 0.0)

        metrics = {
            "stagger/step_min": step_min,
            "stagger/step_max": step_max,
            "stagger/active_groups": jnp.sum(phase_counts > 0),
            "stagger/group_coverage_frac": jnp.mean(phase_counts > 0),
            "stagger/group_frac_min": jnp.min(phase_fracs),
            "stagger/group_frac_max": jnp.max(phase_fracs),
            "stagger/valid_frac": jnp.mean(valid_mask.astype(jnp.float32)),
            "stagger/waiting_frac": waiting_frac,
            "stagger/gate_frac": gate_frac,
        }
        if num_groups <= 16:
            metrics.update(
                {
                    f"stagger/group_{group_idx:02d}_frac": phase_fracs[group_idx]
                    for group_idx in range(num_groups)
                }
            )
        return metrics

    def _activate_env_state(env_state, offset_steps):
        """Stamp the StaggeredResetWrapper bookkeeping fields for training.

        ``group_step = offset_steps`` makes each group's first reset gate
        fire at ``H - j*K`` ticks from warmup end, after which all groups
        run full-length episodes in lockstep (offset by ``K``).
        ``active = True`` switches the wrapper out of AutoReset passthrough.
        Safe no-op when :class:`StaggeredResetWrapper` is not installed
        (the keys simply overwrite with values the wrapper would ignore).
        """
        info = env_state.info
        if "group_step" in info:
            info["group_step"] = offset_steps
        if "is_waiting" in info:
            info["is_waiting"] = jnp.zeros_like(info["is_waiting"])
        if "valid_mask" in info:
            info["valid_mask"] = jnp.ones_like(info["valid_mask"])
        if "active" in info:
            info["active"] = jnp.ones_like(info["active"])
        return env_state

    def stagger_env_state(key, env_state):
        """Advance envs to staggered rollout offsets before training starts."""
        if not args.stagger_envs:
            return env_state, {"enabled": False}

        if args.stagger_mode != "grouped":
            raise ValueError(
                f"Unsupported stagger_mode={args.stagger_mode!r}; "
                "only 'grouped' is implemented."
            )
        if args.stagger_warmup_policy != "random":
            raise ValueError(
                f"Unsupported stagger_warmup_policy={args.stagger_warmup_policy!r}; "
                "only 'random' is implemented."
            )

        stagger_step_size, num_groups, group_idx = compute_stagger_schedule(args)
        max_offset = (num_groups - 1) * stagger_step_size
        offset_steps = group_idx * stagger_step_size

        if max_offset <= 0:
            env_state = _activate_env_state(env_state, offset_steps)
            return env_state, {
                "enabled": True,
                "step_size": stagger_step_size,
                "num_groups": num_groups,
                "max_offset": max_offset,
                "group_idx": group_idx,
                "offset_steps": offset_steps,
            }

        def warmup_step(carry, step_idx):
            rng, state, done_counts = carry
            rng, act_key = jax.random.split(rng)
            action = jax.random.uniform(
                act_key,
                (args.num_envs, env.action_size),
                minval=-1.0,
                maxval=1.0,
            )
            action = jnp.clip(action, -0.999, 0.999)
            next_state = env.step(state, action)
            active = step_idx < offset_steps
            done_mask = jnp.asarray(next_state.done > 0, dtype=jnp.bool_)
            done_counts = done_counts + (active & done_mask).astype(jnp.int32)
            state = select_active_envs(active, next_state, state)
            return (rng, state, done_counts), None

        (_, env_state, warmup_done_counts), _ = jax.lax.scan(
            warmup_step,
            init=(key, env_state, jnp.zeros(args.num_envs, dtype=jnp.int32)),
            xs=jnp.arange(max_offset, dtype=jnp.int32),
        )
        env_state = _activate_env_state(env_state, offset_steps)
        return env_state, {
            "enabled": True,
            "step_size": stagger_step_size,
            "num_groups": num_groups,
            "max_offset": max_offset,
            "group_idx": group_idx,
            "offset_steps": offset_steps,
            "warmup_done_counts": warmup_done_counts,
        }

    return (
        select_active_envs,
        stagger_env_state,
        compute_stagger_debug_metrics,
        maybe_print_stagger_debug_summary,
    )


# =============================================================================
# Losses + epoch runners
# =============================================================================


def make_loss_fns(args: "Args", actor: "Actor", critic: "Critic", action_size: int):
    """Factory returning (critic_loss, actor_loss, compute_policy_kl,
    update_critic, update_actor, run_epoch, run_epoch_separate)."""

    def _actor_dist(params, obs):
        mean, log_std = actor.apply(params, obs)
        return action_dist(mean, log_std)

    def critic_loss(params, train_state, minibatch):
        """HL-Gauss categorical cross-entropy loss plus next-emb / reward aux."""
        critic_output = critic.apply(params, minibatch.obs, minibatch.action)
        target_values = minibatch.extras["target_values"]
        # Project scalar TD-lambda targets into categorical HL-Gauss bins.
        target_cat = jax.vmap(hl_gauss, in_axes=(0, None, None, None))(
            target_values, args.num_bins, args.vmin, args.vmax
        )
        critic_pred = critic_output["logits"]
        # Main critic loss: cross-entropy between logits and soft two-hot target.
        critic_update_loss = optax.softmax_cross_entropy(critic_pred, target_cat)
        value = critic_output["value"]
        critic_mse = jnp.mean(optax.squared_error(value, target_values))

        # Auxiliary losses: predict the next-step fused embedding and the
        # next-step reward, regressed by MSE against stop-grad targets.
        # Masked by (1 - done) so terminal steps do not train the predictors.
        pred = critic_output["pred_features"]
        pred_rew = critic_output["pred_rew"]
        aux_loss = optax.squared_error(pred, minibatch.extras["next_emb"])
        aux_rew_loss = optax.squared_error(pred_rew, minibatch.reward.reshape(-1, 1))
        aux_loss = jnp.mean(
            (1 - minibatch.done.reshape(-1, 1))
            * jnp.concatenate([aux_loss, aux_rew_loss], axis=-1),
            axis=-1,
        )

        # Weighted mean over the minibatch: w = 1 for original samples,
        # w = 1/m_t (or 1) for hindsight samples. (1 - truncated) zeroes
        # out time-limit and invalid-slot transitions.
        w = minibatch.extras["weight"]
        loss = jnp.sum(
            w
            * (1.0 - minibatch.truncated)
            * (critic_update_loss + args.aux_loss_coeff * aux_loss)
        ) / jnp.maximum((w * (1.0 - minibatch.truncated)).sum(), 1.0)
        return loss, dict(
            value_mse=critic_mse,
            cross_entropy_loss=critic_update_loss.mean(),
            loss=loss,
            aux_loss=aux_loss.mean(),
            rew_aux_loss=aux_rew_loss.mean(),
            q=value.mean(),
            abs_batch_action=jnp.abs(minibatch.action).mean(),
            reward_mean=minibatch.reward.mean(),
            target_values=target_values.mean(),
        )

    def compute_policy_kl(minibatch, pi, old_pi):
        """Sample-based KL(old_pi || pi) estimator using 4 action samples."""
        # Draw 4 actions under old_pi; clamp them to the valid tanh range.
        old_pi_action, old_pi_act_log_prob = old_pi.sample_and_log_prob(
            sample_shape=(4,), seed=minibatch.extras["kl_key"]
        )
        old_pi_action = jnp.clip(old_pi_action, -1 + 1e-4, 1 - 1e-4)
        # KL(old || pi) ~ mean_a[ log old_pi(a) - log pi(a) ], a ~ old_pi.
        old_pi_act_log_prob = old_pi_act_log_prob.mean(0)
        pi_act_log_prob = pi.log_prob(old_pi_action).mean(0)
        kl = old_pi_act_log_prob - pi_act_log_prob
        return kl

    def actor_loss(params, train_state, minibatch):
        """Actor loss with KL trust region, entropy target, and Lagrangian update.

        Four additive pieces:
          * per_actor_loss  (Q-branch, pathwise): alpha * log pi(a') - Q(x, a')
              where a' ~ pi(.|x). Trains pi toward high soft-Q.
          * kl * stopgrad(lagrangian)  (KL branch, active when the estimated
              KL(old_pi || pi) >= kl_bound): pulls pi back toward the frozen
              behavior policy.
          * target_entropy_loss = temperature * stopgrad(entropy
              - entropy_target): grows alpha when entropy drops below target.
          * lagrangian_loss    = -lagrangian * stopgrad(kl - kl_bound):
              grows beta when the estimated KL exceeds kl_bound.
        """
        critic_params = train_state.critic_state.params
        actor_target_params = train_state.actor_target_params

        # Current and frozen behavior policies at the same observations.
        pi = _actor_dist(params, minibatch.obs)
        old_pi = _actor_dist(actor_target_params, minibatch.obs)

        w = minibatch.extras["weight"]
        w_sum = jnp.maximum(w.sum(), 1.0)

        # Per-sample KL(old_pi || pi) estimate.
        kl = compute_policy_kl(minibatch=minibatch, pi=pi, old_pi=old_pi)
        alpha = jax.lax.stop_gradient(jnp.exp(params["params"]["log_temperature"]))
        # Pathwise DPG-style Q-branch: sample a' from pi, evaluate Q(x, a').
        pred_action, log_prob = pi.sample_and_log_prob(
            seed=minibatch.extras["action_key"]
        )
        critic_pred = critic.apply(critic_params, minibatch.obs, pred_action)
        value = critic_pred["value"]
        # Q-branch loss: - Q(x, a') + alpha * log pi(a')
        per_actor_loss = log_prob * alpha - value
        entropy = -log_prob
        action_size_target = action_size * args.ent_target_mult
        lagrangian = jnp.exp(params["params"]["log_lagrangian"])

        # If KL < kl_bound: train with the Q-branch (pathwise).
        # Otherwise: train with the KL-branch, scaled by the detached Lagrangian.
        per_sample_loss = jnp.where(
            kl < args.kl_bound,
            per_actor_loss,
            kl * jax.lax.stop_gradient(lagrangian),
        )
        loss = jnp.sum(w * per_sample_loss) / w_sum

        # Entropy-temperature update: alpha grows when entropy drops below target.
        target_entropy = entropy - action_size_target
        temperature = jnp.exp(params["params"]["log_temperature"])
        target_entropy_loss = temperature * jax.lax.stop_gradient(target_entropy)

        # Lagrangian update: beta grows when the estimated KL exceeds kl_bound.
        lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl - args.kl_bound)

        loss += jnp.sum(w * target_entropy_loss) / w_sum
        loss += jnp.sum(w * lagrangian_loss) / w_sum
        return loss, dict(
            actor_loss=per_actor_loss.mean(),
            loss=loss.mean(),
            temp=temperature,
            abs_batch_action=jnp.abs(minibatch.action).mean(),
            abs_pred_action=jnp.abs(pred_action).mean(),
            reward_mean=minibatch.reward.mean(),
            kl=kl.mean(),
            lagrangian=lagrangian,
            lagrangian_loss=lagrangian_loss.mean(),
            entropy=entropy.mean(),
            entropy_loss=target_entropy_loss.mean(),
            target_values=minibatch.extras["target_values"].mean(),
            valid_frac=(w > 0).astype(jnp.float32).mean(),
        )

    def update_critic(train_state, minibatch):
        """Critic-only gradient step."""
        critic_grad_fn = jax.value_and_grad(critic_loss, has_aux=True)
        output, grads = critic_grad_fn(
            train_state.critic_state.params, train_state, minibatch
        )
        critic_train_state = train_state.critic_state.apply_gradients(grads=grads)
        train_state = train_state.replace(critic_state=critic_train_state)
        return train_state, prefix_dict("critic", output[1])

    def update_actor(train_state, minibatch):
        """Actor-only gradient step with NaN-to-num (not optax.zero_nans)."""
        actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
        output, grads = actor_grad_fn(
            train_state.actor_state.params, train_state, minibatch
        )
        grad_norm = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), grads)
        grad_norm = jax.tree_util.tree_reduce(lambda x, y: x + y, grad_norm)
        grads = jax.tree_util.tree_map(
            lambda x: jnp.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0), grads
        )
        actor_train_state = train_state.actor_state.apply_gradients(grads=grads)
        train_state = train_state.replace(actor_state=actor_train_state)
        return train_state, {
            **prefix_dict("actor", output[1]),
            "grad_norm": grad_norm,
        }

    def run_epoch(key, train_state, batch):
        """One epoch with joint critic+actor updates on the same minibatches."""
        batch_size = batch.obs.shape[0]
        mini_batch_size = (args.num_steps * args.num_envs) // args.num_mini_batches
        num_minibatches = batch_size // mini_batch_size
        key, shuffle_key, act_key, kl_key = jax.random.split(key, 4)
        indices = jax.random.permutation(shuffle_key, batch_size)
        minibatch_idxs = indices.reshape((num_minibatches, mini_batch_size))
        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.take(x, minibatch_idxs, axis=0), batch
        )
        minibatches.extras["action_key"] = jax.random.split(act_key, num_minibatches)
        minibatches.extras["kl_key"] = jax.random.split(kl_key, num_minibatches)

        def update_both(train_state, minibatch):
            train_state, critic_metrics = update_critic(train_state, minibatch)
            train_state, actor_metrics = update_actor(train_state, minibatch)
            return train_state, {**critic_metrics, **actor_metrics}

        train_state, metrics = jax.lax.scan(update_both, train_state, minibatches)
        metrics_mean = jax.tree_util.tree_map(lambda x: x.mean(0), metrics)
        return train_state, metrics_mean

    def run_epoch_separate(key, train_state, critic_batch, actor_batch):
        """One epoch with separate critic (1+her_k)·S·E and actor S·E loops."""
        mini_batch_size = (args.num_steps * args.num_envs) // args.num_mini_batches
        critic_size = (1 + args.her_k) * args.num_steps * args.num_envs
        actor_size = args.num_steps * args.num_envs
        num_critic_minibatches = critic_size // mini_batch_size
        num_actor_minibatches = actor_size // mini_batch_size

        key, critic_shuffle_key = jax.random.split(key)
        critic_indices = jax.random.permutation(critic_shuffle_key, critic_size)
        critic_mb_idxs = critic_indices.reshape(
            (num_critic_minibatches, mini_batch_size)
        )
        critic_minibatches = jax.tree_util.tree_map(
            lambda x: jnp.take(x, critic_mb_idxs, axis=0), critic_batch
        )

        key, actor_shuffle_key, act_key, kl_key = jax.random.split(key, 4)
        actor_indices = jax.random.permutation(actor_shuffle_key, actor_size)
        actor_mb_idxs = actor_indices.reshape(
            (num_actor_minibatches, mini_batch_size)
        )
        actor_minibatches = jax.tree_util.tree_map(
            lambda x: jnp.take(x, actor_mb_idxs, axis=0), actor_batch
        )
        actor_minibatches.extras["action_key"] = jax.random.split(
            act_key, num_actor_minibatches
        )
        actor_minibatches.extras["kl_key"] = jax.random.split(
            kl_key, num_actor_minibatches
        )

        def update_both(train_state, minibatches):
            critic_mb, actor_mb = minibatches
            train_state, critic_metrics = update_critic(train_state, critic_mb)
            train_state, actor_metrics = update_actor(train_state, actor_mb)
            return train_state, {**critic_metrics, **actor_metrics}

        paired_critic = jax.tree_util.tree_map(
            lambda x: x[:num_actor_minibatches], critic_minibatches
        )
        remaining_critic = jax.tree_util.tree_map(
            lambda x: x[num_actor_minibatches:], critic_minibatches
        )

        train_state, paired_metrics = jax.lax.scan(
            update_both, train_state, (paired_critic, actor_minibatches)
        )

        train_state, remaining_critic_metrics = jax.lax.scan(
            update_critic, train_state, remaining_critic
        )

        all_critic_metrics = jax.tree_util.tree_map(
            lambda p, r: jnp.concatenate([p, r], axis=0).mean(0),
            {k: v for k, v in paired_metrics.items() if k.startswith("critic")},
            remaining_critic_metrics,
        )
        actor_metrics_mean = jax.tree_util.tree_map(
            lambda x: x.mean(0),
            {k: v for k, v in paired_metrics.items() if not k.startswith("critic")},
        )

        return train_state, {**all_critic_metrics, **actor_metrics_mean}

    return (
        critic_loss,
        actor_loss,
        compute_policy_kl,
        update_critic,
        update_actor,
        run_epoch,
        run_epoch_separate,
    )


# =============================================================================
# Training state + checkpoint I/O
# =============================================================================


@flax.struct.dataclass
class ReppoTrainingState:
    """Training state: networks, optimizer state, RMS stats, and last env state."""
    env_steps: jnp.ndarray
    valid_env_steps: jnp.ndarray
    iteration: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    actor_target_params: Any
    normalization_state: NormalizationState
    last_env_state: State


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    """Save parameters in flax format (pickle)."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


# =============================================================================
# Training driver
#
# train(args): end-to-end trainer. Builds env, networks, normalizer,
# optimizers and helper closures, then runs an eval-interleaved outer loop:
#
#     for eval_epoch in range(num_evals):
#         training_epoch       # jit-scanned train_step over eval_interval iters
#         evaluator.run_evaluation
#         log + checkpoint
#
# Each train_step = one rollout + one learner_fn call (N_epochs of SGD
# over the collected batch).
# =============================================================================


def train(args: Args):
    # --- Validation -------------------------------------------------------
    if args.use_her_actor and not args.use_her_critic:
        raise ValueError("use_her_actor=True requires use_her_critic=True")
    if args.use_her_td_lambda and not args.use_her_critic:
        raise ValueError("use_her_td_lambda=True requires use_her_critic=True")

    assert (args.num_steps * args.num_envs) % args.num_mini_batches == 0, (
        f"num_steps*num_envs ({args.num_steps * args.num_envs}) must be "
        f"divisible by num_mini_batches ({args.num_mini_batches})"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- Env + eval env ---------------------------------------------------
    env_raw, obs_dim, goal_start_idx, goal_end_idx = make_env(args.env_id)
    args.obs_dim = obs_dim
    args.goal_start_idx = goal_start_idx
    args.goal_end_idx = goal_end_idx
    goal_dim = goal_end_idx - goal_start_idx
    goal_indices = jnp.arange(goal_start_idx, goal_end_idx)
    goal_reach_thresh = args.goal_reach_thresh

    if args.stagger_envs:
        _, _, stagger_group_idx = compute_stagger_schedule(args)
    else:
        stagger_group_idx = None
    env = wrap_for_training(
        env_raw,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        stagger_envs=args.stagger_envs,
        group_idx=stagger_group_idx,
    )

    eval_env_id = args.eval_env_id if args.eval_env_id else args.env_id
    eval_env_raw, _, _, _ = make_env(eval_env_id)
    eval_env = envs.training.wrap(eval_env_raw, episode_length=args.episode_length)

    obs_size = env.observation_size
    action_size = env.action_size
    print(f"obs_size: {obs_size}, action_size: {action_size}", flush=True)
    print(
        f"obs_dim: {args.obs_dim}, "
        f"goal_start_idx: {args.goal_start_idx}, goal_end_idx: {args.goal_end_idx}",
        flush=True,
    )

    # --- Networks + normalizer -------------------------------------------
    normalizer = Normalizer()
    actor = Actor(
        action_size=action_size,
        network_width=args.actor_network_width,
        network_depth=args.actor_depth,
        skip_connections=args.actor_skip_connections,
        use_relu=args.use_relu,
        kl_start=args.kl_start,
        ent_start=args.ent_start,
    )
    critic = Critic(
        action_size=action_size,
        network_width=args.critic_network_width,
        network_depth=args.critic_depth,
        skip_connections=args.critic_skip_connections,
        use_relu=args.use_relu,
        num_bins=args.num_bins,
        vmin=args.vmin,
        vmax=args.vmax,
    )

    # --- Optimizers -------------------------------------------------------
    if args.anneal_lr:
        actor_lr = optax.linear_schedule(
            init_value=args.actor_lr,
            end_value=0.0,
            transition_steps=args.total_time_steps,
        )
        critic_lr = optax.linear_schedule(
            init_value=args.critic_lr,
            end_value=0.0,
            transition_steps=args.total_time_steps,
        )
    else:
        actor_lr = args.actor_lr
        critic_lr = args.critic_lr
    actor_opt = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm), optax.adam(actor_lr)
    )
    critic_opt = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm), optax.adam(critic_lr)
    )

    # --- Helper factories -------------------------------------------------
    (
        replace_goal_in_obs,
        compute_goal_reward,
        her_augment,
        td0_targets,
    ) = make_her_helpers(args, goal_indices, goal_dim, goal_reach_thresh)
    (
        nstep_lambda,
        compute_extras,
        compute_extras_alt,
        compute_alt_td_lambda_targets_k,
    ) = make_td_lambda_helpers(
        args,
        actor,
        critic,
        normalizer,
        goal_indices,
        goal_reach_thresh,
        replace_goal_in_obs,
        fusion_dim=args.critic_network_width,
    )
    (
        select_active_envs,
        stagger_env_state,
        compute_stagger_debug_metrics,
        maybe_print_stagger_debug_summary,
    ) = make_stagger_helpers(args, env)
    (
        critic_loss,
        actor_loss,
        compute_policy_kl,
        update_critic,
        update_actor,
        run_epoch,
        run_epoch_separate,
    ) = make_loss_fns(args, actor, critic, action_size)

    # --- init_train_state -------------------------------------------------
    def init_train_state(key: jax.random.PRNGKey) -> ReppoTrainingState:
        key, actor_key, critic_key, env_key, stagger_key = jax.random.split(key, 5)

        dummy_obs = jnp.zeros((1, obs_size), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, action_size), dtype=jnp.float32)
        actor_params = actor.init(actor_key, dummy_obs)
        critic_params = critic.init(critic_key, dummy_obs, dummy_action)

        actor_state = TrainState.create(
            apply_fn=actor.apply, params=actor_params, tx=actor_opt
        )
        critic_state = TrainState.create(
            apply_fn=critic.apply, params=critic_params, tx=critic_opt
        )

        normalization_state = normalizer.init(jnp.zeros(obs_size, dtype=jnp.float32))

        env_keys = jax.random.split(env_key, args.num_envs)
        env_state = env.reset(env_keys)
        env_state, stagger_info = stagger_env_state(stagger_key, env_state)
        if stagger_info.get("enabled"):
            print(
                "Training env staggering enabled: "
                f"step_size={stagger_info['step_size']}, "
                f"num_groups={stagger_info['num_groups']}, "
                f"max_offset={stagger_info['max_offset']}",
                flush=True,
            )
            maybe_print_stagger_debug_summary(env_state, stagger_info)

        return ReppoTrainingState(
            env_steps=jnp.int32(0),
            valid_env_steps=jnp.int32(0),
            iteration=jnp.int32(0),
            actor_state=actor_state,
            critic_state=critic_state,
            actor_target_params=actor_params,
            normalization_state=normalization_state,
            last_env_state=env_state,
        )

    # --- Policy + rollout -------------------------------------------------
    def policy(key, obs, train_state: ReppoTrainingState):
        """Stochastic rollout policy: normalize obs, sample from tanh-Gaussian."""
        obs = normalizer.normalize(train_state.normalization_state, obs)
        mean, log_std = actor.apply(train_state.actor_state.params, obs)
        pi = action_dist(mean, log_std)
        return pi.sample(seed=key)

    def collect_rollout(key, train_state: ReppoTrainingState):
        """Scan num_steps env steps with the current stochastic policy.

        Each tick: sample a ~ pi(.|x), clamp to the tanh-Gaussian's valid
        range, step the env, and emit a RolloutTransition. ``next_obs`` is
        the env's ``raw_obs`` (pre-auto-reset next observation), so HER can
        still read the true achieved goal even on ticks that just terminated.
        """
        def step_env(carry, _):
            key, env_state = carry
            key, act_key = jax.random.split(key)
            action = policy(act_key, env_state.obs, train_state)
            # Clip strictly inside the tanh range for a finite log-prob.
            action = jnp.clip(action, -0.999, 0.999)
            next_env_state = env.step(env_state, action)
            # valid_mask is only meaningful under staggered resets; it is
            # all-True under AutoResetWrapper (the default).
            valid_mask = next_env_state.info.get(
                "valid_mask",
                jnp.ones_like(next_env_state.done, dtype=jnp.bool_),
            )
            transition = RolloutTransition(
                obs=env_state.obs,
                action=action,
                reward=next_env_state.reward,
                next_obs=next_env_state.info["raw_obs"],
                done=next_env_state.done,
                truncated=next_env_state.info["truncation"],
                valid_mask=valid_mask,
                extras={
                    **next_env_state.info,
                    "train_metrics": next_env_state.metrics,
                    "train_done_mask": next_env_state.done.astype(jnp.bool_),
                },
            )
            return (key, next_env_state), transition

        rollout_state, transitions = jax.lax.scan(
            f=step_env,
            init=(key, train_state.last_env_state),
            xs=None,
            length=args.num_steps,
        )
        _, last_env_state = rollout_state
        # Track both total env steps and valid-only env steps (they differ
        # only under staggered resets, where frozen ticks do not count).
        valid_count = jnp.sum(transitions.valid_mask.astype(jnp.int32))
        train_state = train_state.replace(
            last_env_state=last_env_state,
            env_steps=train_state.env_steps + args.num_steps * args.num_envs,
            valid_env_steps=train_state.valid_env_steps + valid_count,
        )
        return transitions, train_state

    # --- Learner ----------------------------------------------------------
    def _learner_fn_her_per_epoch(
        key, train_state: ReppoTrainingState, batch: RolloutTransition
    ):
        """Per-epoch HER variant of ``learner_fn`` (v7b).

        Resamples HER goals and recomputes hindsight TD-lambda targets at the
        top of every inner epoch, instead of once per outer iteration. The
        original-goal TD-lambda targets still use the start-of-iteration
        critic, but the hindsight targets pick up within-iteration critic
        updates (by design). The target policy pi_{theta'} stays frozen
        across epochs -- we swap it in at every hindsight-target call so the
        existing helper keeps reading ``train_state.actor_state.params``.

        See ``plans/algorithms/reppo_gcrl_algorithm_v7b.md`` for the
        pseudocode. Activated by ``--her-per-epoch`` combined with
        ``--use-her-critic``.
        """
        S, E = args.num_steps, args.num_envs
        N = S * E
        k = args.her_k

        # Keep the un-normalized batch for per-epoch HER + hindsight TD-lambda.
        batch_raw = batch

        # --- Update observation normalizer, then normalize with *pre-update*
        #     stats so every obs fed to the networks this iteration matches
        #     the stats the targets were computed against (matches v6). ---
        new_norm_state = normalizer.update_masked(
            train_state.normalization_state, batch.obs, batch.valid_mask
        )
        norm_state = train_state.normalization_state
        batch = batch.replace(
            obs=normalizer.normalize(norm_state, batch.obs),
            next_obs=normalizer.normalize(norm_state, batch.next_obs),
        )
        train_state = train_state.replace(normalization_state=new_norm_state)

        # --- v7b Step 2: Original-goal TD-lambda targets (once per iter). ---
        key, orig_key = jax.random.split(key)
        extras_orig = compute_extras(key=orig_key, train_state=train_state, batch=batch)
        batch.extras.update(extras_orig)
        batch.extras["target_values"] = jax.lax.stop_gradient(nstep_lambda(batch=batch))

        # --- Freeze pi_{theta'} for the rest of this iteration. ---
        train_state = train_state.replace(
            actor_target_params=train_state.actor_state.params
        )

        # --- Flatten the per-iteration original slice once. ---
        batch.extras["weight"] = batch.valid_mask.astype(jnp.float32)
        orig_flat = jax.tree_util.tree_map(
            lambda x: x.reshape((N, *x.shape[2:])), batch
        )
        orig_stripped = orig_flat.replace(
            extras={
                "target_values": orig_flat.extras["target_values"],
                "weight": orig_flat.valid_mask.astype(jnp.float32),
                "next_emb": orig_flat.extras["next_emb"],
            }
        )

        # --- v7b Step 4: Epoch loop, with HER + hindsight TD-lambda refreshed
        #     at the top of every epoch. ---
        key, train_key = jax.random.split(key)

        def epoch_step(ts, epoch_key):
            her_key, alt_key, sgd_key = jax.random.split(epoch_key, 3)

            # 4.1: Fresh HER goals this epoch (policy-independent).
            (
                obs_alt_raw,
                next_obs_alt_raw,
                reward_alt,
                done_alt,
                truncated_alt,
                weight_alt,
                g_alt,
                end_idx,
            ) = her_augment(her_key, batch_raw)
            obs_alt = normalizer.normalize(norm_state, obs_alt_raw)
            next_obs_alt = normalizer.normalize(norm_state, next_obs_alt_raw)

            # 4.2: Hindsight TD targets use the *current* critic but the
            # *frozen* actor. compute_alt_td_lambda_targets_k (and
            # compute_extras_alt) read actor params from
            # train_state.actor_state.params, so we locally swap in
            # actor_target_params without touching the helper itself.
            ts_for_targets = ts.replace(
                actor_state=ts.actor_state.replace(params=ts.actor_target_params)
            )

            if args.use_her_td_lambda:
                def _compute_alt_targets_lambda(rng, g_alt_single_k):
                    rng, k_key = jax.random.split(rng)
                    tv_k, mean_rho_k, next_emb_k = compute_alt_td_lambda_targets_k(
                        key=k_key,
                        train_state=ts_for_targets,
                        batch_raw=batch_raw,
                        norm_state=norm_state,
                        g_alt_k=g_alt_single_k,
                        end_idx=end_idx,
                    )
                    return rng, (tv_k, mean_rho_k, next_emb_k)

                _, (target_values_alt_k, mean_rho_per_k, next_emb_alt_k) = (
                    jax.lax.scan(
                        _compute_alt_targets_lambda,
                        alt_key,
                        jnp.moveaxis(g_alt, 2, 0),
                    )
                )
                is_ratio_mean_epoch = jnp.mean(mean_rho_per_k)
                target_values_alt = jnp.moveaxis(target_values_alt_k, 0, 2)
                next_emb_alt = jnp.moveaxis(next_emb_alt_k, 0, 2)
            else:
                def _compute_alt_targets(rng, inputs):
                    obs_k, next_obs_k, reward_k, done_k = inputs
                    rng, k_key = jax.random.split(rng)
                    extras_k = compute_extras_alt(
                        key=k_key,
                        train_state=ts_for_targets,
                        obs_alt=obs_k,
                        next_obs_alt=next_obs_k,
                        reward_alt=reward_k,
                        done_alt=done_k,
                    )
                    tv_k = td0_targets(
                        extras_k["soft_reward"], extras_k["next_value"], done_k
                    )
                    return rng, (tv_k, extras_k["next_emb"])

                _, (target_values_alt_k, next_emb_alt_k) = jax.lax.scan(
                    _compute_alt_targets,
                    alt_key,
                    (
                        jnp.moveaxis(obs_alt, 2, 0),
                        jnp.moveaxis(next_obs_alt, 2, 0),
                        jnp.moveaxis(reward_alt, 2, 0),
                        jnp.moveaxis(done_alt, 2, 0),
                    ),
                )
                is_ratio_mean_epoch = None
                target_values_alt = jnp.moveaxis(target_values_alt_k, 0, 2)
                next_emb_alt = jnp.moveaxis(next_emb_alt_k, 0, 2)

            # 4.3: Build the hindsight slice and concatenate with the
            # per-iteration original slice.
            action_alt = jnp.broadcast_to(
                batch.action[:, :, None, :],
                (S, E, k, batch.action.shape[-1]),
            )
            valid_mask_alt = jnp.broadcast_to(
                batch.valid_mask[:, :, None], (S, E, k)
            )
            alt_batch = RolloutTransition(
                obs=obs_alt,
                action=action_alt,
                reward=reward_alt,
                next_obs=next_obs_alt,
                done=done_alt,
                truncated=truncated_alt,
                valid_mask=valid_mask_alt,
                extras={
                    "target_values": jax.lax.stop_gradient(target_values_alt),
                    "weight": weight_alt * valid_mask_alt.astype(jnp.float32),
                    "next_emb": jax.lax.stop_gradient(next_emb_alt),
                },
            )
            alt_flat = jax.tree_util.tree_map(
                lambda x: x.reshape((k * N, *x.shape[3:])), alt_batch
            )
            combined_batch = jax.tree_util.tree_map(
                lambda o, a: jnp.concatenate([o, a], axis=0),
                orig_stripped,
                alt_flat,
            )

            # 4.4: Shuffle + mini-batch critic/actor updates (unchanged
            # helpers).
            if args.use_her_actor:
                ts, epoch_metrics = run_epoch(sgd_key, ts, combined_batch)
            else:
                ts, epoch_metrics = run_epoch_separate(
                    sgd_key, ts, combined_batch, orig_flat
                )

            if args.use_her_td_lambda:
                epoch_metrics["critic/is_ratio_mean"] = is_ratio_mean_epoch
            return ts, epoch_metrics

        train_state, update_metrics = jax.lax.scan(
            epoch_step,
            train_state,
            jax.random.split(train_key, args.num_epochs),
        )
        update_metrics = jax.tree_util.tree_map(lambda x: x[-1], update_metrics)
        return train_state, update_metrics

    def learner_fn(key, train_state: ReppoTrainingState, batch: RolloutTransition):
        """Process one rollout batch: build targets, then run N_epochs of SGD."""
        if args.her_per_epoch and args.use_her_critic:
            return _learner_fn_her_per_epoch(key, train_state, batch)

        S, E = args.num_steps, args.num_envs
        N = S * E
        k = args.her_k

        # --- 1. HER augmentation: sample k hindsight goals per step ---
        if args.use_her_critic:
            key, her_key = jax.random.split(key)
            (
                obs_alt_raw,
                next_obs_alt_raw,
                reward_alt,
                done_alt,
                truncated_alt,
                weight_alt,
                g_alt,
                end_idx,
            ) = her_augment(her_key, batch)

        # Save the raw (un-normalized) batch; compute_alt_td_lambda_targets_k
        # needs it because it normalizes obs internally with different goals.
        if args.use_her_td_lambda:
            batch_raw = batch

        # --- 2. Update running observation normalizer (mask-aware) ---
        # The update uses the fresh batch; the batch itself is normalized
        # below with the *pre-update* stats so targets and losses are
        # consistent with the stats the networks were trained on.
        new_norm_state = normalizer.update_masked(
            train_state.normalization_state, batch.obs, batch.valid_mask
        )
        norm_state = train_state.normalization_state

        # --- 3. Normalize observations for the networks ---
        if args.use_her_critic:
            obs_alt = normalizer.normalize(norm_state, obs_alt_raw)
            next_obs_alt = normalizer.normalize(norm_state, next_obs_alt_raw)
        batch = batch.replace(
            obs=normalizer.normalize(norm_state, batch.obs),
            next_obs=normalizer.normalize(norm_state, batch.next_obs),
        )
        train_state = train_state.replace(normalization_state=new_norm_state)

        # --- 4. Original trajectory: soft rewards, V(next), and TD-lambda targets ---
        key, orig_key = jax.random.split(key)
        extras_orig = compute_extras(key=orig_key, train_state=train_state, batch=batch)
        batch.extras.update(extras_orig)
        batch.extras["target_values"] = jax.lax.stop_gradient(nstep_lambda(batch=batch))

        # --- 5. Freeze the target policy used by the actor's KL branch ---
        train_state = train_state.replace(
            actor_target_params=train_state.actor_state.params
        )

        is_ratio_mean = None
        # --- 6. Hindsight targets: IS-corrected TD-lambda or plain TD(0) ---
        if args.use_her_critic:
            key, alt_key = jax.random.split(key)

            if args.use_her_td_lambda:
                # Full TD-lambda target over the remaining segment with
                # per-decision IS ratio. One scan step per hindsight slot.
                def _compute_alt_targets_lambda(rng, g_alt_single_k):
                    rng, k_key = jax.random.split(rng)
                    tv_k, mean_rho_k, next_emb_k = compute_alt_td_lambda_targets_k(
                        key=k_key,
                        train_state=train_state,
                        batch_raw=batch_raw,
                        norm_state=norm_state,
                        g_alt_k=g_alt_single_k,
                        end_idx=end_idx,
                    )
                    return rng, (tv_k, mean_rho_k, next_emb_k)

                _, (target_values_alt_k, mean_rho_per_k, next_emb_alt_k) = jax.lax.scan(
                    _compute_alt_targets_lambda,
                    alt_key,
                    jnp.moveaxis(g_alt, 2, 0),
                )
                is_ratio_mean = jnp.mean(mean_rho_per_k)
                target_values_alt = jnp.moveaxis(target_values_alt_k, 0, 2)
                next_emb_alt = jnp.moveaxis(next_emb_alt_k, 0, 2)
            else:
                # Simpler one-step TD(0) target using a fresh action under g_alt.
                def _compute_alt_targets(rng, inputs):
                    obs_k, next_obs_k, reward_k, done_k = inputs
                    rng, k_key = jax.random.split(rng)
                    extras_k = compute_extras_alt(
                        key=k_key,
                        train_state=train_state,
                        obs_alt=obs_k,
                        next_obs_alt=next_obs_k,
                        reward_alt=reward_k,
                        done_alt=done_k,
                    )
                    tv_k = td0_targets(
                        extras_k["soft_reward"], extras_k["next_value"], done_k
                    )
                    return rng, (tv_k, extras_k["next_emb"])

                _, (target_values_alt_k, next_emb_alt_k) = jax.lax.scan(
                    _compute_alt_targets,
                    alt_key,
                    (
                        jnp.moveaxis(obs_alt, 2, 0),
                        jnp.moveaxis(next_obs_alt, 2, 0),
                        jnp.moveaxis(reward_alt, 2, 0),
                        jnp.moveaxis(done_alt, 2, 0),
                    ),
                )
                target_values_alt = jnp.moveaxis(target_values_alt_k, 0, 2)
                next_emb_alt = jnp.moveaxis(next_emb_alt_k, 0, 2)

            # --- 7. Build the combined batch: original (S*E rows, w=1) plus
            #        hindsight (k*S*E rows, w = 1 or 1/m_t) flattened over time/env ---
            batch.extras["weight"] = batch.valid_mask.astype(jnp.float32)
            orig_flat = jax.tree_util.tree_map(lambda x: x.reshape((N, *x.shape[2:])), batch)

            # The behavior action is the same for every hindsight slot at a
            # given (t, e), so broadcast it across the k axis.
            action_alt = jnp.broadcast_to(
                batch.action[:, :, None, :], (S, E, k, batch.action.shape[-1])
            )
            valid_mask_alt = jnp.broadcast_to(batch.valid_mask[:, :, None], (S, E, k))
            alt_batch = RolloutTransition(
                obs=obs_alt,
                action=action_alt,
                reward=reward_alt,
                next_obs=next_obs_alt,
                done=done_alt,
                truncated=truncated_alt,
                valid_mask=valid_mask_alt,
                extras={
                    "target_values": jax.lax.stop_gradient(target_values_alt),
                    "weight": weight_alt * valid_mask_alt.astype(jnp.float32),
                    "next_emb": jax.lax.stop_gradient(next_emb_alt),
                },
            )
            alt_flat = jax.tree_util.tree_map(
                lambda x: x.reshape((k * N, *x.shape[3:])), alt_batch
            )

            orig_stripped = orig_flat.replace(
                extras={
                    "target_values": orig_flat.extras["target_values"],
                    "weight": orig_flat.valid_mask.astype(jnp.float32),
                    "next_emb": orig_flat.extras["next_emb"],
                }
            )
            combined_batch = jax.tree_util.tree_map(
                lambda o, a: jnp.concatenate([o, a], axis=0), orig_stripped, alt_flat
            )

            # --- 8. Run num_epochs of SGD on the combined batch ---
            if args.use_her_actor:
                # Actor sees hindsight samples too: both critic and actor use
                # the full combined batch.
                key, train_key = jax.random.split(key)
                train_state, update_metrics = jax.lax.scan(
                    f=lambda ts, k_: run_epoch(k_, ts, combined_batch),
                    init=train_state,
                    xs=jax.random.split(train_key, args.num_epochs),
                )
            else:
                # Actor sees only the original S*E samples; critic sees the
                # combined (1 + k)*S*E batch.
                key, train_key = jax.random.split(key)
                train_state, update_metrics = jax.lax.scan(
                    f=lambda ts, k_: run_epoch_separate(
                        k_, ts, combined_batch, orig_flat
                    ),
                    init=train_state,
                    xs=jax.random.split(train_key, args.num_epochs),
                )
        else:
            # No HER: flatten the S x E batch and run num_epochs of joint
            # critic + actor SGD on the original trajectory alone.
            batch.extras["weight"] = batch.valid_mask.astype(jnp.float32)
            batch = jax.tree_util.tree_map(lambda x: x.reshape((N, *x.shape[2:])), batch)

            key, train_key = jax.random.split(key)
            train_state, update_metrics = jax.lax.scan(
                f=lambda ts, k_: run_epoch(k_, ts, batch),
                init=train_state,
                xs=jax.random.split(train_key, args.num_epochs),
            )

        update_metrics = jax.tree_util.tree_map(lambda x: x[-1], update_metrics)
        if args.use_her_td_lambda and is_ratio_mean is not None:
            update_metrics["critic/is_ratio_mean"] = is_ratio_mean
        return train_state, update_metrics

    # --- train_step + training_epoch -------------------------------------
    def train_step(state: ReppoTrainingState, key):
        """One iteration = 1 rollout (num_steps env steps) + 1 learner_fn call."""
        key, rollout_key, learn_key = jax.random.split(key, 3)
        transitions, state = collect_rollout(key=rollout_key, train_state=state)

        done_mask = transitions.extras["train_done_mask"]
        train_episode_metrics = jax.tree_util.tree_map(
            lambda x: jnp.mean(x, where=done_mask),
            transitions.extras["train_metrics"],
        )

        state, update_metrics = learner_fn(
            key=learn_key, train_state=state, batch=transitions
        )
        metrics = {
            **update_metrics,
            **compute_stagger_debug_metrics(transitions),
            **prefix_dict("episode", train_episode_metrics),
        }
        state = state.replace(iteration=state.iteration + 1)
        return state, metrics

    @jax.jit
    def training_epoch(train_state, keys):
        """Scan train_step for eval_interval iterations under a single jit."""
        train_state, metrics = jax.lax.scan(train_step, train_state, keys)
        metrics = jax.tree_util.tree_map(lambda x: x[-1], metrics)
        return train_state, metrics

    # --- Eval wiring (CrlEvaluator) --------------------------------------
    def deterministic_actor_step(training_state: ReppoTrainingState, env_, env_state, extra_fields=()):
        """Evaluator step: pick the tanh-squashed mean action (no sampling)."""
        obs = normalizer.normalize(training_state.normalization_state, env_state.obs)
        actions = actor.apply(
            training_state.actor_state.params, obs, deterministic=True
        )
        actions = jnp.clip(actions, -0.999, 0.999)
        nstate = env_.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )

    # --- Checkpointing helpers ------------------------------------------
    save_dir = Path(args.wandb_dir) / "checkpoints" / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint(train_state: ReppoTrainingState, tag: str):
        """Pickle actor/critic/target/normalizer params to save_dir/step_<tag>.pkl."""
        if not args.checkpoint:
            return
        params = (
            train_state.actor_state.params,
            train_state.critic_state.params,
            train_state.actor_target_params,
            train_state.normalization_state,
        )
        path = str(save_dir / f"step_{tag}.pkl")
        save_params(path, params)

    # --- capture_vis -----------------------------------------------------
    def capture_vis_to_wandb(train_state: ReppoTrainingState, step: int):
        """Roll out num_render deterministic episodes and log Brax HTML viz to W&B."""
        if not args.capture_vis:
            return
        print(f"Capturing vis at step {step}...", flush=True)

        vis_env = env_raw

        @jax.jit
        def reset_one(k):
            return vis_env.reset(k)

        @jax.jit
        def policy_step_vis(env_state, norm_state, actor_params):
            obs = normalizer.normalize(norm_state, env_state.obs)
            actions = actor.apply(actor_params, obs, deterministic=True)
            actions = jnp.clip(actions, -0.999, 0.999)
            next_state = vis_env.step(env_state, actions)
            return next_state, env_state

        rng = jax.random.PRNGKey(args.seed + step)
        for render_idx in range(args.num_render):
            rng, reset_key = jax.random.split(rng)
            env_state = reset_one(reset_key)
            states = [env_state]
            for _ in range(args.vis_length):
                env_state, prev_state = policy_step_vis(
                    env_state,
                    train_state.normalization_state,
                    train_state.actor_state.params,
                )
                states.append(env_state)
            try:
                html_str = html.render(vis_env.sys, [s.pipeline_state for s in states])
                if wandb.run is not None:
                    wandb.log(
                        {f"vis/render_{render_idx}": wandb.Html(html_str)}, step=step
                    )
            except Exception as exc:
                print(f"capture_vis render failed: {exc}", flush=True)

    # --- Training loop ---------------------------------------------------
    key = jax.random.PRNGKey(args.seed)
    key, init_key, eval_key = jax.random.split(key, 3)
    train_state = init_train_state(init_key)

    evaluator = CrlEvaluator(
        deterministic_actor_step,
        eval_env,
        num_eval_envs=args.num_eval_envs,
        episode_length=args.episode_length,
        key=eval_key,
    )

    total_iterations = args.total_time_steps // (args.num_steps * args.num_envs)
    eval_interval = max(-(-total_iterations // args.num_eval), 1)
    num_evals = -(-total_iterations // eval_interval)

    trigger_sync = None
    if wandb.run is not None and args.wandb_mode == "offline":
        trigger_sync = TriggerWandbSyncHook()

    training_walltime = 0.0
    for eval_epoch in range(num_evals):
        t = time.time()
        epoch_key, key = jax.random.split(key)
        train_state, train_metrics = training_epoch(
            train_state, jax.random.split(epoch_key, eval_interval)
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), train_metrics)
        epoch_time = time.time() - t
        training_walltime += epoch_time

        sps = (eval_interval * args.num_steps * args.num_envs) / epoch_time
        env_step_count = int(train_state.env_steps)
        valid_step_count = int(train_state.valid_env_steps)
        training_metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            "training/env_steps": env_step_count,
            "training/valid_env_steps": valid_step_count,
            "training/valid_env_steps_frac": (
                valid_step_count / max(1, env_step_count)
            ),
            **{f"training/{k}": v for k, v in train_metrics.items()},
        }

        metrics = evaluator.run_evaluation(train_state, training_metrics)
        current_step = valid_step_count if args.stagger_envs else env_step_count
        log_metrics(current_step, eval_epoch, metrics)

        # Checkpoint cadence: first 5 eval epochs, last 5, and every 10th.
        should_checkpoint = (
            eval_epoch < 5
            or eval_epoch >= num_evals - 5
            or eval_epoch % 10 == 0
        )
        if should_checkpoint:
            checkpoint(train_state, tag=str(env_step_count))

        if trigger_sync is not None:
            trigger_sync()

    # Final checkpoint + optional capture_vis
    final_env_step = int(train_state.env_steps)
    final_valid_step = int(train_state.valid_env_steps)
    checkpoint(train_state, tag=f"final_{final_env_step}")
    if args.capture_vis:
        vis_step = final_valid_step if args.stagger_envs else final_env_step
        capture_vis_to_wandb(train_state, step=vis_step)
        if trigger_sync is not None:
            trigger_sync()

    return train_state


# =============================================================================
# main (tyro.cli entrypoint)
# =============================================================================


def main():
    args = tyro.cli(Args)

    # Build a compact run_name that summarizes the HER flags, target type,
    # k (number of hindsight goals per step), and actor depth.
    name_parts = []
    if args.use_her_critic:
        name_parts.append("c")
    if args.use_her_actor:
        name_parts.append("a")
    name_parts.append("tdL" if args.use_her_td_lambda else "td0")
    name_parts.append(f"k{args.her_k}")
    name_parts.append(f"d{args.actor_depth}")
    run_name = "_".join(name_parts)

    group_parts = list(name_parts)
    if args.stagger_envs:
        group_parts.append("stag")
    if args.sample_new_action_for_tdL:
        group_parts.append("sample-new-action")
    group_parts.append(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    auto_group = "_".join(group_parts)

    if args.track:
        wandb_group = auto_group if args.wandb_group == "." else args.wandb_group
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            group=wandb_group,
            mode=args.wandb_mode,
            dir=args.wandb_dir,
            config=vars(args),
        )

    try:
        train(args)
    finally:
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
