import math
import re

import flax.linen as nn
import flax.struct as struct
import jax.numpy as jnp

import openpi.shared.array_typing as at


@struct.dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    # LoRA rank.
    rank: int
    # LoRA scaling factor.
    alpha: float = 1.0
    # Initialization function for LoRA parameters.
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732
    rslora: bool = False
    # Axes in the weight to apply LoRA to. Should typically be the last two axes.
    axes: tuple[int, int] = (-2, -1)
    # Axis label which is used by LoRA in einsum equations. Must not be present in the original equation.
    label: str = "L"
    # Conditional LoRA (LoRAdapter, Stracke et al. 2024): if set, the rank-r bottleneck A x is
    # FiLM-modulated by a per-sample condition c of this size, (1 + gamma(c)) * A x + beta(c), with
    # gamma/beta bias-free linear maps c -> r ("lora_film_*" params, so the LoRA freeze filter and
    # weight loader treat them like the other LoRA params). The condition is passed at call time.
    cond_dim: int | None = None

    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


class Einsum(nn.Module):
    """Einsum with LoRA support. Can be used as a drop-in replacement for the Gemma Einsum."""

    # Shape of the weight.
    shape: tuple[int, ...]
    # Initialization function for the weight.
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w = self.param("w", self.init_fn, self.shape)

        if config := self.lora_config:
            # Setup LoRA parameters.
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            self.w_b = self.param("lora_b", config.init_fn, shape_b)
            if config.cond_dim:
                self.film = _film_params(self, config.cond_dim, config.rank)

    @nn.compact
    def __call__(self, eqn: str, x, cond=None):
        dtype = x.dtype  # original dtype, could be half-precision
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if config := self.lora_config:
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            if config.cond_dim and cond is not None:
                # broadcast the per-sample (B, r) FiLM over every other axis of A x
                a_out = eqn_a.split("->")[1]
                shape = [1] * len(a_out)
                shape[a_out.index("B")] = cond.shape[0]
                shape[a_out.index(config.label)] = config.rank
                lora = _apply_film(lora, cond, self.film, shape)
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            result = result + lora * config.scaling_value

        return result

    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()

        assert self.lora_config is not None
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)
        label = self.lora_config.label

        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b


class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.lora_config:
            # Setup LoRA parameters.
            # TODO: follow up with a simplified init_fn api.
            self.w_gating_lora = (
                self.param("gating_einsum_lora_a", self.lora_config.init_fn, (2, self.features, self.lora_config.rank)),
                self.param(
                    "gating_einsum_lora_b", self.lora_config.init_fn, (2, self.lora_config.rank, self.hidden_dim)
                ),
            )
            self.w_linear_lora = (
                self.param("linear_lora_a", self.lora_config.init_fn, (self.hidden_dim, self.lora_config.rank)),
                self.param("linear_lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features)),
            )
        self.film_gating = self.film_linear = None
        if self.lora_config and self.lora_config.cond_dim:
            c, r = self.lora_config.cond_dim, self.lora_config.rank
            self.film_gating = (
                _film_params(self, c, r, "lora_film_gating_0"),
                _film_params(self, c, r, "lora_film_gating_1"),
            )
            self.film_linear = _film_params(self, c, r, "lora_film_linear")

    @nn.compact
    def __call__(self, x, cond=None):
        dtype = x.dtype  # original dtype, could be half-precision
        use_film = self.film_gating is not None and cond is not None
        ff_gate = self._dot(
            x,
            self.w_gating[0],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
            (cond, self.film_gating[0]) if use_film else None,
        )
        gate_value = nn.gelu(ff_gate)

        ff1 = self._dot(
            x,
            self.w_gating[1],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
            (cond, self.film_gating[1]) if use_film else None,
        )
        activations = gate_value * ff1

        outputs = self._dot(
            activations, self.w_linear, self.w_linear_lora, (cond, self.film_linear) if use_film else None
        )
        assert outputs.dtype == dtype
        return outputs

    def _dot(self, x: at.Array, w: at.Array, lora_weights: tuple[at.Array, at.Array] | None, film=None) -> at.Array:
        base = jnp.dot(x, w.astype(x.dtype))
        if lora_weights is None:
            return base
        a = jnp.dot(x, lora_weights[0].astype(x.dtype))
        if film is not None:
            cond, params = film
            shape = (cond.shape[0],) + (1,) * (a.ndim - 2) + (a.shape[-1],)
            a = _apply_film(a, cond, params, shape)
        return base + jnp.dot(a, lora_weights[1].astype(x.dtype))


def _film_params(module: nn.Module, cond_dim: int, rank: int, prefix: str = "lora_film"):
    """(gamma, beta) weights (cond_dim, rank), initialised like LoRAdapter's nn.Linear (U(+-1/sqrt(c)))."""
    init = nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform")
    return (module.param(f"{prefix}_gamma", init, (cond_dim, rank)), module.param(f"{prefix}_beta", init, (cond_dim, rank)))


def _apply_film(a: at.Array, cond: at.Array, params, shape) -> at.Array:
    """(1 + gamma(c)) * a + beta(c), with the (B, r) FiLM reshaped to broadcast against a."""
    gamma_w, beta_w = params
    cond = cond.astype(a.dtype)
    gamma = jnp.dot(cond, gamma_w.astype(a.dtype)).reshape(shape)
    beta = jnp.dot(cond, beta_w.astype(a.dtype)).reshape(shape)
    return a * (1 + gamma) + beta
