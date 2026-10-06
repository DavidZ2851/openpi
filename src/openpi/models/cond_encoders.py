"""Condition encoders for conditional LoRA (lora.LoRAConfig.cond_dim): observation -> (B, cond_dim).

- DepthEncoder: metric depth map(s) (B, n_cams, H, W), 0 = invalid; per camera [depth / max_depth,
  valid mask] through a ConvMapEncoder.
- PluckerEncoder: the exterior camera's Plücker ray map (B, 6, H, W) in the robot base frame
  (direction, moment) through a ConvMapEncoder.
- ConvMapEncoder: a conv stack shaped like LoRAdapter's structure mapper (3x3 convs, SiLU, four
  stride-2 stages), global average pooling; the per-map vectors (shared weights) are concatenated,
  projected and LayerNorm'd (LoRAdapter's SimpleMapper).
- PointNet2Encoder: PointNet++ with single-scale grouping on the exterior camera's point cloud (B, N, 3) in the
  robot base frame, normalised with a fixed workspace centre/scale: two set-abstraction levels
  (farthest point sampling + ball query + shared MLP + max pool) and a global one.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at


class ConvMapEncoder(nnx.Module):
    """(B, n_maps, H, W, in_channels) -> (B, cond_dim)."""

    def __init__(self, cond_dim: int, rngs: nnx.Rngs, in_channels: int, n_maps: int = 1, width: int = 128):
        chans = [(in_channels, 16, 1), (16, 16, 1), (16, 32, 2), (32, 32, 1), (32, 64, 2), (64, 64, 1),
                 (64, width, 2), (width, width, 1), (width, width, 2), (width, width, 1)]  # fmt: skip
        # named attributes, not a list: openpi flattens param paths with string keys
        self.n_convs = len(chans)
        for i, (cin, cout, s) in enumerate(chans):
            setattr(self, f"conv{i}", nnx.Conv(cin, cout, (3, 3), strides=(s, s), padding="SAME", rngs=rngs))
        self.proj = nnx.Linear(n_maps * width, cond_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(cond_dim, rngs=rngs)

    def __call__(self, maps: at.Float[at.Array, "b n h w c"]) -> at.Float[at.Array, "b d"]:
        b, n, h, w, c = maps.shape
        x = maps.reshape(b * n, h, w, c).astype(jnp.float32)
        for i in range(self.n_convs):
            x = nnx.silu(getattr(self, f"conv{i}")(x))
        feats = x.mean(axis=(1, 2)).reshape(b, -1)
        return self.norm(self.proj(feats))


class DepthEncoder(nnx.Module):
    def __init__(self, cond_dim: int, rngs: nnx.Rngs, n_cams: int = 1, max_depth: float = 4.0):
        self.max_depth = max_depth
        self.net = ConvMapEncoder(cond_dim, rngs, in_channels=2, n_maps=n_cams)

    def __call__(self, depth: at.Float[at.Array, "b n h w"]) -> at.Float[at.Array, "b d"]:
        d = depth.astype(jnp.float32)[..., None]
        return self.net(jnp.concatenate([d / self.max_depth, (d > 0).astype(jnp.float32)], axis=-1))


class PluckerEncoder(nnx.Module):
    def __init__(self, cond_dim: int, rngs: nnx.Rngs):
        self.net = ConvMapEncoder(cond_dim, rngs, in_channels=6, n_maps=1)

    def __call__(self, plucker: at.Float[at.Array, "b 6 h w"]) -> at.Float[at.Array, "b d"]:
        return self.net(jnp.moveaxis(plucker.astype(jnp.float32), 1, -1)[:, None])


def farthest_point_sample(xyz: at.Float[at.Array, "b n 3"], n_samples: int) -> at.Int[at.Array, "b s"]:
    """Deterministic FPS starting at point 0."""
    b, n, _ = xyz.shape

    def step(i, carry):
        idx, dist, farthest = carry
        idx = idx.at[:, i].set(farthest)
        centroid = jnp.take_along_axis(xyz, farthest[:, None, None], axis=1)  # (b, 1, 3)
        dist = jnp.minimum(dist, jnp.sum((xyz - centroid) ** 2, axis=-1))
        return idx, dist, jnp.argmax(dist, axis=-1)

    init = (jnp.zeros((b, n_samples), jnp.int32), jnp.full((b, n), jnp.inf), jnp.zeros((b,), jnp.int32))
    idx, _, _ = jax.lax.fori_loop(0, n_samples, step, init)
    return idx


def _gather(points, idx):
    """points (b, n, c), idx (b, ...) -> (b, ..., c)."""
    b = points.shape[0]
    flat = idx.reshape(b, -1)
    out = jnp.take_along_axis(points, flat[..., None], axis=1)
    return out.reshape(*idx.shape, points.shape[-1])


def ball_query(xyz, centers, radius: float, k: int):
    """k neighbours of each center within radius (the nearest one repeated where fewer)."""
    d2 = jnp.sum((centers[:, :, None, :] - xyz[:, None, :, :]) ** 2, axis=-1)  # (b, s, n)
    neg, idx = jax.lax.top_k(-d2, k)
    return jnp.where(-neg > radius**2, idx[..., :1], idx)


class _MLP(nnx.Module):
    def __init__(self, dims: list[int], rngs: nnx.Rngs):
        self.n_layers = len(dims) - 1
        for i, (cin, cout) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
            setattr(self, f"linear{i}", nnx.Linear(cin, cout, rngs=rngs))
            setattr(self, f"norm{i}", nnx.LayerNorm(cout, rngs=rngs))

    def __call__(self, x):
        for i in range(self.n_layers):
            x = nnx.relu(getattr(self, f"norm{i}")(getattr(self, f"linear{i}")(x)))
        return x


class _SetAbstraction(nnx.Module):
    def __init__(self, n_samples: int | None, radius: float, k: int, in_dim: int, dims: list[int], rngs: nnx.Rngs):
        self.n_samples, self.radius, self.k = n_samples, radius, k
        self.mlp = _MLP([in_dim + 3, *dims], rngs)

    def __call__(self, xyz, feats):
        if self.n_samples is None:  # global level
            grouped = xyz if feats is None else jnp.concatenate([xyz, feats], axis=-1)
            return None, self.mlp(grouped).max(axis=1)
        centers = _gather(xyz, farthest_point_sample(xyz, self.n_samples))
        nbr = ball_query(xyz, centers, self.radius, self.k)
        rel = (_gather(xyz, nbr) - centers[:, :, None, :]) / self.radius
        grouped = rel if feats is None else jnp.concatenate([rel, _gather(feats, nbr)], axis=-1)
        return centers, self.mlp(grouped).max(axis=2)


class PointNet2Encoder(nnx.Module):
    def __init__(self, cond_dim: int, rngs: nnx.Rngs, center=(0.45, 0.2, 0.8), scale: float = 0.6):
        self.center = tuple(float(c) for c in center)  # python floats: not a parameter
        self.scale = scale
        self.sa1 = _SetAbstraction(256, 0.1, 32, 0, [64, 64, 128], rngs)
        self.sa2 = _SetAbstraction(64, 0.25, 32, 128, [128, 128, 256], rngs)
        self.sa3 = _SetAbstraction(None, 0.0, 0, 256, [256, 512], rngs)
        self.proj = nnx.Linear(512, cond_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(cond_dim, rngs=rngs)

    def __call__(self, points: at.Float[at.Array, "b n 3"]) -> at.Float[at.Array, "b c"]:
        xyz = (points.astype(jnp.float32) - jnp.asarray(self.center)) / self.scale
        xyz1, f1 = self.sa1(xyz, None)
        xyz2, f2 = self.sa2(xyz1, f1)
        _, g = self.sa3(xyz2, f2)
        return self.norm(self.proj(g))
