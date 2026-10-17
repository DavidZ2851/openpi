import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import geo_adapter as _geo


def _mapanything_inputs(b=4, p=256, width=2048, feat=1538, seed=0):
    k1, k2 = jax.random.split(jax.random.key(seed))
    tokens = jax.random.normal(k1, (b, p, width), jnp.bfloat16)
    g = jax.random.normal(k2, (b, p, feat))
    frac = jnp.ones((b, p, 2)).at[:, :48].set(0.0)  # padding rows: no image
    return tokens, g.at[..., -2:].set(frac)


def test_mapanything_adapter_identity_at_init_and_shapes():
    tokens, g = _mapanything_inputs()
    ad = _geo.GeoAdapter(2048, "mapanything", nnx.Rngs(0), feature_dim=1538)
    out = ad(tokens, geo_features=g)
    assert out.shape == tokens.shape and out.dtype == tokens.dtype
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.asarray(tokens, np.float32))
    # the trainable spatial projection: (B, 256, 128)
    feat = ad.feat_norm(g[..., :-2]) * (g[..., -2:-1] > 0)
    proj = ad.geo_norm(ad.geo_out(nnx.gelu(ad.geo_in(jnp.concatenate([feat, g[..., -2:]], -1)))))
    assert proj.shape == (4, 256, 128)


def test_mapanything_adapter_requires_features():
    tokens, _ = _mapanything_inputs(b=1)
    ad = _geo.GeoAdapter(2048, "mapanything", nnx.Rngs(0), feature_dim=1538)
    try:
        ad(tokens)
    except ValueError:
        return
    raise AssertionError("missing geo_features must raise, not fall back to zeros")


def test_projection_and_adapter_get_gradients():
    tokens, g = _mapanything_inputs(b=2)
    ad = _geo.GeoAdapter(2048, "mapanything", nnx.Rngs(0), feature_dim=1538)
    ad.up.kernel.value = 1e-3 * jnp.ones_like(ad.up.kernel.value)  # past the zero init, as after one step
    target = jax.random.normal(jax.random.key(1), tokens.shape)

    def loss(m):
        return jnp.mean(m(tokens, geo_features=g).astype(jnp.float32) * target)

    grads = nnx.grad(loss)(ad)
    for name in ("geo_in", "geo_out", "down", "up", "gate", "feat_norm"):
        leaves = jax.tree.leaves(grads[name])
        assert any(float(jnp.abs(x).max()) > 0 for x in leaves), name


def test_repr_losses():
    z = jax.random.normal(jax.random.key(0), (8, 128))
    same = jnp.repeat(z[0::2], 2, axis=0)  # A == B for every pair
    assert float(_geo.repr_losses(same, "inv")["inv"]) == 0.0
    t = _geo.repr_losses(z, "vicreg")
    assert set(t) == {"inv", "var", "cov"} and float(t["inv"]) > 0
    collapsed = _geo.repr_losses(jnp.zeros((8, 128)), "vicreg")
    assert float(collapsed["inv"]) == 0.0 and float(collapsed["var"]) > 0.9  # the variance hinge catches collapse


def test_repr_loss_reaches_adapter():
    tokens, g = _mapanything_inputs(b=4)
    ad = _geo.GeoAdapter(2048, "mapanything", nnx.Rngs(0), feature_dim=1538)
    ad.up.kernel.value = 1e-3 * jax.random.normal(jax.random.key(2), ad.up.kernel.value.shape)
    head = _geo.ReprHead(2048, nnx.Rngs(1))
    prompt = jax.random.normal(jax.random.key(3), (4, 12, 2048))
    mask = jnp.ones((4, 12), bool)

    def loss(a, h):
        out = a(tokens, geo_features=g)
        t = _geo.repr_losses(h(out, prompt, mask), "vicreg")
        return t["inv"] + t["var"] + t["cov"]

    grads = nnx.grad(loss, argnums=(0, 1))(ad, head)
    adapter_grads = jax.tree.leaves(grads[0])
    assert any(float(jnp.abs(x).max()) > 0 for x in adapter_grads)
