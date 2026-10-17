"""Pre-training checks of a MapAnything geo-adapter config (M1 / M2 / M3) on the real model and data.

  CUDA_VISIBLE_DEVICES=<free gpu> PI05_JOINTPOS_DIR=... HF_LEROBOT_HOME=... .venv/bin/python \
      mapanything_features/smoke_openpi.py pi05_droid_jointpos_molmospaces_mapanything_m3 [--steps 3]

Checks: cache rows <-> dataset frames; paired batches are one timestep seen by two views; the trainable set
is exactly Action Expert LoRA + projection / adapter (+ repr head); flow matching reaches the LoRA, the
projection and the adapter, the repr loss reaches the adapter; frozen weights stay put over a few steps;
wrist / language tokens are untouched by the adapter; inference runs without the repr head.
"""

import argparse
import dataclasses
import functools
import importlib.util
import json
import pathlib
import sys

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train", ROOT / "scripts/train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        sys.exit(1)


def flat(state):
    return {"/".join(map(str, k)): v.value for k, v in state.flat_state().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()
    config = dataclasses.replace(_config.get_config(args.config), wandb_enabled=False, batch_size=8)
    m = config.model
    data_cfg = config.data.create(config.assets_dirs, m)

    # -- cache <-> dataset frames ---------------------------------------------------------------------
    cache_path = config.data.geo_feature_cache
    meta = json.loads(pathlib.Path(cache_path).with_suffix(".json").read_text())
    ds = _data_loader.create_torch_dataset(data_cfg, m.action_horizon, m)
    check(len(ds) == sum(meta["episode_lengths"]) == meta["shape"][0], f"cache rows {meta['shape'][0]} == dataset frames {len(ds)}")
    starts = np.cumsum([0, *meta["episode_lengths"]])
    for i in [0, 1234, len(ds) - 1]:
        item = ds[i]
        e, f, idx = int(item["episode_index"]), int(item["frame_index"]), int(item["index"])
        check(idx == starts[e] + f, f"frame {i}: global index {idx} == cache row of (episode {e}, frame {f})")

    # -- paired batches -----------------------------------------------------------------------------------
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    obs, actions = next(iter(loader))
    st, act, tok = np.asarray(obs.state), np.asarray(actions), np.asarray(obs.tokenized_prompt)
    img, gf = np.asarray(obs.images["base_0_rgb"]), np.asarray(obs.geo_features)
    check(gf.shape == (8, 256, m.geo_feature_dim), f"geo_features {gf.shape}")
    check(np.array_equal(st[0::2], st[1::2]) and np.array_equal(act[0::2], act[1::2]) and np.array_equal(tok[0::2], tok[1::2]),
          "rows 2i / 2i+1: same state, action chunk and prompt (one timestep)")
    check(all(not np.array_equal(img[2 * i], img[2 * i + 1]) and not np.array_equal(gf[2 * i], gf[2 * i + 1]) for i in range(4)),
          "rows 2i / 2i+1: different exterior image and geo_features (two views)")
    check(len({st[2 * i].tobytes() for i in range(4)}) == 4, "the 4 timesteps of a batch are distinct states")

    # -- trainable set ------------------------------------------------------------------------------------
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state, state_sharding = train.init_train_state(config, jax.random.key(0), mesh, resume=False)
    train.log_trainable_params(config, state)
    trainable = flat(state.params.filter(config.trainable_filter))
    allp = flat(state.params)
    check(all("lora" in k for k in trainable), "every trainable param is a lora_* param")
    llm_lora = [k for k in trainable if k.startswith("PaliGemma/llm")]
    check(len(llm_lora) > 0 and all("_1" in k for k in llm_lora), f"{len(llm_lora)} LLM LoRA tensors, all in the action expert (*_1)")
    check(not any(k.startswith("PaliGemma/img") for k in trainable), "SigLIP frozen")
    check(not any(k.startswith(("action_in_proj", "action_out_proj", "time_mlp")) for k in trainable), "action / time projections frozen")
    check(any(k.startswith("lora_geo_adapter/geo_in") for k in trainable), "spatial projection trainable")
    check(any(k.startswith("lora_repr_head") for k in trainable) == (m.repr_loss is not None), f"repr head trainable iff repr_loss ({m.repr_loss})")
    lora_shapes = sorted({tuple(v.shape) for k, v in trainable.items() if k.startswith("PaliGemma/llm")})
    print("  expert LoRA shapes:", lora_shapes)

    # -- a few real train steps (no donation: keep the old state to compare) ------------------------------
    step = jax.jit(functools.partial(train.train_step, config), in_shardings=(replicated, state_sharding, data_sharding),
                   out_shardings=(state_sharding, replicated))
    before = {k: np.asarray(v, np.float32) for k, v in allp.items()
              if k.startswith(("PaliGemma/img/embedding", "PaliGemma/llm/layers/attn/q_einsum/w", "PaliGemma/llm/layers/attn/q_einsum_1/w", "action_out_proj/kernel"))
              or k.startswith(("lora_geo_adapter/up/kernel", "lora_geo_adapter/geo_in/kernel"))}
    s = state
    for i in range(args.steps):
        s, info = step(jax.random.key(1), s, (obs, actions))
        print(f"  step {i}: " + ", ".join(f"{k}={float(v):.4f}" for k, v in info.items()), flush=True)
    after = flat(s.params)
    for k, v in before.items():
        moved = not np.array_equal(v, np.asarray(after[k], np.float32))
        check(moved == ("lora" in k), f"{k}: {'updated' if moved else 'unchanged'}")

    # -- gradients ----------------------------------------------------------------------------------------
    model = nnx.merge(s.model_def, s.params)
    model.train()
    diff = nnx.DiffState(0, config.trainable_filter)

    def fm_loss(mdl):
        fm, _, _ = mdl.compute_loss_and_aux(jax.random.key(2), obs, actions, train=True)
        return jnp.mean(fm)

    g = flat(nnx.grad(fm_loss, argnums=diff)(model))
    nz = lambda pref: any(float(jnp.abs(v).max()) > 0 for k, v in g.items() if k.startswith(pref))
    check(nz("PaliGemma/llm") , "flow matching -> Action Expert LoRA")
    check(nz("lora_geo_adapter/geo_in") and nz("lora_geo_adapter/feat_norm"), "flow matching -> spatial projection")
    check(nz("lora_geo_adapter/down") and nz("lora_geo_adapter/up") and nz("lora_geo_adapter/gate"), "flow matching -> adapter")
    if m.repr_loss is not None:
        def repr_only(mdl):
            return mdl.compute_loss_and_aux(jax.random.key(2), obs, actions, train=True)[1]

        g = flat(nnx.grad(repr_only, argnums=diff)(model))
        check(nz("lora_geo_adapter/down") and nz("lora_geo_adapter/geo_in"), f"repr loss ({m.repr_loss}) -> adapter + projection")
        check(not nz("PaliGemma/llm"), "repr loss does not touch the action expert")

    # -- the adapter only changes the exterior tokens -------------------------------------------------------
    model.eval()
    from openpi.models import model as _model

    pobs = _model.preprocess_observation(None, obs, train=False)
    tokens, _, _, parts = model.embed_prefix(pobs, return_parts=True)
    ext = parts["exterior"]
    raw, _ = model.PaliGemma.img(pobs.images["base_0_rgb"], train=False)
    n_img = 256
    ref = []
    for name in pobs.images:
        t, _ = model.PaliGemma.img(pobs.images[name], train=False)
        ref.append(t)
    ref.append(model.PaliGemma.llm(pobs.tokenized_prompt, method="embed"))
    ref = jnp.concatenate(ref, axis=1)
    check(bool(jnp.array_equal(tokens[:, n_img:], ref[:, n_img:])), "wrist and language tokens identical to the no-adapter path")
    check(not bool(jnp.array_equal(ext, raw)), "exterior tokens changed by the (trained) adapter")

    # -- inference: no repr head, no pairs -------------------------------------------------------------------
    one = jax.tree.map(lambda x: x[:1], obs)
    acts = model.sample_actions(jax.random.key(3), one, num_steps=2)
    check(acts.shape == (1, m.action_horizon, m.action_dim) and bool(jnp.isfinite(acts).all()), f"sample_actions on one view: {acts.shape}")
    print("peak GPU memory: %.1f GB" % (jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 1e9))
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
