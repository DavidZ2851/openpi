"""Donation-bug check for a geo-adapter config with a trainable SigLIP (see TrainConfig.donate_train_state).

Compares train.py's donated train step against a forward-only reference loss on the same batch and rng
(step 0 and a few more steps; each reference uses the params the donated step is about to update).

  CUDA_VISIBLE_DEVICES=<gpu> PI05_JOINTPOS_DIR=... HF_LEROBOT_HOME=... .venv/bin/python \
      mapanything_features/donation_check.py <config> [--steps 5] [--batch-size 8]
"""

import argparse
import dataclasses
import functools
import importlib.util
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train", ROOT / "scripts/train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-donate", action="store_true")
    ap.add_argument("--fsdp-devices", type=int, default=1)
    ap.add_argument("--geo-feature-cache", help="override the config's cache path (other machines)")
    args = ap.parse_args()
    config = dataclasses.replace(_config.get_config(args.config), wandb_enabled=False, batch_size=args.batch_size,
                                 donate_train_state=not args.no_donate, fsdp_devices=args.fsdp_devices)
    if args.geo_feature_cache:
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, geo_feature_cache=args.geo_feature_cache))
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = iter(_data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True))
    state, state_sharding = train.init_train_state(config, jax.random.key(0), mesh, resume=False)
    print("donate_train_state:", config.donate_train_state)

    @jax.jit
    def ref_loss(params, rng, step, batch):
        model = nnx.merge(state.model_def, params)
        model.train()
        obs, actions = batch
        fm, extra, _ = model.compute_loss_and_aux(jax.random.fold_in(rng, step), obs, actions, train=True)
        return jnp.mean(fm) + extra

    step = jax.jit(functools.partial(train.train_step, config), in_shardings=(replicated, state_sharding, data_sharding),
                   out_shardings=(state_sharding, replicated), donate_argnums=(1,) if config.donate_train_state else ())
    rng = jax.random.key(1)
    worst = 0.0
    for i in range(args.steps):
        batch = next(loader)
        ref = float(ref_loss(state.params, rng, state.step, batch))  # before the donated step consumes state
        state, info = step(rng, state, batch)
        jax.block_until_ready(state)
        got = float(info["loss"])
        worst = max(worst, abs(got - ref) / abs(ref))
        print(f"step {i}: train_step loss {got:.4f}   reference {ref:.4f}", flush=True)
    print("peak GPU memory: " + ", ".join("%.1f GB" % (d.memory_stats().get("peak_bytes_in_use", 0) / 1e9) for d in jax.devices()))
    print(f"max relative difference {worst:.2e} ->", "OK" if worst < 1e-2 else "MISMATCH (donation bug)")


if __name__ == "__main__":
    main()
