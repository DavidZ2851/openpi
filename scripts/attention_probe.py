"""Does pi05 look at the exterior camera, or only the wrist camera? (potato two-random-view data)

For each checkpoint, on the same fixed training frames:
  1. attention share: the action expert's queries (15 action tokens, every denoising step, every layer/head) split
     their softmax mass over key groups [exterior, wrist, (masked 2nd wrist), BOS, task text, state, rest, actions];
     also per-token (share / #tokens) and the exterior/wrist share by layer;
  2. causal ablations with the same noise: drop a view (image mask off) or swap it with another episode's frame,
     swap the MapAnything features (geo models) -- change in the predicted chunk and error to the ground truth.
Geometry in the adapter models is added onto the exterior tokens (no separate tokens), so its use shows up in the
ablations, not as its own attention group.

  CUDA_VISIBLE_DEVICES=0 HF_LEROBOT_HOME=/data/haotian/lerobot_home .venv/bin/python scripts/attention_probe.py \
      --out /data/haotian/attention_probe
"""

import argparse
import dataclasses
import json
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.gemma as gemma
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer

S = "/data/haotian/openpi_store/checkpoints"
# (name, config, checkpoint or "pretrained", potato2v success over seeds 1111/2222/3333)
MODELS = [
    ("pretrained pi05", "pi05_droid_jointpos_molmospaces_lora_frozenvis", "pretrained", None),
    ("two-view LoRA", "pi05_droid_jointpos_molmospaces_lora", f"{S}/pi05_droid_jointpos_molmospaces_lora/pick_place_potato_onehouse_two_random_lora/19999", 64.0),
    ("frozenvis LoRA", "pi05_droid_jointpos_molmospaces_lora_frozenvis", f"{S}/pi05_droid_jointpos_molmospaces_lora_geoadapter/two_random_lora_frozenvis_paired/19999", 67.3),
    ("RGB adapter + frozenvis LoRA", "pi05_droid_jointpos_molmospaces_lora_geoadapter_rgb", f"{S}/pi05_droid_jointpos_molmospaces_lora_geoadapter/two_random_lora_frozenvis_geoadapter_rgb_paired/19999", 68.7),
    ("E2 MapAnything + frozenvis LoRA", "pi05_droid_jointpos_molmospaces_lora_frozenvis_mapanything", f"{S}/pi05_droid_jointpos_molmospaces_lora_frozenvis_mapanything/e2_frozenvis_mapanything/19999", 66.7),
    ("M1 MapAnything + LoRA", "pi05_droid_jointpos_molmospaces_mapanything_m1_lora", f"{S}/pi05_droid_jointpos_molmospaces_mapanything_m1_lora/m1_lora_b8/19999", 52.0),
    ("M1 MapAnything adapter only", "pi05_droid_jointpos_molmospaces_mapanything_m1", f"{S}/pi05_droid_jointpos_molmospaces_mapanything_m1/m1_b16/9999", 44.0),
]
DATA_CONFIG = "pi05_droid_jointpos_molmospaces_lora_frozenvis_mapanything"  # same frames + cached MapAnything features
EXT, WRIST, WRIST2 = "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"
ACT_DIMS = 8  # 7 joints + gripper (the rest is padding)


def load_frames(n, batch, episode=None, stride=1):
    cfg = _config.get_config(DATA_CONFIG)
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = _data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
    ds = _data_loader.transform_dataset(raw, data_cfg)
    if episode is None:
        idx = np.linspace(0, len(ds) - 1, n).astype(int)  # spread over episodes and progress
    else:  # every `stride`-th frame of one episode, padded with its last frame to whole batches
        lengths = json.loads(pathlib.Path(cfg.data.geo_feature_cache).with_suffix(".json").read_text())["episode_lengths"]
        start = int(np.sum(lengths[:episode]))
        idx = np.arange(start, start + lengths[episode], stride)
        idx = np.concatenate([idx, np.full(-len(idx) % batch, idx[-1])])
    items = [ds[int(i)] for i in idx]
    episodes = [int(raw[int(i)]["episode_index"]) for i in idx]
    batches = []
    for b in range(0, len(items), batch):
        d = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *items[b : b + batch])
        batches.append((_model.Observation.from_dict(d), np.asarray(d["actions"])))
    return batches, episodes


def token_groups(obs, max_len):
    """Key index -> group id, from the prefix layout (3 x 256 image tokens, then the pi05 prompt)."""
    sp = _tokenizer.PaligemmaTokenizer(max_len)._tokenizer
    ids = np.asarray(obs.tokenized_prompt[0])
    pieces = [sp.id_to_piece(int(t)) for t in ids]
    n_valid = int(np.asarray(obs.tokenized_prompt_mask[0]).sum())
    s0 = next(i for i, p in enumerate(pieces) if "State" in p)
    s1 = next(i for i in range(s0, n_valid) if ";" in pieces[i]) + 1
    names = ["exterior", "wrist", "wrist2 (masked)", "BOS", "task text", "state", "rest of prompt", "action tokens"]
    g = np.concatenate([np.full(256, 0), np.full(256, 1), np.full(256, 2), [3], np.full(s0 - 1, 4),
                        np.full(s1 - s0, 5), np.full(max_len - s1, 6)])
    text = "".join(pieces[:n_valid]).replace("▁", " ")
    return names, g, {"prompt": text, "task_text": [1, s0], "state": [s0, s1], "valid": n_valid}


def restore(name, cfg, ckpt):
    if ckpt != "pretrained":
        return cfg.model.load(_model.restore_params(pathlib.Path(ckpt) / "params", dtype=jnp.bfloat16))
    # pretrained pi05 (no fine-tuning) in the LoRA model: lora_b is zero-initialised, so zero LoRA == base model
    base = _model.restore_params(pathlib.Path(_config.PI05_JOINTPOS_DIR) / "params", dtype=jnp.bfloat16)
    want = nnx.state(nnx.eval_shape(cfg.model.create, jax.random.key(0))).to_pure_dict()
    flat_b = dict(jax.tree_util.tree_flatten_with_path(base)[0])
    leaves, treedef = jax.tree_util.tree_flatten_with_path(want)
    filled, n_missing = [], 0
    for path, leaf in leaves:
        if path in flat_b:
            filled.append(flat_b[path])
        else:
            assert "lora" in jax.tree_util.keystr(path), f"{name}: missing non-LoRA param {jax.tree_util.keystr(path)}"
            filled.append(jnp.zeros(leaf.shape, jnp.bfloat16))
            n_missing += 1
    print(f"  pretrained: {n_missing} LoRA tensors set to zero")
    return cfg.model.load(jax.tree_util.tree_unflatten(treedef, filled))


def probe(name, cfg_name, ckpt, batches, groups, n_groups, num_steps, rng, ablate=True):
    cfg = _config.get_config(cfg_name)
    model = restore(name, cfg, ckpt)
    graphdef, state = nnx.split(model)
    seg = jax.nn.one_hot(jnp.asarray(groups), n_groups - 1)  # prefix keys -> groups; suffix keys handled below
    rec = []

    def hook(probs):  # (B, K, G, T, S); T == S is the prefix pass (no action queries)
        if probs.shape[3] == probs.shape[4]:
            return
        p = probs.astype(jnp.float32).mean(axis=3)[:, 0]  # (B, heads, S), mean over the action queries
        P = seg.shape[0]
        share = jnp.concatenate([jnp.einsum("bhs,sg->bhg", p[..., :P], seg), p[..., P:].sum(-1, keepdims=True)], -1)
        jax.debug.callback(lambda s, m: rec.append((np.asarray(s), np.asarray(m))), share, p[..., :512].mean(1), ordered=True)

    def sample(st, obs, noise):
        return nnx.merge(graphdef, st).sample_actions(None, obs, num_steps=num_steps, noise=noise)

    gemma.ATTN_PROBS_HOOK = hook
    sample_attn = jax.jit(lambda st, obs, noise: sample(st, obs, noise))
    o0 = jax.tree.map(jnp.asarray, batches[0][0])  # trace now: the hook is read at trace time
    jax.block_until_ready(sample_attn(state, o0, jnp.zeros((len(batches[0][1]), cfg.model.action_horizon, cfg.model.action_dim))))
    jax.effects_barrier()
    gemma.ATTN_PROBS_HOOK = None
    sample_plain = jax.jit(lambda st, obs, noise: sample(st, obs, noise))

    has_geo = cfg.model.geo_adapter == "mapanything"

    def roll(x):
        return jnp.roll(x, 1, axis=0)

    conds = {
        "drop exterior": lambda o: dataclasses.replace(o, image_masks={**o.image_masks, EXT: jnp.zeros_like(o.image_masks[EXT])}),
        "drop wrist": lambda o: dataclasses.replace(o, image_masks={**o.image_masks, WRIST: jnp.zeros_like(o.image_masks[WRIST])}),
        "swap exterior": lambda o: dataclasses.replace(o, images={**o.images, EXT: roll(o.images[EXT])},
                                                       geo_features=None if o.geo_features is None else roll(o.geo_features)),
        "swap wrist": lambda o: dataclasses.replace(o, images={**o.images, WRIST: roll(o.images[WRIST])}),
        "swap state": lambda o: dataclasses.replace(o, tokenized_prompt=roll(o.tokenized_prompt), tokenized_prompt_mask=roll(o.tokenized_prompt_mask)),
    }
    if not ablate:
        conds = {}
    if has_geo and ablate:
        conds["swap MapAnything only"] = lambda o: dataclasses.replace(o, geo_features=roll(o.geo_features))
    err = {k: [] for k in ["full", *conds]}
    delta = {k: [] for k in conds}
    shares, maps = [], []
    for bi, (obs, act) in enumerate(batches):
        obs = jax.tree.map(jnp.asarray, obs)
        noise = jax.random.normal(jax.random.fold_in(rng, bi), (len(act), cfg.model.action_horizon, cfg.model.action_dim))
        rec.clear()
        full = np.asarray(sample_attn(state, obs, noise))
        jax.effects_barrier()
        layers = gemma.get_config(cfg.model.paligemma_variant).depth
        assert len(rec) == num_steps * layers, (len(rec), num_steps, layers)
        shares.append(np.stack([s for s, _ in rec]).reshape(num_steps, layers, *rec[0][0].shape))  # (step, layer, B, heads, g)
        maps.append(np.stack([m for _, m in rec]).mean(0))  # (B, 512)
        a = act[..., :ACT_DIMS]
        err["full"].append(np.square(full[..., :ACT_DIMS] - a).mean((1, 2)))
        for k, f in conds.items():
            pred = np.asarray(sample_plain(state, f(obs), noise))[..., :ACT_DIMS]
            err[k].append(np.square(pred - a).mean((1, 2)))
            delta[k].append(np.square(pred - full[..., :ACT_DIMS]).mean((1, 2)))
    shares = np.concatenate(shares, axis=2)  # (step, layer, N, heads, g)
    out = {
        "share": shares.mean((0, 1, 2, 3)).tolist(),
        "share_by_layer": shares.mean((0, 2, 3)).tolist(),
        "share_by_step": shares.mean((1, 2, 3)).tolist(),
        "share_by_head_layer": shares.mean((0, 2)).tolist(),
        "share_per_frame": shares.mean((0, 1, 3)).tolist(),
        "share_per_frame_layer": shares.mean((0, 3)).transpose(1, 0, 2).tolist(),  # (frame, layer, group)
        "mse_gt": {k: float(np.concatenate(v).mean()) for k, v in err.items()},
        "mse_vs_full": {k: float(np.concatenate(v).mean()) for k, v in delta.items()},
    }
    np.save(OUT / f"maps_{name.replace(' ', '_')}.npy", np.concatenate(maps))
    del model, state
    return out


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/data/haotian/attention_probe"))
    ap.add_argument("--frames", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--episode", type=int, default=None, help="all frames of one episode instead of --frames")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no-ablate", action="store_true")
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names")
    args = ap.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    batches, episodes = load_frames(args.frames, args.batch, args.episode, args.stride)
    obs0 = batches[0][0]
    names, groups, prompt_info = token_groups(obs0, obs0.tokenized_prompt.shape[1])
    n_tok = [int((groups == i).sum()) for i in range(len(names) - 1)] + [15]
    print("prompt:", prompt_info, "group sizes:", dict(zip(names, n_tok)))
    print(f"{len(episodes)} frames from {len(set(episodes))} episodes")
    imgs = np.concatenate([np.asarray(o.images[EXT]) for o, _ in batches])
    np.save(OUT / "exterior_images.npy", ((imgs + 1) * 127.5).astype(np.uint8))
    np.save(OUT / "wrist_images.npy", ((np.concatenate([np.asarray(o.images[WRIST]) for o, _ in batches]) + 1) * 127.5).astype(np.uint8))
    results = {"groups": names, "group_tokens": n_tok, "prompt": prompt_info, "episodes": episodes, "models": {}}
    path = OUT / "results.json"
    if path.exists():
        results["models"] = json.loads(path.read_text())["models"]
    for name, cfg_name, ckpt, sr in MODELS:
        if args.models and name not in args.models:
            continue
        print(f"== {name}", flush=True)
        r = probe(name, cfg_name, ckpt, batches, groups, len(names), args.steps, jax.random.key(0), ablate=not args.no_ablate)
        r["success"] = sr
        results["models"][name] = r
        print("  share:", {n: round(s, 3) for n, s in zip(names, r["share"])})
        print("  mse_gt:", {k: round(v, 4) for k, v in r["mse_gt"].items()})
        print("  mse_vs_full:", {k: round(v, 4) for k, v in r["mse_vs_full"].items()}, flush=True)
        path.write_text(json.dumps(results))


if __name__ == "__main__":
    main()
