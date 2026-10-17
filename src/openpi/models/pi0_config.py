import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    # 4D RoPE (PyTorch model only): rotate every token by a continuous (x, y, z, t) position
    # instead of its 1D sequence index. Needs Observation.patch_xyz and Observation.eef_xyz.
    rope_4d: bool = False
    # Metres -> RoPE position units (centimetres, as in ArticuBot).
    rope_4d_xyz_scale: float = 100.0

    # Conditional LoRA (LoRAdapter): FiLM the low-rank bottleneck of every LoRA layer (needs *_lora
    # Gemma variants) with a per-sample condition from a geometry encoder (models/cond_encoders.py):
    # "depth" (Observation.depth, exterior camera, DepthEncoder), "pointnet" (Observation.point_cloud,
    # PointNet2Encoder) or "plucker" (Observation.plucker, PluckerEncoder). None = plain LoRA.
    cond_lora: str | None = None
    cond_lora_dim: int = 256

    # Geometry-conditioned residual adapter (models/geo_adapter.py) on the exterior camera's image tokens,
    # between SigLIP and the VLM: F' = F + sigmoid(gate(z)) * up(gelu(down(z))), z = [LN(F), g], with g a
    # per-patch geometry embedding from the exterior depth map + its intrinsics (Observation.depth,
    # .intrinsics; camera frame, no extrinsics). geo_adapter picks g: "rgb" (none), "depth" (the patch's
    # raw depths), "xyz" (camera-frame patch centroid), "relative" (scale-normalised local shape, normal,
    # relative depth). None = no adapter.
    # "mapanything": g from frozen MapAnything features of the exterior view (Observation.geo_features,
    # geo_feature_dim channels: mapanything_features/extractor.py), through a trainable projection.
    geo_adapter: str | None = None
    geo_adapter_bottleneck: int = 64
    geo_feature_dim: int = 1538
    # Spatial augmentation (crop / rotate) of the exterior image in training. Off for inputs that are
    # pixel-aligned with it (geo_features; depth for the depth-based adapters), so they stay aligned.
    exterior_spatial_aug: bool = True

    # Cross-view representation loss on the adapted exterior tokens (needs geo_adapter and paired-view
    # batches: DataConfig.paired_views=2, rows 2i / 2i+1 = the same timestep seen by views A / B).
    # Z = h(TaskPool(F', prompt)) per sample (models/geo_adapter.py ReprHead; pi0.5's prompt holds the
    # task and the discretised state). "inv": lambda_inv * mean_i ||Z_A^i - Z_B^i||^2; "vicreg": that plus
    # lambda_var * variance hinge + lambda_cov * off-diagonal covariance (per branch, averaged). None = off.
    repr_loss: str | None = None
    repr_dim: int = 128
    repr_lambda_inv: float = 0.01
    repr_lambda_var: float = 1.0
    repr_lambda_cov: float = 0.04
    repr_var_gamma: float = 1.0

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.repr_loss not in (None, "inv", "vicreg"):
            raise ValueError(f"repr_loss must be None, 'inv' or 'vicreg', got {self.repr_loss!r}")
        if self.repr_loss is not None and self.geo_adapter is None:
            raise ValueError("repr_loss needs geo_adapter (it trains the adapter's output tokens)")
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
