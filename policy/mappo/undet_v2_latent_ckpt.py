"""
Load pretrained UndeterminedTargetHeadV2 weights from another run's actor.pt.

Used when training or rendering with --enable_undetermined_goal_v2 and without
--use_attn_comm_actor: the latent / preference head can be warm-started from a
checkpoint saved under a folder such as `undet_v2_target_latent` (run root with
models/actor.pt or a direct path to actor.pt).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

UNDETERMINED_HEAD_PREFIX = "undetermined_head."

# MAPPO/policy/mappo/undet_v2_latent_ckpt.py -> repo root is parents[2]
_MAPPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_latent_model_base(raw: Union[str, Path, None]) -> Optional[Path]:
    """
    Interpret undet_v2_target_latent_model_dir relative to the MAPPO repo root so that
    default ``../undet_v2_target_latent/checkpoints`` works regardless of process cwd.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(s).expanduser()
    if p.is_absolute():
        return p
    return (_MAPPO_ROOT / p).resolve()


def resolve_actor_pt_under_dir(base: Union[str, Path]) -> Optional[Path]:
    """Resolve export layout: run dir containing models/actor.pt, or path to a .pt file."""
    p = Path(base).expanduser()
    s = str(p)
    if not s.strip():
        return None
    if p.is_file() and p.suffix == ".pt":
        return p
    c = p / "models" / "actor.pt"
    if c.is_file():
        return c
    c2 = p / "actor.pt"
    if c2.is_file():
        return c2
    return None


def _subset_undetermined_head(sd: dict) -> dict[str, torch.Tensor]:
    return {k: v for k, v in sd.items() if k.startswith(UNDETERMINED_HEAD_PREFIX)}


def _standalone_selector_state_to_head_subset(model_sd: dict) -> dict[str, torch.Tensor]:
    """Map TargetLatentSelector state_dict keys to MAPPO actor keys under undetermined_head.*."""
    return {UNDETERMINED_HEAD_PREFIX + k: v for k, v in model_sd.items()}


def _assert_head_shapes_match(actor: nn.Module, subset: dict[str, torch.Tensor], ckpt_path: Path) -> None:
    cur = actor.state_dict()
    bad: list[str] = []
    for k, v in subset.items():
        if k not in cur:
            bad.append(f"missing in current actor: {k}")
        elif tuple(cur[k].shape) != tuple(v.shape):
            bad.append(f"{k}: current {tuple(cur[k].shape)} vs checkpoint {tuple(v.shape)}")
    if bad:
        raise ValueError(
            f"undetermined_head shape/key mismatch when loading {ckpt_path}:\n" + "\n".join(bad)
        )


def apply_undet_v2_target_latent_heads(
    all_args,
    actors: Sequence[nn.Module],
    device: torch.device,
) -> bool:
    """
    If args request it, load undetermined_head.* from undet_v2_target_latent_model_dir into each actor.

    :return: True if weights were loaded; False if the option is unset (no-op).
    """
    raw = getattr(all_args, "undet_v2_target_latent_model_dir", None)
    if raw is None or not str(raw).strip():
        return False
    if not (
        bool(getattr(all_args, "enable_undetermined_goal_v2", False))
        and not bool(getattr(all_args, "use_attn_comm_actor", False))
    ):
        return False
    base = resolve_latent_model_base(raw)
    if base is None:
        return False
    ap = resolve_actor_pt_under_dir(base)
    arch = str(getattr(all_args, "undet_v2_head_arch", "dot_product"))
    if ap is None:
        st = base / "target_latent_selector.pt" if base.is_dir() else None
        if st is not None and st.is_file() and arch == "pair_mlp":
            ap = st
    if ap is None:
        st_only = base / "target_latent_selector.pt" if base.is_dir() else None
        if st_only is not None and st_only.is_file() and arch != "pair_mlp":
            raise ValueError(
                f"undet_v2_target_latent_model_dir={raw!r}: found {st_only.name} only; "
                f"use --undet_v2_head_arch pair_mlp (and matching M/hidden/d_emb) or add MAPPO actor.pt."
            )
        raise FileNotFoundError(
            f"undet_v2_target_latent_model_dir={raw!r} (resolved {base}): need models/actor.pt, actor.pt, "
            f"target_latent_selector.pt (with --undet_v2_head_arch pair_mlp), or a direct *.pt path"
        )
    actors_list = list(actors)
    if not actors_list:
        return False
    for a in actors_list:
        if getattr(a, "undetermined_head", None) is None:
            raise RuntimeError(
                "undet_v2_target_latent_model_dir is set but actor has no undetermined_head "
                "(requires --enable_undetermined_goal)"
            )
    sd = torch.load(str(ap), map_location=device)
    subset = _subset_undetermined_head(sd)
    if not subset and isinstance(sd, dict) and "model" in sd:
        if arch != "pair_mlp":
            raise ValueError(
                f"No {UNDETERMINED_HEAD_PREFIX!r} keys in {ap}: file is a standalone TargetLatentSelector "
                f"checkpoint; use --undet_v2_head_arch pair_mlp and matching M / hidden / d_emb."
            )
        subset = _standalone_selector_state_to_head_subset(sd["model"])
        if isinstance(sd.get("model_layout"), dict):
            ml = sd["model_layout"]
            print(f"[undet_v2_target_latent] checkpoint model_layout: {ml}")
    if not subset:
        raise ValueError(f"No {UNDETERMINED_HEAD_PREFIX!r} keys in {ap}")
    for actor in actors_list:
        _assert_head_shapes_match(actor, subset, ap)
        actor.load_state_dict(subset, strict=False)
    _mode = str(getattr(all_args, "undet_v2_latent_train_mode", "finetune_all"))
    print(
        f"[undet_v2_target_latent] Loaded {len(subset)} tensors from {ap} "
        f"into undetermined_head of {len(actors_list)} actor(s); train_mode={_mode}"
    )
    return True


def undetermined_v2_slot_supervision_loss(
    robot_obs: torch.Tensor,
    logits: torch.Tensor,
    *,
    goal_rr: float,
    active_mask: torch.Tensor | None = None,
    match_tol: float = 0.35,
) -> torch.Tensor:
    """
    Supervised cross-entropy: label = nearest-M slot whose reconstructed goal (from obs geometry)
    matches the current assigned goal (gx, gy) from the first two obs channels. Skips pending rows
    and invalid slots (k_norm < 0).
    """
    ro = robot_obs
    B, M = logits.shape[0], logits.shape[1]
    px = ro[:, -2]
    py = ro[:, -1]
    gx = ro[:, 0] + px
    gy = ro[:, 1] + py
    g = ro[:, 7 : 7 + 5 * M].reshape(B, M, 5)
    invalid = g[:, :, 4] < -0.05
    scale = max(float(goal_rr), 1.0)
    tx = px.unsqueeze(1) + g[:, :, 0] * scale
    ty = py.unsqueeze(1) + g[:, :, 1] * scale
    d2 = (tx - gx.unsqueeze(1)).pow(2) + (ty - gy.unsqueeze(1)).pow(2)
    d2 = d2.masked_fill(invalid, 1e12)
    min_d2, label = d2.min(dim=1)
    tol = float(match_tol) ** 2
    good = min_d2 < tol
    pend = ro[:, 7 + 5 * M]
    if pend.dim() > 1:
        pend = pend.squeeze(-1)
    good = good & (pend < 0.5)
    if active_mask is not None:
        am = active_mask.reshape(B)
        good = good & (am > 0.5)
    if not bool(good.any().item()):
        return robot_obs.new_zeros(())
    ce = F.cross_entropy(logits, label, reduction="none")
    w = good.float()
    return (ce * w).sum() / w.sum().clamp(min=1.0)
