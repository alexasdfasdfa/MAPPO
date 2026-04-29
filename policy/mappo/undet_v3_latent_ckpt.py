"""
Load pretrained undetermined v3 selector head weights into MAPPO actor.undetermined_head.

This loader is intentionally strict on shape match and only loads head tensors.
It supports:
1) MAPPO actor checkpoints containing `undetermined_head.*` keys;
2) Standalone state_dict containing direct head keys;
3) Dict checkpoints with {"model": state_dict}.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

UNDETERMINED_HEAD_PREFIX = "undetermined_head."
_MAPPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_base(raw: Union[str, Path, None]) -> Optional[Path]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(s).expanduser()
    if p.is_absolute():
        return p
    return (_MAPPO_ROOT / p).resolve()


def _resolve_pt(base: Union[str, Path]) -> Optional[Path]:
    p = Path(base).expanduser()
    if p.is_file() and p.suffix == ".pt":
        return p
    c1 = p / "models" / "actor.pt"
    if c1.is_file():
        return c1
    c2 = p / "actor.pt"
    if c2.is_file():
        return c2
    c3 = p / "target_latent_selector.pt"
    if c3.is_file():
        return c3
    c4 = p / "checkpoints_decoupled_equal"
    if c4.is_dir():
        pts = sorted(c4.glob("v3decoupled_equal_n*.pt"))
        if pts:
            return pts[-1]
    c5 = p / "checkpoints"
    if c5.is_dir():
        pts = sorted(c5.glob("v3global_rank_latent_n*.pt"))
        if pts:
            return pts[-1]
    return None


def _subset_prefixed(sd: dict) -> dict[str, torch.Tensor]:
    return {k: v for k, v in sd.items() if isinstance(k, str) and k.startswith(UNDETERMINED_HEAD_PREFIX)}


def _as_prefixed_subset(sd: dict, actor: nn.Module) -> dict[str, torch.Tensor]:
    """
    Try deriving undetermined_head.* subset from a raw state_dict.
    """
    subset = _subset_prefixed(sd)
    if subset:
        return subset
    head = getattr(actor, "undetermined_head", None)
    if head is None:
        return {}
    head_keys = set(head.state_dict().keys())
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not isinstance(k, str):
            continue
        if k in head_keys:
            out[UNDETERMINED_HEAD_PREFIX + k] = v
    return out


def _assert_match(actor: nn.Module, subset: dict[str, torch.Tensor], ckpt: Path) -> None:
    cur = actor.state_dict()
    bad: list[str] = []
    for k, v in subset.items():
        if k not in cur:
            bad.append(f"missing in actor: {k}")
        elif tuple(cur[k].shape) != tuple(v.shape):
            bad.append(f"{k}: actor {tuple(cur[k].shape)} vs ckpt {tuple(v.shape)}")
    if bad:
        raise ValueError(
            f"undetermined v3 head mismatch for {ckpt}:\n" + "\n".join(bad[:40])
        )


def apply_undet_v3_target_latent_heads(
    all_args,
    actors: Sequence[nn.Module],
    device: torch.device,
) -> bool:
    raw = getattr(all_args, "undet_v3_target_latent_model_dir", None)
    if raw is None or not str(raw).strip():
        return False
    if not bool(getattr(all_args, "enable_undetermined_goal_v3", False)):
        return False
    base = _resolve_base(raw)
    if base is None:
        return False
    ap = _resolve_pt(base)
    if ap is None:
        raise FileNotFoundError(
            f"undet_v3_target_latent_model_dir={raw!r} (resolved {base}): "
            "need models/actor.pt, actor.pt, target_latent_selector.pt, or direct *.pt path"
        )
    actors_list = list(actors)
    if not actors_list:
        return False
    for a in actors_list:
        if getattr(a, "undetermined_head", None) is None:
            raise RuntimeError(
                "undet_v3_target_latent_model_dir is set but actor has no undetermined_head"
            )

    sd = torch.load(str(ap), map_location=device)
    raw_sd: dict
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        raw_sd = sd["model"]
    elif isinstance(sd, dict):
        raw_sd = sd
    else:
        raise ValueError(f"Unsupported checkpoint format for {ap}")

    subset = _as_prefixed_subset(raw_sd, actors_list[0])
    if not subset:
        sample_keys = list(raw_sd.keys())[:20]
        print(
            "[undet_v3_target_latent][warn] incompatible checkpoint format; "
            "no undetermined_head.* tensors found, skip latent head loading. "
            f"path={ap}, sample_keys={sample_keys}"
        )
        return False

    for actor in actors_list:
        _assert_match(actor, subset, ap)
        actor.load_state_dict(subset, strict=False)

    mode = str(getattr(all_args, "undet_v3_latent_train_mode", "finetune_all"))
    print(
        f"[undet_v3_target_latent] Loaded {len(subset)} tensors from {ap} "
        f"into undetermined_head of {len(actors_list)} actor(s); train_mode={mode}"
    )
    return True

