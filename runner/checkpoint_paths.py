"""
Resolve MAPPO shared-policy checkpoint paths.

Supports the usual layout ``<model_dir>/actor.pt`` and ``<model_dir>/critic.pt``,
or ``model_dir`` pointing directly to a ``*.pt`` actor file (e.g. ``4.pt``).
If the directory has no ``actor.pt``, falls back to ``4.pt`` when present.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def resolve_shared_actor_checkpoint_path(model_dir: Union[str, Path, None]) -> Optional[Path]:
    """
    :return: Path to actor state dict file, or None if not found.
    """
    if model_dir is None:
        return None
    md = Path(str(model_dir)).expanduser()
    s = str(md)
    if not s.strip():
        return None
    if md.is_file() and md.suffix.lower() == ".pt":
        return md
    if md.is_dir():
        for name in ("actor.pt", "4.pt"):
            p = md / name
            if p.is_file():
                return p
    return None


def resolve_shared_critic_checkpoint_path(
    model_dir: Union[str, Path, None], actor_path: Optional[Path]
) -> Optional[Path]:
    """
    Critic lives next to the actor when model_dir is a file, or under model_dir when it is a directory.
    """
    if model_dir is None:
        return None
    md = Path(str(model_dir)).expanduser()
    if actor_path is not None and actor_path.is_file():
        c = actor_path.parent / "critic.pt"
        if c.is_file():
            return c
    if md.is_dir():
        c = md / "critic.pt"
        if c.is_file():
            return c
    return None
