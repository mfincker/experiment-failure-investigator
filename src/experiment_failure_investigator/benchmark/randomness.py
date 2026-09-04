"""Stable named random streams for reproducible benchmark generation."""

from __future__ import annotations

from hashlib import sha256


def derive_child_seed(root_seed: int, namespace: str) -> int:
    """Derive a stable uint32 child seed without consuming another stream."""
    digest = sha256(f"{root_seed}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)
