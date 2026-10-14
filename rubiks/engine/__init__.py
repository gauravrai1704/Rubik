"""Shared engine (technical basis, Section 3). Everything else codes against this API."""
from .checker import is_reachable
from .moves import canonical_successor_mask, canonicalize, metrics, parse_notation, to_notation
from .orbits import Orbit, clusters, orbit_summary, orbits
from .puzzle import FACE_NAMES, Move, PuzzleSpec, cube, make, square
from .random_states import random_state, random_walks, scramble

__all__ = [
    "FACE_NAMES", "Move", "PuzzleSpec", "cube", "square", "make",
    "canonicalize", "metrics", "canonical_successor_mask", "to_notation", "parse_notation",
    "Orbit", "orbits", "clusters", "orbit_summary",
    "is_reachable", "scramble", "random_state", "random_walks",
]
