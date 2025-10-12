from __future__ import annotations

from typing import Dict, Type

from .base import ObservationBuilder
from .cnn import CNNObservationBuilder
from .flat import FlatObservationBuilder


_OBSERVATION_BUILDERS: Dict[str, Type[ObservationBuilder]] = {
    "cnn": CNNObservationBuilder,
    "flat": FlatObservationBuilder,
}


def get_observation_builder(name: str) -> ObservationBuilder:
    """Instantiate an observation builder by name."""
    key = name.lower()
    if key not in _OBSERVATION_BUILDERS:
        raise ValueError(f"Unknown observation builder '{name}'. Available: {sorted(_OBSERVATION_BUILDERS)}")
    builder_cls = _OBSERVATION_BUILDERS[key]
    return builder_cls()


__all__ = [
    "ObservationBuilder",
    "CNNObservationBuilder",
    "FlatObservationBuilder",
    "get_observation_builder",
]
