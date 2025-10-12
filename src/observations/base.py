from __future__ import annotations

from abc import ABC, abstractmethod
from gym.spaces import Space


class ObservationBuilder(ABC):
    """Base interface for building observations for :class:`GridWorldEnv`."""

    #: Indicates whether the produced observations are image-like (CxHxW).
    is_image_based: bool = False

    @abstractmethod
    def get_observation_space(self, env) -> Space:
        """Return the Gym space describing observations for ``env``."""

    @abstractmethod
    def get_observation(self, env):
        """Construct an observation for the provided ``env`` instance."""
