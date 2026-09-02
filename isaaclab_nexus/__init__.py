"""Isaac Lab physics backend for Nexus (dimforge, Haixuantao fork) on CUDA.

Backend spike. What is real:
  * ``NexusCfg`` / ``NexusManager`` -- a ``PhysicsManager`` that owns one
    headless Nexus CUDA state and steps it.
  * ``assets.articulation.Articulation`` / ``ArticulationData`` -- serve
    joint / root / body state as **zero-copy torch views** of Nexus GPU memory
    (``__cuda_array_interface__``), no staging buffer, no host round-trip.

What is not: every ``BaseArticulation`` method that is not implemented raises
``NotImplementedError`` naming itself, so a caller hits a clear wall instead
of a silent wrong answer. Contacts, raycasts, terrain, MJCF->Isaac name maps
and the write path are the remaining work; see /workspace/bench/nexus_port.

Layout is dictated by ``isaaclab.utils.backend_utils.FactoryBase``: for a
factory defined in ``isaaclab.<subpath>.<mod>`` it imports
``isaaclab_nexus.<subpath>`` and looks up a class of the factory's own name.
"""

from .physics.nexus_manager_cfg import NexusCfg, NexusMjcfCfg

__all__ = ["NexusCfg", "NexusMjcfCfg"]
