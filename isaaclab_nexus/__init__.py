"""Isaac Lab physics backend for Nexus (dimforge, Haixuantao fork) on CUDA.

Runs Isaac Lab manager-based environments on Nexus instead of PhysX/Newton:
no USD, no Fabric, no cloner. Robots load from MJCF straight into the Nexus
state, one Nexus batch per Isaac Lab environment.

Validated end to end: WBC-AGILE's ``HeightTracking-G1-v0`` (Unitree G1 29-DOF,
rough terrain, height-scan + contact sensors) trains with ``rsl_rl`` PPO on
this backend -- see README.md and /workspace/bench/nexus_port/PORT_SPEC.md.

State is served as **zero-copy torch views** of Nexus GPU memory
(``__cuda_array_interface__``): no staging buffer, no host round-trip. Every
``BaseArticulation`` method that is not implemented raises
``NotImplementedError`` naming itself, so a caller hits a clear wall instead
of a silent wrong answer.

Layout is dictated by ``isaaclab.utils.backend_utils.FactoryBase``: for a
factory defined in ``isaaclab.<subpath>.<mod>`` it imports
``isaaclab_nexus.<subpath>`` and looks up a class of the factory's own name.
"""

from .physics.nexus_manager_cfg import NexusCfg, NexusMjcfCfg

__all__ = ["NexusCfg", "NexusMjcfCfg"]
