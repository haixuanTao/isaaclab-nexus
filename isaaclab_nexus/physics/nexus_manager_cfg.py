from __future__ import annotations

from dataclasses import dataclass, field

from isaaclab.physics import PhysicsCfg
from isaaclab.utils.configclass import configclass


@configclass
class NexusMjcfCfg:
    """Spawn description for a Nexus articulation: one MJCF, replicated per env.

    Replaces Isaac Lab's USD ``spawn`` for this backend. Nothing here touches a
    USD stage; the robot is inserted straight into the Nexus state with
    ``NexusState.insert_mjcf_headless(path, env)`` once per environment.
    """

    mjcf_path: str = ""
    """Path to the MJCF scene."""

    num_envs: int = 1
    """Number of batched environments (Nexus batches == Isaac Lab envs)."""

    translation: tuple[float, float, float] | None = None
    """Spawn translation applied to every robot body (e.g. to start above terrain). None = as in the MJCF."""

    auto_floor: bool = True
    """Let the loader add a flat floor under the robot. Set False when the scene provides terrain."""


@configclass
class NexusCfg(PhysicsCfg):
    """Physics configuration selecting the Nexus CUDA backend.

    ``SimulationContext`` reads ``class_type`` and calls
    ``class_type.initialize(self)``; ``__post_init__`` binds it to
    :class:`NexusManager` so the manager is resolved without string lookup.
    ``backend_utils._get_backend`` maps the manager name to ``"nexus"`` and
    the factories then import ``isaaclab_nexus.<subpath>``.
    """

    backend_kind: str = "cuda"
    """Nexus backend kind: ``"cuda"`` (zero-copy views) or ``"webgpu"`` (staging copies)."""

    substeps: int = 1
    """Physics substeps per ``step()``. Isaac Lab's ``decimation`` sits above this."""

    solver_iterations: int = 1
    """SUBSTEPS per ``step()``, not PGS iterations: the engine divides ``dt`` by this and runs
    that many full integrate + dynamics passes (``set_visible_dt``). The engine's own default
    is 4, which integrates 4x faster than the ``dt`` Isaac Lab asked for and costs 2.15x the
    wall clock at 4096 envs (166 vs 454 ms of physics per control step) for no measured change
    in how the robot rests on terrain. 1 matches PhysX's integration rate at the same ``dt``.
    The engine's contact sensor reports a per-iteration impulse, so the backend scales the
    sensor readout by this value."""

    implicit_coriolis: bool = False
    """Engine default is True: the multibody mass matrix is rebuilt WITH a dt*C Coriolis term
    (`gpu_mb_compute_dynamics_pre` with the coriolis gemms), which is both the dominant kernel
    and a fidelity problem (over/under-damps with substep count; Zealot turns it off, MuJoCo
    and PhysX linearize once per step). False = explicit Coriolis, the MuJoCo/PhysX-like mode."""

    contact_reduction: bool = True
    """Merge every collider pair's manifolds into one deepest-``MAX_MANIFOLD_POINTS``
    manifold before the solvers. Off in the engine by default; needed on terrain, where
    a foot touches many triangles and each emits its own manifold."""

    cuda_graph_warmup: int = 0
    """Capture one physics step into a CUDA graph after this many normal steps, then replay it
    (0 = off). Replay skips the engine's buffer auto-resize and freezes the solver's coloring
    loop, so the scene must have settled first -- a robot still crumpling onto terrain grows
    the contact buffers and would be replayed with the old, too-small ones."""

    collisions_capacity: int = 256
    """Rigid contact-manifold capacity per env. Nexus's default (4096) costs ~11 MiB/env;
    a humanoid on terrain uses well under 256, which costs ~0.8 MiB/env."""

    def __post_init__(self):
        from .nexus_manager import NexusManager

        self.class_type = NexusManager
