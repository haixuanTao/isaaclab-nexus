"""``PhysicsManager`` implementation driving one headless Nexus CUDA state.

Lifecycle mirrors the base class: ``initialize`` -> ``reset`` -> ``step``* ->
``close``. All state is class-level, as in ``isaaclab_newton``, because
``SimulationContext`` holds the manager *type*, not an instance.
"""

from __future__ import annotations

import warnings

from typing import TYPE_CHECKING, Any, ClassVar

from isaaclab.physics import PhysicsManager
from isaaclab.scene_data.scene_data_backend import SceneDataBackend

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext


class NexusSceneDataBackend(SceneDataBackend):
    """No USD stage exists on this backend, so there are no transforms to sync."""

    @property
    def transforms(self):
        return None

    @property
    def transform_count(self) -> int:
        return 0

    @property
    def transform_paths(self) -> list[str]:
        return []


class NexusManager(PhysicsManager):
    """Owns the Nexus backend, state and pipeline; steps them in lock-step."""

    _backend: ClassVar[Any] = None   # nexus3d.NexusBackend
    _state: ClassVar[Any] = None     # nexus3d.NexusState
    _pipeline: ClassVar[Any] = None  # nexus3d.NexusPipeline
    _num_envs: ClassVar[int] = 0
    _finalized: ClassVar[bool] = False
    _scene_data: ClassVar[NexusSceneDataBackend | None] = None
    _assets: ClassVar[dict[str, Any]] = {}     # prim_path prefix -> Articulation
    _terrain: ClassVar[Any] = None
    _graph: ClassVar[bool] = False   # a CUDA graph of one physics step has been captured
    _steps: ClassVar[int] = 0

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        import nexus3d

        super().initialize(sim_context)
        kind = getattr(PhysicsManager._cfg, "backend_kind", "cuda")
        cls._backend = nexus3d.NexusBackend(kind)
        cls._state = nexus3d.NexusState()
        cls._pipeline = nexus3d.NexusPipeline()
        cls._num_envs = 0
        cls._finalized = False
        cls._graph = False
        cls._steps = 0
        cls._scene_data = NexusSceneDataBackend()

    @classmethod
    def reset(cls, soft: bool = False) -> None:
        PhysicsManager._sim_time = 0.0
        if not soft:
            cls.finalize()

    @classmethod
    def forward(cls) -> None:
        # Nexus recomputes forward kinematics inside the step; nothing to do.
        return None

    @classmethod
    def step(cls) -> None:
        if not cls._finalized:
            cls.finalize()
        # One physics step as a replayed CUDA graph, once the scene has settled.
        # Capture freezes buffer addresses and the solver's coloring loop and
        # skips `auto_resize_buffers`, so it only happens after `warmup` normal
        # steps -- and never before the contact/coloring buffers have stopped
        # growing (`cuda_graph_warmup` in NexusCfg).
        if cls._graph:
            if cls._pipeline.replay_cuda_graph():
                PhysicsManager._sim_time += cls.get_physics_dt()
                return
            cls._graph = False                                  # graph lost; fall back
        cls._pipeline.simulate_headless(cls._backend, cls._state, None)
        cls._steps += 1
        if cls._steps % 200 == 0 and hasattr(cls._state, "rbd_resize_stats"):
            cls._log_resize_ratchet()
        warmup = int(getattr(PhysicsManager._cfg, "cuda_graph_warmup", 0) or 0)
        if warmup and cls._steps >= warmup and not cls._graph and hasattr(cls._pipeline, "capture_cuda_graph_headless"):
            try:
                cls._graph = bool(cls._pipeline.capture_cuda_graph_headless(cls._backend, cls._state))
            except RuntimeError as e:
                # Capture fails on anything that allocates or reads back inside the step.
                # Known case: the deterministic contact sort re-creates a 4-byte tensor
                # every step (its cache key holds the contact count), so capture needs
                # NEXUS_DETERMINISTIC=0. Not fatal -- keep re-encoding each step.
                warnings.warn(f"Nexus CUDA-graph capture failed, staying on the encoded path: {e}")
                cls._graph = False
                PhysicsManager._cfg.cuda_graph_warmup = 0
        PhysicsManager._sim_time += cls.get_physics_dt()

    _resize_seen: ClassVar[tuple] = ()
    _pairs_peak: ClassVar[int] = 0

    @classmethod
    def _log_resize_ratchet(cls) -> None:
        """Every 200 physics steps: record the peak read-back pair count and warn when the
        engine grew its collision buffers or its colour count (both are permanent under the
        default policies and each one slows every later step)."""
        st = cls._state.rbd_resize_stats()
        cls._pairs_peak = max(cls._pairs_peak, int(st.get("pairs_len", 0)))
        key = (int(st.get("capacity_per_batch", 0)), int(st.get("max_colors", 0)))
        if cls._resize_seen and key != cls._resize_seen:
            warnings.warn(f"[nexus] buffer ratchet at physics step {cls._steps}: capacity/batch "
                          f"{cls._resize_seen[0]} -> {key[0]}, max_colors {cls._resize_seen[1]} -> {key[1]} "
                          f"(peak pairs/batch so far {cls._pairs_peak})")
        cls._resize_seen = key

    @classmethod
    def close(cls) -> None:
        cls._pipeline = None
        cls._state = None
        cls._backend = None
        cls._finalized = False

    @classmethod
    def get_scene_data_backend(cls) -> SceneDataBackend:
        return cls._scene_data

    @classmethod
    def get_device(cls) -> str:
        return "cuda:0" if cls.is_cuda() else PhysicsManager._device

    # ------------------------------------------------------------------ nexus API
    @classmethod
    def is_cuda(cls) -> bool:
        return bool(cls._backend is not None and cls._backend.is_cuda())

    @classmethod
    def state(cls):
        return cls._state

    @classmethod
    def backend(cls):
        return cls._backend

    @classmethod
    def register_asset(cls, prim_path: str, asset: Any) -> None:
        cls._assets[prim_path] = asset

    @classmethod
    def find_asset(cls, prim_path: str):
        """Longest registered prim_path prefix matching `prim_path` (e.g. '.../Robot/pelvis' -> '.../Robot')."""
        best = None
        for k, v in cls._assets.items():
            if prim_path == k or prim_path.startswith(k.rstrip("/") + "/"):
                if best is None or len(k) > len(best[0]):
                    best = (k, v)
        return best

    @classmethod
    def set_terrain(cls, terrain: Any) -> None:
        cls._terrain = terrain

    @classmethod
    def terrain(cls):
        return cls._terrain

    @classmethod
    def ensure_envs(cls, n: int) -> None:
        """Create batched environments 0..n-1 once; later callers reuse them."""
        while cls._num_envs < int(n):
            cls.add_env()
        if cls._num_envs != int(n):
            raise RuntimeError(f"scene already has {cls._num_envs} envs, asked for {n}")

    @classmethod
    def num_envs(cls) -> int:
        return cls._num_envs

    @classmethod
    def add_env(cls) -> int:
        """Reserve one more batched environment; returns its index."""
        if cls._finalized:
            raise RuntimeError("cannot add environments after finalize()")
        env = 0 if cls._num_envs == 0 else cls._state.add_environment()
        cls._num_envs += 1
        return env

    @classmethod
    def finalize(cls) -> None:
        """Upload the scene to the GPU, applying Isaac Lab's dt / gravity. Idempotent."""
        if not cls._finalized:
            import nexus3d

            sim_cfg = getattr(PhysicsManager._sim, "cfg", None)
            dt = float(getattr(sim_cfg, "dt", 1.0 / 60.0))
            substeps = int(getattr(PhysicsManager._cfg, "substeps", 1))
            cap = int(getattr(PhysicsManager._cfg, "collisions_capacity", 256))
            cls._state.set_rbd_collisions_capacity(cap)
            if hasattr(cls._state, "set_rbd_resize_policy"):
                cls._state.set_rbd_resize_policy(str(getattr(PhysicsManager._cfg, "collisions_resize_policy", "grow")))
            cls._state.set_rbd_solver_iterations(int(getattr(PhysicsManager._cfg, "solver_iterations", 4)))
            cls._state.set_rbd_dt(dt / max(substeps, 1))
            cls._state.set_rbd_steps_per_frame(max(substeps, 1))
            cls._state.finalize_headless(cls._backend)
            if hasattr(cls._state, "set_implicit_coriolis"):          # needs the finalized rbd state
                cls._state.set_implicit_coriolis(bool(getattr(PhysicsManager._cfg, "implicit_coriolis", False)))
            if getattr(PhysicsManager._cfg, "contact_reduction", False) and hasattr(cls._pipeline, "set_contact_reduction"):
                cls._pipeline.set_contact_reduction(cls._backend, True)
            g = getattr(sim_cfg, "gravity", (0.0, 0.0, -9.81))
            cls._state.set_rbd_gravity_headless(cls._backend, nexus3d.Vec3(*[float(x) for x in g]))
            cls._finalized = True

    @classmethod
    def synchronize(cls) -> None:
        """Block until queued GPU work is done; required before reading views."""
        if cls._backend is not None:
            cls._backend.synchronize()
