"""`InteractiveScene` replacement for the Nexus backend (no USD, no cloner).

Builds the scene straight from an `InteractiveSceneCfg`: terrain (generator) -> per-env tile
colliders, articulations from MJCF (the cfg's `spawn` must be a `NexusMjcfCfg`), then sensors.
Envs are Nexus batches in local coordinates, so `env_origins` are zeros. Lights and other
visual-only assets are skipped.
"""

from __future__ import annotations

import torch

from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.terrains import TerrainImporterCfg

from .physics.nexus_manager import NexusManager
from .terrain import NexusTerrainImporter

_ENV_NS, _ENV_REGEX_NS = "/World/envs", "/World/envs/env_.*"


class NexusScene:
    def __init__(self, cfg):
        from isaaclab.sim import SimulationContext
        self.cfg = cfg
        self.sim = SimulationContext.instance()
        self.device = "cuda:0"
        self.num_envs = int(cfg.num_envs); self.env_spacing = float(cfg.env_spacing)
        self.physics_backend = "nexus"; self.replicate_physics = True
        self.stage = None; self.extras = {}; self.clone_plan = None
        self.articulations, self.rigid_objects, self.rigid_object_collections = {}, {}, {}
        self.deformable_objects, self.sensors, self.surface_grippers = {}, {}, {}
        self._terrain = None
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        NexusManager.ensure_envs(self.num_envs)
        entries = [(k, v) for k, v in cfg.__dict__.items() if not k.startswith("_") and k not in ("num_envs", "env_spacing", "lazy_sensor_update", "replicate_physics", "filter_collisions", "clone_in_fabric")]
        # 1. terrain
        for name, ecfg in entries:
            if isinstance(ecfg, TerrainImporterCfg):
                self._terrain = NexusTerrainImporter(ecfg, self.num_envs, self.device)
        # 2. articulations
        for name, ecfg in entries:
            if isinstance(ecfg, ArticulationCfg):
                ecfg = ecfg.replace(prim_path=self._path(ecfg.prim_path))
                if not getattr(ecfg.spawn, "mjcf_path", ""):
                    raise ValueError(f"scene.{name}: on the Nexus backend spawn must be a NexusMjcfCfg (got {type(ecfg.spawn).__name__})")
                ecfg.spawn.num_envs = self.num_envs
                if self._terrain is not None and getattr(ecfg.spawn, "auto_floor", True):
                    ecfg.spawn.auto_floor = False
                from isaaclab.assets import Articulation
                self.articulations[name] = Articulation(ecfg)
        # 3. sensors
        for name, ecfg in entries:
            if isinstance(ecfg, ContactSensorCfg):
                from isaaclab.sensors import ContactSensor
                self.sensors[name] = ContactSensor(ecfg.replace(prim_path=self._path(ecfg.prim_path)))
            elif isinstance(ecfg, RayCasterCfg):
                from isaaclab.sensors import RayCaster
                self.sensors[name] = RayCaster(ecfg.replace(prim_path=self._path(ecfg.prim_path), mesh_prim_paths=[self._path(p) for p in ecfg.mesh_prim_paths]))
        self._entities = {**self.articulations, **self.rigid_objects, **self.sensors}
        if self._terrain is not None:
            self._entities["terrain"] = self._terrain

    @staticmethod
    def _path(p: str) -> str:
        return p.replace("{ENV_REGEX_NS}", _ENV_REGEX_NS).replace("{ENV_NS}", _ENV_NS)

    # ---- InteractiveScene API ----
    @property
    def terrain(self): return self._terrain
    @property
    def env_ns(self) -> str: return _ENV_NS
    @property
    def env_regex_ns(self) -> str: return _ENV_REGEX_NS
    @property
    def physics_dt(self) -> float: return float(NexusManager.get_physics_dt())
    @property
    def physics_scene_path(self): return None
    def keys(self): return list(self._entities.keys())
    def __getitem__(self, key: str):
        if key not in self._entities:
            raise KeyError(f"Scene entity with key {key!r} not found. Available: {self.keys()}")
        return self._entities[key]
    def get(self, key, default=None): return self._entities.get(key, default)
    def __str__(self): return f"NexusScene(num_envs={self.num_envs}, entities={self.keys()})"

    def initialize_renderers(self): pass
    def clone_environments(self, copy_from_source=False): pass
    def filter_collisions(self, global_prim_paths=None): pass

    def reset(self, env_ids=None, env_mask=None):
        for a in self.articulations.values(): a.reset(env_ids, env_mask)
        for s in self.sensors.values(): s.reset(env_ids, env_mask)

    def write_data_to_sim(self):
        for a in self.articulations.values(): a.write_data_to_sim()

    def update(self, dt: float):
        for a in self.articulations.values(): a.update(dt)
        for s in self.sensors.values(): s.update(dt)

    def get_state(self, is_relative: bool = False):
        st = {"articulation": {}}
        for n, a in self.articulations.items():
            st["articulation"][n] = {"root_pose": a.data.root_pose_w.torch.clone(), "root_velocity": a.data.root_vel_w.torch.clone(),
                                     "joint_position": a.data.joint_pos.torch.clone(), "joint_velocity": a.data.joint_vel.torch.clone()}
        return st

    def reset_to(self, state, env_ids=None, env_mask=None, is_relative: bool = False):
        for n, s in state.get("articulation", {}).items():
            a = self.articulations[n]
            a.write_root_pose_to_sim(s["root_pose"], env_ids, env_mask); a.write_root_velocity_to_sim(s["root_velocity"], env_ids, env_mask)
            a.write_joint_state_to_sim(s["joint_position"], s["joint_velocity"], env_ids=env_ids, env_mask=env_mask)
