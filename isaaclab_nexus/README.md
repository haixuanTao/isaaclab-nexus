# `isaaclab_nexus` — Nexus as an Isaac Lab physics backend

Runs Isaac Lab (3.0.0b2) manager-based environments on [Nexus](https://github.com/dimforge/nexus)
(GPU rigid-body engine, CUDA backend) instead of PhysX or Newton. No USD, no Fabric,
no cloner: robots are loaded from MJCF straight into the Nexus state, one Nexus batch
per Isaac Lab environment.

Validated end to end on WBC-AGILE's `HeightTracking-G1-v0` (Unitree G1, 29 DOF, rough
terrain, height-scan + contact sensors) training with `rsl_rl` PPO.

## Install

1. Build the Nexus fork's Python bindings with the CUDA feature into the Isaac Lab venv.
   The engine side of this port lives on
   [`Haixuantao/nexus@isaac-backend`](https://github.com/Haixuantao/nexus/tree/isaac-backend)
   (zero-copy CUDA views, batched env control, MJCF name capture, batched MJCF spawn,
   contact-sensor readout, contact reduction):

   ```bash
   git clone -b isaac-backend https://github.com/Haixuantao/nexus.git nexus-fork
   cd nexus-fork
   env -u CONDA_PREFIX VIRTUAL_ENV=<isaaclab-venv> \
     CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=<...>/nexus_rbd_shaders3d.cubin \
     CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=<...>/vortx_shaders.cubin \
     maturin develop --release --features cuda -m crates/nexus_python3d/Cargo.toml
   ```

2. Put this package on the venv's path (`echo /path/to/parent > <site-packages>/isaaclab_nexus.pth`).

3. Teach Isaac Lab's backend factory about the name — one line in
   `isaaclab/utils/backend_utils.py::_get_backend`:

   ```python
   if manager_name.startswith("nexus"):
       return "nexus"
   ```

   Every `FactoryBase` subclass (`Articulation`, `ContactSensor`, `RayCaster`, ...) then
   imports `isaaclab_nexus.<subpath>` on its own.

## Use

```python
from isaaclab_nexus.envs import nexusify

env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
env_cfg.scene.num_envs = 4096
nexusify(env_cfg, "/path/to/robot.xml")     # physics -> NexusCfg, USD spawns -> MJCF
env = gym.make(TASK, cfg=env_cfg)           # ManagerBasedRLEnv on Nexus
```

`nexusify` rewrites the config in place: `sim.physics` becomes `NexusCfg`, every
`ArticulationCfg.spawn` becomes `NexusMjcfCfg(mjcf_path=...)`, visual-only assets are
dropped, and event/curriculum terms that need PhysX-only views
(`randomize_rigid_body_material`, `randomize_rigid_body_com`) are removed with a warning.
It also installs a scene dispatch so `ManagerBasedEnv` builds a `NexusScene`
(no USD stage) whenever the sim config selects Nexus.

## Layout

| file | what it provides |
|---|---|
| `physics/nexus_manager_cfg.py` | `NexusCfg(PhysicsCfg)`, `NexusMjcfCfg` (MJCF spawn description) |
| `physics/nexus_manager.py` | `NexusManager(PhysicsManager)` — owns the backend/state/pipeline, `initialize/reset/forward/step` |
| `assets/articulation/` | `Articulation` + `ArticulationData` — zero-copy torch views over the Nexus workspace, joint targets/efforts, resets, name/DOF maps from MJCF |
| `scene.py` | `NexusScene` — `InteractiveScene` replacement (terrain, articulations, sensors; no cloner) |
| `terrain.py` | `NexusTerrain` / `NexusTerrainImporter` — Isaac `TerrainGenerator` meshes as per-env trimesh colliders + a GPU height grid |
| `sensors/ray_caster/` | `RayCaster` — height scan against the terrain grid |
| `sensors/contact_sensor/` | `ContactSensor` — per-link normal impulse from the engine's contact-sensor kernel |
| `envs.py` | `nexusify()` / `install()` |

## Config knobs (`NexusCfg`) — and why the defaults are not the engine's

| field | default | meaning |
|---|---|---|
| `backend_kind` | `"cuda"` | `"cuda"` (zero-copy views) or `"webgpu"` (staging copies) |
| `substeps` | 1 | physics substeps per `step()` (Isaac's `decimation` sits above this) |
| `solver_iterations` | **1** | *substeps* inside the engine (it divides `dt` by this and runs that many full integrate + dynamics passes — not PGS iterations). Engine default 4 = 800 Hz for a 200 Hz `dt`; 1 matches PhysX's integration rate. 2.15x |
| `implicit_coriolis` | **False** | engine default True rebuilds the mass matrix with a dt·C term (the dominant kernel, and a damping artefact that scales with substeps). False = one Coriolis linearization per step, as MuJoCo/PhysX/Genesis/Zealot. 2.3x |
| `contact_reduction` | `True` | merge a collider pair's manifolds into one deepest-point manifold (engine default off). +7% |
| `collisions_capacity` | 256 | contact-manifold capacity per env (~0.8 MiB/env; the engine default 4096 costs ~11 MiB/env) |
| `cuda_graph_warmup` | 0 | capture one physics step into a CUDA graph after N steps and replay it. Measured: no gain at any stage (kernel-bound); needs `NEXUS_DETERMINISTIC=0` to capture |

And one thing that is not a knob but matters as much: **cap the robot's convex hulls at 64
vertices** (PhysX's default `convexHullVertexLimit`). The unitree G1 MJCF collides its full
visual STLs (mean hull 1,087 vertices, pelvis 5,583); `bench/nexus_port/make_convex_mjcf.py 64`
writes a copy with support-mapped hulls. 1.32x.

Net effect on AGILE's `HeightTracking-G1-v0` on an RTX 5090, PPO iteration time vs PhysX on the
same task: 1024 envs 1.52 vs 2.16 s, 2048 envs 1.89 vs 2.81 s, 4096 envs **2.45 vs 3.99 s**
(40,124 vs 24,638 env-steps/s), with an identical reward curve at every step. Numbers and method in `bench/nexus_port/PORT_SPEC.md`.

## Known limitations

- No USD path, no cloner, no rendering; environments are Nexus batches in local
  coordinates (`env_origins` are zeros).
- Terrain curriculum records level changes but a tile's collider is fixed at
  construction, so an env does not move to a new tile mid-run.
- Domain-randomization terms that drive PhysX views (material, COM) are dropped, so
  friction is fixed: terrain colliders take `TerrainImporterCfg.physics_material.static_friction`
  (rapier's own default is 0.5), robot colliders take the MJCF geom friction (MuJoCo default 1.0).
- `ArticulationData` implements the properties AGILE's task needs; anything else
  raises `NotImplementedError` naming the property.
