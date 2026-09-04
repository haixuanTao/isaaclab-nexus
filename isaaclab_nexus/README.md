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
same task (medians of iterations 5-29, GPU idle, fallen-state resets correct): 1024 envs 1.41 vs
2.16 s, 2048 envs 1.85 vs 2.81 s, 4096 envs **2.71 vs 3.99 s — 36,275 vs 24,638 env-steps/s, 1.47x**. Numbers and method in `bench/nexus_port/PORT_SPEC.md`.

## Training on this backend: two documented deviations from AGILE's config

- **Observation normalization on** (`empirical_normalization=True`; AGILE ships it off). Without it,
  every 10k run collapsed at ~4,000 iterations with a critic divergence that no physics, reward,
  force, optimizer or reset measurement could pin on the engine; with it, training proceeds
  cleanly past 4,500. `train_nexus.py` sets it by default (`NEXUS_EMP_NORM=0` restores AGILE's).
- **Critic contact-force clip 5 kN** (`nexusify(critic_force_clip_n=5000)`): a critic-only
  observation clip; rewards, policy inputs and physics are untouched. Nexus's impulsive contact
  forces are milder than PhysX's, so AGILE's ±25 kN clip is met rarely here. Measured to delay,
  not remove, the divergence above; kept as the milder input.

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

## Fallen-state dataset and terrain: what this backend does differently (and why)

AGILE resets most episodes from a cached dataset of fallen poses collected by 2 m drops. Three things
about that path are backend-specific; all are in `PORT_SPEC.md` with the measurements.

- **Collect the dataset on Nexus.** Pass `agent_cfg` to `nexusify(...)`: it moves
  `fallen_state_dataset_cfg.cache_dir` to `<dir>_nexus`. AGILE's cache key does not include the physics
  backend, and a cache collected on PhysX stores `joint_pos` in the USD breadth-first joint order —
  replayed on the MJCF articulation (depth-first) it scrambles the pose (legs through the terrain).
  The articulation provides the PhysX-view hooks the collection loop calls
  (`_joint_effort_target_sim`, `root_view.set_dof_actuation_forces`, `_ALL_INDICES`).
- **`NEXUS_COLLECTION_VZ_MAX`** (default 3.5 m/s, `0` disables): during the raw collection steps the
  root's downward speed is capped. The terrain trimesh is a thin surface; at the 6 m/s of an uncapped
  2 m drop (3 cm per 200 Hz step, beyond the engine's 2 cm contact margin) limbs tunnel through it and are
  never pushed back out, and the recorded "fallen" state has a leg under the ground.
- **Terrain construction**: one trimesh tile per env (tile-local XY), a flat 1 m-grid apron at the
  tile's lowest height out to ±8 m for robots that roll off the tile, and a deep cuboid slab under both
  as a last resort. The apron matters: falling robots sink into a cuboid, not into a trimesh.

Residual after all of this: after a dataset reset ~4% of envs have a hand or foot pressed up to 16 cm
into the terrain (PhysX's own cache shows the same order of penetration).

The Nexus cache is anchored at the AGILE repo root (override with `NEXUS_DATASET_CACHE_DIR`) because
AGILE resolves `cache_dir` relative to the process cwd. Collection-time diagnostics: `NEXUS_SHIM_LOG=1`.

**Actuators.** Isaac Lab's implicit actuator does not clip torques in Python (PhysX's drive does): this
backend clips to `effort_limit_sim`, enforces `velocity_limit_sim` with a post-step clamp, and re-evaluates
the PD law at every physics substep (`NEXUS_PD_SUBSTEP`, `NEXUS_JOINT_VEL_CLAMP`; both default on).
**Termination deviation.** `nexusify` raises AGILE's `invalid_state.max_ang_vel` from 50 to 100 rad/s
(`invalid_state_max_ang_vel=`; `None` keeps AGILE's value): the pelvis angular-velocity tail is ~2x heavier
here than on PhysX under identical actions (contact coupling), and 50 sits inside it.
