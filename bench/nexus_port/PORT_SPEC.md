# Nexus as an Isaac Lab physics backend — feasibility, measured

Investigated against dimforge/nexus @ main (cloned 2026-09-02) and the shipped
`dimforge-nexus3d` 0.1.0 wheel, versus Isaac Lab 3.0.0b2's backend contract.
NOT a port. This is what a port would have to satisfy and what already exists.

## Corrections to my earlier estimate

Three things I got wrong, all in Nexus's favour:

| I said | Actually |
|---|---|
| WebGPU/SPIR-V only, CUDA interop is the deep problem | A **`cuda` backend feature** exists (`--features cuda`), alongside metal/cpu; webgpu is only the default |
| No batched multi-env concept | `NexusState.add_environment()` / `insert_*_in(env, ...)` is **native**, and `dofs_per_batch()` is first-class |
| USD is load-bearing | Only **10 of 83** files in Isaac Lab's Newton backend touch `pxr`; ingestion funnels through one `builder.add_usd(...)` call. Nexus ships **MJCF and URDF loaders** (`insert_mjcf`, `insert_urdf`) |

And one that matters a lot: **Nexus already sits on vortx**
(`src_rbd/pipeline/rbd_state.rs:25: use vortx::tensor::Tensor;`). Engine state is
already in your own tensor abstraction, so the interop question is vortx<->torch,
not wgpu<->CUDA.

## Scope: the contract is far smaller than the file count suggests

`isaaclab_newton` is 24,338 lines / 83 files, and `ArticulationData` alone declares
**76 properties**. But AGILE's G1 task reads only **16 of those 76**:

| property | refs in agile/ | Nexus source |
|---|---:|---|
| joint_pos | 29 | `dof_state()` |
| joint_vel | 21 | `dof_state()` |
| default_joint_pos | 18 | static, from MJCF import |
| joint_pos_limits | 9 | static, from MJCF import |
| joint_vel_limits | 7 | static |
| joint_acc | 5 | derive (finite difference) |
| applied_torque | 4 | **GAP** (see below) |
| soft_joint_vel_limits | 3 | derive from limits |
| joint_pos_target | 3 | motor `target_pos` |
| default_root_state | 3 | static |
| default_joint_vel | 2 | static |
| body_com_pose_b | 2 | derive from `body_poses()` + com offsets |
| root_link_quat_w | 1 | `body_poses()` |
| root_link_pos_w | 1 | `body_poses()` |
| joint_armature | 1 | in `dof_state()` layout |
| body_link_quat_w | 1 | `body_poses()` |

Allowing for what Isaac Lab reads internally (sensors, actuators, resets), budget
**~25-30 properties**, not 76.

## What Nexus already has (Rust side)

| need | Nexus API |
|---|---|
| batched joint state, read | `MultibodySet::dof_state() -> &Tensor<f32>` |
| batched joint state, write | `dof_state_mut() -> &mut Tensor<f32>` |
| batch geometry | `dofs_per_batch()`, `NexusCounts.num_environments` |
| body poses | `RbdState::body_poses() -> &Tensor<Pose>` |
| collider poses | `collider_poses()`, `collider_world_poses()` |
| contacts | `RbdState::contacts() -> &Tensor<GpuIndexedContact>` |
| raycast | `src_rbd/queries/` |
| joint drive | motors with `target_pos`, `target_vel`, `stiffness`, `damping`, `max_force` |
| asset ingestion | `insert_mjcf`, `insert_urdf` |

## The four real gaps

**1. Python bindings expose almost none of it.**
The shipped `nexus3d` 0.1.0 API is **write-only**: `NexusState` has 20 methods, all
construction (`insert_*`, `add_environment`, `set_multibody_motor_velocity`,
`finalize`); `NexusPipeline` has 2 (`preload_pipelines`, `simulate`). Every
state-shaped accessor in the module is on a *Builder* — initial conditions, not
readback. `RigidBody` has zero public methods. So the Rust has the state and Python
cannot see it. This is a PyO3 binding gap, not an engine gap — much the smaller problem.

**2. No explicit joint-effort input.**
Nexus drives joints through motors: PD with `stiffness`/`damping` and `max_force`
saturation. That maps cleanly onto Isaac Lab's *implicit* actuator. It does NOT map
onto AGILE's G1, which uses `G1_29DOF_DELAYED_DC_MOTOR` — an explicit actuator that
computes torque host-side (delay + torque-speed envelope) and writes
`joint_effort_target`. No `dof_force` / `apply_torque` / external-force entry point
exists in `src_rbd/`. Either add a generalized-force input buffer, or approximate the
DC motor as PD+max_force, which changes the physics and invalidates the comparison.

**3. vortx Tensor -> torch, zero-copy.**
~25-30 properties read every step at 4096 envs. A host round-trip per step would
erase any win. Since Nexus is already on vortx and (per the paper) so is Zealot, this
is the most tractable gap for this team specifically — but it is unwritten today.

**4. Terrain.**
AGILE generates rough terrain procedurally (`STAND_UP_ROUGH_TERRAIN_G1_CFG`, 72 tiles
of 8x8 m) and imports it as USD meshes. Nexus has a heightfield path
(`examples/heightfield3.py`) but nothing consumes Isaac Lab's terrain generator.
Not optional: terrain packing is exactly what produced the 8192-env broadphase wall,
so it is load-bearing for any scaling comparison.

## Revised estimate

Assuming the batched API work you already planned:

| piece | est. |
|---|---|
| PyO3 readback bindings (dof_state, body_poses, contacts, counts) | 1-2 wk |
| vortx <-> torch zero-copy | 1-2 wk, unknown if CUDA-backend-only |
| generalized joint-force input in the solver | 1-3 wk, touches solver internals |
| `isaaclab_nexus` factories (~25-30 props, articulation + rigid object + contact + raycast) | 3-5 wk |
| MJCF ingestion + name->index maps replacing `add_usd` | 1 wk |
| terrain heightfield path | 1-2 wk |
| parity harness + convention debugging | **open-ended** |

**~8-14 weeks**, down from my earlier 3-6 months. The dominant risk is unchanged and
is not lines of code: the property contract is semantically fussy (link vs COM frames,
world vs body, quaternion order, Featherstone vs maximal coordinates). Every mismatch
is a silent physics bug, not a crash. Build the parity harness first — drive both
backends from identical initial states and diff state trajectories per step.

## What is NOT the blocker
- USD (skippable; MJCF path exists both sides)
- CUDA interop (cuda feature exists)
- Multi-env batching (native)
- Line count (~15k, and only ~25-30 properties actually needed)

---

# BUILT AND VALIDATED (spike, 2026-09-02)

State readback from Nexus into Python now works. Patch:
`/workspace/bench/nexus_port/nexus_readback.patch` (against dimforge/nexus @ main).

## Two changes

**1. `src_rbd/dynamics/multibody/multibody_from_rapier.rs`** — added `COPY_SRC` to
the `dof_state` allocation. It was `storage` only, so wgpu rejected any readback:
`Usage flags BufferUsages(COPY_DST | STORAGE) ... do not contain required usage
flags BufferUsages(COPY_SRC)`. The `links_workspace` buffer immediately above it
already carries `COPY_SRC` with the comment "so hosts can read joint/link state
back (observation pipelines)", so this matches existing intent.

**2. `crates/nexus_python3d/src/nexus.rs`** — four new `NexusState` methods:

| method | returns |
|---|---|
| `dofs_per_batch()` | u32 |
| `links_per_batch()` | u32 |
| `dof_state(viewer)` | `(rows, dofs_per_batch)` f32 — velocity section of the 7-section DOF buffer |
| `link_state(viewer)` | `(joint_pos, link_pose_w, link_vel_w)` |

`link_state` returns, one row per link, laid out `env * links_per_batch + link`:
- `joint_pos`   `(n, 6)` generalized coords; first `ndofs` meaningful,
                 linear DOFs first then angular
- `link_pose_w` `(n, 7)` link-to-world pose `x y z qx qy qz qw`
- `link_vel_w`  `(n, 6)` world rigid-body velocity `vx vy vz wx wy wz`

Path: `NexusState.rbd -> multibodies().links_workspace().buffer()`
-> `backend.slow_read_vec()` -> `ws_soa_to_structs()` -> numpy.

## Validation
4 envs x 3-link revolute chains under gravity, 60 steps:
- joint coordinate lands in slot 3 (first angular DOF) = **1.12368 rad**
- angle recovered independently from the link-to-world quaternion = **1.12061 rad**

Two separate readback paths agree to 0.3%, so the decode and the layout are right.

## Maps onto the Isaac Lab contract
| Isaac Lab property | source | status |
|---|---|---|
| `joint_pos` | `link_state().joint_pos` | **done** |
| `joint_vel` | `dof_state()` velocity section / `link_state` | **done** |
| `root_link_pos_w`, `root_link_quat_w` | `link_pose_w` row of link 0 | **done** |
| `body_link_pose_w` | `link_pose_w` | **done** |
| `body_com_vel_w` | `link_vel_w` | **done** |
| `joint_acc` | finite difference | derivable |
| `applied_torque` | `gen_forces` tensor (exists, unbound) | not yet |

That is 5 of the 16 properties AGILE actually reads, covering the two highest-use
ones (`joint_pos` 29 refs, `joint_vel` 21).

## What this spike did NOT establish
- **CUDA backend does not build here.** Rust-CUDA's `rustc_codegen_nvvm` requires
  **LLVM 7**; Ubuntu 24.04 ships 14+. This is the **webgpu** backend.
- **Readback is a staging-buffer copy, not zero-copy.** `slow_read_vec` allocates a
  staging buffer and blocks. At 4096 envs every step this would dominate. The
  zero-copy path is `device_ptr_raw()` (khal `cuda.rs:75`) exposed as
  `__cuda_array_interface__` — needs the CUDA backend, hence LLVM 7 or cuda-oxide.
- No writes back into the sim (actuation), no contacts, no raycast, no MJCF->Isaac
  Lab index mapping, no factory registration.

## Build gotchas (undocumented, all blocked)
| blocker | fix |
|---|---|
| `cargo-gpu` on crates.io | **stub, prints "Coming Soon"** — use `--git https://github.com/Rust-GPU/cargo-gpu` |
| `cargo-cuda` on crates.io | also prints "Coming Soon" |
| rust-gpu nightly consent | khal-builder gives no TTY; pre-run `cargo gpu build --auto-install-rust-toolchain` once |
| stale build-script cache | `cargo clean -p vortx` — cargo silently reuses the stub's output |
| viewer needs a display | `NexusViewer(w, h, headless=True)` |

---

# BUILT: Isaac Lab factory dispatch onto Nexus (fork) on CUDA — 2026-09-02

Stack: Haixuantao/{nexus,khal,vortx,cuda-oxide} forks, cuda-oxide cubins (sm_120),
`dimforge-nexus3d` built with `--features cuda` into the WBC-AGILE venv.

## What runs (test_isaac_backend.py, no Kit / no AppLauncher)
    isaaclab.assets.Articulation(cfg)                 # the REAL Isaac Lab factory
      -> FactoryBase.__new__ -> _get_backend() == "nexus"
      -> isaaclab_nexus.assets.articulation.Articulation
      -> NexusState.insert_mjcf_headless(nv_humanoid.xml, env) x 8 envs
      -> finalize_headless(NexusBackend("cuda")); dt/gravity from sim.cfg
      -> torch.as_tensor(links_workspace_cuda())  # data_ptr == nexus ptr (zero-copy)
    NexusManager.step() x 100  -> every env falls under gravity

    num_instances=8 num_bodies=22 num_joints=27
    joint_coords (8,22,6)  body_link_pose_w (8,22,7)  joint_vel (8,27)  -- all views

## Pieces
| piece | file | state |
|---|---|---|
| `NexusCfg(PhysicsCfg)`, `NexusMjcfCfg` | isaaclab_nexus/physics/nexus_manager_cfg.py | done |
| `NexusManager(PhysicsManager)` | isaaclab_nexus/physics/nexus_manager.py | init/step/reset/close/dt/gravity |
| `Articulation` / `ArticulationData` | isaaclab_nexus/assets/articulation/ | 5 read props zero-copy; rest raise NotImplementedError by name |
| `_get_backend()` "nexus" branch | isaaclab/utils/backend_utils.py (+ .orig backup) | patched |
| fork bindings: `CudaArray`, `links_workspace_cuda`, `dof_state_cuda`, `ws_layout`, `NexusBackend.synchronize/is_cuda` | nexus-fork crates/nexus_python3d | fork_zero_copy_bindings.patch |

## Not done (in the order it blocks AGILE)
1. write path: joint targets / efforts (`gen_forces`), root pose/vel writes, resets
2. MJCF -> Isaac name maps (body_names / joint_names are index placeholders) and the
   per-joint ndof map to compact `joint_coords (n,links,6)` into flat `joint_pos (n,dofs)`
3. ContactSensor, RayCaster (height scan), terrain heightfield
4. running under a real `SimulationContext` + `ManagerBasedRLEnv` decimation loop
5. parity harness vs PhysX

---

# STATUS 2026-09-02 (later): ①②③④ progress on the fork/CUDA backend

## ② names + DOF map — DONE (validated)
- `links_static_host()` -> per-link (rb_id, parent, mb, assembly_id, ndofs, kinematic, locked_axes, motor_axes)
- `mjcf_names()` -> body/joint/actuator names resolved onto Nexus links via rb_id; 22/22 links named
- floating base = 6-DOF free joint on link 0, EXCLUDED from joint vectors (Isaac semantics): num_joints = 21
- per-joint check: coordinate slot == velocity column for all 21 joints (rel err ~0.00-0.09; outliers are joints at limits)

## ① write path — DONE (validated through Isaac Lab's Articulation API)
- position targets: on-GPU `scatter_motor_targets_gpu` into force-based PD motors; gains from `cfg.actuators`
  (`set_motor_gains`, since MJCF `<motor>` actuators leave PD gains at 0). 0.25 -> 0.087 rad in 1 s.
- effort: zero-copy write into `external_gen_forces` (dof-major, batch-innermost). +3.7 rad/s in 5 steps on a 0-gain joint.
- resets: `publish_reset_template` once, `reset_envs(env_ids, offsets, dof_vels)` batched. NOTE: the offset
  translates the WHOLE env snapshot (fixed bodies too) -- Isaac env-origin semantics; do not use it to lift a
  robot above terrain. Spawn placement is `insert_mjcf_headless(..., translation=)` (added).
- `write_root_pose_to_sim` (translation), `reset(env_ids)`, `set_joint_*_target(_index)`, `find_joints/bodies` with regex,
  ProxyArray data (`data.joint_pos.torch`). Unimplemented calls raise NotImplementedError by name.

## ④ env loop on a REAL SimulationContext — DONE (throughput measured)
`SimulationContext(SimulationCfg(physics=NexusCfg(), create_stage_in_memory=True))` constructs in 1.4 s with Kit
in-process, `_get_backend() == "nexus"`, `sim.reset()/step()` drive NexusManager. AGILE-shaped loop
(24 control steps x decimation 4, targets -> write_data_to_sim -> 4x sim.step -> update -> obs cat + readback):

| envs | spawn | iteration (median) | env-steps/s |
|---:|---:|---:|---:|
| 512  | 0.4 s | 517 ms  | 23,788 |
| 1024 | 0.6 s | 910 ms  | 26,994 |
| 2048 | 0.9 s | 1,796 ms | 27,376 |
| 4096 | 1.5 s | 2,946 ms | **33,373** |

AGILE (G1 29-DOF, rough terrain, sensors) at 4096: 3,990 ms / 24,638 env-steps/s. NOT like-for-like:
this is nv_humanoid (21 DOF), flat floor, no terrain/sensors, unoptimized Python loop with a per-step sync.
Memory: default rigid contact capacity costs 11.6 MiB/env (4096 -> OOM on 32 GB); `NexusCfg.collisions_capacity=256`
-> 0.84 MiB/env, 4096 envs = 3.5 GB.

### ④ with terrain + sensors (`bench_env_loop_terrain.py`, plain step loop, no decimation)
4096 envs, per-env 2,048-tri terrain collider, feet ContactSensor (history 3), 121-ray height scanner, obs cat
(172 floats/env) each step: setup 2.8 s, 6.45 GiB GPU, **24.6 steps/s = 101 k env-steps/s**; feet at
+0.019 m, 100 % in contact, obs finite. Reference: flat floor without sensors ran 33 k env-steps/s in the
AGILE-shaped loop (decimation 4 => 4 physics steps per control step); per physics step this terrain loop is
40.6 ms vs 2,946 / 96 = 30.7 ms flat -- terrain + sensors add ~30 % per step at 4096 envs.

## ③ sensors + terrain — DONE (validated through Isaac Lab's sensor API)
Tests: `test_contact_sensor.py`, `test_terrain_v2.py`, `test_isaac_sensors.py` (all pass).
- terrain: `NexusTerrain` = Isaac `TerrainGenerator` (USD-free) -> per-env tile trimesh + GPU height grid
  (rasterized once, 0.05 m). Tile from AGILE's `STAND_UP_ROUGH_TERRAIN_G1_CFG` (+-0.14 m relief). The
  *collider* is re-rasterized at `collider_res=0.25 m` (2,048 tris/env); the height grid keeps full resolution.
- RayCaster (height scanner): grid pattern + yaw alignment, bilinear lookup on the height grid; hits match
  the grid to 1e-4 (`test_isaac_sensors.py`).
- ContactSensor: engine's built-in per-link normal-impulse sensor (`set_contact_sensor_links`, `MAX_CONTACT_SENSORS`
  raised 4 -> 32 in the fork). Isaac semantics: `net_forces_w` (+Z, magnitude), history, air/contact time,
  `compute_first_contact/air`. Summed over every link of a resting humanoid: **1.019 x weight** (408 N vs 401 N).
- Result on AGILE's rough terrain (4 envs, 3 s): foot-terrain gap **+0.082 m** (foot geometry), contact time
  1.05 s / 1.09 s per foot, air time seen during the fall, ray hit == grid height.

### ③ root causes found (each cost a wrong first diagnosis)
1. sensor readout = **per-solver-iteration** impulse: sum/weight = 1.019 / 0.509 / 0.255 / 0.133 at
   1 / 2 / 4 / 8 iterations (`steps_per_frame` has no effect). Backend scales by `NexusCfg.solver_iterations`
   (which it also sets on the engine). Physically the proper fix is to accumulate in the engine.
2. sinking on fine meshes is **not** winding (all 12,800 terrain normals point +z; flipping makes it worse:
   -0.35 -> -0.71 m) and **not** the pair capacity (`collisions_capacity` 256 / 1024 / 4096 identical) and
   **not** the per-multibody cap (89 points used of 256). It is `MAX_MANIFOLD_POINTS = 4` per (link, trimesh)
   pair, filled in BVH-traversal order rather than deepest-first: a foot over 100+ small triangles keeps 4
   arbitrary ones. Sink vs collider resolution: 0.10 m -> -0.353 m, 0.25 m -> -0.089 m (raw), +0.02..0.08 m
   through the backend. Engine fix = per-triangle sub-manifolds or deepest-4 selection in `narrow_phase.rs`.
3. `MAX_MB_CONTACTS_PER_MB` 64 -> 256 *was* needed (89 contact points on a crumpled humanoid > 64).

## ④ nsys of the 4096-env loop (Nexus backend, real SimulationContext, Isaac Lab API)
Same loop shape as the AGILE trace (24 control steps x decimation 4, 6 iterations, 23.5 s window):

| metric | AGILE / PhysX (G1 29-DOF, terrain, sensors) | Nexus backend (nv_humanoid 21-DOF, flat floor) |
|---|---:|---:|
| GPU busy (union of kernels) | 49.6% | **91.7%** |
| kernel launches / s | 28,623 | **5,491** |
| mean kernel duration | 17.6 us | **166.9 us** |
| env-steps / s at 4096 envs | 24,638 | 34,104 (33,373 untraced) |
| torch elementwise share of GPU | 6.2% (51% of launches) | 1.3% |

Top Nexus kernels: `gpu_mb_integrate_and_dynamics_pre` 10.4 s (48%), `gpu_mb_compute_dynamics_pre` 3.5 s,
`gpu_mb_gravity_and_lu` 1.3 s. Profiling overhead ~2%.
CAVEATS: different robot/DOF count, no terrain or sensors on the Nexus side yet, per-step host sync
in the loop. The *shape* comparison (busy fraction, launch rate, kernel granularity) is the valid part.

## ⑤ parity vs MuJoCo (first cut) — DONE
Oracle: MuJoCo 3 on gym's classic `humanoid.xml` (MuJoCo-stable; `nv_humanoid.xml` explodes in MuJoCo
itself at any dt/integrator, so it is not a valid reference). Same MJCF in Nexus (free root written as
`<freejoint/>` for rapier-mjcf; explicit -z gravity; matching flat floor at z=0), same initial state,
no actuation, dt = 1/200, 2 s.

| | MuJoCo | Nexus |
|---|---:|---:|
| root z at 0.25 / 0.50 / 0.75 / 1.0 s | 1.286 / 1.150 / 0.598 / 0.259 | 1.280 / 1.066 / 0.498 / 0.265 |
| final rest height | 0.080 | 0.077 |
| root z mean / max abs diff | — | **0.049 m / 0.195 m** (max at t = 0.69 s, contact transient) |
| joint angle mean / max abs diff (17 hinges) | — | 0.095 rad / 0.99 rad (worst: elbows, unactuated flailing) |

Gravity note: multibodies honor `set_rbd_gravity_headless`; an MJCF without `<option gravity>` otherwise
runs Y-up (Nexus default). `NexusManager.finalize()` sets gravity from `sim.cfg.gravity`.

## ③ findings (contacts / terrain) — root causes, all engine-level constants
- Spawn: Nexus builds the multibody with its root free joint at the world origin (MJCF root pos and
  `MjcfLoaderOptions.shift` both dropped). Placement = write the root's linear free-joint coords in the
  zero-copy workspace view (`ws[0, WS_COORDS, env, :3]`), then one FK step. Also gives
  `write_root_pose_to_sim` (translation) without moving fixed bodies. Every earlier "fell through" was the
  robot spawning intersected with the ground and being depenetrated.
- Trimesh contact works (rigid and multibody), but degrades with triangle count because the multibody
  contact build caps at `MAX_MB_CONTACTS_PER_MB = 64` points per multibody and fine meshes emit a pair
  per touched triangle: 2 tris rest, 128 sag, 3,200 sink, 12,800 fall through (only link 0 keeps contacts).
  Raised to 256 (shader + host slabs derive from it).
- Contact sensor: `substep_solve_no_bias` (which dispatched the sense kernel) has no callers in the fork;
  hooked `gpu_mb_sense_contact_impulses` at the top of `substep_integrate_positions` (after the bias solve,
  `is_last_substep`). Validated nonzero per-foot impulses. `MAX_CONTACT_SENSORS` raised 4 -> 32.
- Memory: `collisions_capacity` 4096 -> 256 per env (11.6 -> 0.84 MiB/env); 4096 envs = 3.5 GB.

## ⑤ parity vs PhysX (Isaac Lab `isaaclab_physx`, in-process Kit) — DONE, with a topology caveat
`test_parity_physx.py`: the SAME gym `humanoid.xml` converted to USD by Isaac Sim 6's `MJCFImporter`
(`import_scene=False`, `fix_base=False`; the extension must be enabled through Kit's extension manager before
it is importable), Isaac Lab `Articulation` with zero-gain implicit actuators (passive), flat ground, dt 1/200,
same initial state as the MuJoCo/Nexus reference (`parity_ref.npz`; knees clamped to -0.036 rad on all
three sides because Isaac Lab refuses out-of-limit default joint positions).

| root z at 0.25 s steps | 0.25 | 0.50 | 0.75 | 1.00 | 1.25 | 1.50 | 1.75 | 2.00 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MuJoCo | 1.288 | 1.177 | 0.690 | 0.355 | 0.079 | 0.081 | 0.081 | 0.080 |
| Nexus  | 1.280 | 1.066 | 0.498 | 0.265 | 0.168 | 0.077 | 0.077 | 0.077 |
| PhysX  | 1.275 | 0.809 | 0.508 | 0.496 | 0.440 | 0.284 | 0.645 | 0.342 |

| pair | root z abs diff mean / max | joint abs diff mean / max (rad) |
|---|---:|---:|
| Nexus - MuJoCo | 0.062 / 0.275 m | 0.111 / 0.982 (17 joints) |
| PhysX - MuJoCo | 0.275 / 0.587 m | 0.323 / 1.527 (5 joints, see caveat) |
| PhysX - Nexus  | 0.241 / 0.590 m | 0.298 / 1.526 (5 joints) |

Caveat (importer, not the solver): the MJCF importer turns each group of stacked hinges (abdomen_z/y,
hip_x/z/y, shoulder1/2) into a 3-axis D6 joint (`abdomen_z:0..2`, `right_hip_x:0..2`, ...), giving PhysX
20 DOF vs 17 and extra free axes. Only the 5 single revolute joints (abdomen_x, knees, elbows) compare 1:1.
The PhysX humanoid is still tumbling at 2 s (z 0.284 -> 0.645 -> 0.342), so its "final z" is not a rest
height. Conclusion that survives the caveat: on the root-height trajectory Nexus tracks the MuJoCo
reference ~4x closer than PhysX-via-importer does, and both Nexus and MuJoCo come to rest at the same
height (0.077 vs 0.080 m).

## Summary of the plan (PLAN.md) — all items executed 2026-09-02
② names/DOF map, ① write path, ③ contacts/raycast/terrain, ④ real SimulationContext + env loop + throughput
(flat 33 k env-steps/s AGILE-shaped; terrain+sensors 101 k env-steps/s plain step loop at 4096), ⑤ parity vs
MuJoCo and PhysX. Not done, by design: USD/cloner, `ManagerBasedRLEnv` end-to-end with AGILE's managers
(needs the G1 MJCF + every mdp term audited against the Nexus data API), and the engine-side fixes listed
under "③ root causes" (per-iteration sensor impulse; 4-point per-pair manifolds).

## Persistence
`/workspace` is not a volume on this instance. Everything is in `/workspace/bench/nexus_port_bundle.tar.gz`
(isaaclab_nexus/, bench/nexus_port/ incl. tests + this spec) and the fork diff `nexus_fork_isaac_backend.patch`
(apply on Haixuantao/nexus at the checked-out commit). The AGILE venv patch is `isaaclab/utils/backend_utils.py`
(one `startswith("nexus")` line; original kept as `backend_utils.py.orig`).
