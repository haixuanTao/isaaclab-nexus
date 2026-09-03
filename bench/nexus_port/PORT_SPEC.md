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

## Where the code is
Engine side: **`Haixuantao/nexus`, branch `isaac-backend`** (8 commits on top of
`27ba35e`). Backend side: `isaaclab_nexus/` in this repo.

## Persistence
`/workspace` is not a volume on this instance. Everything is in `/workspace/bench/nexus_port_bundle.tar.gz`
(isaaclab_nexus/, bench/nexus_port/ incl. tests + this spec) and the fork diff `nexus_fork_isaac_backend.patch`
(apply on Haixuantao/nexus at the checked-out commit). The AGILE venv patch is `isaaclab/utils/backend_utils.py`
(one `startswith("nexus")` line; original kept as `backend_utils.py.orig`).

---

# END TO END: AGILE's G1 task TRAINS on the Nexus backend — 2026-09-02 (final)

`train_nexus.py` = AGILE's `scripts/train.py` without hydra: the real
`HeightTracking-G1-v0` env cfg, the real `rsl_rl` PPO runner, `nexusify()` swapping
physics/spawns, the G1 29-DOF MJCF (`unitree_mujoco/unitree_robots/g1/g1_29dof.xml`)
in place of the USD.

    [nexus] env built in 6.6s
    scene NexusScene(num_envs=N, entities=['robot','contact_forces','height_measurement_sensor','terrain'])
    obs groups ['policy','critic'] | action dim 29
    Learning iteration 0..9   (24 rollout steps x N envs, 5 epochs x 4 minibatches)
    TRAIN ON NEXUS OK

What runs: `ManagerBasedRLEnv` with AGILE's own action/observation/reward/termination/
event/curriculum managers, `DelayedDCMotor` actuators, fallen-state reset dataset,
rough-terrain curriculum, height-scan `RayCaster` and `ContactSensor`, PPO update.
Dropped by `nexusify` (no Nexus equivalent, warned): `randomize_physics_material`,
`randomize_base_com`.

## Spawn cost: MJCF parsed once per batch, not once per env
`insert_mjcf_headless` re-parses the XML and rebuilds every convex hull per environment
(~0.45 s/env for the G1: 512 envs = 230 s, 4096 = 30 min). Added
`NexusState.insert_mjcf_headless_range(path, env_start, env_end, translation, auto_floor)`
— one parse, N inserts:

| envs | scene build, per-env parse | scene build, one parse |
|---:|---:|---:|
| 128 | 59.1 s | — |
| 512 | ~230 s (extrapolated) | **6.1 s** |
| 4096 | ~30 min (extrapolated) | **~8 s** |

## Throughput: the same task, same hardware, PhysX vs Nexus
`train_nexus.py N 10` vs `bench/results/train_HeightTracking-G1-v0_n*.log`; both are
rsl_rl's own timers, median of the steady iterations (Nexus: iters 5..9, PhysX: 5..29).
One iteration = 24 rollout steps x N envs + 5 epochs x 4 minibatches.

| envs | PhysX iter (s) | Nexus iter (s) | PhysX env-steps/s | Nexus env-steps/s | ratio |
|---:|---:|---:|---:|---:|---:|
| 1024 | 2.160 | 5.580 | 11,378 | 4,404 | 0.39x |
| 2048 | 2.810 | 9.120 | 17,492 | 5,389 | 0.31x |
| 4096 | 3.990 | 16.130 | 24,638 | 6,094 | 0.25x |

Rollout share: PhysX 94%, Nexus **98%** (PPO update 0.15-0.32 s, same as PhysX).
At 4096 envs: 12.7 GB GPU memory, 91-100% utilization, **314-334 W** (PhysX: 182 W / 60%).

So the Nexus backend is a working but slower path for this task today: it burns nearly
twice the power at 4x the wall-clock. That is the opposite of the flat-floor comparison
(§④ nsys: 34 k vs 24.6 k env-steps/s for a 21-DOF humanoid on a plane) and the
difference is the terrain: every env carries its own 8 x 8 m tile collider (2,048
triangles after the backend's 0.25 m collider rasterization), the Nexus narrow phase emits
one manifold per touched triangle, and the per-multibody contact loop walks them serially. Iteration time also
grows within a run (11.2 s -> 16.1 s over 10 iterations) as episodes progress and more
links crumple onto the mesh, which is the same signature.

## New engine knob: per-collider-pair contact reduction
`RbdPipeline.contact_reduction` (upstream default `false`) merges every manifold a
collider pair emits into one manifold of the deepest `MAX_MANIFOLD_POINTS` points — the
fix for a trimesh emitting one manifold per touched triangle. It had no Python entry
point; added `NexusPipeline.set_contact_reduction(backend, enabled)` and wired it to
`NexusCfg.contact_reduction` (default on, applied in `NexusManager.finalize`).

## Where the Nexus-backed env step actually goes (`profile_env_step.py 4096 30`)
Wall time per control step (4096 envs, decimation 4, zero actions), sections timed with
`torch.cuda.synchronize()` around them:

| section | ms / control step | share | with contact reduction |
|---|---:|---:|---:|
| physics (`NexusManager.step` x4) | 426.8 | 90.0% | **393.7 (-7.7%)** |
| everything else (obs/reward/termination managers, PPO-side torch) | 33.4 | 7.0% | 33.7 |
| `Articulation.write_data_to_sim` (actuators + scatter) | 9.4 | 2.0% | 9.5 |
| sensors (contact + ray caster) | 4.5 | 0.9% | 4.5 |
| `Articulation.update` | 0.3 | 0.1% | 0.3 |
| **total** | **474.4** (8,634 env-steps/s) | | **441.8** (9,272 env-steps/s) |

**The backend glue is not the cost — the engine step is.** Zero-copy views, the write
path and both sensors together are 3% of the step; 90% is inside
`NexusPipeline.simulate`. Any further gain has to come from the engine
(narrow phase / contact solve on the per-env terrain trimesh), not from `isaaclab_nexus`.

Contact reduction is worth +7.4% throughput here and costs nothing in fidelity
(foot-to-terrain gap p05/p50/p95 = 0.013 / 0.033 / 1.143 m with, 0.015 / 0.034 / 1.145
without — the robots rest ON the terrain in the real env; the deep-penetration case from
the earlier standalone drop test does not appear here). It is on by default in `NexusCfg`.

## Fidelity fix: terrain friction
Terrain colliders were built with rapier's default friction (0.5) while AGILE's
`TerrainImporterCfg.physics_material` asks for 1.0 (the robot's own colliders already get
1.0 from the MJCF geoms, which the rapier MJCF loader applies). `NexusTerrainImporter` now
passes `physics_material.static_friction` down to every tile collider and the fallback floor.

## What is still missing for a like-for-like AGILE run
1. **Terrain curriculum is static.** An env's tile collider is chosen at construction;
   `update_env_origins` records level changes but the geometry does not follow. AGILE's
   curriculum therefore does not bite on this backend.
2. **Dropped domain randomization**: `randomize_rigid_body_material` (friction 0.2-1.5 per env)
   and `randomize_base_com`. Both need per-collider / per-body writes after finalize.
3. **Engine, not backend**: the contact-sensor readout is a per-solver-iteration impulse
   (the backend scales it by `solver_iterations`); a proper fix accumulates in the engine.

## nsys: which kernels the 90% is (4096 envs, AGILE G1 task on Nexus, contact reduction on)
`traces/nexus_agile_g1_n4096.nsys-rep` — 34.8 s window of `profile_env_step.py 4096`,
CUDA tracing only. GPU busy **93.9%** of wall, 167,127 launches, mean kernel 196 us.

| kernel | ms | % GPU busy | calls |
|---|---:|---:|---:|
| `gpu_narrow_phase_pfm_pfm` | 10,398 | **31.8%** | 203 |
| `gpu_mb_integrate_and_dynamics_pre` | 9,866 | **30.2%** | 609 |
| `gpu_mb_finalize_contact_constraints` | 4,488 | 13.7% | 812 |
| `gpu_mb_compute_dynamics_pre` | 3,333 | 10.2% | 204 |
| `gpu_mb_gravity_and_lu` | 963 | 2.9% | 813 |
| `gpu_mb_init_contact_constraints` | 772 | 2.4% | 812 |
| `gpu_mb_solve_constraints` | 767 | 2.3% | 1,624 |
| `gpu_mb_init_joint_constraints` | 659 | 2.0% | 812 |
| `gpu_bf_compute_aabbs` | 589 | 1.8% | 204 |
| `gpu_mb_sense_contact_impulses` (contact sensor) | 81 | 0.2% | 406 |
| `gpu_reduce_contacts` (the new knob) | 68 | 0.2% | 203 |
| all torch elementwise/index kernels | 54 | 0.2% | 23,868 |

Grouped: **contacts ~50%** (narrow phase 31.8 + finalize/init/warmstart 16.4 + AABBs 1.8),
**multibody dynamics ~43%** (integrate_and_dynamics_pre 30.2 + compute_dynamics_pre 10.2 +
gravity_and_lu 2.9), everything else < 3%. One `gpu_narrow_phase_pfm_pfm` call costs
**51 ms** — one dispatch per physics step across all 4096 envs against their terrain tiles.

Two consequences:
1. The Isaac Lab layer is free here — torch is 0.2% of GPU time against 6.2% in the
   PhysX/AGILE trace (which spent 51% of its *launches* on torch elementwise work).
   Nexus's problem is the opposite of PhysX's: few, long kernels rather than many short ones.
2. Closing the 4x gap to PhysX on this task means the trimesh narrow phase and the
   multibody dynamics pre-pass, in that order. Neither is reachable from `isaaclab_nexus`.

## Final configuration, measured (4096 envs, contact reduction + terrain friction 1.0)
`train_nexus_final_HeightTracking-G1-v0_n4096.log`: 15.48 s/iter median (iters 5-9),
**6,350 env-steps/s**, PPO update 0.32 s. Against the same run without either fix
(16.13 s, 6,094 env-steps/s) that is +4.2%; against PhysX's 3.99 s / 24,638 it is 0.26x.

## CUDA graphs: wired up, measured, and worth nothing here
The backend was calling `simulate_headless` every step — no graph capture, no replay. The
engine has `capture_rbd_graph`, but its Python binding took a `NexusViewer`, so the headless
path had no way in. Added `NexusPipeline.capture_cuda_graph_headless(backend, state)` and
`NexusCfg.cuda_graph_warmup` (capture after N settled steps, then replay; 0 = off, default).

**Capture fails as-is**: `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`. `KHAL_CUDA_ALLOC_TRACE=1`
names it — `RadixSort::dispatch` → `TensorBuilder::build_init::<u32>`, a 4-byte allocation
*inside* the step, because the sort's uniform cache key holds `total_n` (the contact count),
which changes every step. That sort is the deterministic contact-order pass, so today
**bit-exact reruns and CUDA graphs are mutually exclusive** in the engine. The fix is to make
`n_sort_flat` a persistent buffer written per step instead of re-created.

**With `NEXUS_DETERMINISTIC=0` capture succeeds — and buys nothing** (4096 envs, 30 control
steps, median):

| configuration | ms / control step | env-steps/s |
|---|---:|---:|
| deterministic sort on, no graph (shipped default) | 596.0 | 6,872 |
| deterministic sort off, no graph | 587.8 | 6,969 |
| deterministic sort off, **graph replayed** | 589.9 | 6,944 |

Determinism costs 1.4%; the graph is inside the noise. That is what 93.9% GPU busy already
said: this workload is kernel-bound, not launch-bound, so removing host encoding removes
nothing from the critical path. The other CUDA-specific optimization *is* already on —
`KHAL_TRACE_INDIRECT=1` prints nothing, so fixed-grid dispatch is active and no dispatch is
paying the indirect-dispatch stream drain.

Capture failure is non-fatal in the backend (warn, drop back to the encoded path).

## The real regression: `solver_iterations` is a SUBSTEP count
`NexusCfg.solver_iterations` was left at the engine's default of 4. It is not a PGS
iteration count: `GpuMultibodySet::set_visible_dt` divides `dt` by it and runs that many
full integrate + dynamics passes per step. So the G1 was being integrated at **800 Hz**
while Isaac Lab and PhysX were asked for 200 Hz — 4x the dynamics work, for an integration
rate nobody requested. The nsys call counts said so plainly: 4 `gpu_mb_gravity_and_lu` and
8 `gpu_mb_solve_constraints` per `gpu_narrow_phase_pfm_pfm`.

Plain step loop, 4096 envs, 30 control steps:

| substeps | physics ms/ctrl step | total ms | env-steps/s | foot-terrain gap p05/p50/p95 |
|---:|---:|---:|---:|---|
| 4 | 454.0 | 537.0 | 7,628 | 0.013 / 0.032 / 1.143 |
| 2 | 260.1 | 341.8 | 11,984 | 0.016 / 0.033 / 1.142 |
| **1** | **166.0** | **249.5** | **16,415** | 0.019 / 0.031 / 1.142 |

2.15x, with the robots resting on the terrain exactly as before. In the full training loop
(`train_nexus.py 4096 10`): **15.48 -> 11.32 s/iteration, 6,350 -> 8,684 env-steps/s**, and
the reward curve is unchanged (-46/-100/-154/-212/-265/-315 vs -46/-102/-157/-216/-269/-322),
i.e. this removes work, not fidelity. 1 is also the value at which the engine's contact
sensor reports 1.019x body weight without the backend's per-iteration rescaling.
`NexusCfg.solver_iterations` now defaults to 1.

Note what the two numbers above imply: at 1 substep the plain loop runs at 16,415 env-steps/s
but training only reaches 8,684, so **more than half of the training step is no longer
physics**. That is the next thing to profile (actuator model, resets, managers under real
actions — the step-breakdown above was measured with zero actions and is no longer
representative).

## The second setup bug: 1,087-vertex collision hulls
The unitree G1 MJCF has no simplified collision geoms — its 25 colliding geoms are the full
visual STLs, and `MeshConverter::ConvexHull` keeps every hull vertex. Measured hulls:
mean **1,087 vertices**, pelvis **5,583 vertices / 11,162 faces**. PhysX's
`convexHullVertexLimit` defaults to **64**, so the AGILE/PhysX G1 collides hulls an order of
magnitude simpler than the ones we handed the GPU narrow phase — which then clips those
features against every terrain triangle a fallen robot touches.

`make_convex_mjcf.py N` rewrites the MJCF's colliding mesh assets as support-mapped hulls of
at most N vertices (support point per direction on a Fibonacci sphere, then hull): total hull
vertices 27,179 -> 1,481 (18.4x), max per hull 5,583 -> 64.

## Cumulative: 2x in training, and where the two fixes landed
4096 envs, `HeightTracking-G1-v0`, same seed. Plain step loop (30 control steps, zero actions)
and the full rsl_rl training loop (10 iterations, median of 5..9):

| configuration | physics ms/step | loop env-steps/s | train s/iter | train env-steps/s |
|---|---:|---:|---:|---:|
| 4 substeps, full hulls | 454.0 | 7,628 | 15.48 | 6,350 |
| 1 substep, full hulls | 166.0 | 16,415 | 11.32 | 8,684 |
| **1 substep, 64-vertex hulls** | **107.0** | **21,746** | **7.76** | **12,668** |
| PhysX / AGILE baseline | — | — | 3.99 | 24,638 |

Reward curves are identical across all three (-46/-101/-154/-212/-264/-315), and the
foot-to-terrain gap is unchanged (p05/p50 0.016/0.030 m), so neither fix trades physics for
speed — both remove work the task never asked for. Remaining gap to PhysX: **1.95x**, down
from 3.9x.

## Training-loop breakdown (4096 envs, 1 substep, full hulls)
Real PPO loop, every manager term timed (`profile_train_step.py`):

| section | ms/ctrl step | % |
|---|---:|---:|
| physics | 340.5 | 69.2% |
| `_reset_idx` total (18.5 of it `Articulation.reset`, rest Isaac's managers) | 46.2 | 9.4% |
| write + actuators | 28.5 | 5.8% |
| reward / obs / sensors / events / actions | 55.9 | 11.5% |
| PPO + policy | 28.2 | 5.7% |

Physics costs 340 ms here against 166 ms in the settled zero-action loop, because AGILE's
stand-up task **spawns robots fallen** — lying on the mesh is the steady state, not an edge
case. Note also that no GPU heightfield exists (`TypedShape::HeightField => todo!()` in
`src_rbd/shapes/shape.rs`), so terrain must be a trimesh; a heightfield collider would be the
structural fix for the narrow phase.

## Is it the FFI? No — measured three ways
1. **CUDA graph replay removes all host encoding for the physics step, and changes nothing.**
   Before the substep/hull fixes: 356.9 ms/control step with the graph active vs 355.9 without.
   After them, at the corrected config: identical again. If per-step FFI or dispatch encoding
   were on the critical path, replaying a recorded graph would have shown it.
2. **Compilation is fully optimized.** Cubins: `opt -passes=default<O3>` -> `llc -mcpu=sm_120 -O3
   -fp-contract=fast` -> `ptxas -arch=sm_120 -O3`, built `--release` with
   `unsafe_remove_boundchecks`, and **embedded at build time** by `khal-builder`'s build script
   (`cargo:rerun-if-env-changed=CUDA_OXIDE_SHADERS_PTX_*`) — so processes that never set those
   env vars still run the optimized cubins. Host extension: `maturin develop --release`.
   The one gap is that the fork declares no `[profile.release]` (stock `codegen-units=16`, no
   LTO); that is a host-side knob, and the host is not the bottleneck.
3. **After the fixes the trace is no longer GPU-saturated**: GPU busy fell 93.9% -> **35.2%**,
   but the idle is dominated by two multi-second gaps during scene build/settle, and the
   remaining per-step idle sits in Isaac Lab's own manager layer between physics steps — not
   inside `NexusPipeline.simulate`, which is what a graph covers.

## Why Zealot looks so much faster: it is a different workload
From Zealot's own `docs/benchmarks.md` (same RTX 5090):

| N envs | Zealot (native CUDA + cuTile) | Isaac / PhysX 5 |
|---:|---:|---:|
| 4,096 | 91.4 k | **126 k** |
| 8,192 | 99.5 k | **201 k** |

That benchmark is the **LeRobot bipedal: `NUM_JOINTS = 12`** (18 DOF with the free root) on the
**flat** `VelocityFlatTask`. This port runs the **G1: 29 joints / 35 DOF on rough terrain**, in
AGILE's stand-up task, which *spawns robots fallen* — the contact-heavy regime.

Two things follow. First, on Zealot's own numbers Nexus runs at **0.5-0.7x PhysX**, the same
ratio measured here on the G1 task (12,668 vs 24,638 = 0.51x) — Nexus is not behaving
differently in Isaac Lab than in Zealot. Second, the remaining cost is exactly what that DOF
difference predicts: after the fixes the top kernel is `gpu_mb_compute_dynamics_pre` at
**60.3% of GPU time, 18.7 ms per physics step** — the articulated-body dynamics pre-pass, not
contacts. It scales with DOF^2..DOF^3, and 35 DOF vs 18 is 3.8-7.3x per multibody. Nothing in
`isaaclab_nexus` reaches it.

## The third engine default: implicit Coriolis — and the number that flips the comparison
After the substep and hull fixes the trace put **60.3% of GPU time in `gpu_mb_compute_dynamics_pre`**
(18.7 ms per physics step). That kernel's cost is a *mode*: the engine defaults to
`implicit_coriolis = true`, rebuilding the mass matrix WITH a dt·C Coriolis term (extra
`gemm_inertia_lhs_par` / `gemm_skew_*` / `gemm_tr_par` chains per link, and under multiple
substeps a full M/LU rebuild per substep). Zealot's own `biped_env_nexus.rs` switches it OFF and
says why: it over/under-damps with substep count (the sim-to-real foot-slip bug), MuJoCo's
`implicitfast`, Genesis, PhysX and Bullet all linearize Coriolis once per step, and with it on
"compute_dynamics_pre + gravity_and_lu were 51% of ALL GPU time". The setter already existed in
the Python binding; this backend never called it. `NexusCfg.implicit_coriolis` now defaults to
`False`.

Plain step loop, 4096 envs, 64-vertex hulls, 1 substep (same process conditions):

| Coriolis | physics ms/ctrl step | env-steps/s | foot-terrain gap p05/p50 |
|---|---:|---:|---|
| implicit (engine default) | 94.1 | 28,970 | 0.016 / 0.030 |
| **explicit** | **40.6** | **46,613** | 0.016 / 0.030 |

## Final: the same task, 4096 envs, rsl_rl's own timers (median of iterations 5..9)

| configuration | iter | collect | learn | env-steps/s | rewards, iters 0-5 |
|---|---:|---:|---:|---:|---|
| this morning: 4 substeps, full hulls, implicit Coriolis | 15.48 s | 15.16 | 0.32 | 6,350 | -46 -102 -157 -216 -269 -322 |
| 1 substep | 11.32 s | 10.80 | 0.47 | 8,684 | -46 -100 -154 -212 -265 -315 |
| + 64-vertex hulls | 7.76 s | 7.18 | 0.59 | 12,668 | -46 -101 -154 -212 -264 -315 |
| **+ explicit Coriolis (shipped default)** | **2.45 s** | **2.14** | 0.32 | **40,124** | -46 -101 -154 -212 -262 -314 |
| PhysX / AGILE baseline | 3.99 s | 3.75 | 0.23 | 24,638 | |

**6.3x since this morning; 1.63x faster than PhysX on AGILE's own G1 task**, with the reward
curve unchanged across every configuration. None of the three changes touched the port's own
code path in a way that alters physics semantics the task asked for — each was an engine default
(800 Hz integration nobody requested, unlimited hull vertices, per-substep implicit Coriolis) that
Zealot, MuJoCo and PhysX all set differently. The earlier "Nexus is 0.5-0.7x PhysX" reading of
Zealot's benchmark table stands for *that* configuration of the engine; this one is not it.

What CUDA graphs did NOT do still holds (measured at every stage: identical with and without),
and `deterministic_contacts` still costs only 1.4% and stays on.

## Final scaling table (shipped defaults: 1 substep, explicit Coriolis, 64-vertex hulls, contact reduction)
Same task, same box, rsl_rl's own timers, median of steady iterations (Nexus 5..9, PhysX 5..29):

| envs | PhysX iter | Nexus iter | PhysX env-steps/s | Nexus env-steps/s | Nexus / PhysX |
|---:|---:|---:|---:|---:|---:|
| 1024 | 2.160 s | **1.360 s** | 11,378 | **18,071** | **1.59x** |
| 2048 | 2.810 s | **1.750 s** | 17,492 | **28,087** | **1.61x** |
| 4096 | 3.990 s | **2.450 s** | 24,638 | **40,124** | **1.63x** |

Nexus now scales better with batch size than PhysX does on this task (the PhysX row flattens
toward its 8192-env broadphase wall documented in `bench/results/FINDINGS.md`).

Fidelity check of the shipped mode (`check_contact_sensor_env.py`, 512 envs, robots at rest on
the terrain after 3 s): summed contact-sensor normal force / body weight = **0.996 median**
(p10 0.988, p90 1.127) — the per-step impulse readout is correctly scaled under explicit Coriolis.

## Does the step get cheaper as the policy learns? Measured over 100 iterations — no, it is flat
`train_nexus.py 4096 100`, shipped config, per-10-iteration means:

| iters | iter time | env-steps/s | reward | episode length |
|---|---:|---:|---:|---:|
| 0-9 | 2.69 s | 36,504 | -286 | 124 |
| 10-19 | 3.11 s | 31,568 | -778 | 363 |
| 20-29 | 2.44 s | 40,371 | -1207 | 603 |
| 30-39 | 2.43 s | 40,421 | -1414 | 749 |
| 50-59 | 2.42 s | 40,588 | -1180 | 750 |
| 70-79 | 2.41 s | 40,739 | -951 | 750 |
| 90-99 | 2.57 s | 38,206 | -664 | 750 |

Like-for-like with PhysX's own 5..29 median: **2.45 s vs 3.99 s, 1.63x** (unchanged). Steady
state (iters 50-99): **2.42 s, 40,621 env-steps/s**. The policy is learning (reward bottoms at
-1414 and climbs to -664; episodes reach the full 750 steps) but the step cost does not follow
it. That is the expected shape *after* the fixes: this morning the iteration time GREW within a
run (11.2 -> 16.1 s) because the contact path dominated and crumpled robots emit more contacts;
now the narrow phase is 8% of GPU time and the remaining cost (articulated-body dynamics, even
explicit) is per-link and state-independent. A cheaper step from "the robots stand up" would
need the contact path to matter again, and it no longer does.

## Correction: the like-for-like Zealot comparison is the G1-on-terrain row, not the 12-DOF headline
Zealot's `docs/benchmarks.md` (lines 275-279) has exactly this comparison — full-body G1 with
AGILE-matched actuator delay/history and a port of AGILE's terrain curriculum — and it is the
one to quote, not the 99.5 k biped number I quoted earlier:

| N | Zealot full-body +realism +terrain | WBC-AGILE (terrain) | this port (Isaac Lab on Nexus) |
|---:|---:|---:|---:|
| 2048 | 25.4 k | 20.6 k | **28.1 k** |
| 4096 | 30.7 k | 32.3 k | **40.1 k** |

Caveat that makes this a fair reading rather than a win: Zealot's G1 rows run `solver-iters 8`
(its choice to mirror PhysX's TGS budget — in Nexus those are substeps, 8x the integration
work); this port runs 1 at the same 200 Hz `dt`, with the reward curve unchanged against the
engine's default of 4. Per substep the two are roughly the same engine efficiency. The task also
differs (AGILE's `Velocity-G1-History-v0` there, `HeightTracking-G1-v0` here).

## Video
`record_nexus_policy.py <ckpt>` rolls a checkpoint on the Nexus backend and stores per-step root
pose + joint angles for 4 envs plus env 0's terrain tile; `render_nexus_video.py` renders that
with MuJoCo's offscreen renderer (OSMesa; EGL is not usable headless on this box) using the G1's
real visual meshes on the actual tile. Nexus's own viewer path (`insert_mjcf` + `NexusViewer`)
would render natively, but this backend builds its state through `insert_mjcf_headless`, which
registers no render data — wiring that is the cleaner long-term route.

## Long run
`train_nexus.py 4096 10000` started 2026-09-02 17:30 UTC, shipped config, detached (`setsid`);
log `bench/results/train_nexus_10k_HeightTracking-G1-v0_n4096.log`, checkpoints every 250
iterations under `bench/nexus_port/logs/<timestamp>_nexus/`. Expected ~7 h at 2.45 s/iter.
Note `/workspace` is not a volume on this instance: copy checkpoints off-box.

## The long run slowed 1.9x at iteration 73 — what it was not, and the restart
`train_nexus.py 4096 10000` ran flat at 2.4-2.5 s/iter, then at iteration 73 the PPO update
spiked (0.32 -> 1.33 s), iteration 74's collection took 12.5 s, and every iteration after sat
at 4.4-4.7 s with GPU memory up 12.2 -> 16.3 GB — a one-time step, not a trend, and no task
metric moved (episode length, curricula, rewards all continuous).

First suspect was the engine's collision-buffer ratchet (`auto_resize_buffers`, default policy
`Grow`: any single env's pair spike reallocates nine buffers to 1.5x and never shrinks, and
with fixed-grid dispatch every capacity-gridded kernel then launches over the larger capacity).
To test it I exposed the knobs — `NexusState.set_rbd_resize_policy("grow"|"fit"|"fixed")`,
`NexusState.rbd_resize_stats()`, `NexusCfg.collisions_resize_policy` — and rolled the run's
`model_250` checkpoint with them (`probe_resize_ratchet.py`). **Falsified**: `pairs_len` stays
0, capacity stays 256, `rb_contacts_inert == True` — in a robot-only scene the rigid-body pair
readback never runs, so that ratchet cannot fire here. The knobs stay (they are the right
interface, and Zealot pre-sizes and fixes for the same reason), and the backend now warns if
capacity or `max_colors` ever changes mid-run.

The dataset is pre-collected and fixed, GPU memory did not keep growing, and the timing of the
onset coincides with the video render + ffmpeg encode I launched in the same command as the run
(CPU-saturating for ~3 min). Learn-time-first, then one huge collection, then a permanent
plateau with +4 GB reserved is the fingerprint of the CUDA caching allocator fragmenting after a
transient. Cause not proven — the decisive test is the restart: `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True`, no concurrent jobs, `NEXUS_STATS_EVERY=250` logging torch allocator
stats (`num_alloc_retries`, reserved) and engine resize stats every 250 iterations, and a watch
that flags any iteration over 3.5 s. Started 2026-09-02 18:20 UTC as
`train_nexus_10k_v2_HeightTracking-G1-v0_n4096.log`; the first run's `model_250.pt` is kept.

**Resolved (18:25 UTC): the 1.9x was GPU sharing.** `nvidia-smi --query-compute-apps` shows a
second 4096-env training of the same task on this GPU — `scripts/train.py --task
HeightTracking-G1-v0 --num_envs 4096 --headless --max_iterations 10000`, launched from
`/workspace/WBC-AGILE-NEWTON` (a parallel session), 6 GB. Its fallen-state dataset cache was
written at 17:34 — the exact minute the Nexus run's iteration 73 slowed — and it was restarted at
~18:16, four minutes before the v2 Nexus run began, which is why v2 was "slow from iteration 1".
Neither the buffer ratchet nor the allocator was involved; the earlier "+4 GB" was that process.
Every throughput number in this document was measured with the GPU otherwise idle.

## Long run, iteration 3000 (shared GPU, 4.70 s/iter): the policy is learning to get up
Per-term episode rewards from the training log — the height terms tripled and the tilt /
illegal-contact penalties vanished, i.e. the base is rising and staying upright:

| term | it 100 | it 1500 | it 3030 |
|---|---:|---:|---:|
| base_height_fine | 0.43 | 2.49 | **3.43** |
| base_height_medium | 0.60 | 1.66 | 2.21 |
| severely_tilted | −0.22 | −0.005 | −0.001 |
| forward_pitch | −1.60 | −0.04 | −0.14 |
| illegal_contacts | −1.49 | −1.52 | −0.26 |
| flat_feet | −3.68 | −0.80 | −0.93 |
| joint_tracking_error | −0.93 | −0.33 | −0.27 |

Mean reward −511 (it 100) → −139 (it 2000) → −135 (it 3000); `terrain_levels` stays 0 on this
backend (curriculum is static, stated limitation). Allocator and engine stats are flat every
250 iterations: `torch reserved 2.81 GiB, retries 0 | engine cap/batch 256, max_colors 8`.

## BUG (fixed 22:15 UTC): fallen-state resets dropped 78% of robots through the terrain
Found while checking whether the 3000-iteration policy stands. With training-style resets, the
recorded base height across 64 envs went 0.48 m at t=0 to **-0.58 m mean from 0.5 s on** (78%
of envs below zero, resting at -0.80 m on the backstop floor half a metre under the tile).
Cause: AGILE's `reset_from_fallen_dataset` places the robot at
`root_pos_rel + terrain.terrain_origins[level, type]` — Isaac's **global** tile origins, tens of
metres apart — while this backend keeps every env in tile-local coordinates with its collider
centred at XY = 0. Dataset resets therefore landed outside the 8 x 8 m collider and fell
through. Fix: `NexusTerrainImporter.terrain_origins` now has XY = 0 and keeps the generator's Z
(the tile meshes keep world Z), which covers every AGILE reader of it (dataset collection, the
reset, the out-of-bounds termination all work origin-relative). Verified: 0% below ground at
t = 0 / 0.5 / 2 / 8 s, min 0.06 m, median 0.17 m.

Consequences, stated plainly:
- The zero-action benchmarks (`profile_env_step.py`, the substep / hull / Coriolis A/Bs, the
  contact-sensor and foot-gap checks) never load the dataset and were **not affected**.
- Every **training-loop** number (the 2.45 s/iter, 40,124 env-steps/s headline, the 1024/2048
  points, the 100-iteration trend) ran with most robots colliding with a flat cuboid instead of
  the trimesh. Those must be re-measured with the fix; they are likely worse. Not restated until
  measured on an idle GPU (currently shared with a second 4096-env run from another session).
- The 10k checkpoints from `logs/2026-09-02_17-53-58_nexus` (3,250 iterations) were learned
  under the bug and are not meaningful. Restarted as `train_nexus_10k_v3_...log`.
- The rising `base_height_*` rewards at 3000 iterations were consistent with robots on a flat
  floor; the recorded `model_3000` policy does not stand (0% of envs above 0.5 m at 8 s).

**Cost of the fix, like-for-like (same GPU sharing, same partner process):** buggy v2 ran at
4.67 s/iter; the corrected v3 runs at **4.91 s/iter** (iters 5-24; collection 4.44 s) — **+5%**.
Robots now collide with the trimesh instead of the flat backstop, and it barely shows, which is
consistent with the earlier finding that after the hull/Coriolis fixes the step is dominated by
state-independent articulated dynamics, not contacts. Extrapolated to an idle GPU that is
~2.57 s/iter (~38 k env-steps/s) against PhysX's 3.99 s — but that stays an extrapolation until
measured with the GPU idle.
**Confirmation from training itself (v3, iteration 500):** `Episode_Reward/completely_airborne`
went from -1.53 under the bug to **-0.36** with the fix — the "robots in free fall / on the
backstop" signal is gone — and mean reward at 500 is -205 vs the buggy run's -175 at the same
point: the corrected task is harder, as a rough-terrain stand-up should be.

## Corrected training throughput (reset fix in, GPU idle) — the number that replaces 2.45 s
The other session's baseline finished at ~iteration 2,300 of v3, leaving the GPU to the Nexus run
alone. v3's steady iterations with correct resets, robots on the trimesh, mid-training state
distribution (iterations ~2,470-2,535, 1-iteration granularity flat at 2.87-2.90 s):

| | iter | collect | learn | env-steps/s | vs PhysX (3.99 s / 24,638) |
|---|---:|---:|---:|---:|---:|
| Nexus, corrected, mid-run | **2.89 s** | 2.57 | 0.32 | **34,015** | **1.38x** |

The withdrawn 2.45 s was measured with 78% of robots on a flat backstop; the honest figure is
2.89 s. The extrapolation from the like-for-like +5% (2.57 s) was optimistic — the shared-GPU
comparison was also at a different point in training. A fresh iteration-5..29 measurement on the
idle GPU (PhysX's own window) follows below.

## FINAL corrected like-for-like: idle GPU, iterations 5..29 (PhysX's own window)
v3 paused with SIGSTOP on the CUDA process (verified state `T`, GPU 0%), fresh
`train_nexus.py 4096 30`, reset fix in, 64-vertex hulls, 1 substep, explicit Coriolis:

| | iter | collect | learn | env-steps/s | vs PhysX |
|---|---:|---:|---:|---:|---:|
| **Nexus, corrected** | **2.710 s** | 2.394 | 0.317 | **36,275** | **1.47x** |
| PhysX / AGILE | 3.990 s | 3.754 | 0.234 | 24,638 | |

Rewards over the first six iterations with correct resets: -39, -87, -138, -190, -240, -293
(the buggy runs read -46, -101, -154, -212, -264, -315 — a different curve, as it should be).
This replaces the withdrawn 2.45 s / 40,124 / 1.63x. (A first attempt at this measurement read
6.30 s: `pgrep` had returned the setsid wrapper, not the CUDA process, so v3 was never paused and
the two shared the GPU — the learn time doubling to 0.63 s gave it away. Discarded.)

## FINAL corrected scaling table (idle GPU, iterations 5..29, resets fixed)

| envs | PhysX iter | Nexus iter | PhysX env-steps/s | Nexus env-steps/s | Nexus / PhysX |
|---:|---:|---:|---:|---:|---:|
| 1024 | 2.160 s | **1.410 s** | 11,378 | **17,430** | **1.53x** |
| 2048 | 2.810 s | **1.850 s** | 17,492 | **26,569** | **1.52x** |
| 4096 | 3.990 s | **2.710 s** | 24,638 | **36,275** | **1.47x** |

This supersedes every earlier training-loop table in this document (all taken under the reset
bug). The zero-action loop numbers, the substep / hull / Coriolis A/Bs and the fidelity checks
were never affected and stand as written.

## v3 diverged at iteration ~4044 — diagnosed, physics exonerated, resumed from model_4000
Mean reward -166 (it 4000) -> -3,900 (it 5000). Onset is a **critic** event: `Mean value loss`
spikes briefly at 3956 (12) and 3972 (67) with reward unchanged, then 4036 (80) -> 4044 (**3,740**)
and never recovers; the policy decays afterwards while action std and entropy stay flat. Every
physics-driven penalty (`joint_vel_limits` 0, `torque_limits` -0.01, `joint_tracking_error` -0.5,
`completely_airborne` -0.45) is flat through all three spikes and only rises after 4044 — the
environment did not blow up first. No NaN, no invalid-state terminations, engine buffers untouched.

Ruled out:
- **the fast physics config** — `model_4000` rolled 64 envs x 750 steps at 1 vs 4 substeps: max
  |joint_vel| 19.5 vs 17.6 rad/s, max root speed 3.68 vs 3.71 m/s, z range 0.02-0.92 vs
  0.06-0.92, identical height trajectories. No outliers at either setting.
- **the chunked stats logging** — the earlier chunked run showed no such spikes in 3,250
  iterations, and this rsl_rl's `learn()` re-initializes nothing between calls.

What remains is an RL-side instability of AGILE's PPO setup at this point (adaptive LR, reward
normalization) — engine-independent as far as these measurements can tell. Action: stopped v3,
resumed from `model_4000.pt` (last good) as **v4**, unchunked, same config, 6,000 iterations, with
the watch flagging any value loss over 100 so a recurrence is caught at onset.

**v4 reproduced it.** Resumed from `model_4000` with a fresh random stream, unchunked: value loss
crosses 100 at iteration **4231** and the reward slides (-200 -> -315 within 15 iterations). v3
diverged at 4044 from the same weights. Systematic, not stochastic. Since a 64-env x 15 s rollout
cannot see a one-in-ten-thousand-episode outlier while training sees 24M env-steps per 250
iterations, the rare-physics-outlier hypothesis is back on the table; `probe_outliers.py` rolls
`model_4000` on 4096 envs with training-style resets and logs, per step, the minimum reward over
envs, the physics extremes, and — for the worst (env, step) pairs — which reward terms produced
them. v4 stopped at 4245.

## Divergence root cause candidate: contact forces into the critic (and a sensor scaling bug)
Training-scale probe (`probe_contact_extremes.py`, `model_4000`, 4096 envs x 300 steps):

| substeps | peak contact force | median per-step max | p99.9 | critic input max (scale 5e-3, clip ±25 kN) |
|---:|---:|---:|---:|---:|
| 1 | 17,049 N (49x weight) | 11,507 N | 3,741 N | 85.2 |
| 4 | 91,110 N (263x weight) | 71,278 N | 17,897 N | **125.0 — pinned at the clip** |

Rewards and physics extremes were bounded in the same rollouts; the only consumer of raw force
magnitude in AGILE's task is the **critic** observation `contact_force_norm` over all 30 bodies —
so a critic-only blow-up with flat reward penalties is exactly what these numbers predict.

**Bug found:** the backend scaled the sensed impulse by `solver_iterations` unconditionally. That
is the implicit-Coriolis rule (constraints rebuilt per substep, readout = last substep's impulse).
In explicit mode the impulse accumulates over the whole step and must not be scaled (Zealot's
`sensor_inv_dt` makes the same distinction). Fixed: scale only when `implicit_coriolis`. This is
why 4 substeps read ~4x higher than 1.

Open: whether PhysX reports comparable per-step impulsive forces under the same policy (the
task's ±25 kN clip suggests spikes are expected there too). Control probe on the stock PhysX env
with the same checkpoint follows.

**PhysX control (stock env, same `model_4000` policy, 2048 envs x 200 steps):** peak contact force
**95,460 N**, median per-step max 31,130 N, p99.9 12,479 N — critic input pinned at the ±25 kN clip
every step. Isaac Lab's PhysX sensor is the same quantity (`get_net_contact_forces(dt=physics_dt)`,
impulse over one physics step). So tens-of-kN impulsive forces are AGILE-normal, and this
backend's force inputs to the critic are *milder* than PhysX's (1 substep: 17 kN peak). The
scaling fix is verified: 4 substeps now reads 24,704 N peak instead of 91,110 N. **Contact forces
are not the Nexus-specific cause of the divergence.**

Status of the divergence: not physics extremes, not rewards, not forces (milder than PhysX), not
chunked logging. v4's reproduction was not an independent trial — it resumed with the same env
seed (42), replaying a correlated reset sequence. Next: resume from `model_4000` with a different
seed (v5, `NEXUS_SEED=7`), value-loss alarm armed. Recurrence near iteration 4000-4300 would make
it systematic to this config; a clean continuation would make it a stochastic PPO blow-up.

**v5 (seed 7) diverged 16 iterations after the resume** — three for three, systematic. Ruled out
since: the adaptive learning rate (1.7e-4 - 3.8e-4 at every checkpoint, inside the schedule's
bounds) and AGILE's polish curriculum (its weights never changed; it is gated on
`terrain_levels >= 4`, which this backend's static terrain curriculum never reaches — a semantic
gap in its own right: **AGILE's polish phase never starts on this backend**).

Remaining candidate, consistent with everything measured: the critic's `contact_force_norm`
input. On PhysX the ±25 kN clip is hit routinely (typical per-step max 31 kN), so the critic is
trained on 125s; on Nexus the typical max is 11 kN, so a clipped 125 is a rare outlier the critic
has never fit — and a critic meeting rare 125s after 4,000 iterations of ~60s extrapolates. The
milder engine is the *worse* one here. Diagnostic run v6: resume `model_4000`, seed 7, critic
contact-force clip tightened to ±5 kN (`NEXUS_DIAG_FORCE_CLIP`, a diagnostic, not the shipped
config). Trains on => mechanism confirmed; diverges => it is something else.

## DIVERGENCE ROOT CAUSE — confirmed: the critic's raw contact-force observation
v6 (resume `model_4000`, seed 7 — identical to v5 except the critic's `contact_forces` clip
tightened ±25 kN -> ±5 kN) trained straight through the region where every other continuation
died: iteration 4100 value loss 0.039, 4200 0.044, **4300 0.039**, reward -153 / -163 / -156.
v5 (same seed, ±25 kN) was at value loss 1,900 by 4020; v4 diverged at +231; v3 (fresh) at 4044.

Mechanism: AGILE's critic observes `contact_force_norm` over all 30 bodies, scale 5e-3, clip
±25 kN. On PhysX the per-step impulsive force routinely exceeds the clip (measured under the same
policy: typical per-step max 31 kN, peak 95 kN), so the critic is trained on the clipped 125 as a
common input. On Nexus the same quantity is milder (typical max 11 kN, peak 17 kN at 1 substep),
so the clipped 125 is a rare outlier arriving after thousands of iterations of ~60s — the critic
extrapolates, the value loss explodes, the policy follows. Rewards, physics and policy inputs
were bounded throughout; nothing in the engine broke. A task tuned to the other engine's spikes.

Shipped: `nexusify(..., critic_force_clip_n=5000.0)` — a critic-only observation clip; rewards,
policy observations and physics are unchanged. This is a documented deviation from AGILE's cfg
on this backend. Every throughput number in this document was measured with the unmodified cfg
(the clip does not affect step cost).

Along the way: the contact-sensor substep scaling bug (fixed, `ddb889f`), and the fact that
AGILE's polish curriculum is gated on `terrain_levels >= 4`, which this backend's static terrain
curriculum never reaches — AGILE's polish phase does not run here (stated limitation, now with
a concrete consequence).

**Correction: the ±5 kN clip delays the collapse, it does not remove it.** v6 held to 4400
(value loss 0.04-1.5, reward -136..-183) and then went at **4451**: value loss 10-20, reward
-310..-493 — the same shape, 451 iterations after the resume instead of 16-231. The critic's
contact-force input clearly moves the cliff, so it is *a* driver, but either ±5 kN (inputs up to
25 against O(1)) is still too large or it is not the only one. Diagnostic v8: same resume, same
seed, the critic's `contact_forces` term **removed** (`NEXUS_DIAG_NO_FORCE_OBS=1`). Trains through
=> the term is the whole cause and the shipping fix is a matter of scale; diverges => another
critic input (candidates: `base_height_from_sensor` from this backend's ray caster, `base_lin_vel`).
The `critic_force_clip_n=5000` default stays until v8 decides its final form.
