> STATUS 2026-09-02: all five items executed; results and caveats in PORT_SPEC.md (sections ①-⑤).

# Nexus Isaac Lab backend — execution plan (2026-09-02)

Baseline (done): factory dispatch, NexusCfg/NexusManager, 5 read properties zero-copy,
CUDA, 8-env MJCF humanoid stepping. Everything below is gated on a test that must pass
before the next step starts. Order is by dependency.

## ② Name + DOF maps  (unblocks ① and ④)
- Rust: read `links_static` once at finalize -> per link `(rb_id, parent, multibody_id,
  assembly_id, ndofs, kinematic)`. `assembly_id` is the flat DOF column, `ndofs` the width.
- Rust: MJCF body/joint/actuator names from `rapier3d_mjcf::MjcfRobotHandles`.
- Python: `joint_pos` becomes flat `(num_envs, num_dofs)` by gathering
  `coords[link, 0:ndofs]` into columns `assembly_id..assembly_id+ndofs` (one index tensor,
  built once; the gather is a view-free `index_select`). `joint_vel` already flat.
- Python: `body_names`, `joint_names`, `find_joints`/`find_bodies` with Isaac regex semantics.
- Python: wrap data props in Isaac's `ProxyArray` so `data.joint_pos.torch[...]` works.
- Test: names round-trip; flat joint_pos matches per-link coords; regex lookup.

## ① Write path
- Motors (implicit actuator): `set_joint_position_target_index` / `..velocity..` ->
  batched writes of `target_pos`/`target_vel` (+ stiffness/damping from cfg) into
  `links_static` motor params. Zero-copy if the struct layout is exposed; else one
  H2D per step (bounded, documented).
- Effort (explicit actuator, AGILE's `DelayedDCMotor`): needs a generalized-force input.
  If the fork has the gen-force tensor the docs reference, expose it as a CudaArray and
  `set_joint_effort_target_index` becomes a zero-copy torch write. If not: implement in
  Rust (add `gen_forces` accumulate into the solver RHS) — flagged as the one real engine change.
- Resets: `reset(env_ids)` -> `reset_envs_from_templates`; `write_root_pose_to_sim` /
  `write_root_velocity_to_sim` -> `reset_env_from_snapshot_offset` semantics
  (template + pose/vel offset), batched over env_ids.
- `write_data_to_sim`: flush pending motor targets / efforts once per env step.
- Test: command a joint target, step, joint tracks it; effort in known direction accelerates
  the right DOF; reset returns to template within tolerance.

## ③ Sensors + terrain
- ContactSensor: `net_forces_w (num_envs, bodies, 3)` = per-body sum of contact impulses/dt
  from Nexus contact manifolds (`contacts` + `contacts_len`), scattered by body id in torch.
  `net_forces_w_history`, `current_contact_time`/`current_air_time` derived in Python
  (same formulas as Isaac's base class).
- RayCaster (height scanner): Nexus has a point-projection query; project scan points onto
  colliders -> `ray_hits_w`. If the query is not bound/batched, first version computes hits
  in torch against the terrain heightfield directly (terrain is ours to generate).
- Terrain: Isaac's `TerrainGenerator` is numpy+trimesh (USD-free) -> `ColliderBuilder.trimesh`
  fixed body per env at its grid origin. `TerrainImporter` is USD-only and is bypassed.
- Test: humanoid standing on generated rough terrain: feet contact forces ~ weight,
  height scan returns terrain heights under the robot.

## ④ Real SimulationContext + env loop
- `SimulationContext(SimulationCfg(physics=NexusCfg(), create_stage_in_memory=True))` —
  Kit bootstraps on import; no AppLauncher.
- `InteractiveScene` clones through USD (`cloner.usd_replicate`) then a backend cloner;
  a `isaaclab_nexus.cloner` is NOT in scope for this pass. Instead: a thin scene that
  spawns via our Articulation, `env_origins` from `cloner.grid_transforms` (pure tensor
  math), and a step loop identical to `ManagerBasedRLEnv.step` (action -> write_data_to_sim
  -> sim.step x decimation -> update -> obs). Stated limitation: the full manager-based env
  needs the cloner.
- Test: 4096 envs, 24-step rollout, throughput + GPU busy measured the same way as AGILE.

## ⑤ Parity vs PhysX
- Same MJCF humanoid in Isaac Lab/PhysX (via the MJCF importer already in isaacsim) and in
  Nexus; identical initial state; compare root height / joint trajectories over N steps;
  report max divergence per step and where it starts.
- Test: report only. Divergence is expected (different solvers); the point is to bound it.

## Out of scope for this pass (stated)
- `isaaclab_nexus.cloner` (USD replicate path), RayCasterCamera, IMU/PVA/joint-wrench,
  deformables, tendons, `ManagerBasedRLEnv` end-to-end.
