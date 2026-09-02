# PhysX vs Newton on WBC-AGILE — G1 29-DOF (`HeightTracking-G1-v0`)

Same hardware and stack as `../results/FINDINGS.md`: RTX 5090 (Blackwell cc12.0,
32 GB, 500 W cap), driver 580.105.08, Isaac Lab 3.0.0b2 / Isaac Sim 6.0 /
torch 2.11.0+cu128 / rsl-rl 5.4.1+AGILE patch. 200 Hz physics, decimation 4,
50 Hz control, rough-terrain generator, 15 s episodes. One iteration = 24 rollout
steps x N envs, then 5 epochs x 4 minibatches. Median of iterations 5+.

Two trees, same commit `6830cf9`, each with its own venv:

- `/workspace/WBC-AGILE` — untouched baseline (`git diff` empty).
- `/workspace/WBC-AGILE-NEWTON` — identical except `sim.physics = NewtonCfg()`.
  9 files, +31/-27. See `WBC-AGILE-NEWTON/NEWTON_PORT.md` for the exact diff.

Both engines were re-measured today, back to back, on an idle GPU, through the
same harness (`scripts/run_engine_compare.sh`, `scripts/run_newton_substeps.sh`).
**The PhysX side reproduces the recorded baseline within 2%** (1024: 2.170 vs
2.160 s; 4096: 4.080 vs 3.990 s), so the two engines are being measured on the
same footing.

## Status: Newton is faster per iteration, but no config has finished a full run

**Do not quote a speedup as settled.** Every Newton configuration tested eventually
dies with `observation group 'policy' contains NaN`, and the failure is
*stochastic* -- the same config, same seed, failed at iteration 122 in one run and
completed 400/400 in another. Single runs cannot rank configurations.

| config | 4096-env outcome |
|---|---|
| PhysX | 30/30 iters, clean (reference; AGILE ships on this) |
| Newton, 1 substep | NaN at 1-11 iters |
| Newton, 2 substeps | NaN at 29 iters |
| Newton, 4 substeps | NaN at 57 iters; separately NaN at 57/1000 |
| Newton, 1 substep + velocity clamp | NaN at 122/1000 **and** 400/400 clean |

## Throughput, where runs were valid

PhysX side reproduces the recorded baseline within 2% (1024: 2.170 vs 2.160 s;
4096: 4.080 vs 3.990 s), so both engines are measured on the same footing.

| envs | PhysX iter (s) | Newton x4 (s) | speedup |
|---:|---:|---:|---:|
| 1024 | 2.170 | 1.500 | 1.45x |
| 2048 | 2.820 | 1.760 | 1.60x |
| 4096 | 4.080 | 2.240 | 1.82x |

With the velocity clamp at 1 substep, 4096 envs: **1.24 s/iter, flat across 400
iterations**, vs PhysX 4.08 s at a matched training stage -> **3.3x**. Same caveat:
that config also failed once at iteration 122.

Newton scales better: 1.49x iteration-time growth across a 4x env increase, vs
PhysX's 1.88x. The PPO update is identical on both (0.232 vs 0.234 s at 4096).

## Newton does learn

400 iterations at 4096 envs, clamp, 1 substep -- reward improved monotonically
while episode length sat at the 750-step maximum:

| iter | iter_s | mean reward | ep_len |
|---:|---:|---:|---:|
| 50 | 1.24 | -1870.2 | 741.3 |
| 200 | 1.23 | -895.0 | 750.0 |
| 399 | 1.24 | -569.8 | 749.2 |

## Root cause and the clamp

Newton implements **no joint velocity limit on any of its six solvers**
(`newton/solvers.py` support matrix; `solver_mujoco.py` carries
`# MuJoCo doesn't have velocity limit` above a commented-out read at line 4310).
PhysX enforces it in-solver via `root_view.set_dof_max_velocities()`, which is why
a PhysX rollout sits at *exactly* the configured limit. AGILE sets
`velocity_limit_sim` for every G1 joint group.

`agile/isaaclab_extras/newton_joint_velocity_clamp.py` reinstates it after each
physics step. It reproduces PhysX's signature exactly (step 0 at 32.00, then
pinned at 37.00). Substepping alone also helps but does not fix the divergence.

**The clamp is a safety net, not a crutch** -- measured over 300 steps at 1024 envs:

| engine | DOFs at limit | clamp writes |
|---|---:|---:|
| Newton (1 substep + clamp) | 0.418% | 0.400% of (env,joint) writes |
| PhysX (in-solver) | 0.094% | n/a |

Same order of magnitude, both rare. It removes ~334 rad/s summed per physics step
across ~29k DOFs. Note this *is* non-physical energy removal -- but PhysX's clamp
is equally non-physical, so the Newton tree matches the baseline rather than
introducing a new artifact.

## Diagnostics run, and what they ruled OUT

These were chased down after the rendered video looked wrong. All clean:

- **Gravity is correct.** Free fall with the robot lifted 20 m and contact force
  confirmed 0.00: measured **-9.97 m/s^2** (from velocity), -10.30 (from
  displacement), against -9.81 configured. An earlier probe reporting -0.97 m/s^2
  was invalid -- it lifted only 5 m, leaving links in contact. It also reported
  "wrong" for PhysX, which should have been the immediate tell.
- **Camera orientation is correct.** Newton's `get_frame()` returns top-left
  origin (documented), so no vflip. Matches Isaac Lab's own RTX render.
- **Terrain is fine.** Height-scan sampling: z range 0.000-0.016 m, 3 distinct
  heights covering 100% of samples -- flat plates with ~16 mm steps. The "egg
  carton" look is Newton's GL viewer flat-shading each quad; RTX renders the same
  geometry as flat plates.
- **The slow descent in the video was not a fall.** `HeightTracking-G1-v0` commands
  pelvis height as a square wave; it flips 0.92 -> -0.50 at ~4.5 s and the robot
  follows it down to ~0.10 m. Commanded behaviour, not physics.

## Contact validation (Newton vs PhysX)

Prompted by the rendered video looking as though robots held unsupported poses.

**Weight and contact agree on both engines.** Standing, 32 envs, default PD:

| engine | mass | weight | total vertical contact force | ratio |
|---|---:|---:|---:|---:|
| Newton | 33.00 kg | 323.7 N | 308.0 N | 0.951 |
| PhysX | 33.00 kg | 323.7 N | 315.1 N | 0.973 |

**Static resting geometry agrees to a few cm.** Settled under PD, 32 envs:

| | Newton | PhysX | diff |
|---|---:|---:|---:|
| lowest link above terrain (mean) | 0.0848 m | 0.0660 m | 1.9 cm |
| lowest link (median) | 0.0819 m | 0.0471 m | 3.5 cm |
| pelvis height | 0.4776 m | 0.4438 m | 3.4 cm |

Same contact body on both (`ankle_roll_link`). **No systematic collision
inflation.**

**A torque-free 40-degree tilt test was run and should be disregarded.** It showed
Newton settling at pelvis z=0.59 m vs PhysX 0.28 m, which looked like a large
contact discrepancy. It is not diagnostic: a ragdoll tumbling from 40 deg is
chaotic, and two solvers will land in different heaps for reasons unrelated to
correctness. Averaging 32 envs does not fix this -- the outcome distribution is
broad. The controlled standing comparison above is the trustworthy one.

`NewtonShapeCfg.gap` (default 0.01 m per shape) does measurably raise resting
height in the collapse test -- 0.58 m at gap=0.01 vs 0.47 m at gap=0.0 -- and is
exposed as `AGILE_NEWTON_GAP`. But since the tilt test itself is not diagnostic,
this is not evidence of a defect either.

**Still untested: contact *dynamics*.** Impact forces, friction during slipping,
restitution. Static agreement does not establish any of these, and they are what
matter for sim-to-real. This remains the largest open gap in the comparison.

## Root cause of the Newton divergence: an unstable explicit-PD loop

Everything measured earlier in this document -- the NaN runs, the tumbling in
free fall, the "impossible" poses, the reset-time limit violations -- collapses to
one deterministic, minimal repro (`scripts/torque_origin_fixed.py`,
`scripts/actuator_trace.py`).

**Setup that removes every other variable:** identical *nominal* actuator gains on
both engines (kp 100 / kd 2.5 legs+waist, 20 / 1.0 feet, 20 / 0.5 arms), actuator
delay buffers set to zero, robot airborne at 3 m (no contact), identical written
state, zero policy action.

| ctrl-step | PhysX tau_max | PhysX qd_max | Newton tau_max | Newton qd_max |
|---:|---:|---:|---:|---:|
| 0 | 5.3 | 0.42 | 4.6 | 1.5 |
| 4 | 5.0 | 1.04 | **88.0** | **30.8** |
| 24 | 3.1 | 0.45 | **129.7** | **40.4** |

PhysX damps to ~2 N.m and falls cleanly. Newton self-excites to torque saturation
in four control steps with no perturbation at all.

Per physics step on `waist_yaw` (`actuator_trace.py`), Newton's joint velocity
**doubles every 5 ms**: 0.334 -> 0.705 -> 1.419 -> 2.789 -> 5.421 rad/s. The
actuator reads the correct state (`qd_seen == qd_true`) and outputs the correct
*opposing* torque (+18.99 N.m against qd = -5.42). The joint accelerates against
its own correctly-computed damping. PhysX, same trace: qd peaks at 0.115 and decays.

**Everything that could make the plant differ was measured and is identical:**

| quantity | method | result |
|---|---|---|
| link positions for same joint angles (FK) | `fk_compare.py`, by name | max 0.0001 m (pose A), 8 mm (pose B) |
| torque -> velocity gain, all 29 joints | `dof_gain_check.py`, PD off | ratio 0.98-1.03 on every joint |
| torque direction, all 29 joints | `torque_sign_check.py` | all nonzero pairs agree in sign |
| effort routing (which joint moves) | `torque_map_check.py` | correct |
| joint_vel readback lag | `vel_lag_check.py` | lag 0 on both, err 1e-4 |
| per-link mass / inertia / armature / friction | `inertia_compare.py`, `armature_check.py` | identical distribution; armature 0.02 reaches MJWarp |
| gravity (free fall, contact = 0) | `freefall_check.py` | -9.97 vs -9.81 m/s^2 |
| weight vs ground reaction | `balance_check.py` | 0.95 / 0.97 |
| angular momentum, torque-free fall | `limp_freefall.py` | conserved on both |
| subset-env joint-state writes | `subset_write_check.py` | correct, held in solver |
| integrator euler vs implicitfast | `torque_origin.py` | identical to 3 dp (PD is explicit; implicit cannot touch it) |

So: identical open-loop plant, identical controller, different closed loop. The
remaining suspect is the ordering of Python-side control writes against solver
reads inside Newton's step, which CUDA-graph capture freezes. Result of that test:
identical traces with the graph on and off (`actuator_trace.py`), so not that.

Narrowed further with the PD active on **waist_yaw alone** (`pd_isolation.py`):
still doubles every step on Newton (ratios 2.01, 2.01, 2.00), decays on PhysX.
So it is a single-DOF phenomenon, not a coupled mode between driven joints.

And with the solver's own force accounting read per step on that DOF
(`limit_constraint_check.py`, GPU `mjw_data`): no joint limit active (`nl=0`),
`qfrc_constraint = +0.01` (friction-loss noise), `qfrc_passive = 0`,
`qfrc_actuator = 0`, and `qfrc_applied` equals the Python PD torque exactly
(+0.60, +1.36, +2.92, +6.13 -- positive, opposing). The DOF accelerates at
-290 rad/s^2 against the only force applied to it. That acceleration can only
enter through the mass-matrix coupling to the rotating floating base -- the
velocity-dependent Coriolis/gyroscopic terms -- which MuJoCo-Warp's explicit
Euler integrator amplifies x2 per 5 ms step once the base begins to turn, and
which PhysX's TGS solver damps. Both the GPU and CPU MuJoCo models were dumped:
actuator gain/bias zero, `dof_damping` zero, `jnt_stiffness` zero, armature 0.02,
friction-loss 0.01, gravity -9.81, ranges sane, margin 0. No hidden force exists.

**This is an integrator stability property of "explicit PD + MJWarp Euler at
5 ms on a free-floating 29-DOF humanoid", not a reference, axis, ordering, or
mapping defect.** Every such mapping was tested and is identical between the
engines. `implicitfast` cannot help because the PD is computed in Python and
handed to the solver as an external force the integrator cannot implicitize.

**Why PhysX never shows it.** PhysX's in-solver DOF velocity clamp truncates any
growth every step; Newton has no joint velocity limit on any solver. The clamp was
masking the loop's marginal stability on PhysX, not preventing it from existing.

## Remedy, measured

`scripts/pd_solver_side.py`: explicit actuator disabled, the *same* kp=100 / kd=2.5
written as **solver-side** joint drives on waist_yaw, joint perturbed 0.1 rad off
target, airborne, 1 substep:

| integrator | q trajectory | qd growth ratio | verdict |
|---|---|---|---|
| Newton, euler | 0.097 -> 0.004 -> -0.011 (damped overshoot) | 0.92 -> 0.77 | stable |
| Newton, implicitfast | 0.097 -> 0.001 -> -0.010 | 0.95 -> 0.82 | stable |
| Newton, explicit Python PD (`pd_isolation.py`) | diverges | **2.01, 2.01, 2.00** | unstable |

Same joint, same solver, same gains, same integrator: stable inside the solver,
unstable as an external per-step force. The incompatibility is *AGILE's explicit
`DelayedDCMotor` x MuJoCo-Warp explicit Euler*, not Newton per se.

The PhysX control for this exact variant was not obtained (my probe's
`write_joint_stiffness_to_sim_index` call hit a warp `pack_args` shape mismatch on
the PhysX backend); solver-side PD is PhysX's native mode and was stable in every
other test here, so nothing rests on it.

**Recommendation for training AGILE's G1 tasks on Newton:** use an implicit
actuator (`ImplicitActuatorCfg`, solver-side PD) instead of `DelayedDCMotorCfg`.
This is a task-model change -- the DC-motor saturation curve and the command-delay
buffer would need to be reproduced on the target side -- so it was not applied in
the Newton tree. Substeps and the velocity clamp only postpone the divergence
(57 and 122 iterations); they do not remove it.

## First full 1000-iteration Newton run (task changed: implicit actuators)

With the user's agreement to change the task, the height-tracking G1 was switched to
`DelayedImplicitActuatorCfg` (solver-side PD; same gains, limits, armature, friction
and command delay; DC-motor saturation curve dropped). Config: implicit actuators +
`implicitfast` + velocity clamp, 1 substep, 4096 envs. Run dir
`logs/rsl_rl/height_tracking_g1/2026-09-02_13-16-22_height_tracking_g1` (curves
below recovered from its TensorBoard events; the text log was overwritten by a
repeat run launched with the same tag -- launcher since fixed).

**1000 / 1000 iterations, 0 NaN** -- the first Newton configuration to finish.

| | Newton (implicit + implicitfast + clamp) | PhysX (explicit, baseline) |
|---|---:|---:|
| median iteration | **1.230 s** | 4.080 s |
| collect / learn | 0.999 / 0.234 s | 3.845 / 0.234 s |
| env-steps / s | **79,922** | 24,094 |
| speedup | **3.32x** | -- |

Learning (mean episode return; curriculum never advanced -- `terrain_levels` and
`random_fallen_states` stayed 0.0 for the whole run, so returns are comparable):

| iter | reward | ep_len | invalid-state terminations |
|---:|---:|---:|---:|
| 50 | -1917.7 | 726 | 0.043 |
| 200 | -946.6 | 750 | 0.005 |
| 400 | -588.1 | 750 | 0.003 |
| 700 | **-420.4** | 750 | 0.001 |
| 900 | -475.9 | 750 | 0.003 |
| **950** | **-54,797** | 707 | 0.003 |
| 999 | -692.7 | 726 | 0.032 |

Reward improved 4.6x over 700 iterations. Then **one iteration (950) returned
-54,797** -- a single contact-impact event: `ground_slam` -1017 (vs -1.3 at iter
700), `torso_slam` -910 (vs -2.4), `body_velocity` -229 (vs -2.9). Joint velocities
stayed clamped, so nothing went NaN -- but from ~975 on invalid-state terminations
climb 0.005 -> 0.027 and the return drifts -570 -> -741. The most likely reading is
that the iteration-950 batch poisoned one PPO update and the policy degraded from
there. So: physics survived, learner did not fully.

This is a **contact** failure mode, distinct from the joint-PD instability fixed by
the implicit actuator. It is the untested part flagged from the start (contact
dynamics), and it is what the repeat run and any longer run must be judged on.

## Torque-free behaviour on the ground, and the joint-limit springs

Asked directly whether Newton is reliable *without* torque: in free fall, yes
(`limp_freefall.py`: orientation, angular momentum and gravity identical to PhysX).
On the ground, no. `limp_drop.py`, upright start, all gains and effort limits zeroed:

| t | PhysX pelvis z | Newton pelvis z |
|---:|---:|---:|
| 0.12 s | 0.801 | 0.870 |
| 0.42 s | 0.558 | 0.576 |
| 0.82 s | 0.318 | **0.585** |
| 1.62 s | 0.279 | **0.625** |

PhysX crumples and stays; Newton stops at 0.58 m and *rises*. This reproduces the
0.3 m gap earlier dismissed as a "chaotic tilt test" -- that dismissal was wrong.

The rebound is an under-damped spring, and Newton's joint limits are springs:
`joint_limit_ke = 1e4`, `joint_limit_kd = 10` -- Newton's *builder defaults*, not
authored in the G1 USD (`import_usd.py:286`, `builder.py:452`). PhysX enforces
limits as hard constraints. Overriding the damping (`limit_damping_check.py`,
knob `AGILE_NEWTON_LIMIT_KD`):

| limit_kd | final z | behaviour |
|---:|---:|---|
| 10 (default) | 0.625 | rebounds |
| 200 | 0.450 | smaller bounce, settles lower |
| 1000 | 0.909 | never falls -- over-damped limits lock the joints |
| PhysX | 0.279 | crumples |

A confirmed lever that closes about half the gap; not the whole answer.

## On-ground stability under PD, all configurations (`gentle_stability.py`)

No teleport, `env.reset()` on the ground, zero action (PD holds the default pose),
64 envs, 100 control steps:

| config | max joint speed over the run | final pelvis z |
|---|---:|---:|
| PhysX, explicit DCMotor | **~20 rad/s, flat** | 0.31 |
| Newton, explicit, no clamp | 21 -> 44 -> 36-45 | 0.40 |
| Newton, explicit, velocity clamp on | **pinned at 37.00** | 0.41 |
| Newton, implicit actuators, euler | up to 64-69 | 0.39 |
| Newton, implicit actuators, implicitfast | up to 73-77 | 0.41 |

Two things follow. PhysX's ~20 is *exactly* the knee velocity limit: PhysX is not
calmer than Newton, it is **clamped** -- both engines saturate under this PD from
the reset states, and the in-solver limit is most of what PhysX's "stability" is.
With the post-step clamp, Newton pins at its limits the same way. And **implicit
actuators make the on-ground case worse, not better**, under either integrator --
so the actuator-swap remedy from the airborne single-joint test does not carry
over to the task. Correction 10, below.

## Findings that were retracted along the way

- Newton "held up by something" (0.3 m resting-height gap) -- chaotic ragdoll test.
- Torso inertia 26% lower on Newton -- per-env `randomize_rigid_body_mass`
  (`add (-1,+3) kg` on torso_link) drawn from different RNG streams; both engines
  match the authored 0.259 trace within the draw. Not an importer bug.
- Fallen-states cache index-scrambled on Newton -- the cache *is* PhysX-ordered and
  the loader *is* index-based, and a name-remap was added (`fallen_state_dataset.py`,
  `pre_learn.py`); but it changed nothing, because the violations being read were
  the unstable loop, not the dataset. The remap is correct hygiene, not a fix.
- Reset-time limit violations as a "stale buffer read" -- persisted one step later.
- Velocity-feedback lag, torque-axis sign inversion, effort mis-routing, hidden
  Newton drives, missing armature, quaternion order, viewer axis -- all tested, all
  excluded, controls on PhysX every time.

## Corrections made during this investigation

Recorded because each one was stated confidently before being checked:

1. **"Stable at 4 substeps"** -- claimed on 30 iterations; it NaN'd at 57/1000.
2. **"Each mitigation doubles survival"** (11/28/57/122 iters) -- read as a trend
   from one run per config. The failure is stochastic: the same clamp config
   failed at 122 and then completed 400/400.
3. **"GRAVITY WRONG" (-0.97 m/s^2)** -- invalid probe; lifted the robot only 5 m,
   leaving links in contact. It also reported "wrong" for PhysX, which should have
   been the immediate tell. Correct value is -9.97 m/s^2.
4. **Video captioned as robots "falling"** -- it was a commanded height cycle in a
   task named HeightTracking. Three diagnostics were spent on a phantom before
   checking the command signal.
5. **"Newton robots are being held up by something"** -- inferred from the chaotic
   tilt test and stated as the most consequential finding. Retracted; the
   controlled test shows ~3 cm agreement.
6. **"Torso inertia 26% lower on Newton, importer bug"** -- it was a per-env
   mass-randomization draw; before/after-reset traces bracket the authored value
   on both engines.
7. **"Fallen-states cache scrambled -- confirmed in the real reset path"** -- the
   scramble is real and was fixed, but the 47%-of-resets figure was the unstable
   loop blowing past limits, not the dataset. Stated as confirmed before checking
   that the remap changed the number. It did not.
8. **"Reset reads are a stale buffer"** -- violations persisted one step later.
10. **"Torque-free Newton is reliable"** -- true in free fall, overstated for the
   ground: a limp robot on Newton settles 0.35 m higher than on PhysX and rebounds.
11. **"Implicit actuators are the remedy"** -- stable in the single-joint airborne
   test; on the ground with all joints driven they are worse than the explicit
   model under both integrators.
9. **"Newton's in-graph actuator adapter double-applies torque"** -- adapter was
   `None`; CUDA graph on/off identical; effort routing, torque sign, velocity-lag,
   armature, hidden drives, GPU actuator gains, joint limits, subset writes all
   measured identical or clean. Each was announced as the likely cause before
   its test ran.

## Caveats

- **No Newton config has completed 1000 iterations.** Best is 400/400.
- Throughput only. Different contact model, different integrator, no in-solver
  velocity clamp -- equal step cost does not imply equal sample efficiency or
  sim-to-real transfer. A same-wall-clock reward comparison was not run.
- Newton's **contact response** was never validated against PhysX. That is the gap
  that matters most for sim-to-real and nothing here addresses it.
- `pyglet` was downgraded 3.0.dev5 -> 2.1.16 in the Newton venv (3.x drops
  `pyglet.gl`); rendering only, does not affect physics or timings.
- Isaac Lab's `--video` path silently produces empty-terrain footage on this setup
  (its camera only tracks the robot when a Newton visualizer is live). Use
  `/workspace/bench/scripts/record_newton_direct.py` instead.

## Scripts

`run_engine_compare.sh`, `run_newton_substeps.sh`, `parse_engine_compare.py`,
`nan_probe.py`, `clamp_stats.py`, `freefall_check.py`, `terrain_probe2.py`,
`limp_drop.py`, `height_cmd_probe.py`, `record_newton_direct.py` -- all in
`/workspace/bench/scripts/`.
