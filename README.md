# isaaclab-nexus

An Isaac Lab physics backend for [Nexus](https://github.com/dimforge/nexus) on
CUDA, plus the benchmark harness used to measure it against PhysX and Newton on
[WBC-AGILE](https://github.com/nvidia-isaac/WBC-AGILE)'s G1 29-DOF tasks.

## `isaaclab_nexus/`

A backend spike, not a finished backend. What is real:

* `NexusCfg` / `NexusManager` — a `PhysicsManager` owning one headless Nexus
  CUDA state and stepping it.
* `assets.articulation.Articulation` / `ArticulationData` — joint, root and body
  state served as **zero-copy torch views** of Nexus GPU memory (via
  `__cuda_array_interface__`): no staging buffer, no host round-trip.
* `envs.nexusify(env_cfg, mjcf_path)` — rewrites an existing
  `ManagerBasedRLEnvCfg` onto Nexus in place (physics cfg, MJCF in place of USD
  spawns, dropping event terms that need PhysX-only views).

Every `BaseArticulation` method that is not implemented raises
`NotImplementedError` naming itself, so a caller hits a clear wall rather than a
silent wrong answer.

The layout is dictated by `isaaclab.utils.backend_utils.FactoryBase`: for a
factory defined in `isaaclab.<subpath>.<mod>` it imports
`isaaclab_nexus.<subpath>` and looks up a class of the factory's own name.

Requires the engine-side work on
[haixuanTao/nexus](https://github.com/haixuanTao/nexus), branch
`isaac-lab-backend` — the CUDA array views, batched reset, motor target groups
and MJCF name resolution this package binds to.

## `bench/`

* `scripts/` — the measurement harness (throughput sweeps, Nsight capture and
  trace analysis, power sampling) and the single-purpose probes written during
  the stability investigation (gravity, free-fall, terrain, contact, joint
  clamp, solver-side PD).
* `nexus_port/` — `PORT_SPEC.md` and `PLAN.md` for the Isaac-Lab-onto-Nexus
  port, with the backend tests that check it (zero-copy views, write path,
  contacts, terrain, articulation, and parity against MuJoCo and PhysX).
* `results/` — PhysX baseline: throughput sweep, ablations, power, and
  `FINDINGS.md`.
* `results_newton/` — the PhysX vs Newton comparison and
  `FINDINGS_NEWTON.md`, the evidence behind the WBC-AGILE Newton port
  (branch `newton-port` on [haixuanTao/WBC-AGILE](https://github.com/haixuanTao/WBC-AGILE)).

Nsight traces and recorded video are deliberately not tracked; regenerate them
with `scripts/run_nsys.sh` and `scripts/record_*.sh`.
