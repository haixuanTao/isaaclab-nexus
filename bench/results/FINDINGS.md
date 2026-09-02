# WBC-AGILE benchmark — G1 29-DOF full-body (`HeightTracking-G1-v0`)

Hardware: RTX 5090 (Blackwell cc12.0, 32 GB, **500 W cap**), driver 580.105.08,
16-core host, unprivileged container.
Stack: Isaac Lab 3.0.0b2 / Isaac Sim 6.0 / torch 2.11.0+cu128 / rsl-rl 5.4.1+AGILE patch.

Task actuates all 29 joints (`joint_names=[".*"]`); 200 Hz physics, decimation 4,
50 Hz control; rough-terrain generator; 15 s episodes.
One iteration = 24 rollout steps x N envs, then 5 epochs x 4 minibatches.

## Throughput (rsl_rl's own timers, median of iters 5..29)

| envs | iter (s) | collect (s) | learn (s) | collect % | env-steps/s | GPU power (med) | util (med) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 2.160 | 2.033 | 0.122 | 94.1% | 11,378 | 139 W | 69% |
| 2048 | 2.810 | 2.663 | 0.151 | 94.8% | 17,492 | 161 W | 70% |
| 4096 | 3.990 | 3.754 | 0.234 | 94.1% | 24,638 | 182 W | 60% |
| 8192 | see below — PhysX broadphase overflow at default config |

**Rollout is ~94% of every iteration; the PPO update is ~6%**, flat across batch size.

## Power: high reported utilization, low actual work
At 4096 envs the GPU draws **182 W of a 500 W cap (36%)** while nvidia-smi reports
**60% "utilization"**. nvidia-smi utilization is the fraction of time >=1 kernel is
resident, not occupancy or throughput, so this pair is the signature of a GPU that is
nominally busy and actually idle-ish -- many small serialized kernels, not saturation.

## Scaling cliff at 8192 envs (default config)
Out of the box, 8192 envs produces:

    [Error] [omni.physx.plugin] PhysX error: The application needs to increase
    PxGpuDynamicsMemoryConfig::foundLostAggregatePairsCapacity to 172733764,
    otherwise, the simulation will miss interactions

Isaac Lab default is `gpu_found_lost_aggregate_pairs_capacity = 2**25` (33.5 M);
PhysX asks for 172.7 M (~2**27.4). Iteration time was **55.2 s** vs 3.99 s at 4096 --
a 14x jump for a 2x batch increase -- AND the sim was silently dropping interactions,
so that number is not a valid throughput point in either direction.

Cause is structural, not just a buffer size: the terrain generator makes a fixed
8-level x 9-type tile grid, so raising env count past ~4096 packs more robots onto the
same tiles, and broadphase aggregate pairs grow superlinearly.

## Caveats on the measurement environment
- `perf_event_paranoid=4`, `perf_event_open` unavailable -> nsys CPU sampling,
  context-switch tracing and `--gpu-metrics` are all unavailable. CUDA/CUPTI tracing works.
- GPU power sampled via `nvidia-smi` polling (100 ms sweep / 50 ms trace) instead.
- AGILE ships no NVTX ranges, so trace windows are cut on kernel pattern + rsl_rl timers,
  not on markers.
- 500 W card. The paper's "156 W of 600 W" figure came from a different GPU and is not
  comparable to these numbers.

## 8192 envs: a real scaling wall, not a buffer size

Re-ran 8192 with the broadphase buffers raised 8x
(`gpu_found_lost_aggregate_pairs_capacity=2**28`, `gpu_total_aggregate_pairs_capacity=2**23`):

| 8192 config | iter (s) | collect (s) | learn (s) | env-steps/s | verdict |
|---|---:|---:|---:|---:|---|
| default buffers | 55.2 | - | - | ~3,560 | PhysX dropping interactions (wrong) |
| 8x buffers | 68.4 | 67.808 | 0.539 | 2,876 | correct, and *slower* |

Raising the buffer made it **slower** (68.4 s vs 55.2 s) because PhysX then actually
computes the interactions it had been silently discarding. Throughput vs 4096 envs:
24,638 -> 2,876 env-steps/s, an **8.6x collapse for a 2x batch increase**.

### Mechanism
`STAND_UP_ROUGH_TERRAIN_G1_CFG` = `size=(8.0, 8.0)`, `num_rows=8`, `num_cols=9`
-> a fixed grid of **72 tiles of 8 x 8 m**, independent of `num_envs`.

| envs | robots per 8x8 m tile | area per robot |
|---:|---:|---:|
| 1024 | 14 | 4.5 m^2 |
| 4096 | 57 | 1.1 m^2 |
| 8192 | 114 | 0.56 m^2 |

At 8192 the robots are packed shoulder-to-shoulder on shared tiles, so broadphase
aggregate pairs grow superlinearly. This is a property of the *task configuration*,
not of PhysX: scaling this task past ~4096 envs requires growing the terrain grid
(`num_rows`/`num_cols`) with the env count, which no CLI flag does.

**Consequence for a scaling comparison:** 4096 is the largest batch at which AGILE's
own stand-up task is both correct and performant out of the box. A Zealot-vs-AGILE
curve that runs to 8192 is not comparing like with like unless the terrain grid is
scaled too -- and that changes the task.

## nsys trace: G1 29-DOF, 4096 envs, 45 s steady-state window

nsys 2026.1.3, `--trace=cuda,osrt,nvtx --sample=none --cpuctxsw=none`, 103 MB report.
Profiling overhead: iteration 4.28 s traced vs 3.99 s untraced (~7%).

### Headline: the GPU is idle half the time
| | ms | % of wall |
|---|---:|---:|
| wall (window) | 44,977 | 100% |
| **GPU busy** (union of kernel intervals) | **22,289** | **49.6%** |
| **GPU idle** | **22,689** | **50.4%** |

- 1,287,406 kernel launches in 45 s = **28,623 launches/s**
- **mean kernel duration 17.6 us** — the GPU never gets a big enough unit of work
- 337,251 memcpys totalling 765 ms (tiny transfers, launch-bound not bandwidth-bound)

### Rollout vs update (segmented on PhysX-kernel presence)
Independently reproduces rsl_rl's own timers by a different method:

| phase | ms | % of iteration | GPU busy in phase | launches |
|---|---:|---:|---:|---:|
| rollout | 4,353.7 | **94.0%** | 49.5% | 120,752 |
| PPO update | 276.7 | **6.0%** | 63.4% | 11,646 |
| iteration | 4,630.4 | 100% | 50.3% | 132,398 |

rsl_rl said 94.1% / 5.9%. Two independent methods agree.
**132,398 kernel launches per iteration = ~5,031 launches per env step** (24 steps/iter).

### Where GPU time actually goes
| family | ms | % busy | % wall | launches |
|---|---:|---:|---:|---:|
| PhysX articulation/solver | 14,949 | 67.1% | 33.2% | 378,371 |
| PhysX collision (BP/NP) | 4,463 | 20.0% | 9.9% | 39,990 |
| torch elementwise/reduce/index | 1,387 | 6.2% | 3.1% | **662,324** |
| NN (GEMM/activations) | 1,117 | 5.0% | 2.5% | 23,689 |
| unclassified (mostly PhysX) | 597 | 2.7% | 1.3% | 179,540 |
| Warp (height scan / raycast) | 197 | 0.9% | 0.4% | 3,492 |

**PhysX is ~90% of GPU busy time. The neural network is 5.0%.**

**The task-layer tax, measured:** torch elementwise/reduce/index kernels are
**662,324 launches — 51% of every kernel launch in the run — for 6.2% of GPU work**,
averaging **2.1 us each**. This is the Isaac Lab manager-based MDP layer (observation
terms, reward terms, resets) expressed as tensor-op chains. It contributes almost no
arithmetic and roughly half the launch pressure.

### Idle is dispatch latency, not one big stall
1,248,095 idle gaps: mean 18.2 us, median **5.6 us**, p99 94.6 us, max 13.5 ms.

| gap size | count | total ms | % of idle |
|---|---:|---:|---:|
| <5 us | 590,902 | 1,156 | 5.1% |
| 5-20 us | 513,624 | 5,335 | 23.5% |
| 20-100 us | 132,298 | 4,818 | 21.2% |
| 0.1-1 ms | 9,392 | 1,443 | 6.4% |
| >1 ms | 1,879 | 9,938 | 43.8% |

~50% of the idle is micro-gaps under 100 us (per-launch dispatch), ~44% is 1,879
larger stalls (host sync points, ~179 per iteration).

### Host side
CUDA API time 6,697 ms over the 45 s span = 14.9% of one host thread:

| api | ms | calls | us/call |
|---|---:|---:|---:|
| cudaLaunchKernel | 2,108 | 715,691 | 2.95 |
| cuLaunchKernel | 1,493 | 571,715 | 2.61 |
| cudaStreamSynchronize | 1,294 | 115,495 | 11.21 |
| cudaMemcpyAsync | 570 | 143,764 | 3.96 |
| cuStreamSynchronize | 391 | 8,374 | 46.66 |

**3.60 s of the 45 s window is spent purely issuing kernel launches** (1.29 M launch
calls), and 115,495 stream synchronizations cost another 1.29 s.

### Incidental
PPO GEMMs dispatch to `cutlass_80_tensorop_s1688gemm_*` — SM80 (Ampere) TF32 kernels
running on an SM120 Blackwell card via compatibility, not Blackwell-native GEMMs.
Only 5% of GPU time here, so it is a small lever, but it means the baseline is not
using this GPU's tensor cores at their native path.

## Ablation: stub the physics step, subtract

The Nexus swap was not runnable (see below), so the same question was answered by
no-op'ing the engine instead of replacing it. `SimulationContext.step()` is monkey-
patched to return immediately; everything else in the pipeline runs unchanged.
The WBC-AGILE clone is **not modified** — the harness is two files in
`/workspace/bench/scripts/` (`ablate_patch.py`, `train_ablate.py`) and `git status`
is clean. Run with `./.venv/bin/python`, not `uv run` (which builds a separate env
for an out-of-tree script).

G1 29-DOF, `HeightTracking-G1-v0`, 4096 envs, 20 iterations, median of iters 5+.

| rung | removed | iter (s) | collect (s) | learn (s) |
|---|---|---:|---:|---:|
| `none` | — | 4.200 | 3.961 | 0.234 |
| `nophysics` | `SimulationContext.step()` | 0.970 | 0.732 | 0.233 |
| `nophysics_noreadback` | + `InteractiveScene.update()` | 0.930 | 0.699 | 0.233 |

**Validity check:** learn time is 0.234 / 0.233 / 0.233 across all three rungs.
Stubbing physics does not touch the learner, and it didn't — the ablation is
isolating what it claims to.

### Decomposition of the 4.200 s iteration
| component | seconds | share |
|---|---:|---:|
| **PhysX solve** | **3.229** | **76.9%** |
| task layer + policy inference | 0.699 | 16.6% |
| sensor / state readback | 0.033 | 0.8% |
| PPO update | 0.234 | 5.6% |

**Removing physics makes the iteration 4.3x faster (4.200 -> 0.970 s).**

### Reading this against the trace
These two results are consistent and answer different questions:
- The nsys trace said the task layer is **51% of kernel launches but 6.2% of GPU work**.
- The ablation says physics is **76.9% of wall time**.

So the task layer dominates *dispatch pressure* while PhysX dominates *time*. The
launch flood is real but it is not what the iteration is waiting on. For a stack
comparison this matters: fusing the task layer into kernels removes half the launches
and at most ~17% of the iteration, whereas the engine is the 77%.

The floor also matters: even with physics entirely removed, an iteration still costs
0.93 s at 4096 envs — 0.699 s of task layer and inference plus a 0.234 s update. Any
engine dropped into this framework inherits that floor.

### Caveat that bounds the claim
With physics stubbed the robots never move, so contacts stay zero and terminations
stop firing. Branch-dependent work (resets, terrain curriculum) therefore differs from
a live run, and the `noreadback` rung additionally runs on stale observations. These
numbers **bound** the framework's cost rather than measure it exactly; the physics
share is if anything *understated*, since a live run does more reset work.

## Why the Nexus swap was not run
1. Zealot is not on this machine, so the pinned Nexus revision could not be read.
2. `isaaclab.utils.backend_utils.FactoryBase._get_backend()` hardcodes exactly
   `physx`, `ovphysx`, `newton` and raises on anything else. A fourth backend means
   patching Isaac Lab plus an `isaaclab_nexus` package on the scale of the existing
   ones (`isaaclab_physx` 87 files / 26,923 lines; `isaaclab_newton` 83 / 24,338).
3. dimforge/nexus documents no batched multi-environment API and no batched tensor
   state access — the exact contract Isaac Lab's views require. Its README leads with
   "still under heavy development and is still missing many features"; no tagged releases.
4. NOTE: `nexus3d` on PyPI is an unrelated glTF/STL mesh utility, not dimforge's Nexus.
