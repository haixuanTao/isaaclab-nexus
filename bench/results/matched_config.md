# WBC-AGILE matched configuration (read from repo, not from the paper)

Repo: github.com/nvidia-isaac/WBC-AGILE @ $(git rev-parse --short HEAD)
Stack: Isaac Lab 3.0.0b2 / Isaac Sim 6.0 / torch 2.11.0+cu128 / rsl-rl-lib 5.4.1 (+AGILE patch)

## G1 task taxonomy (actuated DOF != simulated DOF)
| Task id | Robot cfg | Actuated joints | Terrain |
|---|---|---|---|
| Velocity-G1-Teacher-v0 | G1_29DOF_DELAYED_DC_MOTOR | LEG only (12) | LESS_ROUGH generator |
| Velocity-Height-G1-Teacher-v0 | G1_29DOF | LEG only (12) | generator |
| **HeightTracking-G1-v0** | G1_29DOF_HEIGHT_TRACKING | **`.*` = all 29** | STAND_UP_ROUGH generator |
| MotionTracking-G1-v0 | G1_29DOF_BeyondMimic | all 29 | flat; needs motion dataset |
| PickPlace-G1-* | G1_W_HANDS_AGILE_CFG | 29 + hands | flat |

NOTE: every G1 task simulates the full 29-DOF articulation in PhysX. The
"12 DOF" tasks differ only in the size of the action space and the policy head,
not in physics cost. A 12-vs-29 DOF comparison is therefore a comparison of
*policy/action* width, not of *simulation* width.

## One PPO iteration (identical for both tasks)
- rollout: num_steps_per_env = 24  ->  24 * num_envs env steps
- update:  num_learning_epochs = 5 x num_mini_batches = 4  =  20 gradient steps
- actor MLP [256,256,128], critic MLP [512,256,128], ELU
- symmetry: lr_mirror_G1 data augmentation ON (doubles the update batch)
- empirical_normalization = False

## Physics budget (identical for both tasks)
- sim.dt = 1/200 s (200 Hz), decimation = 4  ->  control 50 Hz
- 4 physics substeps per control step
- HeightTracking-G1-v0: episode_length_s = 15.0, gpu_max_rigid_patch_count = 2^20
- Velocity-G1-Teacher-v0: episode_length_s = 30.0
- default scene: num_envs = 4096, env_spacing = 2.5

## Determinism posture (scripts/train.py)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.deterministic    = False
    torch.backends.cudnn.benchmark        = False
AGILE opts out of determinism in its own trainer by default.

## Measurement environment
- GPU: RTX 5090 (Blackwell, cc 12.0), 32 GB, 500 W cap, driver 580.105.08
- CPU: 16 cores, 30 GB RAM; unprivileged container
- perf_event_paranoid = 4, perf_event_open unavailable
  -> nsys CPU sampling / context-switch tracing / --gpu-metrics NOT available
  -> CUDA (CUPTI) + OSRT tracing IS available; GPU power sampled via nvidia-smi
