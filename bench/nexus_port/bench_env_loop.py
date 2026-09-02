"""④ throughput: an AGILE-shaped rollout loop on the Nexus backend through Isaac Lab's API,
with a REAL SimulationContext (Kit in-process, no AppLauncher).
One iteration = 24 control steps x decimation 4 (200 Hz physics / 50 Hz control), each control
step: action -> set_joint_position_target -> write_data_to_sim -> sim.step x4 -> update -> obs.
Usage: bench_env_loop.py <num_envs> [iters]"""
import sys, time, torch
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_nexus import NexusCfg, NexusMjcfCfg
from isaaclab_nexus.physics.nexus_manager import NexusManager

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
MJCF = "/workspace/WBC-AGILE/.venv/lib/python3.12/site-packages/newton/examples/assets/nv_humanoid.xml"
STEPS, DECIM = 24, 4

t0 = time.time()
sim = SimulationContext(SimulationCfg(dt=1 / 200, device="cuda:0", physics=NexusCfg(), create_stage_in_memory=True))
robot = Articulation(ArticulationCfg(prim_path="/World/envs/env_.*/Robot", spawn=NexusMjcfCfg(mjcf_path=MJCF, num_envs=N),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=80.0, damping=5.0, effort_limit=200.0)}))
sim.reset()
print(f"spawn+finalize {N} envs: {time.time()-t0:.1f}s | joints {robot.num_joints} bodies {robot.num_bodies}")
d = robot.data; J = robot.num_joints
torch.cuda.synchronize()

def iteration():
    obs_acc = 0.0
    for s in range(STEPS):
        act = 0.3 * torch.randn(N, J, device="cuda")                      # policy stand-in
        robot.set_joint_position_target(act)
        robot.write_data_to_sim()
        for _ in range(DECIM):
            sim.step()
        robot.update(sim.get_physics_dt())
        obs = torch.cat([d.joint_pos.torch, d.joint_vel.torch, d.root_link_pos_w.torch, d.root_link_quat_w.torch, d.root_com_vel_w.torch], -1)
        obs_acc += float(obs[0, 0])                                        # force a tiny readback like a real obs pipeline
    return obs_acc

for _ in range(2): iteration()                                             # warmup
torch.cuda.synchronize(); NexusManager.synchronize()
times = []
for i in range(ITERS):
    t = time.perf_counter(); iteration(); NexusManager.synchronize(); torch.cuda.synchronize()
    times.append(time.perf_counter() - t)
import statistics as st_
it = st_.median(times)
print(f"num_envs={N} | iteration median {it*1000:.1f} ms (min {min(times)*1000:.1f}) | env-steps/s {N*STEPS/it:,.0f} | physics substeps/s {N*STEPS*DECIM/it:,.0f}")
