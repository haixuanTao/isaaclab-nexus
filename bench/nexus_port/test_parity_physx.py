"""⑤ parity vs PhysX: the SAME gym humanoid MJCF, imported to USD with Isaac Sim's MJCF importer, dropped
passively (zero gains) on Isaac Lab's PhysX backend from the same state as the MuJoCo/Nexus reference
(parity_ref.npz written by test_parity_mujoco.py). Runs inside a headless Kit process."""
import os, sys, numpy as np
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import torch, isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import omni.kit.app
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)   # puts the importer on sys.path
from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
V = "/workspace/WBC-AGILE/.venv/lib/python3.12/site-packages"; MJCF = f"{V}/gym/envs/mujoco/assets/humanoid.xml"
ref = np.load("/workspace/bench/nexus_port/parity_ref.npz"); DT = float(ref["dt"]); STEPS = len(ref["z_mj"]); names = [str(n) for n in ref["names"]]
out_dir = "/workspace/bench/nexus_port/physx_usd"; os.makedirs(out_dir, exist_ok=True); usd = f"{out_dir}/humanoid.usd"
last = None
for kw in ({}, {"run_asset_transformer": False}, {"run_asset_transformer": False, "run_multi_physics_conversion": False}):
    try:
        res = MJCFImporter(MJCFImporterConfig(mjcf_path=MJCF, usd_path=usd, import_scene=False, fix_base=False, **kw)).import_mjcf(); print("importer ->", res, kw); break
    except Exception as e: last = e; print("importer failed with", kw, "->", repr(e)[:200])
else: raise SystemExit(f"MJCF import failed: {last}")
usd_path = res if isinstance(res, str) and res.endswith((".usd", ".usda", ".usdc")) else usd
from pxr import Usd, UsdPhysics
_stage = Usd.Stage.Open(usd_path)
usd_joint_names = [pr.GetName() for pr in _stage.Traverse() if pr.IsA(UsdPhysics.RevoluteJoint) or pr.IsA(UsdPhysics.PrismaticJoint)]
print("USD joints:", usd_joint_names)
def to_usd(n):   # MJCF joint name -> USD joint prim name (exact, else unique suffix/containment match)
    if n in usd_joint_names: return n
    c = [u for u in usd_joint_names if u.endswith(n) or n in u]
    return c[0] if len(c) == 1 else None
name_map = {n: to_usd(n) for n in names}; print("unmapped:", [n for n, u in name_map.items() if u is None])
sim = SimulationContext(SimulationCfg(dt=DT, device="cuda:0", gravity=(0, 0, -9.81)))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
robot = Articulation(ArticulationCfg(prim_path="/World/Robot", spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={name_map[str(n)]: float(v) for n, v in zip(ref["names"], ref["q0"]) if name_map.get(str(n))}),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)}))
sim.reset(); robot.update(DT)
T = lambda x: getattr(x, "torch", x)
jn = list(robot.joint_names); print("Isaac Lab DOF names:", jn)
def to_dof(n):   # MJCF hinge -> Isaac Lab DOF name (D6-converted joints carry the axis name)
    if n in jn: return n
    c = [u for u in jn if u.endswith(n) or n in u]
    return c[0] if len(c) == 1 else None
name_map = {n: to_dof(n) for n in names}; common = [n for n in names if name_map.get(n)]; idx = [jn.index(name_map[n]) for n in common]
print(f"PhysX joints {len(jn)} | common with reference {len(common)}/{len(names)} | missing {[n for n in names if n not in jn]}")
z = []; q = []
for i in range(STEPS):
    sim.step(); robot.update(DT)
    z.append(float(T(robot.data.root_pos_w)[0, 2])); q.append(T(robot.data.joint_pos)[0, idx].cpu().numpy().copy())
z = np.array(z); q = np.array(q); sel = np.arange(49, STEPS, 50); r = lambda a: [round(float(v), 3) for v in a]
print(f"root z0 physx {z[0]:.3f} (ref mujoco {ref['z_mj'][0]:.3f})")
print("root z every 0.25s  mujoco:", r(ref["z_mj"][sel])); print("                    physx :", r(z[sel])); print("                    nexus :", r(ref["z_nx"][sel]))
ci = [names.index(n) for n in common]
for lbl, zz, qq in (("physx-mujoco", ref["z_mj"], ref["q_mj"][:, ci]), ("physx-nexus", ref["z_nx"], ref["q_nx"][:, ci]), ("nexus-mujoco", None, None)):
    if zz is None: dz = np.abs(ref["z_nx"] - ref["z_mj"]); dq = np.abs(ref["q_nx"] - ref["q_mj"])
    else: dz = np.abs(z - zz); dq = np.abs(q - qq)
    print(f"{lbl:13s}: root z |diff| mean {dz.mean():.3f} max {dz.max():.3f} m | joint |diff| mean {dq.mean():.3f} max {dq.max():.3f} rad | final z {(z if zz is None else z)[-1]:.3f} vs {(ref['z_mj'] if zz is None else zz)[-1]:.3f}")
np.savez("/workspace/bench/nexus_port/parity_physx.npz", z=z, q=q, names=np.array(common))
print("PHYSX PARITY DONE"); app.close()
