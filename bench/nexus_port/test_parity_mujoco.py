"""⑤ parity: Nexus (fork, CUDA) vs MuJoCo on the SAME MJCF (gym humanoid, MuJoCo-stable), same initial
state, no actuation, flat floor at z=0 on both sides. Divergence is expected; the point is to bound it."""
import re, tempfile, numpy as np, torch, mujoco
from nexus3d import NexusBackend, NexusState, NexusPipeline, RigidBodyBuilder, ColliderBuilder, Vec3
V="/workspace/WBC-AGILE/.venv/lib/python3.12/site-packages"; DT, STEPS = 1/200, 400
xml=open(f"{V}/gym/envs/mujoco/assets/humanoid.xml").read()
# ---- MuJoCo (reference) ----
m=mujoco.MjModel.from_xml_string(xml); m.opt.timestep=DT; m.opt.integrator=0; d=mujoco.MjData(m); mujoco.mj_forward(m,d)
jn=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(m.njnt)]; hinge=[j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
fz=float(m.geom_pos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"floor")][2]); z_mj=[]; q_mj=[]
# initial hinge pose clamped just inside the joint limits (PhysX/Isaac Lab refuses out-of-limit defaults; MuJoCo/Nexus accept q=0)
q0_clamped={jn[j]: float(np.clip(0.0, m.jnt_range[j,0]+1e-3, m.jnt_range[j,1]-1e-3)) if m.jnt_limited[j] else 0.0 for j in hinge}
for j in hinge: d.qpos[m.jnt_qposadr[j]]=q0_clamped[jn[j]]
mujoco.mj_forward(m,d); print("clamped initial joints:", {k: round(v,3) for k,v in q0_clamped.items() if v!=0.0})
for i in range(STEPS):
    mujoco.mj_step(m,d); z_mj.append(float(d.qpos[2])); q_mj.append([float(d.qpos[m.jnt_qposadr[j]]) for j in hinge])
z_mj=np.array(z_mj); q_mj=np.array(q_mj)
# ---- Nexus (free root as <freejoint/> for rapier-mjcf; explicit -z gravity; matching floor) ----
g=re.sub(r'<joint[^>]*type="free"[^>]*/>','<freejoint name="root"/>',xml); p=tempfile.NamedTemporaryFile("w",suffix=".xml",delete=False); p.write(g); p.close()
be=NexusBackend("cuda"); st=NexusState(); st.insert_mjcf_headless(p.name,0,None,False)
st.insert_rigid_body_in(0, RigidBodyBuilder.fixed().translation(Vec3(0,0,fz-0.5)).build(), ColliderBuilder.cuboid(20,20,0.5).build())
st.set_rbd_collisions_capacity(256); st.set_rbd_dt(DT); st.finalize_headless(be); st.set_rbd_gravity_headless(be,Vec3(0,0,-9.81)); be.synchronize()
lay=st.ws_layout(); ws=torch.as_tensor(st.links_workspace_cuda(),device="cuda"); stat=torch.as_tensor(st.links_static_host(be)).long()
asm,nd,locked=stat[:,3],stat[:,4],stat[:,6]; ljn=st.mjcf_names(be)["link_joint_names"]; L=len(ljn)
jl=[k for k in range(L) if nd[k]>0 and nd[k]!=6]; free=[[a for a in range(6) if not (int(locked[k])>>a)&1][:int(nd[k])] for k in range(L)]
rows=torch.tensor([k for k in jl for _ in range(int(nd[k]))],device="cuda"); slot=torch.tensor([free[k][j] for k in jl for j in range(int(nd[k]))],device="cuda")
nx_names=[ljn[k] for k in jl for _ in range(int(nd[k]))]
def q_nx(): c=torch.cat([ws[:,lay["WS_COORDS"],:,:],ws[:,lay["WS_COORDS"]+1,:,:2]],-1); return c[rows,0,slot].cpu().numpy()
qc=torch.tensor([q0_clamped.get(n,0.0) for n in nx_names],device="cuda")
for r,sl,val in zip(rows.tolist(),slot.tolist(),qc.tolist()):
    ws[r, lay["WS_COORDS"] + (1 if sl>=3 else 0), 0, sl if sl<3 else sl-3] = val
be.synchronize()
pipe=NexusPipeline(); z_nx=[]; q_nxt=[]
for i in range(STEPS):
    pipe.simulate_headless(be,st,None); be.synchronize(); z_nx.append(float(ws[0,lay["WS_LTW"]+1,0,2])); q_nxt.append(q_nx())
z_nx=np.array(z_nx); q_nxt=np.array(q_nxt)
mj_names=[jn[j] for j in hinge]; common=[n for n in nx_names if n in mj_names]; im=[mj_names.index(n) for n in common]; ix=[nx_names.index(n) for n in common]
print(f"common joints {len(common)}/{len(mj_names)} | floor z {fz} | root z0 mujoco {z_mj[0]:.3f} nexus {z_nx[0]:.3f}")
sel=np.arange(49,STEPS,50); print("root z every 0.25s  mujoco:", [round(v,3) for v in z_mj[sel]]); print("                   nexus :", [round(v,3) for v in z_nx[sel]])
dz=np.abs(z_mj-z_nx); dq=np.abs(q_mj[:,im]-q_nxt[:,ix])
print(f"root z |diff|: mean {dz.mean():.3f} m, max {dz.max():.3f} m at t={DT*(dz.argmax()+1):.2f}s | final z: mujoco {z_mj[-1]:.3f} nexus {z_nx[-1]:.3f}")
print(f"joint |diff|: mean {dq.mean():.3f} rad, max {dq.max():.3f} rad | worst joints: {sorted([(round(float(dq[:,i].mean()),3),common[i]) for i in range(len(common))],reverse=True)[:3]}")
np.savez("/workspace/bench/nexus_port/parity_ref.npz", q0=np.array([q0_clamped.get(n,0.0) for n in common]), z_mj=z_mj, z_nx=z_nx, q_mj=q_mj[:, [mj_names.index(n) for n in common]], q_nx=q_nxt[:, [nx_names.index(n) for n in common]], names=np.array(common), dt=DT, floor_z=fz)
print("saved parity_ref.npz (MuJoCo + Nexus trajectories for the PhysX comparison)")
print("PARITY REPORT DONE")
