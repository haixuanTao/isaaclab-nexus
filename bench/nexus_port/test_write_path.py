"""Steps ② and ① of the plan, tested against the raw Nexus bindings (fork, CUDA).

② names + DOF map:  MJCF body/joint names land on the right Nexus links, and the
   (assembly_id, ndofs) map compacts per-link coords into a flat DOF vector that
   agrees with the flat generalized-velocity vector's layout.
① write path:      (a) position targets scattered on-GPU are tracked by the PD motors;
                   (b) an external generalized force accelerates the addressed DOF and
                       only that DOF's chain; (c) a batched template reset restores the
                       initial state (with per-env offsets) for the chosen envs only.
"""
import torch
from nexus3d import NexusBackend, NexusState, NexusPipeline

MJCF = "/workspace/WBC-AGILE/.venv/lib/python3.12/site-packages/newton/examples/assets/nv_humanoid.xml"
NENV, DT = 4, 1.0 / 200.0

be = NexusBackend("cuda"); assert be.is_cuda()
st = NexusState()
for e in range(NENV):
    env = 0 if e == 0 else st.add_environment()
    st.insert_mjcf_headless(MJCF, env)
st.set_rbd_dt(DT)
st.finalize_headless(be)
st.publish_reset_template(be)          # template 0 = initial pose
be.synchronize()

lay = st.ws_layout()
L, NB, D = lay["links_per_batch"], lay["num_batches"], lay["dofs_per_batch"]
ws = torch.as_tensor(st.links_workspace_cuda(), device="cuda")     # (L, 15, NB, 4)
dof = torch.as_tensor(st.dof_state_cuda(), device="cuda")          # (sections, D, NB)
stat = torch.as_tensor(st.links_static_host(be)).long().cuda()     # (L, 8) as int64 (cuda has no u32 compare)
names = st.mjcf_names(be)

# ---------------- ② names + DOF map ----------------
rb, parent, mbid, asm, ndofs, kin, locked, motor_axes = (stat[:, i] for i in range(8))
print(f"links={L} dofs={D} envs={NB} | actuated links: {(ndofs > 0).sum().item()} | sum ndofs = {ndofs.sum().item()}")
assert ndofs.sum().item() == D, "sum of per-link ndofs must equal dofs_per_batch"
lbn, ljn = names["link_body_names"], names["link_joint_names"]
named = sum(1 for n in lbn if n)
print(f"body names resolved on {named}/{L} links; e.g. {[n for n in lbn if n][:5]}")
print(f"joint names on links:  {[n for n in ljn if n][:6]}")
assert named == L, "every link should map to an MJCF body"
# the floating base is a 6-DOF free joint with no MJCF <joint>; Isaac Lab excludes it from joint_* vectors
is_root = (parent > L) | (ndofs == 6)
root_links = torch.nonzero(is_root).flatten().tolist()
print(f"root link(s): {root_links} with ndofs {[int(ndofs[k]) for k in root_links]} -> excluded from joint vectors")
joint_links = [k for k in range(L) if ndofs[k] > 0 and not is_root[k]]
assert all(ljn[k] for k in joint_links), "every non-root DOF-bearing link should carry its joint name"
NJ = int(sum(int(ndofs[k]) for k in joint_links))
print(f"Isaac num_joints = {NJ} (of {D} generalized DOFs)")

# flat DOF gather over non-root links: column asm[k]+i <- coords[k, i]  (linear slots first, then angular)
cols = torch.cat([asm[k] + torch.arange(int(ndofs[k]), device="cuda") for k in joint_links])
rows = torch.cat([torch.full((int(ndofs[k]),), k, device="cuda") for k in joint_links])
# which coord slot holds DOF i of link k: unlocked axes in order (lin 0..2 then ang 3..5)
free_axes = [[a for a in range(6) if not (int(locked[k]) >> a) & 1][: int(ndofs[k])] for k in range(L)]
slot = torch.tensor([free_axes[k][j] for k in joint_links for j in range(int(ndofs[k]))], device="cuda")
order = torch.argsort(cols)                     # joint order = ascending DOF column
rows, slot, cols = rows[order], slot[order], cols[order]

def joint_pos_flat():                            # (NB, D)
    c = torch.cat([ws[:, lay["WS_COORDS"], :, :], ws[:, lay["WS_COORDS"] + 1, :, :2]], dim=-1)  # (L, NB, 6)
    return c[rows, :, slot].T                     # gather -> (D, NB) -> (NB, D)

def joint_vel_flat():                            # (NB, NJ): velocity section, root columns dropped
    return dof[0][cols].T

pipe = NexusPipeline()
for _ in range(20): pipe.simulate_headless(be, st, None)
be.synchronize()
q0, v0 = joint_pos_flat().clone(), joint_vel_flat().clone()
for _ in range(1): pipe.simulate_headless(be, st, None)
be.synchronize()
q1 = joint_pos_flat()
fd = (q1 - q0) / DT
err = (fd - joint_vel_flat()).abs()
print(f"flat joint_pos vs joint_vel layout check: median |dq/dt - v| = {err.median():.4f} rad/s, "
      f"median |v| = {joint_vel_flat().abs().median():.4f}")
assert err.median() < 0.5 * max(joint_vel_flat().abs().median().item(), 1e-3) + 0.05, "DOF map disagrees with velocity layout"
print("② OK: names + DOF map")
act_links = torch.tensor(joint_links, device="cuda")
first_col = torch.tensor([int((cols == asm[k]).nonzero()[0]) for k in act_links.tolist()], device="cuda")  # joint index of each link's first DOF

# ---------------- ① (b) external generalized force ----------------
tau = torch.as_tensor(st.external_gen_forces_cuda(), device="cuda")   # (D, NB)
st.reset_envs(be, list(range(NB)), [[0, 0, 0]] * NB, [0.0] * (NB * D)); be.synchronize()
k_j = int(first_col[0]); k_dof = int(cols[k_j])   # joint index -> generalized DOF column
tau.zero_(); tau[k_dof, :] = 40.0                 # torque on one DOF, all envs
v_before = joint_vel_flat()[:, k_j].clone()
for _ in range(5): pipe.simulate_headless(be, st, None)
be.synchronize()
v_after = joint_vel_flat()[:, k_j]
tau.zero_()
print(f"① (b) effort on dof {k_dof}: v {v_before.mean():+.3f} -> {v_after.mean():+.3f} rad/s")
assert (v_after - v_before).mean() > 0.05, "external generalized force did not accelerate the DOF"
print("① (b) OK: zero-copy effort input")

# DOF map vs velocity layout, now that there is motion
qa = joint_pos_flat().clone()
pipe.simulate_headless(be, st, None); be.synchronize()
qb, vb = joint_pos_flat(), joint_vel_flat()
e = ((qb - qa) / DT - vb).abs()
print(f"layout under motion: median |dq/dt - v| = {e.median():.4f}, max = {e.max():.4f}, median |v| = {vb.abs().median():.4f}")
assert e.max() < 0.25 * vb.abs().max().item() + 0.05, "flat DOF map disagrees with velocity layout under motion"
print("② OK (under motion): DOF map matches velocity layout")

# ---------------- ① (c) batched reset with offsets ----------------
for _ in range(100): pipe.simulate_headless(be, st, None)
be.synchronize()
root0 = ws[0, lay["WS_LTW"] + 1, :, :3].clone()   # root translation per env before reset
st.reset_envs(be, [0, 2], [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], [0.0] * (2 * D)); be.synchronize()
root1 = ws[0, lay["WS_LTW"] + 1, :, :3]
print("root xy after reset(env0 +1x, env2 +2y):", [[round(v, 2) for v in r] for r in root1[:, :2].tolist()])
assert abs(root1[0, 0] - 1.0) < 0.05 and abs(root1[2, 1] - 2.0) < 0.05, "offset reset did not land on the requested origin"
assert torch.allclose(root1[1], root0[1]) and torch.allclose(root1[3], root0[3]), "non-reset envs must be untouched"
print("① (c) OK: batched template reset")
# ---------------- ① (a) position targets ----------------
axis_of = torch.tensor([free_axes[int(k)][0] for k in act_links.tolist()], device="cuda")
groups = {}
for a in sorted(set(axis_of.tolist())):
    ids = act_links[axis_of == a].tolist()
    gid, view = st.motor_target_group(be, [int(i) for i in ids], int(a))
    groups[a] = (gid, ids, torch.as_tensor(view, device="cuda"))   # (n_links, NB)
print(f"motor target groups by axis: { {a: len(g[1]) for a, g in groups.items()} }")
# nv_humanoid uses MJCF <motor> actuators, so rapier-mjcf leaves PD gains at zero:
# set force-based PD gains once (Isaac's ImplicitActuatorCfg semantics), BEFORE scattering targets.
for a, (gid, ids, view) in groups.items():
    st.set_motor_gains(be, [int(i) for i in ids], int(a), 60.0, 4.0, 150.0)
target = 0.3
for a, (gid, ids, view) in groups.items():
    view.fill_(target)
    st.scatter_motor_targets(be, gid)
for _ in range(200): pipe.simulate_headless(be, st, None)
be.synchronize()
q = joint_pos_flat()
# DOF column of each actuated link's first DOF
tracked = q[:, first_col]
print(f"after 200 steps toward target {target}: mean |q - target| over actuated DOFs = {(tracked - target).abs().mean():.3f} rad")
assert (tracked - target).abs().mean() < (q0[:, first_col] - target).abs().mean(), "PD targets did not pull joints toward target"
print("① (a) OK: batched position targets")

print("WRITE PATH OK")
