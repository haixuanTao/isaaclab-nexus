"""End-to-end proof: Nexus (fork, CUDA backend) -> torch, zero copies.

Builds 4 envs x 3-link revolute chains, steps under gravity, and reads joint
state through `__cuda_array_interface__` views that alias the simulator's
memory. Validates the SoA decode against an independent quantity (the joint
angle recovered from the link-to-world quaternion) and proves the tensor is a
live view (no re-fetch between steps, values still move).
"""
import torch
from nexus3d import (
    NexusBackend, NexusState, NexusPipeline,
    RigidBodyBuilder, ColliderBuilder, RevoluteJointBuilder, Vec3,
)

NENV, NLINK = 4, 3
be = NexusBackend("cuda")
assert be.is_cuda(), "backend is not CUDA"

state = NexusState()
for e in range(NENV):
    env = 0 if e == 0 else state.add_environment()
    parent = state.insert_rigid_body_in(
        env, RigidBodyBuilder.fixed().translation(Vec3(0, 0, 0)).build(),
        ColliderBuilder.cuboid(0.1, 0.1, 0.1).build())
    for i in range(NLINK):
        child = state.insert_rigid_body_in(
            env, RigidBodyBuilder.dynamic().translation(Vec3(0, 0, 0.5 * (i + 1))).build(),
            ColliderBuilder.cuboid(0.1, 0.1, 0.2).build())
        state.insert_multibody_joint_in(
            env, parent, child,
            RevoluteJointBuilder.new(Vec3(1, 0, 0))
            .local_anchor1(Vec3(0, 0, 0.25)).local_anchor2(Vec3(0, 0, -0.25)))
        parent = child
state.finalize_headless(be)

lay = state.ws_layout()
print("layout:", {k: lay[k] for k in ("WS_COORDS", "WS_LTW", "WS_JOINT_VEL", "WS_QUADS",
                                       "links_per_batch", "dofs_per_batch", "num_batches")})

# ---- zero-copy views -------------------------------------------------------
view = state.links_workspace_cuda()
ws = torch.as_tensor(view, device="cuda")            # aliases simulator memory
assert ws.data_ptr() == view.ptr, "torch copied the buffer"
assert ws.dtype == torch.float32 and ws.shape == tuple(view.shape)
print("links_workspace view:", tuple(ws.shape), "data_ptr == nexus ptr:", ws.data_ptr() == view.ptr)

dv = state.dof_state_cuda()
dof = torch.as_tensor(dv, device="cuda")
assert dof.data_ptr() == dv.ptr
print("dof_state view      :", tuple(dof.shape))

C, LTW, JV = lay["WS_COORDS"], lay["WS_LTW"], lay["WS_JOINT_VEL"]

def joint_pos():        # (num_batches, links, 4): generalized coords, first 4 slots
    return ws[:, C, :, :].permute(1, 0, 2)

def link_pose_w():      # (num_batches, links, 7): x y z qx qy qz qw
    rot = ws[:, LTW, :, :]          # quad LTW   = rotation, stored x y z w
    tr = ws[:, LTW + 1, :, :3]      # quad LTW+1 = translation
    return torch.cat([tr, rot], dim=-1).permute(1, 0, 2)

def joint_vel_w():      # (num_batches, links, 6): linear then angular
    return torch.cat([ws[:, JV, :, :3], ws[:, JV + 1, :, :3]], dim=-1).permute(1, 0, 2)

pipe = NexusPipeline()
be.synchronize()
q0 = joint_pos().clone()
p0 = link_pose_w().clone()

for _ in range(60):
    pipe.simulate_headless(be, state, None)
be.synchronize()

q1, p1, v1 = joint_pos(), link_pose_w(), joint_vel_w()

# ---- validation ------------------------------------------------------------
# revolute about X: coordinate lands in slot 3 (first angular DOF).
coord = q1[:, 1, 3]                                   # link 1, all envs
ang = 2 * torch.atan2(p1[:, 1, 3], p1[:, 1, 6])       # from qx, qw of link 1
print("coord slot3  link1 per env:", [round(x, 5) for x in coord.tolist()])
print("angle-from-quat link1     :", [round(x, 5) for x in ang.tolist()])
err = (coord - ang).abs().max().item()
print(f"max |coord - angle| = {err:.5f} rad")
assert (q1 - q0).abs().max() > 0, "joint_pos did not change under gravity"
assert (p1 - p0).abs().max() > 0, "link_pose did not change under gravity"
assert err < 0.02, "SoA decode disagrees with pose quaternion"

# ---- live-view proof: step again WITHOUT re-fetching any view --------------
snap = q1.clone()
for _ in range(20):
    pipe.simulate_headless(be, state, None)
be.synchronize()
moved = (joint_pos() - snap).abs().max().item()
print(f"after 20 more steps, same tensor object moved by {moved:.5f} rad (live view)")
assert moved > 0
print("ang vel wx link1:", [round(x, 4) for x in v1[:, 1, 3].tolist()])
print("ZERO-COPY PATH OK")
