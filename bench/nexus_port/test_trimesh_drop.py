"""Does a dynamic ball rest on a fixed TRIMESH on the fork's CUDA backend? Control: cuboid.
Builder-only scenes default to Y-up; gravity is set to -Z explicitly (the MJCF loader does this itself)."""
import torch
from nexus3d import NexusBackend, NexusState, NexusPipeline, RigidBodyBuilder, ColliderBuilder, Vec3
be = NexusBackend("cuda")
def run(kind):
    st = NexusState()
    v = [(-2, -2, 0.0), (2, -2, 0.0), (2, 2, 0.0), (-2, 2, 0.0)]
    if kind == "trimesh":           c = ColliderBuilder.trimesh(v, [(0, 1, 2), (0, 2, 3)]).build(); z = 0.0
    elif kind == "trimesh_flipped": c = ColliderBuilder.trimesh(v, [(0, 2, 1), (0, 3, 2)]).build(); z = 0.0
    else:                           c = ColliderBuilder.cuboid(2, 2, 0.05).build(); z = -0.05
    st.insert_rigid_body_in(0, RigidBodyBuilder.fixed().translation(Vec3(0, 0, z)).build(), c)
    st.insert_rigid_body_in(0, RigidBodyBuilder.dynamic().translation(Vec3(0.3, 0.1, 1.0)).build(), ColliderBuilder.ball(0.2).build())
    st.set_rbd_dt(1 / 200); st.finalize_headless(be); st.set_rbd_gravity_headless(be, Vec3(0, 0, -9.81)); be.synchronize()
    poses = torch.as_tensor(st.body_poses_cuda(), device="cuda")
    pipe = NexusPipeline()
    for _ in range(400): pipe.simulate_headless(be, st, None)
    be.synchronize(); return poses.clone()
for kind in ("cuboid", "trimesh", "trimesh_flipped"):
    p = run(kind); tr = p[1, 4:7]
    print(f"{kind:16s} ball final pos = {[round(x,3) for x in tr.tolist()]}   (resting => z ~ 0.20; fell through => z << 0)")
