"""③ contacts via the engine's contact force sensor (MAX_CONTACT_SENSORS raised to 32):
sense EVERY link, drop the humanoid onto a flat trimesh, let it settle, and check that the
summed normal impulse / dt over all links equals its weight (mass from the MJCF via mujoco)."""
import re, torch, mujoco, numpy as np
from nexus3d import NexusBackend, NexusState, NexusPipeline, RigidBodyBuilder, ColliderBuilder, Vec3
MJCF = "/workspace/WBC-AGILE/.venv/lib/python3.12/site-packages/newton/examples/assets/nv_humanoid.xml"
NENV, DT, G, ITERS = 4, 1 / 200, 9.81, 4   # ITERS: engine reports the per-solver-iteration impulse
xml = re.sub(r"<sensor>.*?</sensor>", "", open(MJCF).read(), flags=re.S)
m_total = float(mujoco.MjModel.from_xml_string(xml).body_subtreemass[1])

def grid(n, half=4.0):
    xs = np.linspace(-half, half, n + 1); X, Y = np.meshgrid(xs, xs, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1); F = []
    for i in range(n):
        for j in range(n): a = i * (n + 1) + j; F += [(a, a + 1, a + n + 2), (a, a + n + 2, a + n + 1)]
    return [tuple(map(float, v)) for v in V], F

be = NexusBackend("cuda"); st = NexusState(); v, f = grid(8)
for e in range(NENV):
    env = 0 if e == 0 else st.add_environment()
    st.insert_rigid_body_in(env, RigidBodyBuilder.fixed().build(), ColliderBuilder.trimesh(v, f).build())
    st.insert_mjcf_headless(MJCF, env, None, False)
st.set_rbd_collisions_capacity(256); st.set_rbd_dt(DT); st.set_rbd_solver_iterations(ITERS); st.finalize_headless(be); st.set_rbd_gravity_headless(be, Vec3(0, 0, -9.81)); be.synchronize()
lay = st.ws_layout(); L, NB = lay["links_per_batch"], lay["num_batches"]
ws = torch.as_tensor(st.links_workspace_cuda(), device="cuda")
ws[0, lay["WS_COORDS"], :, :3] = torch.tensor([0.0, 0.0, 1.4], device="cuda"); be.synchronize()
n = st.set_contact_sensor_links(be, list(range(L)))
out = torch.as_tensor(st.contact_sensor_out_cuda(), device="cuda")            # (mbs, NB, MAX)
print(f"sensors accepted: {n}/{L} links | out shape {tuple(out.shape)} | weight {m_total*G:.1f} N")
pipe = NexusPipeline()
for _ in range(600): pipe.simulate_headless(be, st, None)                       # 3 s: fall + settle
be.synchronize()
F = out[0, :, :L] * ITERS / DT                                                # (NB, L) normal force per link
tot = F.sum(1)
print("total sensed normal force per env:", [round(x, 1) for x in tot.tolist()], "N")
print("ratio to weight:", [round(x, 3) for x in (tot / (m_total * G)).tolist()], "| links loaded per env:", (F > 1.0).sum(1).tolist())
print("min link z per env:", [round(x, 3) for x in ws[:, lay["WS_LTW"] + 1, :, 2].min(0).values.tolist()])
ratio = tot / (m_total * G)
assert (ratio > 0.85).all() and (ratio < 1.15).all(), "sensed contact forces do not balance weight"
print("③ contact sensor OK: summed normal forces balance weight within 15%")
