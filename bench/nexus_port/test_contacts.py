"""③ contacts: per-link net contact force from Nexus's SOLVED rigid contact slab.
Oracle: humanoid mass from the same MJCF via mujoco; at rest, sum_links F_z == m_total * g."""
import re, torch, mujoco
from nexus3d import NexusBackend, NexusState, NexusPipeline
MJCF = "/workspace/WBC-AGILE/.venv/lib/python3.12/site-packages/newton/examples/assets/nv_humanoid.xml"
NENV, DT, G = 4, 1.0 / 200.0, 9.81
xml = re.sub(r"<sensor>.*?</sensor>", "", open(MJCF).read(), flags=re.S)
m_total = float(mujoco.MjModel.from_xml_string(xml).body_subtreemass[1])
print(f"humanoid mass from MJCF: {m_total:.3f} kg -> weight {m_total*G:.1f} N")
be = NexusBackend("cuda"); st = NexusState()
for e in range(NENV): st.insert_mjcf_headless(MJCF, 0 if e == 0 else st.add_environment())
st.set_rbd_dt(DT); st.finalize_headless(be); be.synchronize()
lay = st.ws_layout(); L, NB = lay["links_per_batch"], lay["num_batches"]
rb_of_link = torch.as_tensor(st.links_static_host(be)).long().cuda()[:, 0]
cl = st.rigid_contact_layout()
slab = torch.as_tensor(st.rigid_contacts_cuda(), device="cuda")
clen = torch.as_tensor(st.contacts_len_cuda(), device="cuda").view(torch.int32)
print("rigid slab:", tuple(slab.shape), "| layout:", {k: cl[k] for k in ("stride","off_dir_a","off_solver_body_a","off_solver_body_b","off_len","off_elements","elem_stride","off_elem_normal_impulse","per_batch")})
pipe = NexusPipeline()
for _ in range(400): pipe.simulate_headless(be, st, None)
be.synchronize()
def net_forces_w():
    S = slab.view(NB, cl["per_batch"], cl["stride"]); I = S.view(torch.int32)
    n = clen.long()
    valid = torch.arange(cl["per_batch"], device="cuda")[None, :] < n[:, None]
    ba, bb = I[..., cl["off_solver_body_a"]].long(), I[..., cl["off_solver_body_b"]].long()
    dir_a = S[..., cl["off_dir_a"]:cl["off_dir_a"] + 3]
    nel = I[..., cl["off_len"]].long().clamp(0, cl["max_elements"])
    imp = torch.zeros(NB, cl["per_batch"], device="cuda")
    for e in range(cl["max_elements"]):
        col = cl["off_elements"] + e * cl["elem_stride"] + cl["off_elem_normal_impulse"]
        imp += S[..., col] * (e < nel).float()
    imp = imp * valid.float()
    f_a = dir_a * imp[..., None] / DT
    out = torch.zeros(NB, L, 3, device="cuda")
    for sign, bid in ((1.0, ba), (-1.0, bb)):
        match = (bid[..., None] == rb_of_link[None, None, :])
        out += torch.einsum("npl,npc->nlc", match.float(), sign * f_a)
    return out, int(valid.sum().item()), int(nel[valid].sum().item())
F, n_live, n_pts = net_forces_w()
tot = F.sum(dim=1)
print(f"live constraints: {n_live} ({n_pts} contact points) | contacts_len per env: {clen.tolist()}")
print(f"per-env total contact force z: {[round(v,1) for v in tot[:,2].tolist()]} N | expected {m_total*G:.1f} N")
ratio = tot[:, 2] / (m_total * G)
print(f"ratio: {[round(v,3) for v in ratio.tolist()]} | links loaded per env: {(F[...,2].abs() > 1.0).sum(1).tolist()}")
assert n_live > 0, "no live rigid contact constraints"
assert (ratio > 0.8).all() and (ratio < 1.2).all(), "net contact force does not balance weight"
assert (tot[:, :2].abs() < 0.2 * m_total * G).all(), "large lateral net force"
print("③ contacts OK: net_forces_w balances weight within 20%")
