"""Cap the G1's collision hulls at N vertices (PhysX's default limit is 64).

The unitree MJCF collides full visual STLs (mean convex hull: 1087 verts, pelvis
5583). Rapier's MeshConverter::ConvexHull keeps every one of them, so the GPU
narrow phase clips features against hulls an order of magnitude heavier than the
ones PhysX's G1 uses. This writes a copy of the MJCF whose COLLIDING mesh assets
are replaced by support-mapped hulls of at most N vertices.

usage: make_convex_mjcf.py [n_dirs]
"""
import os, sys, shutil
import xml.etree.ElementTree as ET
import numpy as np, trimesh

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
SRC = "/workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
SRCD = os.path.dirname(SRC)
OUT = f"/workspace/bench/nexus_port/g1_29dof_convex{N}.xml"
OUTD = f"/workspace/bench/nexus_port/meshes_convex{N}"
os.makedirs(OUTD, exist_ok=True)

def fibonacci_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)

tree = ET.parse(SRC); root = tree.getroot()
MESHDIR = next((c.get("meshdir") for c in root.iter("compiler") if c.get("meshdir")), "")
colliding = {g.get("mesh") for g in root.iter("geom") if g.get("contype") != "0" and g.get("mesh")}
dirs = fibonacci_dirs(N)
stats = []
for asset in root.iter("mesh"):
    name = asset.get("name") or os.path.splitext(os.path.basename(asset.get("file")))[0]
    if name not in colliding:
        continue
    src = os.path.join(SRCD, MESHDIR, asset.get("file"))
    if not os.path.exists(src):
        src = os.path.join(SRCD, "meshes", os.path.basename(asset.get("file")))
    m = trimesh.load(src, process=False)
    V = np.asarray(m.vertices, dtype=np.float64)
    keep = np.unique(np.argmax(V @ dirs.T, axis=0))          # support point per direction
    hull = trimesh.Trimesh(vertices=V[keep], process=False).convex_hull
    dst = os.path.join(OUTD, os.path.basename(asset.get("file")))
    hull.export(dst)
    asset.set("file", dst)
    stats.append((name, len(m.convex_hull.vertices), len(hull.vertices)))

# meshdir (if any) must not be re-applied to our absolute/relative rewrite
if MESHDIR:
    for asset in root.iter("mesh"):
        name = asset.get("name") or os.path.splitext(os.path.basename(asset.get("file")))[0]
        if name not in colliding:
            asset.set("file", os.path.join(SRCD, MESHDIR, asset.get("file")))
    for c in root.iter("compiler"):
        c.attrib.pop("meshdir", None)
tree.write(OUT)
before = sum(s[1] for s in stats); after = sum(s[2] for s in stats)
print(f"rewrote {len(stats)} colliding meshes -> {OUT}")
print(f"hull vertices total {before} -> {after}  ({before/max(after,1):.1f}x fewer), "
      f"max per hull {max(s[1] for s in stats)} -> {max(s[2] for s in stats)}")
