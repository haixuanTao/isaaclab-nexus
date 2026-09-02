#!/usr/bin/env python3
"""Refined family classification + idle-gap distribution + launch-rate stats."""
import sqlite3, sys
from collections import defaultdict
import statistics as st

db = sys.argv[1]
con = sqlite3.connect(db); cur = con.cursor()
S = {i: v for i, v in cur.execute("SELECT id, value FROM StringIds")}
rows = list(cur.execute(
    "SELECT start, end, demangledName, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"))

per = defaultdict(lambda: [0, 0])
for s, e, dn, sn in rows:
    nm = S.get(dn) or S.get(sn) or f"id{dn}"
    per[nm][0] += e - s; per[nm][1] += 1

iv = sorted((r[0], r[1]) for r in rows)
busy = 0; cs, ce = iv[0]
for s, e in iv[1:]:
    if s > ce: busy += ce - cs; cs, ce = s, e
    else: ce = max(ce, e)
busy += ce - cs
wall = iv[-1][1] - iv[0][0]

def fam(nm):
    n = nm.lower()
    if any(k in n for k in ("arti", "tgs", "solveblock", "solvecontact", "constraint",
                            "updatebodies", "computeunconstrained", "markactiveslab",
                            "integratecore", "solverstep", "px", "rigid")):
        return "PhysX: articulation/solver"
    if any(k in n for k in ("nphase", "narrowphase", "broadphase", "aggpair", "midphase",
                            "convex", "trimesh", "aabb", "sorttriangle", "contactgen",
                            "bpsap", "pairmanagement", "postprocess")):
        return "PhysX: collision (BP/NP)"
    if any(k in n for k in ("cutlass", "gemm", "elu", "addmm", "tanh")):
        return "NN (GEMM/activations)"
    if any(k in n for k in ("raycast", "warp", "wp_", "mesh_")):
        return "Warp (height scan / raycast)"
    if any(k in n for k in ("reduce", "elementwise", "copy", "cat_", "index", "gather",
                            "scatter", "fill", "memset", "norm", "clamp", "mul", "add",
                            "sum", "where", "compare", "unrolled")):
        return "torch elementwise/reduce/index"
    return "unclassified"

F = defaultdict(int); FC = defaultdict(int)
unc = []
for nm, (ns, c) in per.items():
    f = fam(nm); F[f] += ns; FC[f] += c
    if f == "unclassified": unc.append((ns, c, nm))

print(f"window wall {wall/1e6:.1f} ms | GPU busy {busy/1e6:.1f} ms ({100*busy/wall:.1f}%) "
      f"| idle {(wall-busy)/1e6:.1f} ms ({100*(wall-busy)/wall:.1f}%)")
print(f"kernel launches {len(rows):,}  -> {len(rows)/(wall/1e9):,.0f} launches/s")
print(f"mean kernel duration {sum(e-s for s,e in iv)/len(iv)/1e3:.1f} us\n")

print(f"{'ms':>10} {'%busy':>7} {'%wall':>7} {'count':>10}  family")
for f, ns in sorted(F.items(), key=lambda x: -x[1]):
    print(f"{ns/1e6:10.1f} {100*ns/busy:6.1f}% {100*ns/wall:6.1f}% {FC[f]:10,d}  {f}")

print(f"\ntop unclassified:")
for ns, c, nm in sorted(unc, reverse=True)[:12]:
    print(f"{ns/1e6:10.2f} {c:9,d}  {nm[:88]}")

# gap distribution
gaps = []; prev = iv[0][1]
for s, e in iv[1:]:
    if s > prev: gaps.append(s - prev)
    prev = max(prev, e)
gaps.sort()
tot = sum(gaps)
print(f"\n=== GPU idle gap distribution ({len(gaps):,} gaps, {tot/1e6:.1f} ms total) ===")
print(f"mean {tot/len(gaps)/1e3:.1f} us | median {gaps[len(gaps)//2]/1e3:.1f} us | "
      f"p90 {gaps[int(len(gaps)*.9)]/1e3:.1f} us | p99 {gaps[int(len(gaps)*.99)]/1e3:.1f} us | max {gaps[-1]/1e6:.2f} ms")
for lo, hi, lab in [(0,5e3,"<5us"),(5e3,20e3,"5-20us"),(20e3,100e3,"20-100us"),
                    (100e3,1e6,"0.1-1ms"),(1e6,1e12,">1ms")]:
    sel = [g for g in gaps if lo <= g < hi]
    if sel:
        print(f"  {lab:>9}: {len(sel):9,d} gaps  {sum(sel)/1e6:9.1f} ms  ({100*sum(sel)/tot:5.1f}% of idle)")
