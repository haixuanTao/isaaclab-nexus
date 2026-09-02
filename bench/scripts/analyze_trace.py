#!/usr/bin/env python3
"""Summarize an nsys sqlite export: GPU busy vs wall, kernel families, API gaps.

Usage: analyze_trace.py <trace.sqlite> [--window-start-ns N] [--window-end-ns N]
"""
import sqlite3, sys, argparse
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("db")
ap.add_argument("--top", type=int, default=25)
a = ap.parse_args()

con = sqlite3.connect(a.db)
cur = con.cursor()

def tables():
    return {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
T = tables()

def strings():
    m = {}
    if "StringIds" in T:
        for i, v in cur.execute("SELECT id, value FROM StringIds"):
            m[i] = v
    return m
S = strings()

# ---- GPU kernels
kern_tbl = "CUPTI_ACTIVITY_KIND_KERNEL"
rows = []
if kern_tbl in T:
    rows = list(cur.execute(
        f"SELECT start, end, demangledName, shortName FROM {kern_tbl} ORDER BY start"))
if not rows:
    print("No kernel rows found. Tables:", sorted(T)[:40]); sys.exit(1)

t0 = min(r[0] for r in rows); t1 = max(r[1] for r in rows)
wall = t1 - t0

# busy time = union of kernel intervals (kernels can overlap across streams)
iv = sorted((r[0], r[1]) for r in rows)
busy = 0; cs, ce = iv[0]
for s, e in iv[1:]:
    if s > ce: busy += ce - cs; cs, ce = s, e
    else: ce = max(ce, e)
busy += ce - cs

# sum of kernel durations (can exceed busy if overlapping)
ksum = sum(e - s for s, e in iv)

per = defaultdict(lambda: [0, 0])   # name -> [total_ns, count]
for s, e, dn, sn in rows:
    nm = S.get(dn) or S.get(sn) or f"id{dn}"
    per[nm][0] += e - s; per[nm][1] += 1

# ---- memcpy
memcpy_ns = memcpy_n = 0
if "CUPTI_ACTIVITY_KIND_MEMCPY" in T:
    for s, e in cur.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
        memcpy_ns += e - s; memcpy_n += 1

print(f"=== window ===")
print(f"wall (first kernel start -> last kernel end) : {wall/1e6:10.2f} ms")
print(f"GPU busy (union of kernel intervals)        : {busy/1e6:10.2f} ms  ({100*busy/wall:5.1f}% of wall)")
print(f"GPU idle                                    : {(wall-busy)/1e6:10.2f} ms  ({100*(wall-busy)/wall:5.1f}% of wall)")
print(f"sum of kernel durations (overlap counted)   : {ksum/1e6:10.2f} ms")
print(f"kernel launches                             : {len(rows)}")
print(f"memcpy                                      : {memcpy_ns/1e6:10.2f} ms over {memcpy_n} ops")

print(f"\n=== top {a.top} kernels by total GPU time ===")
print(f"{'ms':>10} {'%busy':>7} {'count':>8}  name")
for nm, (ns, c) in sorted(per.items(), key=lambda x: -x[1][0])[:a.top]:
    print(f"{ns/1e6:10.2f} {100*ns/busy:6.1f}% {c:8d}  {nm[:95]}")

# ---- classify into families
FAM = [
    ("physx/solver",  ("physx","Physx","PxgSolver","artiSolve","solveContact","tgs","Tgs","constraint","Constraint")),
    ("physx/bp-np",   ("broadphase","BroadPhase","narrowphase","NarrowPhase","Bp","Np","contactGen","aabb")),
    ("physx/articul", ("articul","Articul","featherstone","Cache")),
    ("nn/gemm",       ("gemm","Gemm","GEMM","cutlass","sm90","sm100","ampere","volta","nn_","addmm","elementwise_kernel")),
    ("warp/isaaclab", ("wp_","warp","Kernel_")),
    ("reduce/copy",   ("reduce","Reduce","copy","Copy","memset","fill","cat_","index")),
]
fam = defaultdict(int)
for nm, (ns, c) in per.items():
    hit = "other"
    for f, keys in FAM:
        if any(k in nm for k in keys): hit = f; break
    fam[hit] += ns
print(f"\n=== by family (heuristic name match) ===")
for f, ns in sorted(fam.items(), key=lambda x: -x[1]):
    print(f"{ns/1e6:10.2f} ms {100*ns/busy:6.1f}%  {f}")

# ---- largest GPU idle gaps (host-bound evidence)
gaps = []
prev_end = iv[0][1]
for s, e in iv[1:]:
    if s > prev_end: gaps.append((s - prev_end, prev_end))
    prev_end = max(prev_end, e)
gaps.sort(reverse=True)
print(f"\n=== top 15 GPU idle gaps (host-bound evidence) ===")
print(f"{'ms':>10}  at_offset_ms")
for g, at in gaps[:15]:
    print(f"{g/1e6:10.3f}  {(at-t0)/1e6:12.2f}")
tot_gap = sum(g for g, _ in gaps)
print(f"total idle in {len(gaps)} gaps: {tot_gap/1e6:.2f} ms")
