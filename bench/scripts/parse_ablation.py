#!/usr/bin/env python3
"""Decompose AGILE's iteration cost from the ablation ladder."""
import re, glob, os, statistics as st

def parse(path, warm=5):
    t = open(path, errors="ignore").read().replace("\r", "\n")
    t = re.sub(r"\x1b\[[0-9;]*m", "", t)
    g = lambda l: [float(x) for x in re.findall(rf"{l}:\s*([0-9.]+)s?", t)]
    it, co, le = g("Iteration time"), g("Collection time"), g("Learning time")
    if len(it) > warm: it, co, le = it[warm:], co[warm:], le[warm:]
    if not it: return None
    return dict(it=st.median(it), co=st.median(co), le=st.median(le), n=len(it))

rows = {}
for m in ("none", "nophysics", "nophysics_noreadback"):
    p = f"/workspace/bench/results/ablate_{m}_n4096.log"
    if os.path.exists(p):
        r = parse(p)
        if r: rows[m] = r

if not rows:
    print("no ablation logs parsed yet"); raise SystemExit

print(f"{'rung':24} {'iter_s':>8} {'collect_s':>10} {'learn_s':>8} {'iters':>6}")
for m, r in rows.items():
    print(f"{m:24} {r['it']:8.3f} {r['co']:10.3f} {r['le']:8.3f} {r['n']:6d}")

if "none" in rows and "nophysics" in rows:
    base, nop = rows["none"], rows["nophysics"]
    phys = base["co"] - nop["co"]
    print(f"\n--- decomposition of the {base['it']:.3f} s iteration ---")
    print(f"  PhysX solve (collect delta)   {phys:7.3f} s  {100*phys/base['it']:5.1f}%")
    if "nophysics_noreadback" in rows:
        nrb = rows["nophysics_noreadback"]
        rb = nop["co"] - nrb["co"]
        print(f"  sensor/state readback         {rb:7.3f} s  {100*rb/base['it']:5.1f}%")
        print(f"  task layer + inference        {nrb['co']:7.3f} s  {100*nrb['co']/base['it']:5.1f}%")
    else:
        print(f"  everything else in rollout    {nop['co']:7.3f} s  {100*nop['co']/base['it']:5.1f}%")
    print(f"  PPO update                    {base['le']:7.3f} s  {100*base['le']/base['it']:5.1f}%")
    print(f"\n  framework share of rollout    {100*nop['co']/base['co']:5.1f}%"
          f"   (rollout without physics / rollout with)")
