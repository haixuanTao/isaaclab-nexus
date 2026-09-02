#!/usr/bin/env python3
"""Parse the PhysX-vs-Newton logs in results_newton/ into one comparison table."""
import glob
import os
import re
import statistics as st
import sys

WARM = 5  # drop warmup iterations


def parse(path):
    txt = re.sub(r"\x1b\[[0-9;]*m", "", open(path, errors="ignore").read().replace("\r", "\n"))

    def grab(label):
        return [float(x) for x in re.findall(rf"{label}:\s*([0-9.]+)s?", txt)]

    return {"coll": grab("Collection time"), "learn": grab("Learning time"), "it": grab("Iteration time")}


def med(xs):
    return st.median(xs) if xs else float("nan")


rows = []
pat = sys.argv[1] if len(sys.argv) > 1 else "/workspace/bench/results_newton/train_*_n*.log"
for path in sorted(glob.glob(pat)):
    m = re.search(r"train_(physx|newton(?:-sub\d+)?)_(.+)_n(\d+)\.log$", os.path.basename(path))
    if not m:
        continue
    engine, task, n = m.group(1), m.group(2), int(m.group(3))
    d = parse(path)
    if not d["it"]:
        print(f"!! no iterations parsed: {os.path.basename(path)}")
        continue
    it, co, le = (d[k][WARM:] or d[k] for k in ("it", "coll", "learn"))
    rows.append(
        dict(engine=engine, task=task, envs=n, n_iter=len(it), it=med(it), it_min=min(it),
             coll=med(co), learn=med(le), sps=24 * n / med(it))
    )

rows.sort(key=lambda r: (r["task"], r["envs"], r["engine"] != "physx", r["engine"]))
hdr = f"{'engine':>7} {'envs':>6} {'iters':>6} {'iter_med_s':>11} {'iter_min_s':>11} {'collect_s':>10} {'learn_s':>9} {'coll%':>6} {'env_steps/s':>12}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['engine']:>7} {r['envs']:6d} {r['n_iter']:6d} {r['it']:11.3f} {r['it_min']:11.3f} "
          f"{r['coll']:10.3f} {r['learn']:9.3f} {100 * r['coll'] / r['it']:5.1f}% {r['sps']:12,.0f}")

# speedup, pairing physx/newton at equal (task, envs)
by_key = {(r["task"], r["envs"], r["engine"]): r for r in rows}
pairs = sorted({(t, n) for t, n, _ in by_key})
if any((t, n, "physx") in by_key and (t, n, "newton") in by_key for t, n in pairs):
    print(f"\n{'envs':>6} {'physx iter_s':>13} {'newton iter_s':>14} {'speedup':>9} {'collect speedup':>16}")
    for t, n in pairs:
        p = by_key.get((t, n, "physx"))
        if not p:
            continue
        for eng in sorted({k[2] for k in by_key if k[2].startswith("newton")}):
            w = by_key.get((t, n, eng))
            if not w:
                continue
            print(f"{n:6d} {p['it']:13.3f} {w['it']:14.3f} {p['it'] / w['it']:8.2f}x {p['coll'] / w['coll']:15.2f}x   {eng}")
