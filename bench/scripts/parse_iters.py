#!/usr/bin/env python3
"""Parse rsl_rl training logs -> per-iteration timing table + steady-state stats."""
import re, sys, glob, os, statistics as st

def parse(path):
    txt = open(path, errors="ignore").read().replace("\r", "\n")
    txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
    iters = [int(m) for m in re.findall(r"Learning iteration (\d+)/", txt)]
    def grab(label):
        return [float(x) for x in re.findall(rf"{label}:\s*([0-9.]+)s?", txt)]
    return {
        "iter":   iters,
        "steps":  [int(x) for x in re.findall(r"Total steps:\s*(\d+)", txt)],
        "sps":    grab("Steps per second"),
        "coll":   grab("Collection time"),
        "learn":  grab("Learning time"),
        "ittime": grab("Iteration time"),
    }

rows = []
for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1
                             else "/workspace/bench/results/train_*.log")):
    d = parse(path)
    if not d["ittime"]:
        print(f"!! no iterations parsed: {os.path.basename(path)}"); continue
    n = int(re.search(r"_n(\d+)\.log", path).group(1))
    warm = 5                       # drop warmup iterations
    it = d["ittime"][warm:]; co = d["coll"][warm:]; le = d["learn"][warm:]
    if not it: it, co, le = d["ittime"], d["coll"], d["learn"]
    steps = 24 * n
    rows.append(dict(
        envs=n, n_iter=len(it),
        it_med=st.median(it), it_min=min(it),
        coll=st.median(co) if co else float("nan"),
        learn=st.median(le) if le else float("nan"),
        sps=steps / st.median(it),
    ))

rows.sort(key=lambda r: r["envs"])
print(f"{'envs':>6} {'iters':>6} {'iter_med_s':>11} {'iter_min_s':>11} "
      f"{'collect_s':>10} {'learn_s':>9} {'coll%':>6} {'env_steps/s':>12}")
for r in rows:
    pct = 100 * r["coll"] / r["it_med"] if r["it_med"] else 0
    print(f"{r['envs']:6d} {r['n_iter']:6d} {r['it_med']:11.3f} {r['it_min']:11.3f} "
          f"{r['coll']:10.3f} {r['learn']:9.3f} {pct:5.1f}% {r['sps']:12,.0f}")
