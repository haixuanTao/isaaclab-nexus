#!/usr/bin/env python3
"""CUDA API (host) time + per-iteration segmentation using PPO update GEMM bursts."""
import sqlite3, sys
from collections import defaultdict
import statistics as st

con = sqlite3.connect(sys.argv[1]); cur = con.cursor()
S = {i: v for i, v in cur.execute("SELECT id, value FROM StringIds")}
T = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}

# ---------- host-side CUDA API ----------
if "CUPTI_ACTIVITY_KIND_RUNTIME" in T:
    api = defaultdict(lambda: [0, 0])
    for s, e, nid in cur.execute(
            "SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        api[S.get(nid, str(nid))][0] += e - s; api[S.get(nid, str(nid))][1] += 1
    tot = sum(v[0] for v in api.values())
    span = None
    r = list(cur.execute("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_RUNTIME"))[0]
    span = r[1] - r[0]
    print(f"=== host CUDA API time: {tot/1e6:.1f} ms over {span/1e6:.1f} ms span "
          f"({100*tot/span:.1f}% of one host thread) ===")
    print(f"{'ms':>10} {'count':>10} {'us/call':>9}  api")
    for nm, (ns, c) in sorted(api.items(), key=lambda x: -x[1][0])[:14]:
        print(f"{ns/1e6:10.1f} {c:10,d} {ns/c/1e3:9.2f}  {nm[:60]}")

# ---------- per-iteration segmentation ----------
rows = list(cur.execute(
    "SELECT start, end, demangledName, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"))
def nm(dn, sn): return S.get(dn) or S.get(sn) or ""

# PPO update = bursts of cutlass GEMM. Find contiguous cutlass clusters.
gem = [(s, e) for s, e, dn, sn in rows if "cutlass" in nm(dn, sn)]
bursts = []
if gem:
    bs, be = gem[0]
    for s, e in gem[1:]:
        if s - be > 50_000_000:      # >50 ms apart -> new burst
            bursts.append((bs, be)); bs, be = s, e
        else: be = max(be, e)
    bursts.append((bs, be))
print(f"\n=== PPO update bursts (cutlass GEMM clusters): {len(bursts)} ===")
if len(bursts) >= 2:
    per_it = [bursts[i+1][0] - bursts[i][0] for i in range(len(bursts)-1)]
    upd = [e - s for s, e in bursts]
    print(f"iteration period : median {st.median(per_it)/1e6:8.1f} ms  (n={len(per_it)})")
    print(f"update burst span: median {st.median(upd)/1e6:8.1f} ms "
          f"({100*st.median(upd)/st.median(per_it):.1f}% of iteration)")
    print(f"rollout span     : median {(st.median(per_it)-st.median(upd))/1e6:8.1f} ms "
          f"({100*(st.median(per_it)-st.median(upd))/st.median(per_it):.1f}% of iteration)")

    # busy/idle within one representative iteration
    a, b = bursts[len(bursts)//2][0], bursts[len(bursts)//2 + 1][0]
    seg = sorted((s, e) for s, e, _, _ in rows if s >= a and e <= b)
    if seg:
        busy = 0; cs, ce = seg[0]
        for s, e in seg[1:]:
            if s > ce: busy += ce - cs; cs, ce = s, e
            else: ce = max(ce, e)
        busy += ce - cs
        wall = b - a
        print(f"\n--- one representative iteration ({wall/1e6:.1f} ms) ---")
        print(f"GPU busy {busy/1e6:8.1f} ms ({100*busy/wall:.1f}%)   "
              f"idle {(wall-busy)/1e6:8.1f} ms ({100*(wall-busy)/wall:.1f}%)   "
              f"launches {len(seg):,}")
