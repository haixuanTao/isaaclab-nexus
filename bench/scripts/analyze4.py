#!/usr/bin/env python3
"""Segment rollout vs PPO update by presence/absence of PhysX kernels."""
import sqlite3, sys, statistics as st

con = sqlite3.connect(sys.argv[1]); cur = con.cursor()
S = {i: v for i, v in cur.execute("SELECT id, value FROM StringIds")}
rows = list(cur.execute(
    "SELECT start, end, demangledName, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"))

def isphysx(n):
    n = n.lower()
    return any(k in n for k in ("arti", "tgs", "solveblock", "nphase", "aggpair", "convex",
                                "trimesh", "midphase", "contactmanager", "lostfound",
                                "updatebodies", "computeunconstrained", "computemassmatrix",
                                "selfcollision", "aabb", "sorttriangle", "transformcache",
                                "aggregateprojection", "markactiveslab", "rigid", "finishcontacts"))
ev = []
for s, e, dn, sn in rows:
    n = S.get(dn) or S.get(sn) or ""
    ev.append((s, e, isphysx(n)))

# physx timeline -> find gaps in PHYSX activity > 15 ms = PPO update windows
px = sorted((s, e) for s, e, p in ev if p)
holes = []; prev = px[0][1]
for s, e in px[1:]:
    if s - prev > 15_000_000: holes.append((prev, s))
    prev = max(prev, e)
print(f"physx-free windows >15ms: {len(holes)}")
if holes:
    d = [b - a for a, b in holes]
    print(f"  median {st.median(d)/1e6:.1f} ms | min {min(d)/1e6:.1f} | max {max(d)/1e6:.1f}"
          f" | total {sum(d)/1e6:.1f} ms")

# iteration boundaries = start of each update window
bounds = [a for a, b in holes]
if len(bounds) >= 3:
    per = [bounds[i+1] - bounds[i] for i in range(len(bounds)-1)]
    print(f"\niteration period: median {st.median(per)/1e6:.1f} ms (n={len(per)})")
    upd = st.median([b - a for a, b in holes])
    print(f"  update  : {upd/1e6:8.1f} ms  ({100*upd/st.median(per):.1f}%)")
    print(f"  rollout : {(st.median(per)-upd)/1e6:8.1f} ms  ({100*(st.median(per)-upd)/st.median(per):.1f}%)")

    def busyfrac(a, b):
        seg = sorted((s, e) for s, e, _ in ev if s >= a and e <= b)
        if not seg: return 0, 0, 0
        bu = 0; cs, ce = seg[0]
        for s, e in seg[1:]:
            if s > ce: bu += ce - cs; cs, ce = s, e
            else: ce = max(ce, e)
        bu += ce - cs
        return bu, b - a, len(seg)

    # representative iteration
    i = len(bounds)//2
    a, b = bounds[i], bounds[i+1]
    ua, ub = holes[i]
    bu, wl, nk = busyfrac(a, b)
    print(f"\n--- representative iteration: {wl/1e6:.1f} ms, {nk:,} launches ---")
    print(f"  GPU busy {bu/1e6:7.1f} ms ({100*bu/wl:.1f}%)  idle {(wl-bu)/1e6:7.1f} ms ({100*(wl-bu)/wl:.1f}%)")
    bu2, wl2, nk2 = busyfrac(ua, ub)
    print(f"  update window : {wl2/1e6:6.1f} ms, busy {100*bu2/wl2:.1f}%, {nk2:,} launches")
    bu3, wl3, nk3 = busyfrac(ub, b)
    print(f"  rollout window: {wl3/1e6:6.1f} ms, busy {100*bu3/wl3:.1f}%, {nk3:,} launches")
