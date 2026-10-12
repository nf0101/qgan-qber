from collections import Counter
from BB84_crn import run_bb84

_, rec = run_bb84(intercept_rate=0.0, verbose=False)

print("p1:", Counter(r["p1"] for r in rec))
print("p2:", Counter(r["p2"] for r in rec))
print("det:", sum(r["det_flip"] for r in rec) / len(rec))