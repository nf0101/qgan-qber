"""Verifica empirica del costo, SPEC-FASE2 §7.1 e §13 punto 3.

La specifica riporta 9,22 ms per una valutazione isolata e 1,37 ms per una
valutazione raggruppata, con un gradiente esatto a circa 0,14 s. Lo script
rimisura le stesse quantita' sulla macchina corrente e stima il costo di
un'epoca in funzione della numerosita' del dataset.

    python scripts/benchmark_epoch.py [--repeat 3]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qgan import QGAN, QuantumGenerator  # noqa: E402


def timeit(fn, repeat=3):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--d-c", type=int, default=0)
    ap.add_argument("--out", default="results/benchmark_fase2.json")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    g = QuantumGenerator(d_c=a.d_c)
    P = g.n_params
    print(json.dumps(g.describe(), indent=2))

    # --- valutazione isolata --------------------------------------
    Z1 = rng.uniform(0, np.pi, (1, 5))
    C1 = rng.random((1, a.d_c)) if a.d_c else None
    th = rng.uniform(-1, 1, P)
    t_iso = timeit(lambda: g.forward(th, Z1, C1), 20)
    print(f"\nvalutazione isolata          {t_iso*1000:8.2f} ms")

    # --- valutazione raggruppata ----------------------------------
    for N in (32, 320, 3232):
        Zn = rng.uniform(0, np.pi, (N, 5))
        Cn = rng.random((N, a.d_c)) if a.d_c else None
        t = timeit(lambda: g.forward(th, Zn, Cn), a.repeat)
        print(
            f"raggruppata N={N:5d}          {t*1000:8.1f} ms  "
            f"-> {t/N*1000:6.3f} ms/circ  (fattore {t_iso/(t/N):5.1f})"
        )
        if N == 3232:
            t_grouped = t / N

    # --- gradiente completo ---------------------------------------
    print()
    grad_costs = {}
    for B in (1, 8, a.batch):
        Zb = rng.uniform(0, np.pi, (B, 5))
        Cb = rng.random((B, a.d_c)) if a.d_c else None
        t = timeit(lambda: g.forward_and_jacobian(th, Zb, Cb), a.repeat)
        n_circ = B * (1 + 2 * P)
        grad_costs[B] = t
        print(
            f"gradiente batch={B:3d}  {n_circ:6d} circuiti  {t:7.3f} s  "
            f"({t/n_circ*1000:.3f} ms/circ)"
        )

    # --- costo di un'epoca ----------------------------------------
    print("\ncosto stimato di un'epoca e di un addestramento da 200 epoche:")
    per_batch = grad_costs[a.batch]
    rows = []
    for n in (53, 200, 630, 2000, 5000):
        nb = int(np.ceil(n / a.batch))
        ep = nb * per_batch
        rows.append({"n": n, "batches": nb, "epoch_s": ep, "train_200_min": ep * 200 / 60})
        print(
            f"  n={n:6d}  {nb:4d} batch  {ep:7.2f} s/epoca  "
            f"{ep*200/60:8.1f} min per 200 epoche"
        )

    # --- epoca reale ----------------------------------------------
    n = 128
    X = rng.random((n, 5))
    C = rng.random((n, a.d_c)) if a.d_c else None
    q = QGAN(d_c=a.d_c, epochs=2, batch_size=a.batch, verbose=0, n_monitor=64, seed=0)
    t0 = time.perf_counter()
    q.fit(X, C)
    t_real = (time.perf_counter() - t0) / 2
    print(f"\nepoca reale misurata (n={n}): {t_real:.2f} s  "
          f"[stima: {int(np.ceil(n/a.batch))*per_batch:.2f} s]")

    out = {
        "generator": g.describe(),
        "isolated_ms": t_iso * 1000,
        "grouped_ms": t_grouped * 1000,
        "speedup": t_iso / t_grouped,
        "gradient_s": {str(k): v for k, v in grad_costs.items()},
        "epoch_estimates": rows,
        "epoch_measured_s": t_real,
        "epoch_measured_n": n,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nscritto {a.out}")


if __name__ == "__main__":
    main()
