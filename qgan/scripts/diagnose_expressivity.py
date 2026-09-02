"""Diagnosi: il generatore riesce a rappresentare il bersaglio?

Il ciclo avversariale confonde due cause possibili di un cattivo risultato:
insufficiente espressivita' del circuito, oppure instabilita' e budget di
passi del gioco fra generatore e discriminatore.

Questo script rimuove il discriminatore e ottimizza theta direttamente sulla
MMD, per cui il gradiente e' analitico e il problema e' convesso rispetto ai
campioni generati. Se la MMD scende, la catena circuito-jacobiana e'
corretta e l'architettura ha l'espressivita' necessaria: la causa di un
eventuale fallimento e' allora nel ciclo avversariale.

    python scripts/diagnose_expressivity.py --steps 200 --target modes
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qgan.circuit import QuantumGenerator  # noqa: E402
from qgan.discriminator import Adam  # noqa: E402
from qgan.metrics import mmd_rbf, median_bandwidth, detectability  # noqa: E402


def mmd_and_grad_wrt_fake(X: np.ndarray, G: np.ndarray, sigma: float):
    """MMD^2 (biased) e suo gradiente rispetto ai campioni generati G."""
    g = 1.0 / (2.0 * sigma**2)
    n, m = len(X), len(G)

    def k(A, B):
        d2 = np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :] - 2 * A @ B.T
        return np.exp(-g * np.maximum(d2, 0.0))

    Kgg, Kxg = k(G, G), k(X, G)
    val = Kgg.mean() - 2.0 * Kxg.mean()
    # d/dG_b: kernel gaussiano, dk(a,b)/db = -2g (b-a) k(a,b)
    dG = np.zeros_like(G)
    # (G[None,:,:] - G[:,None,:])[i,j] = G_j - G_i, cioe' la differenza vista
    # dal campione j rispetto al quale si deriva. L'ordine opposto inverte il
    # segno del termine repulsivo e fa collassare il generatore sulla media.
    dG += (-2 * g) * ((Kgg[:, :, None] * (G[None, :, :] - G[:, None, :])).sum(0)) * 2 / (m * m)
    dG -= (-2 * g) * ((Kxg[:, :, None] * (G[None, :, :] - X[:, None, :])).sum(0)) * 2 / (n * m)
    return float(val), dG


def make_target(kind, n, rng):
    if kind == "unimodal":
        return np.clip(rng.normal(0.45, 0.08, (n, 5)), 0, 1), None
    centers = np.array(
        [
            [0.20, 0.20, 0.75, 0.30, 0.50],
            [0.75, 0.25, 0.25, 0.70, 0.20],
            [0.30, 0.80, 0.40, 0.20, 0.80],
            [0.70, 0.70, 0.70, 0.65, 0.55],
        ]
    )
    m = rng.integers(0, 4, n)
    return np.clip(centers[m] + rng.normal(0, 0.05, (n, 5)), 0, 1), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--target", default="modes", choices=["modes", "unimodal"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/expressivity.json")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    X, modes = make_target(a.target, a.n, rng)
    sigma = median_bandwidth(X, rng=rng)

    gen = QuantumGenerator(d_c=0)
    theta = rng.uniform(-0.1, 0.1, gen.n_params)
    opt = Adam([theta], lr=a.lr)

    hist = []
    t0 = time.time()
    for s in range(a.steps):
        Z = rng.uniform(0, np.pi, (a.batch, 5))
        G, J = gen.forward_and_jacobian(theta, Z, None)
        Xb = X[rng.choice(len(X), a.batch, replace=False)]
        val, dG = mmd_and_grad_wrt_fake(Xb, G, sigma)
        grad = np.einsum("bq,bqp->p", dG, J)
        opt.step([grad])
        hist.append({"step": s, "mmd_batch": val, "grad_var": float(np.var(grad))})
        if s % 20 == 0 or s == a.steps - 1:
            Gfull = gen.forward(theta, rng.uniform(0, np.pi, (256, 5)), None)
            full = mmd_rbf(X, Gfull, sigma)
            hist[-1]["mmd_full"] = full
            print(
                f"  step {s:4d}  MMD={full:.5f}  Var(grad)={np.var(grad):.2e}  "
                f"{time.time()-t0:.0f}s"
            )

    Gfin = gen.forward(theta, rng.uniform(0, np.pi, (len(X), 5)), None)
    out = {
        "target": a.target,
        "mmd_finale": mmd_rbf(X, Gfin, sigma),
        "detectability_auc": detectability(X, Gfin, seed=a.seed),
        "sigma": sigma,
        "media_reale": X.mean(0).tolist(),
        "media_generata": Gfin.mean(0).tolist(),
        "std_reale": X.std(0).tolist(),
        "std_generata": Gfin.std(0).tolist(),
        "history": hist,
    }
    print(f"\nMMD finale {out['mmd_finale']:.5f}  "
          f"detectability {out['detectability_auc']:.3f}")
    print(f"media reale     {np.round(X.mean(0),3)}")
    print(f"media generata  {np.round(Gfin.mean(0),3)}")
    print(f"std reale       {np.round(X.std(0),3)}")
    print(f"std generata    {np.round(Gfin.std(0),3)}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
