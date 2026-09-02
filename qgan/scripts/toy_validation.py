"""Validazione su dati sintetici, SPEC-FASE2 §13 punto 2.

Tre prove, nessuna delle quali usa i domini reali:

  A. non condizionata, miscela di quattro modalita' in [0,1]^5.
     Verifica che l'MMD scenda, e misura copertura delle modalita' e
     detectability. Le quattro modalita' riproducono la struttura della
     classe probe del dominio IDS (§8 della specifica).

  B. condizionata, d_c = 1: il condizionamento sposta la media.
     Verifica che i campioni generati seguano c, non che siano semplicemente
     verosimili. E' la prova che il condizionamento e' effettivo, ed e'
     il prerequisito di RQ2.

  C. autoregressiva, tracce AR(1), d_c = 8 come nel caso QKD.
     Verifica che l'autocorrelazione delle tracce sintetiche si avvicini a
     quella dei dati, cioe' la mitigazione della deriva di §6.

    python scripts/toy_validation.py --epochs 60
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qgan import QGAN, metrics  # noqa: E402
from qgan.autoregressive import build_ar_dataset, sample_traces, drift_gap  # noqa: E402


def clip01(x):
    return np.clip(x, 0.0, 1.0)


# ------------------------------------------------------------------ #
def toy_a(steps, seed, n=96, verbose=0, batch=16, eval_every=25):
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [0.20, 0.20, 0.75, 0.30, 0.50],
            [0.75, 0.25, 0.25, 0.70, 0.20],
            [0.30, 0.80, 0.40, 0.20, 0.80],
            [0.70, 0.70, 0.70, 0.65, 0.55],
        ]
    )
    modes = rng.integers(0, len(centers), n)
    X = clip01(centers[modes] + rng.normal(0, 0.05, (n, 5)))

    q = QGAN(d_c=0, batch_size=batch, max_steps=steps, eval_every=eval_every,
             seed=seed, verbose=verbose, n_monitor=64)
    pre = q.sample(n)
    m0 = metrics.mmd_rbf(X, pre)
    q.fit(X, modes=modes)
    post = q.sample(n)

    ev = metrics.evaluate(X, post, modes=modes, seed=seed)
    ev["mmd_iniziale"] = m0
    ev["mmd_finale"] = ev.pop("mmd")
    ev["best_step"] = q.best_step_
    ev["history"] = q.history
    return q, ev


# ------------------------------------------------------------------ #
def toy_b(steps, seed, n=96, verbose=0, batch=16, eval_every=25):
    rng = np.random.default_rng(seed + 100)
    c = rng.random((n, 1))
    # ogni feature si muove con c, con pendenze diverse
    slope = np.array([0.6, -0.5, 0.3, 0.0, 0.45])
    X = clip01(0.3 + c @ slope[None, :] + rng.normal(0, 0.04, (n, 5)))

    q = QGAN(d_c=1, batch_size=batch, max_steps=steps, eval_every=eval_every,
             seed=seed, verbose=verbose, n_monitor=64)
    q.fit(X, c)

    # risposta al condizionamento: si generano campioni a c basso e c alto
    lo = q.sample(200, np.full((200, 1), 0.1))
    hi = q.sample(200, np.full((200, 1), 0.9))
    d_gen = hi.mean(0) - lo.mean(0)
    d_true = 0.8 * slope
    ev = {
        "mmd": metrics.mmd_rbf(X, q.sample(n, c)),
        "delta_generato": d_gen.tolist(),
        "delta_atteso": d_true.tolist(),
        "correlazione_delta": float(np.corrcoef(d_gen, d_true)[0, 1]),
        "segni_concordi": int(np.sum(np.sign(d_gen[[0, 1, 2, 4]]) == np.sign(d_true[[0, 1, 2, 4]]))),
        "best_step": q.best_step_,
    }
    return q, ev


# ------------------------------------------------------------------ #
def toy_c(steps, seed, n_tr=16, T=16, verbose=0, batch=16, eval_every=25):
    rng = np.random.default_rng(seed + 200)
    phi = 0.75
    cond = rng.random((n_tr, 3))
    tr = np.zeros((n_tr, T, 5))
    base = 0.2 + 0.5 * cond[:, [0]]  # il livello dipende dal primo asse
    w = base + rng.normal(0, 0.05, (n_tr, 5))
    for t in range(T):
        w = phi * w + (1 - phi) * base + rng.normal(0, 0.03, (n_tr, 5))
        tr[:, t, :] = clip01(w)

    X, ceff = build_ar_dataset(tr, cond)
    q = QGAN(d_c=ceff.shape[1], batch_size=batch, max_steps=steps,
             eval_every=eval_every, seed=seed, verbose=verbose, n_monitor=64)
    q.fit(X, ceff)

    fake = sample_traces(q, cond, T=T)
    ev = drift_gap(tr, fake)
    ev["mmd_finestre"] = metrics.mmd_rbf(
        tr.reshape(-1, 5), fake.reshape(-1, 5)
    )
    ev["best_step"] = q.best_step_
    return q, ev


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="abc")
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--out", default="results/toy_validation.json")
    a = ap.parse_args()

    out = {}
    if "a" in a.only:
        print("\n=== A: non condizionata, 4 modalita' ===")
        _, ev = toy_a(a.steps, a.seed, verbose=a.verbose)
        h = ev.pop("history")
        print(f"  MMD  {ev['mmd_iniziale']:.5f} -> {ev['mmd_finale']:.5f}")
        print(f"  detectability AUC   {ev['detectability_auc']:.3f}")
        print(f"  copertura modalita' {ev['mode_coverage']['coverage']:.2f}  "
              f"entropia {ev['mode_coverage']['entropy']:.3f}")
        print(f"  Wasserstein medio   {ev['wasserstein_mean']:.4f}")
        print(f"  KS medio            {ev['ks_mean']:.3f}")
        print(f"  Var(grad) prima/dopo {h['grad_var'][0]:.2e} / {h['grad_var'][-1]:.2e}")
        out["A"] = {**ev, "history": h}

    if "b" in a.only:
        print("\n=== B: condizionata, d_c=1 ===")
        _, ev = toy_b(a.steps, a.seed, verbose=a.verbose)
        print(f"  MMD {ev['mmd']:.5f}")
        print(f"  delta generato {np.round(ev['delta_generato'],3)}")
        print(f"  delta atteso   {np.round(ev['delta_atteso'],3)}")
        print(f"  correlazione   {ev['correlazione_delta']:.3f}   "
              f"segni concordi {ev['segni_concordi']}/4")
        out["B"] = ev

    if "c" in a.only:
        print("\n=== C: autoregressiva, d_c=8 ===")
        _, ev = toy_c(a.steps, a.seed, verbose=a.verbose)
        print(f"  MMD sulle finestre {ev['mmd_finestre']:.5f}")
        print(f"  ACF lag 1 reale {np.round(np.mean(ev['acf_real'][0]),3)}  "
              f"sintetica {np.round(np.mean(ev['acf_fake'][0]),3)}")
        print(f"  MAE ACF {ev['mae_acf']:.3f}  (lag 1: {ev['mae_acf_lag1']:.3f})")
        out["C"] = ev

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nscritto {a.out}")


if __name__ == "__main__":
    main()
