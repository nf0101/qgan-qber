"""Ordine delle componenti PCA sui qubit — decisione aperta di Appendice B.6.

B.6 stabilisce per il dominio QKD che sul qubit meno pilotabile vada
l'osservabile che richiede piu' dispersione, perche' li' il raggio del latente
non riesce comunque a comprimerla. L'accoppiamento misurato fra il latente di
un qubit e la propria marginale vale 1,00 sui primi quattro e 0,21 sul quinto,
che e' terminale nella catena di entanglement lineare.

Nel dominio IDS il criterio non si applica: la configurazione base e' non
condizionata, quindi non esiste una dispersione condizionale da eguagliare e il
raggio del latente resta pi uniforme. Resta pero' il fatto strutturale che il
quinto qubit e' meno pilotabile, e non e' ovvio che l'ordine sia indifferente.

Tre ordinamenti confrontati sullo stesso bersaglio:

  naturale   componenti in ordine di varianza spiegata decrescente, che e'
             l'ordine in cui la PCA le produce
  invertito  la componente a varianza maggiore sul qubit terminale
  casuale    controllo: se i tre coincidono, l'ordine e' indifferente e va
             dichiarato arbitrario invece di essere motivato a posteriori

Il bersaglio imita la classe probe di NSL-KDD: quattro modalita' distinte
(§2.3 di FASE1A) e componenti con dispersione decrescente dopo il riscalamento
in [0,1].

    python scripts/ordine_qubit_ids.py --seeds 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qgan import QGAN, metrics  # noqa: E402
from qgan.metrics import median_bandwidth  # noqa: E402

FROZEN = dict(
    lr_g=0.002, lr_d=0.002, hidden=(32, 16), batch_size=16,
    eval_every=25, patience=20, mmd_smooth=3, val_fraction=0.0,
    n_monitor=256, monitor_repeats=1, latent_hi=np.pi,
)

# Dispersione per componente dopo riscalamento in [0,1]. Il min-max equipara
# l'intervallo ma non la varianza: una componente con code lunghe finisce
# concentrata e ha deviazione minore.
SIGMA_COMP = np.array([0.20, 0.15, 0.11, 0.075, 0.055])

# Quattro modalita', come i quattro attacchi che compongono probe (FASE1A §2.3)
CENTRI = np.array([
    [0.25, 0.30, 0.70, 0.35, 0.50],
    [0.70, 0.25, 0.30, 0.65, 0.25],
    [0.35, 0.75, 0.45, 0.25, 0.75],
    [0.65, 0.65, 0.65, 0.60, 0.55],
])

ORDINI = {
    "naturale": [0, 1, 2, 3, 4],
    "invertito": [4, 3, 2, 1, 0],
    "casuale": [2, 0, 4, 1, 3],
}


def bersaglio(n, rng, ordine):
    """Genera il bersaglio e lo permuta secondo l'assegnazione ai qubit."""
    m = rng.integers(0, len(CENTRI), n)
    X = CENTRI[m] + rng.normal(0, 1, (n, 5)) * SIGMA_COMP[None, :]
    p = np.asarray(ORDINI[ordine])
    return np.clip(X[:, p], 0, 1), m, SIGMA_COMP[p]


def run(ordine, seed, steps, n, verbose):
    rng = np.random.default_rng(9000 + seed)
    X, modi, sig_ord = bersaglio(n, rng, ordine)
    Xv, _, _ = bersaglio(256, rng, ordine)
    Xt, _, _ = bersaglio(512, rng, ordine)

    q = QGAN(d_c=0, max_steps=steps, seed=seed, verbose=verbose, **FROZEN)
    sigma = float(median_bandwidth(Xt, rng=np.random.default_rng(0)))
    mmd0 = metrics.mmd_rbf(Xt, q.sample(512), sigma=sigma)
    q.fit(X, X_val=Xv, modes=modi)
    Xg = q.sample(512)

    ev = metrics.evaluate(Xt, Xg, modes=None, seed=seed, sigma=sigma)
    return {
        "ordine": ordine, "seed": seed, "steps": steps,
        "mmd_iniziale": float(mmd0),
        "mmd_finale": float(ev["mmd"]),
        "riduzione": float(1 - ev["mmd"] / mmd0),
        "detectability": float(ev["detectability_auc"]),
        "wasserstein": float(ev["wasserstein_mean"]),
        "sigma_bersaglio": sig_ord.tolist(),
        "sigma_generata": Xg.std(axis=0).tolist(),
        "rapporto_sigma": (Xg.std(axis=0) / sig_ord).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1350)
    ap.add_argument("--n", type=int, default=1920)
    ap.add_argument("--ordini", default="naturale,invertito,casuale")
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--out", default="results/ordine_qubit_ids.jsonl")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = []
    for o in a.ordini.split(","):
        for s in range(a.seed_offset, a.seed_offset + a.seeds):
            r = run(o, s, a.steps, a.n, a.verbose)
            with open(a.out, "a") as f:
                f.write(json.dumps(r, default=float) + "\n")
            rows.append(r)
            print(f"{o:>10} seme {s}  MMD {r['mmd_iniziale']:.4f} -> "
                  f"{r['mmd_finale']:.4f} ({r['riduzione']:+.1%})  "
                  f"det {r['detectability']:.3f}", flush=True)

    if len(rows) < 2:
        return
    print(f"\n{'ordine':>10} {'semi':>5} {'riduzione MMD':>18} "
          f"{'detectability':>14} {'sigma gen/vera':>15}")
    for o in a.ordini.split(","):
        v = [r for r in rows if r["ordine"] == o]
        if not v:
            continue
        red = np.array([r["riduzione"] for r in v])
        det = np.array([r["detectability"] for r in v])
        rs = np.array([r["rapporto_sigma"] for r in v]).mean(axis=0)
        print(f"{o:>10} {len(v):5d} {red.mean():+9.1%} +- {red.std(ddof=1):.1%} "
              f"{det.mean():14.3f} {np.round(rs, 2)}")
    print("\nSe i tre ordinamenti coincidono entro il rumore, l'ordine e'")
    print("indifferente e va dichiarato arbitrario: usare l'ordine naturale")
    print("della PCA, che e' quello che il preprocessing produce.")


if __name__ == "__main__":
    main()
