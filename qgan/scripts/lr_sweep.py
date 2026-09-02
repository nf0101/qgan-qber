"""Sensibilita' al rapporto fra i learning rate.

La specifica fissa lr_g = 0,01 e lr_d = 0,001, cioe' un generatore dieci
volte piu' rapido del discriminatore. Il toy A mostra che con questa
configurazione il modello addestrato e' peggiore di quello inizializzato,
mentre la minimizzazione diretta della MMD raggiunge 0,010: il problema e'
quindi nel ciclo avversariale, non nell'espressivita'.

Lo script confronta alcune configurazioni sullo stesso bersaglio, con un
insieme di validazione indipendente e sufficientemente numeroso perche' la
MMD di monitoraggio non sia dominata dal rumore.

    python scripts/lr_sweep.py --steps 150
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qgan import QGAN, metrics  # noqa: E402

CENTERS = np.array(
    [
        [0.20, 0.20, 0.75, 0.30, 0.50],
        [0.75, 0.25, 0.25, 0.70, 0.20],
        [0.30, 0.80, 0.40, 0.20, 0.80],
        [0.70, 0.70, 0.70, 0.65, 0.55],
    ]
)


def target(n, rng):
    m = rng.integers(0, 4, n)
    return np.clip(CENTERS[m] + rng.normal(0, 0.05, (n, 5)), 0, 1), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/lr_sweep.json")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    X, modes = target(a.n, rng)
    X_val, _ = target(256, rng)  # indipendente e numeroso
    X_test, _ = target(512, rng)

    configs = [
        ("specifica  g=0,01 d=0,001", 0.01, 0.001),
        ("pari       g=0,005 d=0,005", 0.005, 0.005),
        ("pari lento g=0,001 d=0,001", 0.001, 0.001),
        ("D piu' veloce g=0,002 d=0,005", 0.002, 0.005),
    ]

    out = []
    for name, lg, ld in configs:
        q = QGAN(
            d_c=0,
            lr_g=lg,
            lr_d=ld,
            batch_size=a.batch,
            max_steps=a.steps,
            eval_every=25,
            patience=1000,  # nessun arresto: interessa la traiettoria intera
            val_fraction=0.0,
            n_monitor=256,
            monitor_repeats=1,
            seed=a.seed,
            verbose=0,
        )
        mmd_init = metrics.mmd_rbf(X_test, q.sample(512))
        q.fit(X, X_val=X_val, modes=modes)
        Xg = q.sample(512)
        row = {
            "config": name,
            "lr_g": lg,
            "lr_d": ld,
            "mmd_init": mmd_init,
            "mmd_final": metrics.mmd_rbf(X_test, Xg),
            "mmd_best_monitor": q.best_mmd_,
            "best_step": q.best_step_,
            "detectability": metrics.detectability(X_test, Xg, seed=a.seed),
            "std_real": X_test.std(0).tolist(),
            "std_gen": Xg.std(0).tolist(),
            "traiettoria_mmd": q.history["mmd_val"],
            "loss_d": q.history["loss_d"],
            "loss_g": q.history["loss_g"],
        }
        out.append(row)
        print(
            f"{name:32s}  MMD {row['mmd_init']:.4f} -> {row['mmd_final']:.4f}  "
            f"(migliore {row['mmd_best_monitor']:.4f} al passo {row['best_step']})  "
            f"det {row['detectability']:.3f}  "
            f"std_gen {np.round(np.mean(row['std_gen']),3)}"
        )

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, default=float)
    print(f"\nscritto {a.out}")


if __name__ == "__main__":
    main()
