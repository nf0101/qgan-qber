"""Generazione autoregressiva — validazione di SPEC-FASE2 §6.

Le tracce QBER sono sequenze di finestre. Una mappatura diretta richiederebbe
n_feature x T qubit, non praticabile: §6 prescrive quindi la generazione
autoregressiva, una finestra alla volta, condizionando su `c` e sulla finestra
precedente, con `c_eff = concat(c, w_{t-1})` e quindi `d_c = 3 + 5 = 8`.

Il codice e' in `src/qgan/autoregressive.py` e passa i test strutturali, ma non
e' mai stato addestrato: il toy C previsto da §13 punto 2 non era stato
eseguito. E' la modalita' operativa dell'intero dominio QKD.

TRE COSE DA VERIFICARE, IN ORDINE DI SEVERITA'

  1. QUALITA' DELLE FINESTRE. Con teacher forcing il compito e' un normale
     addestramento condizionato: se fallisce gia' qui, il problema non e'
     l'autoregressione.

  2. DERIVA. §6 prescrive teacher forcing in addestramento e condizionamento
     sulla finestra GENERATA in produzione. Il divario fra i due regimi e' il
     rischio principale: gli errori si accumulano lungo la traccia. Misurato
     confrontando la MMD delle finestre a ciascun passo temporale.

  3. AUTOCORRELAZIONE. E' la misura di deriva indicata dalla specifica:
     l'autocorrelazione delle tracce sintetiche va confrontata con quella del
     simulatore. Una traccia che collassa su un valore fisso ha
     autocorrelazione 1; una che dimentica il passato l'ha a 0.

Il bersaglio e' sintetico, con struttura AR(1) e livello dipendente da `c`,
dispersioni condizionali dell'ordine di quelle misurate sulla griglia reale.

    python scripts/autoregressive_validation.py --seeds 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qgan import QGAN, metrics  # noqa: E402
from qgan.autoregressive import (autocorrelation, build_ar_dataset,  # noqa: E402
                                 drift_gap, sample_traces)
from qgan.metrics import median_bandwidth  # noqa: E402

# Configurazione congelata (Appendice A) piu' il raggio del latente di B.2.
FROZEN = dict(
    lr_g=0.002, lr_d=0.002, hidden=(32, 16), batch_size=16,
    eval_every=25, patience=20, mmd_smooth=3, val_fraction=0.0,
    n_monitor=256, monitor_repeats=1,
)
PHI = 0.75          # persistenza AR(1)
SIGMA = np.array([0.0032, 0.0702, 0.0575, 0.0015, 0.0714])  # come B.2


def genera_tracce(n, T, rng):
    """Tracce AR(1) con livello dipendente da c e rumore per osservabile."""
    c = rng.random((n, 3))
    q = 0.02 + 0.30 * c[:, 0] + 0.05 * c[:, 1] + 0.35 * c[:, 2] ** 1.5
    asym = 0.04 * (c[:, 0] - 0.5)
    livello = np.stack([0.50 - 0.08 * c[:, 1] ** 2, q + asym, q,
                        0.05 + 0.60 * c[:, 1] ** 1.3, q - asym], axis=1)
    tr = np.zeros((n, T, 5))
    w = livello + rng.normal(0, 1, (n, 5)) * SIGMA
    for t in range(T):
        innov = rng.normal(0, 1, (n, 5)) * SIGMA * np.sqrt(1 - PHI ** 2)
        w = PHI * w + (1 - PHI) * livello + innov
        tr[:, t, :] = np.clip(w, 0, 1)
    return tr, c


def run_seed(seed, steps, T, n_tracce, latent_hi, verbose, w0="zero"):
    rng = np.random.default_rng(7000 + seed)
    tr, c = genera_tracce(n_tracce, T, rng)
    tr_v, c_v = genera_tracce(64, T, rng)

    # w0_value fissato sul training e riusato in generazione: usare la media
    # delle tracce di validazione sarebbe una fuga, seppure minima.
    w0v = tr.reshape(-1, 5).mean(axis=0)
    X, ceff = build_ar_dataset(tr, c, w0=w0, w0_value=w0v)
    Xv, ceff_v = build_ar_dataset(tr_v, c_v, w0=w0, w0_value=w0v)

    q = QGAN(d_c=ceff.shape[1], max_steps=steps, seed=seed, verbose=verbose,
             latent_hi=latent_hi, **FROZEN)
    q.fit(X, ceff, X_val=Xv, c_val=ceff_v)

    sig = float(median_bandwidth(X, rng=np.random.default_rng(0)))

    # --- 1. qualita' delle finestre, in regime di teacher forcing --------
    idx = rng.choice(len(Xv), 512)
    mmd_tf = metrics.mmd_rbf(Xv, q.sample(512, ceff_v[idx]), sigma=sig)

    # --- 2. deriva: MMD per passo temporale in generazione libera --------
    fake = sample_traces(q, c_v, T=T, w0=w0, w0_value=w0v)
    per_passo = [metrics.mmd_rbf(tr_v[:, t, :], fake[:, t, :], sigma=sig)
                 for t in range(T)]
    mmd_libera = metrics.mmd_rbf(tr_v.reshape(-1, 5), fake.reshape(-1, 5),
                                 sigma=sig)

    # --- 3. autocorrelazione --------------------------------------------
    g = drift_gap(tr_v, fake, max_lag=min(8, T - 1))
    acf_r = np.array(g["acf_real"])
    acf_f = np.array(g["acf_fake"])

    # controllo di collasso: una traccia costante ha acf 1 e dispersione 0
    disp_t = float(fake.std(axis=1).mean())      # variabilita' nel tempo
    disp_t_real = float(tr_v.std(axis=1).mean())

    return {
        "seed": seed, "T": T, "steps": steps, "w0": w0,
        "latent_hi": float(np.mean(latent_hi)),
        "mmd_teacher_forcing": float(mmd_tf),
        "mmd_generazione_libera": float(mmd_libera),
        "rapporto_deriva": float(mmd_libera / max(mmd_tf, 1e-12)),
        "mmd_per_passo": [float(v) for v in per_passo],
        "acf_reale_lag1": float(np.mean(acf_r[0])),
        "acf_sintetica_lag1": float(np.mean(acf_f[0])),
        "mae_acf": g["mae_acf"],
        "dispersione_temporale_reale": disp_t_real,
        "dispersione_temporale_sintetica": disp_t,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1350)
    ap.add_argument("--traccia", type=int, default=16, help="T, §6 usa 16")
    ap.add_argument("--n-tracce", type=int, default=120)
    ap.add_argument("--latent-hi", type=str, default="0.7342")
    ap.add_argument("--w0", default="zero", choices=["zero", "media", "scarta"],
                    help="trattamento della prima finestra; 'zero' e' §6")
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--out", default="results/autoregressive.jsonl")
    a = ap.parse_args()

    lh = [float(x) for x in a.latent_hi.split(",")]
    lh = lh[0] if len(lh) == 1 else lh
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    rows = []
    for s in range(a.seed_offset, a.seed_offset + a.seeds):
        print(f"[seme {s}]", flush=True)
        r = run_seed(s, a.steps, a.traccia, a.n_tracce, lh, a.verbose, a.w0)
        with open(a.out, "a") as f:
            f.write(json.dumps(r, default=float) + "\n")
        rows.append(r)
        print(f"  finestre   teacher forcing {r['mmd_teacher_forcing']:.5f}   "
              f"generazione libera {r['mmd_generazione_libera']:.5f}   "
              f"deriva x{r['rapporto_deriva']:.1f}", flush=True)
        print(f"  MMD per passo t=0..{a.traccia-1}: "
              + " ".join(f"{v:.3f}" for v in r["mmd_per_passo"]), flush=True)
        print(f"  autocorr.  reale {r['acf_reale_lag1']:+.3f}   "
              f"sintetica {r['acf_sintetica_lag1']:+.3f}   "
              f"MAE {r['mae_acf']:.3f}", flush=True)
        print(f"  variabilita' nel tempo  reale "
              f"{r['dispersione_temporale_reale']:.5f}   sintetica "
              f"{r['dispersione_temporale_sintetica']:.5f}", flush=True)

    if len(rows) < 2:
        return
    d = np.array([r["rapporto_deriva"] for r in rows])
    pp = np.array([r["mmd_per_passo"] for r in rows])
    dt = np.array([r["dispersione_temporale_sintetica"] for r in rows])
    dr = np.array([r["dispersione_temporale_reale"] for r in rows])
    print(f"\nderiva      x{d.mean():.1f} +- {d.std(ddof=1):.1f}")
    print(f"MMD passo 0 {pp[:, 0].mean():.4f}   ultimo passo "
          f"{pp[:, -1].mean():.4f}   rapporto {pp[:, -1].mean()/max(pp[:, 0].mean(),1e-12):.1f}")
    print(f"variabilita' nel tempo, sintetica/reale "
          f"{dt.mean()/max(dr.mean(),1e-12):.2f}")

    print("\nLettura. Il rapporto fra MMD al primo e all'ultimo passo misura")
    print("l'accumulo dell'errore lungo la traccia: se cresce molto, il")
    print("teacher forcing di §6 non basta e serve una mitigazione.")
    print("La variabilita' nel tempo molto sotto 1 indica tracce che")
    print("collassano su un valore fisso, cioe' perdita della dinamica.")


if __name__ == "__main__":
    main()