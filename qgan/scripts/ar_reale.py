"""§6 sui dati reali — la pipeline dichiarata da costruire in B.7.

Finora la validazione autoregressiva ha sempre girato su `genera_tracce`, un
bersaglio sintetico AR(1). Questo script costruisce le tracce dalla griglia
QBER vera e mette alla prova §6 dove conta.

LA DOMANDA. §6 prescrive di condizionare sulla finestra precedente. Serve?
La risposta non si legge da una metrica sola: si legge da un CONFRONTO fra
due modelli addestrati sugli stessi dati e valutati sulle stesse tracce.

    autoregressivo   d_c = 8 = 3 statici + 5 osservabili precedenti
    per finestra     d_c = 3, le finestre sono indipendenti dato c

Se le finestre sono scambiabili data la cella — come misurato sulla griglia
stazionaria, dove l'autocorrelazione del QBER entro cella e' compatibile con
zero — il modello per finestra e' sufficiente e l'autoregressione non serve.
Se invece la traccia ha dinamica, l'autoregressivo deve riprodurla e il
modello per finestra no, per costruzione.

E' quindi un esperimento con un CONTROLLO NEGATIVO incorporato: la stessa
pipeline sulla griglia stazionaria deve mostrare nessun vantaggio. Se lo
mostrasse, il vantaggio verrebbe da qualcos'altro e non dalla memoria.

    python scripts/ar_reale.py --dati ../data/qber_grid.csv
    python scripts/ar_reale.py --dati ../data/qber_grid_ns.csv --profilo deriva_eve
    python scripts/ar_reale.py --dati ../data/qber_grid_ns.csv --profilo intermittente

SCELTE DI DISEGNO

  Tracce: finestre consecutive della STESSA cella, in blocchi di T. Non
  attraversano mai il confine di cella, e l'ordine e' quello temporale di
  `window_idx`. Il resto della cella, sotto T finestre, viene scartato.

  Condizionamento: i tre parametri nominali della cella. `rate_eff` e
  `depol_eff` sono le condizioni EFFETTIVE finestra per finestra, e restano
  fuori: sono target, non ingressi. E' questo che costringe il modello a
  inferire il regime da w_{t-1} invece di leggerlo da c.

  Split per SEME, non per cella. La griglia non stazionaria ha 2x2x2 = 8
  configurazioni per profilo, tutte agli angoli: non esistono celle interne
  e l'estrapolazione condizionale di RQ2 non e' misurabile qui. Non e' lo
  scopo di questo script.

  Riscalamento min-max calcolato sulle sole celle di ADDESTRAMENTO e
  riusato identico su validazione e test, come prescrive la calibrazione del
  latente.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import corsa  # noqa: E402
from qgan import QGAN, metrics  # noqa: E402
from qgan.autoregressive import (autocorrelation,  # noqa: E402
                                 autocorrelation_nulla, build_ar_dataset,
                                 sample_traces, valore_w0)
from qgan.metrics import median_bandwidth  # noqa: E402

# Ordine naturale (B.6)
OSSERVABILI = ["qber", "qber_rect", "qber_diag", "sift_rate", "loss_rate"]
CONDIZIONI = ["intercept_rate", "length_km", "depolar_total"]
CHIAVI = ["intercept_rate", "length_km", "depolar_total", "seed"]

T_DEFAULT = 16
# Validazione e test sono i DUE SEMI PIU' ALTI presenti nel file, ricavati
# dai dati invece che fissati: con 5 semi diventano 4 e 5, con 15 diventano
# 14 e 15, senza toccare il codice. Lo split e' per seme e non per cella
# perche' il numero di semi E' il numero di realizzazioni della deriva, ed
# e' li' che sta la generalizzazione da misurare.

FROZEN = dict(
    lr_g=0.002, lr_d=0.002, hidden=(32, 16), batch_size=16,
    eval_every=25, patience=20, mmd_smooth=3, val_fraction=0.0,
    n_monitor=256, monitor_repeats=1,
)


def carica(percorso, profilo=None):
    d = pd.read_csv(percorso)
    if profilo is not None:
        if "profilo" not in d.columns:
            raise SystemExit("il file non ha la colonna 'profilo': e' la "
                             "griglia stazionaria, usare --profilo assente")
        d = d[d.profilo == profilo]
        if d.empty:
            raise SystemExit(f"nessuna riga con profilo={profilo!r}")
    return d.sort_values(CHIAVI + ["window_idx"])


def tracce(d, T):
    """(tracce, condizione statica, seme, condizioni effettive).

    Le tracce non attraversano il confine di cella: dentro ogni cella si
    prendono blocchi contigui di T finestre e si scarta il resto.
    """
    tr, cond, semi, eff = [], [], [], []
    for chiave, g in d.groupby(CHIAVI, sort=True):
        g = g.sort_values("window_idx")
        X = g[OSSERVABILI].to_numpy(float)
        E = g[["rate_eff", "depol_eff"]].to_numpy(float) if \
            "rate_eff" in g.columns else np.zeros((len(g), 2))
        n = len(g) // T
        for b in range(n):
            tr.append(X[b * T:(b + 1) * T])
            eff.append(E[b * T:(b + 1) * T])
            cond.append([chiave[CHIAVI.index(c)] for c in CONDIZIONI])
            semi.append(chiave[CHIAVI.index("seed")])
    return (np.array(tr), np.array(cond, float),
            np.array(semi, int), np.array(eff))


def scala(v, lo, amp):
    return np.clip((v - lo) / np.where(amp > 0, amp, 1.0), 0.0, 1.0)


def valuta(q, tr_t, c_t, T, sig, w0v, d_c, seed):
    """Metriche sul test. Con d_c=3 le finestre sono generate indipendenti."""
    n, F = tr_t.shape[0], tr_t.shape[2]
    if d_c > 3:
        fake = sample_traces(q, c_t, T=T, w0="media", w0_value=w0v)
    else:
        # Modello per finestra: la traccia e' T campioni indipendenti dalla
        # stessa condizione. Nessuna memoria per costruzione — e' il punto.
        fake = np.stack([q.sample(n, c_t) for _ in range(T)], axis=1)

    per_passo = [metrics.mmd_rbf(tr_t[:, t, :], fake[:, t, :], sigma=sig)
                 for t in range(T)]
    ml = min(8, T - 1)
    ris = {
        "d_c": d_c,
        "mmd_complessiva": float(metrics.mmd_rbf(
            tr_t.reshape(-1, F), fake.reshape(-1, F), sigma=sig)),
        "mmd_per_passo": [float(v) for v in per_passo],
        "rapporto_primo_ultimo": float(per_passo[-1] / max(per_passo[0], 1e-12)),
        "dispersione_temporale": float(fake.std(axis=1).mean()),
    }
    for et, mode in (("intra", "intra"), ("aggregata", "aggregata")):
        ar = autocorrelation(tr_t, ml, mode=mode)
        af = autocorrelation(fake, ml, mode=mode)
        ris[f"acf_reale_{et}"] = float(np.nanmean(ar[0]))
        ris[f"acf_sintetica_{et}"] = float(np.nanmean(af[0]))
        ris[f"mae_acf_{et}"] = float(np.nanmean(np.abs(ar - af)))
        ris[f"acf_nulla_{et}"] = float(np.nanmean(autocorrelation_nulla(
            tr_t, ml, mode=mode, seed=seed, repliche=10)[0]))
    return ris


def run_seed(seed, dati, profilo, T, steps, latent_hi, verbose, backend):
    d = carica(dati, profilo)
    tr, c_st, semi, eff = tracce(d, T)

    unici = np.unique(semi)
    if len(unici) < 3:
        raise SystemExit(f"servono almeno 3 semi, trovati {unici.tolist()}")
    seme_test, seme_val = int(unici[-1]), int(unici[-2])
    m_tr = ~np.isin(semi, [seme_val, seme_test])
    m_va = semi == seme_val
    m_te = semi == seme_test

    # Riscalamento min-max sulle sole celle di addestramento
    piatto = tr[m_tr].reshape(-1, tr.shape[2])
    lo, hi = piatto.min(0), piatto.max(0)
    amp = hi - lo
    trs = scala(tr, lo, amp)
    c_lo = c_st[m_tr].min(0)
    c_amp = c_st[m_tr].max(0) - c_lo
    cs = scala(c_st, c_lo, c_amp)

    w0v = valore_w0(trs[m_tr], "media")
    X, ceff = build_ar_dataset(trs[m_tr], cs[m_tr], w0="media", w0_value=w0v)
    Xv, ceff_v = build_ar_dataset(trs[m_va], cs[m_va], w0="media", w0_value=w0v)
    sig = float(median_bandwidth(X, rng=np.random.default_rng(0)))

    modelli = {}
    for nome, d_c in (("autoregressivo", ceff.shape[1]), ("per_finestra", 3)):
        q = QGAN(d_c=d_c, max_steps=steps, seed=seed, verbose=verbose,
                 latent_hi=latent_hi, backend=backend, **FROZEN)
        if d_c > 3:
            q.fit(X, ceff, X_val=Xv, c_val=ceff_v)
        else:
            # Stesse finestre, stesso budget: cambia solo cosa vede in c.
            q.fit(X, ceff[:, :3], X_val=Xv, c_val=ceff_v[:, :3])
        modelli[nome] = q

    ris = {
        "seed": seed, "T": T, "steps": steps, "profilo": profilo,
        "dati": os.path.basename(dati),
        "tracce_train": int(m_tr.sum()), "tracce_val": int(m_va.sum()),
        "tracce_test": int(m_te.sum()),
        "semi_train": [int(x) for x in unici[:-2]],
        "seme_val": seme_val, "seme_test": seme_test,
        "derive_distinte_train": int(len(unici) - 2),
        "scala_min": lo.tolist(), "scala_ampiezza": amp.tolist(),
        "sigma_mmd": sig,
        "rate_eff_test_std": float(eff[m_te][:, :, 0].std()),
    }
    for nome, q in modelli.items():
        c_t = cs[m_te] if q.d_c == 3 else cs[m_te]
        v = valuta(q, trs[m_te], c_t, T, sig, w0v, q.d_c, seed)
        v["best_step"] = int(q.best_step_)
        v["steps_run"] = int(q.steps_run_)
        ris[nome] = v
    return ris


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dati", required=True)
    ap.add_argument("--profilo", default=None)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--traccia", type=int, default=T_DEFAULT)
    ap.add_argument("--steps", type=int, default=5400)
    ap.add_argument("--latent-hi", type=str, default="0.8200")
    ap.add_argument("--backend", default="denso", choices=["denso", "aer"])
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--forza", action="store_true")
    ap.add_argument("--out", default="results/ar_reale.jsonl")
    a = ap.parse_args()

    lh = [float(x) for x in a.latent_hi.split(",")]
    lh = lh[0] if len(lh) == 1 else lh
    config = dict(FROZEN, steps=a.steps, T=a.traccia, dati=a.dati,
                  profilo=a.profilo, latent_hi=lh, backend=a.backend,
                  split="due semi piu' alti",
                  osservabili=OSSERVABILI, condizioni=CONDIZIONI)
    fr = corsa.firma(config)
    gia = corsa.gia_presenti(a.out)
    print(f"configurazione {fr}   codice {corsa.versione_codice()}")

    for s in range(a.seed_offset, a.seed_offset + a.seeds):
        if (fr, s) in gia and not a.forza:
            print(f"[seme {s}] gia' presente, saltato", flush=True)
            continue
        print(f"[seme {s}]", flush=True)
        r = run_seed(s, a.dati, a.profilo, a.traccia, a.steps, lh,
                     a.verbose, a.backend)
        r.update(corsa.intestazione(config))
        corsa.scrivi(a.out, r)
        print(f"  tracce  train {r['tracce_train']}  val {r['tracce_val']}"
              f"  test {r['tracce_test']}   derive distinte in train "
              f"{r['derive_distinte_train']}", flush=True)
        for nome in ("autoregressivo", "per_finestra"):
            v = r[nome]
            print(f"  {nome:15s} MMD {v['mmd_complessiva']:.5f}   "
                  f"acf intra sintetica {v['acf_sintetica_intra']:+.3f}   "
                  f"(reale {v['acf_reale_intra']:+.3f}, "
                  f"nullo {v['acf_nulla_intra']:+.3f})   "
                  f"passo {v['best_step']}", flush=True)


if __name__ == "__main__":
    main()