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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import corsa  # noqa: E402
from qgan import QGAN, metrics  # noqa: E402
from qgan.autoregressive import (autocorrelation,  # noqa: E402
                                 autocorrelation_nulla, build_ar_dataset,
                                 sample_traces, valore_w0)
from qgan.metrics import median_bandwidth  # noqa: E402

SEED_BASE = 7000
BACKEND = "denso"   # impostato da main; vedi --backend
SCHEMA = "avanti"   # impostato da main; vedi --cond-schema

# Configurazione congelata (Appendice A) piu' il raggio del latente di B.2.
FROZEN = dict(
    lr_g=0.002, lr_d=0.002, hidden=(32, 16), batch_size=16,
    eval_every=25, patience=20, mmd_smooth=3, val_fraction=0.0,
    n_monitor=256, monitor_repeats=1,
)
PHI = 0.75          # persistenza AR(1)
# Ordine NATURALE (B.6): qber, qber_rect, qber_diag, sift_rate, loss_rate.
# Il toy codificava ancora l'ordine b6, ritirato. Dispersioni della griglia
# rigenerata il 16/09/2026; le precedenti avevano sift_rate e loss_rate
# costanti per corsa, quindi 0,0032 e 0,0015 invece di 0,036 e 0,021.
SIGMA = np.array([0.05336, 0.07109, 0.07136, 0.03584, 0.02084])


def genera_tracce(n, T, rng):
    """Tracce AR(1) con livello dipendente da c e rumore per osservabile.

    Restituisce anche ``livello``: sul toy il livello condizionale e' NOTO, ed
    e' l'unico modo di misurare l'autocorrelazione senza pagare il bias di
    centratura su media empirica. L'autocorrelazione teorica dei residui e'
    PHI**lag, quindi il toy fornisce anche il valore vero contro cui
    verificare la metrica stessa.
    """
    c = rng.random((n, 3))
    q = 0.02 + 0.30 * c[:, 0] + 0.05 * c[:, 1] + 0.35 * c[:, 2] ** 1.5
    asym = 0.04 * (c[:, 0] - 0.5)
    livello = np.stack([q, q + asym, q - asym,
                        0.50 - 0.08 * c[:, 1] ** 2,
                        0.05 + 0.60 * c[:, 1] ** 1.3], axis=1)
    tr = np.zeros((n, T, 5))
    w = livello + rng.normal(0, 1, (n, 5)) * SIGMA
    for t in range(T):
        innov = rng.normal(0, 1, (n, 5)) * SIGMA * np.sqrt(1 - PHI ** 2)
        w = PHI * w + (1 - PHI) * livello + innov
        tr[:, t, :] = np.clip(w, 0, 1)
    return tr, c, livello


def run_seed(seed, steps, T, n_tracce, latent_hi, verbose, w0="zero"):
    # Tre flussi per tre ruoli. Con un flusso unico le tracce di validazione
    # erano estratte dopo quelle di addestramento, quindi --n-tracce cambiava
    # anche l'insieme di validazione: stesso difetto della curva rispetto a n.
    rng, rng_val, rng_eval = corsa.flussi(SEED_BASE, seed, 3)
    tr, c, _ = genera_tracce(n_tracce, T, rng)
    tr_v, c_v, liv_v = genera_tracce(64, T, rng_val)

    # w0_value fissato sul training e riusato in generazione: usare la media
    # delle tracce di validazione sarebbe una fuga, seppure minima. Lo STESSO
    # valore va ad addestramento e generazione: sono due fasi distinte e
    # nulla, nel codice, garantisce da solo che partano dallo stesso stato.
    w0v = valore_w0(tr, w0)
    X, ceff = build_ar_dataset(tr, c, w0=w0, w0_value=w0v)
    Xv, ceff_v = build_ar_dataset(tr_v, c_v, w0=w0, w0_value=w0v)

    q = QGAN(d_c=ceff.shape[1], max_steps=steps, seed=seed, verbose=verbose,
             latent_hi=latent_hi, backend=BACKEND, cond_schema=SCHEMA,
             **FROZEN)
    q.fit(X, ceff, X_val=Xv, c_val=ceff_v)

    sig = float(median_bandwidth(X, rng=np.random.default_rng(0)))

    # --- 1. qualita' delle finestre, in regime di teacher forcing --------
    # Reali e sintetiche sullo STESSO sottoinsieme di condizioni: confrontare
    # tutte le finestre reali contro quelle generate per un sottoinsieme
    # ricampionato con reimmissione mescola il divario di distribuzione con
    # quello di campionamento.
    idx = rng_eval.choice(len(Xv), 512, replace=False)
    mmd_tf = metrics.mmd_rbf(Xv[idx], q.sample(512, ceff_v[idx]), sigma=sig)

    # --- 2. deriva: MMD per passo temporale in generazione libera --------
    fake = sample_traces(q, c_v, T=T, w0=w0, w0_value=w0v)
    per_passo = [metrics.mmd_rbf(tr_v[:, t, :], fake[:, t, :], sigma=sig)
                 for t in range(T)]
    mmd_libera = metrics.mmd_rbf(tr_v.reshape(-1, 5), fake.reshape(-1, 5),
                                 sigma=sig)

    # --- 3. autocorrelazione --------------------------------------------
    # Tre letture, perche' misurano cose diverse.
    #
    #   aggregata  concatena le coppie temporali fra TUTTE le tracce. Quando
    #              il livello dipende da c, due finestre della stessa traccia
    #              condividono quel livello anche senza memoria temporale: la
    #              varianza fra tracce domina e il valore resta altissimo.
    #              Conservata solo per confrontabilita' con i numeri passati.
    #   intra      ogni traccia centrata sulla propria media. Paga un bias di
    #              centratura: in assenza di memoria vale ~-1/(T-1), non 0.
    #   residua    centratura sul livello VERO, noto su questo toy. E' la
    #              lettura corretta, e non paga alcun bias.
    #
    # Il valore teorico dei residui e' PHI**lag: il toy verifica quindi anche
    # la metrica, non solo il modello.
    ml = min(8, T - 1)
    acf = {}
    for nome, kw_acf in (("aggregata", {}), ("intra", {}),
                         ("residua", {"livello": liv_v})):
        acf[f"real_{nome}"] = autocorrelation(tr_v, ml, mode=nome, **kw_acf)
        acf[f"fake_{nome}"] = autocorrelation(fake, ml, mode=nome, **kw_acf)
    # Baseline: la stessa metrica quando la memoria temporale e' distrutta a
    # permutazioni. Se coincide con la misura, la metrica non sta misurando
    # memoria.
    nulla = {n: autocorrelation_nulla(tr_v, ml, mode=n, seed=seed, repliche=10)
             for n in ("aggregata", "intra")}
    acf_teorica = np.array([PHI ** k for k in range(1, ml + 1)])
    g = {f"acf_{k}": v.tolist() for k, v in acf.items()}
    g.update({f"acf_nulla_{k}": v.tolist() for k, v in nulla.items()})
    g["mae_acf"] = float(np.nanmean(np.abs(
        acf["real_aggregata"] - acf["fake_aggregata"])))
    g["mae_acf_intra"] = float(np.nanmean(np.abs(
        acf["real_intra"] - acf["fake_intra"])))
    g["mae_acf_residua"] = float(np.nanmean(np.abs(
        acf["real_residua"] - acf["fake_residua"])))
    acf_r = acf["real_residua"]
    acf_f = acf["fake_residua"]

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
        "acf_teorica_lag1": float(acf_teorica[0]),
        "acf_reale_lag1_aggregata": float(np.mean(acf["real_aggregata"][0])),
        "acf_sintetica_lag1_aggregata": float(np.mean(acf["fake_aggregata"][0])),
        "acf_nulla_lag1_aggregata": float(np.mean(nulla["aggregata"][0])),
        "acf_reale_lag1_intra": float(np.mean(acf["real_intra"][0])),
        "acf_sintetica_lag1_intra": float(np.mean(acf["fake_intra"][0])),
        "acf_nulla_lag1_intra": float(np.mean(nulla["intra"][0])),
        "mae_acf": g["mae_acf"],
        "mae_acf_intra": g["mae_acf_intra"],
        "mae_acf_residua": g["mae_acf_residua"],
        "best_step": int(q.best_step_),
        "steps_run": int(q.steps_run_),
        "monitor": q.monitor_,
        "strati_assi": getattr(q, "strati_assi_", None),
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
    ap.add_argument("--latent-hi", type=str, default="0.8200")
    ap.add_argument("--cond-schema", default="avanti", choices=["avanti", "coda"],
                    help="mappatura di c sugli strati di condizionamento, "
                         "solo per d_c > n_qubit. 'coda' mette l'intera "
                         "finestra precedente nel primo strato, che e' il "
                         "piu' influente")
    ap.add_argument("--backend", default="denso", choices=["denso", "aer"],
                    help="simulatore del circuito. 'denso' e' vettorizzato sul batch, verificato contro Aer a 1e-15 su forward e jacobiana e molto piu' rapido; 'aer' e' il riferimento di SPEC §7.2")
    ap.add_argument("--w0", default="media", choices=["zero", "media", "scarta"],
                    help="trattamento della prima finestra. Il default e' "
                         "'media', adottato in FASE2 §17.2; 'zero' e' la "
                         "prescrizione originale di §6 e resta disponibile "
                         "come variante esplicita"),
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--forza", action="store_true",
                    help="riesegue e riaccoda anche semi gia' presenti nel "
                         "file con la stessa configurazione")
    ap.add_argument("--out", default="results/autoregressive.jsonl")
    a = ap.parse_args()

    global BACKEND, SCHEMA
    BACKEND = a.backend
    SCHEMA = a.cond_schema
    lh = [float(x) for x in a.latent_hi.split(",")]
    lh = lh[0] if len(lh) == 1 else lh
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    config = dict(
        FROZEN, steps=a.steps, T=a.traccia, n_tracce=a.n_tracce, w0=a.w0,
        backend=BACKEND, cond_schema=SCHEMA,
        latent_hi=lh if isinstance(lh, list) else [lh],
        phi=PHI, sigma=SIGMA.tolist(), seed_base=SEED_BASE,
        sigma_fonte="griglia rigenerata 16/09/2026",
    )
    fr = corsa.firma(config)
    gia = corsa.gia_presenti(a.out)
    dup = corsa.duplicati(a.out)
    print(f"configurazione {fr}   codice {corsa.versione_codice()}")
    if dup:
        print(f"ATTENZIONE: {a.out} contiene gia' {dup} righe duplicate: "
              "vanno rimosse prima di qualunque analisi.", flush=True)

    rows = []
    for s in range(a.seed_offset, a.seed_offset + a.seeds):
        if (fr, s) in gia and not a.forza:
            print(f"[seme {s}] gia' presente con questa configurazione, "
                  "saltato (--forza per rieseguire)", flush=True)
            continue
        print(f"[seme {s}]", flush=True)
        r = run_seed(s, a.steps, a.traccia, a.n_tracce, lh, a.verbose, a.w0)
        r.update(corsa.intestazione(config))
        corsa.scrivi(a.out, r)
        rows.append(r)
        print(f"  finestre   teacher forcing {r['mmd_teacher_forcing']:.5f}   "
              f"generazione libera {r['mmd_generazione_libera']:.5f}   "
              f"deriva x{r['rapporto_deriva']:.1f}", flush=True)
        print(f"  MMD per passo t=0..{a.traccia-1}: "
              + " ".join(f"{v:.3f}" for v in r["mmd_per_passo"]), flush=True)
        print(f"  autocorr. lag 1, tre letture:", flush=True)
        print(f"    residua (livello noto)  reale "
              f"{r['acf_reale_lag1']:+.3f}  sintetica "
              f"{r['acf_sintetica_lag1']:+.3f}  teorica "
              f"{r['acf_teorica_lag1']:+.3f}  MAE "
              f"{r['mae_acf_residua']:.3f}", flush=True)
        print(f"    intra (media empirica)  reale "
              f"{r['acf_reale_lag1_intra']:+.3f}  sintetica "
              f"{r['acf_sintetica_lag1_intra']:+.3f}  "
              f"baseline senza memoria {r['acf_nulla_lag1_intra']:+.3f}",
              flush=True)
        print(f"    aggregata (storica)     reale "
              f"{r['acf_reale_lag1_aggregata']:+.3f}  sintetica "
              f"{r['acf_sintetica_lag1_aggregata']:+.3f}  "
              f"baseline senza memoria {r['acf_nulla_lag1_aggregata']:+.3f}",
              flush=True)
        if abs(r["acf_nulla_lag1_aggregata"]
               - r["acf_reale_lag1_aggregata"]) < 0.05:
            print("    la lettura aggregata coincide con la baseline a "
                  "memoria distrutta: su questi dati non misura la dinamica "
                  "temporale, misura le differenze di livello fra tracce.",
                  flush=True)
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

    ar = np.array([r["acf_reale_lag1"] for r in rows])
    af = np.array([r["acf_sintetica_lag1"] for r in rows])
    print(f"autocorr.   residua lag 1: reale {ar.mean():+.3f}  sintetica "
          f"{af.mean():+.3f}  teorica {PHI:+.3f}")
    if abs(ar.mean() - PHI) > 0.1:
        print("           la lettura residua sulle tracce REALI si discosta "
              "dal valore teorico: verificare la metrica prima del modello.")

    print("\nLettura. Il rapporto fra MMD al primo e all'ultimo passo misura")
    print("l'accumulo dell'errore lungo la traccia: se cresce molto, il")
    print("teacher forcing di §6 non basta e serve una mitigazione.")
    print("La variabilita' nel tempo molto sotto 1 indica tracce che")
    print("collassano su un valore fisso, cioe' perdita della dinamica.")


if __name__ == "__main__":
    main()