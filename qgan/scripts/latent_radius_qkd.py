"""Raggio del latente per il dominio QKD, secondo la regola dell'Appendice B.2.

    Il raggio del latente si sceglie perche' la dispersione condizionale
    generata eguagli la dispersione entro cella dei dati.

Lo script non addestra nulla: misura la dispersione sui dati reali, misura la
dispersione del generatore a raggi diversi, e trova il raggio che le fa
coincidere. Costa pochi secondi.

DUE DEFINIZIONI DI "DISPERSIONE CONDIZIONALE", E NON SONO INTERCAMBIABILI

Il condizionamento e' `c = (intercept_rate, length_km, depolar_total)`, che NON
include il seme. La distribuzione condizionale a `c` fissato comprende quindi
anche la variabilita' fra le realizzazioni stocastiche:

  per configurazione   raggruppa i 5 semi di ciascuna delle 120 configurazioni.
                       E' la definizione CORRETTA per la regola, perche'
                       corrisponde a cio' che il generatore deve riprodurre
                       dato `c`.
  per cella            raggruppa configurazione x seme, 600 celle. E' piu'
                       piccola, perche' esclude la variabilita' fra semi.
                       Usarla darebbe un raggio troppo stretto.

Lo script riporta entrambe e usa la prima. La differenza fra le due e' essa
stessa informativa: se sono molto diverse, buona parte della dispersione a `c`
fissato e' dovuta al seme, e nessun generatore condizionato su `c` puo'
prevederla.

LO SPAZIO DI RISCALAMENTO CONTA

La QGAN riceve valori in [0,1]. La deviazione standard va quindi calcolata
DOPO il riscalamento, non in unita' grezze di QBER: il rumore binomiale su 200
posizioni setacciate vale circa 0,02-0,03 in unita' di QBER, ma diventa un
altro numero dopo la normalizzazione, e il fattore dipende dall'ampiezza della
griglia su ciascun asse.

Lo script stampa i parametri del riscalamento, che vanno riusati identici a
valle: se il generatore viene addestrato con uno scaler diverso da quello con
cui si e' calcolato il raggio, il raggio e' sbagliato.

Uso:
    python scripts/latent_radius_qkd.py --data data/qber_grid.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qgan.circuit import QuantumGenerator  # noqa: E402

# SPEC-FASE2 §1.2: cinque osservabili. `basis_asymmetry` e' esclusa, essendo
# ridondante per costruzione (§1.4).
FEATURES = {
    "qber": ["qber", "qber_tot", "qber_total"],
    "qber_rect": ["qber_rect", "qber_z", "qber_rectilinear"],
    "qber_diag": ["qber_diag", "qber_x", "qber_diagonal"],
    "sift_rate": ["sift_rate", "sifting_rate", "sift"],
    "loss_rate": ["loss_rate", "channel_loss_rate", "loss"],
}
COND = {
    "intercept_rate": ["intercept_rate", "attack_rate", "eve_rate"],
    "length_km": ["length_km", "length", "distance_km"],
    "depolar_total": ["depolar_total", "depolar", "depolar_rate"],
}
SEED_ALIASES = ["seed", "replicate", "run"]
TEST_SEED = 5


def resolve(df, aliases, kind):
    out, missing = {}, []
    for canon, names in aliases.items():
        hit = next((n for n in names if n in df.columns), None)
        (missing.append(canon) if hit is None else out.__setitem__(canon, hit))
    if missing:
        raise ValueError(f"colonne {kind} non trovate: {missing}")
    return out


def minmax(X):
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (X - lo) / span, lo, span


def _gen(d_c=3, seed=0):
    rng = np.random.default_rng(seed)
    g = QuantumGenerator(d_c=d_c)
    return g, rng.uniform(-0.1, 0.1, g.n_params), rng


def dispersion_at(radii_vec, g, th, rng, d_c=3, n=800):
    """Dispersione per feature a `c` fissato, dato un vettore di raggi."""
    radii_vec = np.atleast_1d(radii_vec)
    if radii_vec.size == 1:
        radii_vec = np.repeat(radii_vec, g.n_qubits)
    Z = rng.uniform(0.0, 1.0, (n, g.n_qubits)) * radii_vec[None, :]
    C = np.repeat(rng.random((1, d_c)), n, axis=0)
    return g.forward(th, Z, C).std(axis=0)


def generator_dispersion(radii, d_c=3, n=800, seed=0):
    """Dispersione media a raggio uniforme, per ciascun valore di `radii`."""
    g, th, rng = _gen(d_c, seed)
    return np.array([float(dispersion_at(r, g, th, rng, d_c, n).mean())
                     for r in radii])


def qubit_coupling(g, th, rng, d_c=3, n=800, base=0.7854):
    """Quanto z_j contribuisce alla dispersione della feature i.

    L'entanglement lineare non e' simmetrico: il qubit terminale della catena
    e' sempre bersaglio e mai controllo, quindi la sua marginale dipende poco
    dal proprio latente. Misurato: diagonale 0,99 sui primi quattro qubit e
    0,26 sul quinto. Ne segue che restringere il raggio del quinto qubit NON
    ne riduce la dispersione, e l'osservabile che richiede la dispersione piu'
    stretta non va collocata li'.
    """
    base_v = np.repeat(base, g.n_qubits)
    d0 = dispersion_at(base_v, g, th, rng, d_c, n)
    M = np.zeros((g.n_qubits, g.n_qubits))
    for j in range(g.n_qubits):
        r = base_v.copy()
        r[j] = 0.02
        M[j] = 1.0 - dispersion_at(r, g, th, rng, d_c, n) / d0
    return M


def fit_radii(sigma_target, g, th, rng, d_c=3, n=800, iters=40):
    """Cerca un raggio per feature che eguagli le dispersioni bersaglio.

    Aggiustamento moltiplicativo: i qubit sono accoppiati, quindi non esiste
    una formula chiusa e la correzione va iterata. La convergenza non e'
    garantita per il qubit terminale, che risponde poco al proprio raggio: il
    residuo riportato dice quali feature restano fuori bersaglio.
    """
    r = np.full(g.n_qubits, 0.7854)
    for _ in range(iters):
        d = dispersion_at(r, g, th, rng, d_c, n)
        ratio = np.clip(sigma_target / np.maximum(d, 1e-9), 0.5, 2.0)
        r = np.clip(r * np.sqrt(ratio), 1e-3, np.pi)
    return r, dispersion_at(r, g, th, rng, d_c, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/qber_grid.csv")
    ap.add_argument("--include-test-seed", action="store_true",
                    help="include il seme riservato; per default e' escluso")
    ap.add_argument("--out", default="results/latent_radius_qkd.json")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    feat = resolve(df, FEATURES, "osservabili")
    cond = resolve(df, COND, "di condizionamento")
    seed_col = next((n for n in SEED_ALIASES if n in df.columns), None)
    if seed_col is None:
        raise ValueError(f"colonna del seme non trovata: {SEED_ALIASES}")

    if not a.include_test_seed:
        df = df[df[seed_col] != TEST_SEED].copy()

    fcols = list(feat.values())
    ccols = list(cond.values())
    print(f"righe {len(df):,}   osservabili {fcols}")
    print(f"condizionamento {ccols}   seme '{seed_col}'")
    print(f"configurazioni {df[ccols].drop_duplicates().shape[0]}   "
          f"celle {df[ccols + [seed_col]].drop_duplicates().shape[0]}")

    # --- riscalamento in [0,1], da riusare identico a valle -------------
    Xs, lo, span = minmax(df[fcols].to_numpy(float))
    print("\nriscalamento min-max (da riusare identico nell'addestramento):")
    for j, c in enumerate(fcols):
        print(f"  {c:<12} min {lo[j]:.6f}  ampiezza {span[j]:.6f}")

    S = pd.DataFrame(Xs, columns=fcols, index=df.index)

    # --- dispersione condizionale --------------------------------------
    per_conf = S.groupby([df[c] for c in ccols]).std(ddof=1).mean(axis=1)
    per_cell = S.groupby([df[c] for c in ccols] + [df[seed_col]]).std(ddof=1).mean(axis=1)
    sigma_conf = float(per_conf.mean())
    sigma_cell = float(per_cell.mean())

    print(f"\ndispersione entro configurazione (c fissato, semi raggruppati)")
    print(f"  media {sigma_conf:.5f}   mediana {float(per_conf.median()):.5f}   "
          f"min {float(per_conf.min()):.5f}   max {float(per_conf.max()):.5f}")
    print(f"dispersione entro cella (configurazione x seme)")
    print(f"  media {sigma_cell:.5f}")
    quota = 1 - (sigma_cell / sigma_conf) ** 2 if sigma_conf > 0 else float("nan")
    print(f"  quota di varianza dovuta al seme: {quota:.1%}")
    if quota > 0.5:
        print("  ATTENZIONE: oltre meta' della dispersione a c fissato viene")
        print("  dalla realizzazione stocastica. Nessun generatore condizionato")
        print("  su c puo' prevederla: e' un limite superiore a RQ2, da")
        print("  riportare accanto al risultato.")

    print("\ndispersione per osservabile (entro configurazione):")
    per_feat = S.groupby([df[c] for c in ccols]).std(ddof=1).mean()
    for c in fcols:
        print(f"  {c:<12} {per_feat[c]:.5f}")

    # --- calibrazione del raggio ----------------------------------------
    radii = np.pi * np.array([1.0, 1/2, 1/4, 1/6, 1/8, 1/12, 1/16, 1/24, 1/32])
    disp = generator_dispersion(radii)
    print(f"\ndispersione del generatore all'inizializzazione (c fissato):")
    print(f"{'raggio':>10} {'pi/x':>7} {'dispersione':>13}")
    for r, d in zip(radii, disp):
        print(f"{r:10.4f} {np.pi/r:7.1f} {d:13.5f}")

    # interpolazione in scala log-log, monotona nel tratto utile
    order = np.argsort(disp)
    r_star = float(np.interp(sigma_conf, disp[order], radii[order]))
    print(f"\nraggio che eguaglia la dispersione dei dati ({sigma_conf:.5f}): "
          f"{r_star:.4f}  =  pi/{np.pi/r_star:.1f}")

    d_check = float(generator_dispersion([r_star])[0])
    print(f"verifica: a quel raggio il generatore ha dispersione {d_check:.5f}, "
          f"rapporto {d_check/sigma_conf:.2f}")

    # ---------------------------------------------------------------- #
    # Raggi per feature
    # ---------------------------------------------------------------- #
    sig = np.array([per_feat[c] for c in fcols])
    disparita = sig.max() / max(sig.min(), 1e-12)
    print(f"\ndisparita' fra osservabili: {disparita:.0f}x "
          f"({fcols[int(sig.argmax())]} {sig.max():.5f} contro "
          f"{fcols[int(sig.argmin())]} {sig.min():.5f})")
    if disparita < 3:
        print("  sotto 3x: un raggio unico e' adeguato, il resto e' opzionale")

    g, th, rng = _gen()
    M = qubit_coupling(g, th, rng)
    diag = np.diag(M)
    print(f"\naccoppiamento qubit-feature (diagonale della matrice di influenza):")
    print("  " + "  ".join(f"q{i} {diag[i]:+.2f}" for i in range(len(diag))))
    debole = int(diag.argmin())
    print(f"  qubit meno pilotabile dal proprio latente: q{debole} "
          f"({diag[debole]:.2f})")

    # Ordinamento: sul qubit meno pilotabile va l'osservabile che richiede PIU'
    # dispersione, perche' li' il raggio non riesce comunque a comprimerla.
    ordine_qubit = np.argsort(diag)              # dal meno al piu' pilotabile
    ordine_feat = np.argsort(-sig)               # dalla piu' alla meno dispersa
    assegnazione = [None] * len(fcols)
    for q, f in zip(ordine_qubit, ordine_feat):
        assegnazione[q] = fcols[f]
    print(f"\nordinamento proposto delle colonne (qubit 0..4):")
    for q, c in enumerate(assegnazione):
        print(f"  q{q}  {c:<12} sigma {per_feat[c]:.5f}  "
              f"accoppiamento {diag[q]:+.2f}")
    print("  criterio: sul qubit meno pilotabile l'osservabile che richiede")
    print("  piu' dispersione, perche' li' il raggio non la comprime comunque")

    sig_ord = np.array([per_feat[c] for c in assegnazione])
    r_vec, d_vec = fit_radii(sig_ord, g, th, rng)
    print(f"\nraggi per feature, con l'ordinamento proposto:")
    print(f"{'qubit':>6} {'osservabile':<13} {'raggio':>8} {'sigma atteso':>13} "
          f"{'ottenuto':>10} {'rapporto':>9}")
    for q, c in enumerate(assegnazione):
        print(f"{q:>6} {c:<13} {r_vec[q]:8.4f} {sig_ord[q]:13.5f} "
              f"{d_vec[q]:10.5f} {d_vec[q]/max(sig_ord[q],1e-12):9.2f}")
    fuori = [assegnazione[q] for q in range(len(sig_ord))
             if not 0.5 <= d_vec[q] / max(sig_ord[q], 1e-12) <= 2.0]
    if fuori:
        print(f"  fuori bersaglio oltre il fattore 2: {fuori}")
        print("  l'accoppiamento fra qubit impedisce di soddisfarle tutte;")
        print("  riportare il residuo invece di forzare il raggio")

    out_extra = {
        "disparita_osservabili": float(disparita),
        "accoppiamento_diagonale": diag.tolist(),
        "ordinamento_proposto": assegnazione,
        "latent_hi_per_feature": r_vec.tolist(),
        "dispersione_ottenuta": d_vec.tolist(),
        "fuori_bersaglio": fuori,
    }

    out = {
        "n_righe": int(len(df)),
        "features": fcols,
        "cond": ccols,
        "scaler_min": lo.tolist(),
        "scaler_span": span.tolist(),
        "sigma_per_configurazione": sigma_conf,
        "sigma_per_cella": sigma_cell,
        "quota_varianza_seme": float(quota),
        "sigma_per_feature": {c: float(per_feat[c]) for c in fcols},
        "raggi": radii.tolist(),
        "dispersione_generatore": disp.tolist(),
        "latent_hi": r_star,
        "verifica_dispersione": d_check,
        **out_extra,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nscritto {a.out}")
    print(f"\nDa passare all'addestramento:")
    print(f"  raggio unico       --latent-hi {r_star:.4f}")
    print(f"  per feature        --latent-hi "
          f"{','.join(f'{v:.4f}' for v in r_vec)}")
    print(f"  ordine colonne     {','.join(assegnazione)}")
    print("Il ramo NON condizionato resta a pi (Appendice B.5).")


if __name__ == "__main__":
    main()