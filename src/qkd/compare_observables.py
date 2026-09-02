"""
Quale osservabile conservare: basis_asymmetry o qber_stability?

Il vincolo di sei feature impone di sceglierne una sola. Lo script
confronta tre configurazioni sullo stesso dataset ridotto, cosi' da
decidere prima di rigenerare l'intera griglia.

    A  qber, qber_rect, qber_diag, basis_asymmetry, sift_rate, loss_rate
    B  qber, qber_rect, qber_diag, qber_stability,  sift_rate, loss_rate
    5  qber, qber_rect, qber_diag,                  sift_rate, loss_rate

La terza configurazione verifica se la sesta feature serva davvero: se A
e B non superano 5, conviene scendere a cinque qubit.

PREREQUISITO
    Rigenerare una griglia RIDOTTA con windows() modificata, in modo che
    il CSV contenga entrambe le colonne candidate. Bastano poche celle e
    un solo seme: serve un confronto relativo, non una stima assoluta.

Uso:
    python src/qkd/compare_observables.py --data data/qber_grid_small.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

BASE = ["qber", "qber_rect", "qber_diag", "sift_rate", "loss_rate"]
CANDIDATES = {
    "A_basis_asymmetry": BASE + ["basis_asymmetry"],
    "B_qber_stability": BASE + ["qber_stability"],
    "5_senza_sesta": BASE,
}

TARGETS = ["f_eve", "f_canale", "f_rivelatore"]
CELL = ["intercept_rate", "length_km", "depolar_total"]
N_REPEATS = 15
OUT = Path("../../results")


def interior_cells(df):
    m = np.ones(len(df), dtype=bool)
    for c in CELL:
        vals = np.sort(df[c].unique())
        m &= df[c].isin(vals[1:-1]) if len(vals) > 2 else False
    return df.loc[m, CELL].drop_duplicates().reset_index(drop=True)


def split_cells(df, cells, rs, frac=0.5):
    rng = np.random.RandomState(rs)
    n = max(1, int(round(len(cells) * frac)))
    pick = cells.iloc[rng.choice(len(cells), size=n, replace=False)]
    merged = df.merge(pick.assign(_te=1), on=CELL, how="left")
    mask = merged["_te"].notna().values
    return df[~mask], df[mask]


def skill(tr, te, feats, rs):
    m = MultiOutputRegressor(
        HistGradientBoostingRegressor(random_state=rs)).fit(tr[feats],
                                                            tr[TARGETS])
    d = DummyRegressor(strategy="mean").fit(tr[feats], tr[TARGETS])
    mae = mean_absolute_error(te[TARGETS], m.predict(te[feats]),
                              multioutput="raw_values")
    mae_d = mean_absolute_error(te[TARGETS], d.predict(te[feats]),
                                multioutput="raw_values")
    return (float(1 - np.mean(mae) / np.mean(mae_d)),
            {t: float(1 - v / dd) for t, v, dd in zip(TARGETS, mae, mae_d)})


def protocol_random(df, rs):
    rng = np.random.RandomState(rs)
    idx = rng.permutation(len(df))
    cut = int(0.8 * len(df))
    return df.iloc[idx[:cut]], df.iloc[idx[cut:]]


def main(path):
    df = pd.read_csv(path)

    required = set(TARGETS) | set(CELL)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"colonne obbligatorie mancanti: {missing}")

    candidates = {k: v for k, v in CANDIDATES.items()
                  if all(c in df.columns for c in v)}
    if not candidates:
        raise ValueError("nessuna configurazione calcolabile su questo file")
    print(f"configurazioni valutabili: {list(candidates)}\n")

    inter = interior_cells(df)
    print(f"dataset: {len(df):,} righe, "
          f"{df[CELL].drop_duplicates().shape[0]} configurazioni, "
          f"{len(inter)} celle interne\n")

    if len(inter) < 4:
        print("ATTENZIONE: troppe poche celle interne, il protocollo "
              "'interior' sara' molto rumoroso.\n")

    # correlazione fra le due candidate: se molto alta, portano la stessa
    # informazione e la scelta e' indifferente
    if {"basis_asymmetry", "qber_stability"} <= set(df.columns):
        r = df["basis_asymmetry"].corr(df["qber_stability"])
        print(f"correlazione basis_asymmetry / qber_stability: {r:.3f}")
        for t in TARGETS:
            ra = df["basis_asymmetry"].corr(df[t])
            rb = df["qber_stability"].corr(df[t])
            print(f"  corr con {t:<14} A={ra:+.3f}  B={rb:+.3f}")
        print()

    results = {}
    for proto, splitter in [
            ("random", lambda d, r_: protocol_random(d, r_)),
            ("interior", lambda d, r_: split_cells(d, inter, r_))]:
        print(f"--- {proto} ---")
        for name, feats in candidates.items():
            runs = []
            for r_ in range(N_REPEATS):
                tr, te = splitter(df, r_)
                if len(tr) == 0 or len(te) == 0:
                    continue
                runs.append(skill(tr, te, feats, r_))
            if not runs:
                continue
            vals = [x[0] for x in runs]
            per = {t: float(np.mean([x[1][t] for x in runs])) for t in TARGETS}
            results[f"{proto}|{name}"] = {
                "skill_mean": float(np.mean(vals)),
                "skill_std": float(np.std(vals)),
                "per_target": per,
            }
            print(f"  {name:<20} skill {np.mean(vals):.3f} ± "
                  f"{np.std(vals):.3f}   "
                  + "  ".join(f"{t.split('_')[1][:3]}={per[t]:.3f}"
                              for t in TARGETS))
        print()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compare_observables.json").write_text(
        json.dumps(results, indent=2))

    print("Lettura:")
    print("  B > A          -> sostituire, rigenerare l'intera griglia")
    print("  B ~ A          -> tenere A, nessuna rigenerazione necessaria")
    print("  A ~ B ~ 5      -> scendere a cinque feature e cinque qubit")
    print(f"\nrisultati in {OUT/'compare_observables.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/qber_grid_small.csv")
    a = ap.parse_args()
    main(a.data)