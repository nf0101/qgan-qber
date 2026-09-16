"""
Attributore baseline classico per il dominio QKD (Fase 1 - Parte B).

Stima il vettore delle frazioni causali dell'errore
    y = (f_eve, f_canale, f_rivelatore)
a partire dalle sole feature osservabili.

PROTOCOLLI
  random      finestre mescolate: limite superiore ottimistico, perche'
              finestre della stessa cella finiscono in train e test
  seed        addestra su un sottoinsieme di semi, valuta sui restanti:
              stabilita' rispetto alla realizzazione stocastica
  interior    esclude celle INTERNE alla griglia: interpolazione fra
              regimi non simulati. Termine di paragone per RQ2
  boundary    esclude celle di BORDO: estrapolazione, capacita' distinta
              e riportata separatamente

MODELLI
  Tre regressori con capacita' e bias induttivi diversi. Se degradano tutti
  allo stesso modo sull'interpolazione, il limite e' del problema e non
  del modello scelto.

METODO
  - il seme di test e' riservato e non viene toccato salvo con --final
  - ogni combinazione protocollo-modello e' ripetuta N_REPEATS volte
  - ogni risultato e' accompagnato dal riferimento banale (media del
    training) e dallo skill score, senza i quali il MAE non e'
    interpretabile in assoluto

Uso:
    python src/qkd/baseline_attributor.py
    python src/qkd/baseline_attributor.py --data path/to/grid.csv
    python src/qkd/baseline_attributor.py --models rf            # solo uno
    python src/qkd/baseline_attributor.py --final                # test finale
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             precision_score, recall_score)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# =====================================================================
# SCHEMA DEL DATASET
# =====================================================================
# Nomi canonici con i rispettivi alias accettati, cosi' che lo script
# resti applicabile a dataset generati con nomenclature diverse.

# SPEC-FASE2 §1.2 fissa CINQUE osservabili. `basis_asymmetry` e' stata
# esclusa in §1.4: e' il valore assoluto della differenza fra qber_rect e
# qber_diag, quindi ridondante per costruzione, e la verifica sperimentale non
# ha trovato differenze distinguibili dal rumore in alcun protocollo.
#
# Il numero di feature vincola il numero di qubit del generatore e deve restare
# allineato al dominio IDS, altrimenti O1 non e' misurabile. Aggiungerne una
# qui invaliderebbe il confronto.
FEATURE_ALIASES = {
    "qber": ["qber", "qber_tot", "qber_total"],
    "qber_rect": ["qber_rect", "qber_z", "qber_rectilinear"],
    "qber_diag": ["qber_diag", "qber_x", "qber_diagonal"],
    "sift_rate": ["sift_rate", "sifting_rate", "sift"],
    "loss_rate": ["loss_rate", "channel_loss_rate", "loss"],
}

# Osservabili verificate e deliberatamente non impiegate, con il riferimento
# alla verifica. Elencate perche' la loro assenza sia una scelta documentata e
# non una dimenticanza.
FEATURE_ESCLUSE = {
    "basis_asymmetry": "SPEC §1.4: ridondante, |qber_rect - qber_diag|",
    "qber_stability": "SPEC §1.4: verificata, nessun contributo marginale",
}

TARGET_ALIASES = {
    "f_eve": ["f_eve", "f_attacco", "f_attack"],
    "f_canale": ["f_canale", "f_rumore", "f_noise", "f_channel"],
    "f_rivelatore": ["f_rivelatore", "f_detector", "f_det"],
}

CELL_ALIASES = {
    "intercept_rate": ["intercept_rate", "attack_rate", "eve_rate"],
    "length_km": ["length_km", "length", "distance_km"],
    "depolar_total": ["depolar_total", "depolar", "depolar_rate"],
}

SEED_ALIASES = ["seed", "replicate", "run"]

# Prefissi di colonne derivate dalla verita' causale: se finissero fra le
# feature, il compito diventerebbe banale in modo silenzioso.
LEAKY_PREFIXES = ("qber_cf_", "cf_", "f_")

TEST_SEED = 5
N_REPEATS = 15
OUT = Path("results")

# Celle interne: nessun asse al proprio valore estremo. Calcolate dai dati,
# non cablate, cosi' da restare valide se la griglia cambia.


# =====================================================================
# CARICAMENTO
# =====================================================================

def resolve(df, aliases, kind):
    """Mappa nome canonico -> nome effettivo nel dataframe."""
    found, missing = {}, []
    for canon, names in aliases.items():
        hit = next((n for n in names if n in df.columns), None)
        if hit is None:
            missing.append(f"{canon} (alias: {', '.join(names)})")
        else:
            found[canon] = hit
    if missing:
        raise ValueError(f"colonne {kind} non trovate:\n  " +
                         "\n  ".join(missing))
    return found


def load(path):
    df = pd.read_csv(path)

    feat = resolve(df, FEATURE_ALIASES, "osservabili")
    targ = resolve(df, TARGET_ALIASES, "target")
    cell = resolve(df, CELL_ALIASES, "di cella")

    seed_col = next((n for n in SEED_ALIASES if n in df.columns), None)
    if seed_col is None:
        raise ValueError(f"colonna del seme non trovata "
                         f"(alias: {', '.join(SEED_ALIASES)})")

    # controllo di fuga: nessuna feature deve derivare dalla verita' causale
    if len(feat) != 5:
        raise ValueError(
            f"{len(feat)} osservabili invece di 5. Il numero di feature e'"
            " congelato da SPEC-FASE2 §1.1 e determina il numero di qubit del"
            " generatore: cambiarlo qui rende O1 non misurabile.")

    leaky = [c for c in feat.values()
             if c.startswith(LEAKY_PREFIXES) and c not in targ.values()]
    if leaky:
        raise ValueError(f"possibile fuga di informazione fra le feature: "
                         f"{leaky}")

    return df, list(feat.values()), list(targ.values()), \
        list(cell.values()), seed_col


def interior_cells(df, cell_cols):
    """Celle non estreme su alcun asse. Gli estremi sono dedotti dai dati."""
    m = np.ones(len(df), dtype=bool)
    for c in cell_cols:
        vals = np.sort(df[c].unique())
        if len(vals) > 2:
            m &= df[c].isin(vals[1:-1])
        else:
            m &= False          # asse senza valori interni
    return df.loc[m, cell_cols].drop_duplicates().reset_index(drop=True)


def boundary_cells(df, cell_cols):
    allc = df[cell_cols].drop_duplicates().reset_index(drop=True)
    inter = interior_cells(df, cell_cols)
    if len(inter) == 0:
        return allc
    merged = allc.merge(inter.assign(_i=1), on=cell_cols, how="left")
    return allc[merged["_i"].isna()].reset_index(drop=True)


def nonextreme_on(df, cell_cols, free_col):
    """Celle non estreme su tutti gli assi tranne free_col, che resta libero.

    Serve alla variante binaria: fra le celle interne non compare mai
    intercept_rate = 0, che e' un estremo, quindi il test non conterrebbe
    la classe negativa.
    """
    m = np.ones(len(df), dtype=bool)
    for c in cell_cols:
        if c == free_col:
            continue
        vals = np.sort(df[c].unique())
        if len(vals) > 2:
            m &= df[c].isin(vals[1:-1])
    return df.loc[m, cell_cols].drop_duplicates().reset_index(drop=True)


# =====================================================================
# MODELLI
# =====================================================================

def make_model(name, rs):
    if name == "rf":
        return MultiOutputRegressor(RandomForestRegressor(
            n_estimators=100, random_state=rs, n_jobs=-1))
    if name == "hgb":
        return MultiOutputRegressor(HistGradientBoostingRegressor(
            random_state=rs))
    if name == "ridge":
        return MultiOutputRegressor(
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)))
    raise ValueError(name)


# =====================================================================
# VALUTAZIONE
# =====================================================================

def fit_predict(tr, te, feat, targ, model_name, rs):
    Xtr, ytr, Xte, yte = tr[feat], tr[targ], te[feat], te[targ]

    m = make_model(model_name, rs).fit(Xtr, ytr)
    pred = m.predict(Xte)
    pred_d = DummyRegressor(strategy="mean").fit(Xtr, ytr).predict(Xte)

    mae = mean_absolute_error(yte, pred, multioutput="raw_values")
    mae_d = mean_absolute_error(yte, pred_d, multioutput="raw_values")
    return {
        "mae": float(np.mean(mae)),
        "mae_dummy": float(np.mean(mae_d)),
        "skill": float(1 - np.mean(mae) / np.mean(mae_d)),
        **{f"mae_{t}": float(v) for t, v in zip(targ, mae)},
        **{f"skill_{t}": float(1 - v / d) for t, v, d in zip(targ, mae, mae_d)},
        "n_train": len(tr), "n_test": len(te),
    }


def split_random(df, rs, **kw):
    rng = np.random.RandomState(rs)
    idx = rng.permutation(len(df))
    cut = int(0.8 * len(df))
    return df.iloc[idx[:cut]], df.iloc[idx[cut:]]


def split_seed(df, rs, seed_col=None, seeds=None, **kw):
    v = seeds[rs % len(seeds)]
    return df[df[seed_col] != v], df[df[seed_col] == v]


def split_cells(df, rs, cells=None, cell_cols=None, frac=0.5, **kw):
    rng = np.random.RandomState(rs)
    n = max(1, int(round(len(cells) * frac)))
    pick = cells.iloc[rng.choice(len(cells), size=n, replace=False)]
    merged = df.merge(pick.assign(_te=1), on=cell_cols, how="left")
    mask = merged["_te"].notna().values
    return df[~mask], df[mask]


def repeat(splitter, df, feat, targ, model_name, n=N_REPEATS, **kw):
    runs = []
    for r in range(n):
        tr, te = splitter(df, r, **kw)
        if len(tr) == 0 or len(te) == 0:
            continue
        runs.append(fit_predict(tr, te, feat, targ, model_name, r))
    if not runs:
        return None
    keys = [k for k in runs[0] if not k.startswith("n_")]
    agg = {k: (float(np.mean([x[k] for x in runs])),
               float(np.std([x[k] for x in runs]))) for k in keys}
    agg["_n"] = (runs[0]["n_train"], runs[0]["n_test"])
    return agg


# =====================================================================
# VARIANTE BINARIA
# =====================================================================

def binary(df, cells, feat, cell_cols, attack_col, n=N_REPEATS):
    runs = []
    for r in range(n):
        tr, te = split_cells(df, r, cells=cells, cell_cols=cell_cols)
        if len(te) == 0:
            continue
        ytr = (tr[attack_col] > 0).astype(int)
        yte = (te[attack_col] > 0).astype(int)
        if yte.nunique() < 2 or ytr.nunique() < 2:
            continue
        clf = RandomForestClassifier(n_estimators=100, random_state=r,
                                     n_jobs=-1).fit(tr[feat], ytr)
        p = clf.predict(te[feat])
        d = DummyClassifier(strategy="most_frequent").fit(tr[feat], ytr)
        runs.append({
            "acc": accuracy_score(yte, p),
            "f1": f1_score(yte, p, average="macro"),
            "precision": precision_score(yte, p, zero_division=0),
            "recall": recall_score(yte, p, zero_division=0),
            "acc_dummy": accuracy_score(yte, d.predict(te[feat])),
        })
    if not runs:
        print("  nessuna ripetizione valida: il test conteneva una sola "
              "classe. Verificare l'insieme di celle scelto.")
        return None
    return {k: (float(np.mean([x[k] for x in runs])),
                float(np.std([x[k] for x in runs]))) for k in runs[0]}


# =====================================================================
# MAIN
# =====================================================================

def main(path, models, final=False):
    df, feat, targ, cell_cols, seed_col = load(path)
    print(f"dataset : {len(df):,} righe")
    print(f"feature : {feat}  ({len(feat)}, SPEC-FASE2 §1.2)")
    presenti = [c for c in FEATURE_ESCLUSE if c in df.columns]
    if presenti:
        print(f"escluse : {presenti}")
        for c in presenti:
            print(f"          {c}: {FEATURE_ESCLUSE[c]}")
    print(f"target  : {targ}")
    print(f"celle   : {cell_cols}, seme '{seed_col}'")
    print(f"config. fisiche: {df[cell_cols].drop_duplicates().shape[0]}, "
          f"semi {sorted(df[seed_col].unique())}")

    if final:
        print(f"\nMODALITA' FINALE: incluso il seme {TEST_SEED}\n")
        work = df
    else:
        work = df[df[seed_col] != TEST_SEED].copy()
        print(f"seme {TEST_SEED} riservato: {len(work):,} righe di lavoro\n")

    seeds = sorted(work[seed_col].unique())
    inter = interior_cells(work, cell_cols)
    bound = boundary_cells(work, cell_cols)
    print(f"celle interne {len(inter)}, di bordo {len(bound)}, "
          f"ripetizioni {N_REPEATS}\n")

    protocols = {
        "random": (split_random, {}),
        "seed": (split_seed, {"seed_col": seed_col, "seeds": seeds}),
        "interior": (split_cells, {"cells": inter, "cell_cols": cell_cols}),
        "boundary": (split_cells, {"cells": bound, "cell_cols": cell_cols,
                                   "frac": 0.3}),
    }

    results = {}
    hdr = f"{'protocollo':<10} {'modello':<6} {'MAE':>16} {'banale':>8} {'skill':>16}"
    print(hdr)
    print("-" * len(hdr))
    for pname, (splitter, kw) in protocols.items():
        for mname in models:
            r = repeat(splitter, work, feat, targ, mname, **kw)
            if r is None:
                continue
            results[f"{pname}|{mname}"] = r
            print(f"{pname:<10} {mname:<6} "
                  f"{r['mae'][0]:>8.4f}±{r['mae'][1]:.4f} "
                  f"{r['mae_dummy'][0]:>8.4f} "
                  f"{r['skill'][0]:>8.3f}±{r['skill'][1]:.3f}")
        print()

    # skill per componente, solo per il primo modello
    m0 = models[0]
    print(f"Skill per componente ({m0}):")
    print(f"{'protocollo':<10} " + " ".join(f"{t:>14}" for t in targ))
    for pname in protocols:
        k = f"{pname}|{m0}"
        if k in results:
            print(f"{pname:<10} " + " ".join(
                f"{results[k]['skill_' + t][0]:>14.3f}" for t in targ))

    # variante binaria
    attack_col = cell_cols[0]
    bcells = nonextreme_on(work, cell_cols, free_col=attack_col)
    print(f"\nVariante binaria (celle non estreme salvo '{attack_col}': "
          f"{len(bcells)}):")
    b = binary(work, bcells, feat, cell_cols, attack_col)
    if b:
        print(f"  accuracy {b['acc'][0]:.4f} ± {b['acc'][1]:.4f}  "
              f"(banale {b['acc_dummy'][0]:.4f})")
        print(f"  F1 macro {b['f1'][0]:.4f} ± {b['f1'][1]:.4f}  "
              f"precision {b['precision'][0]:.4f}  "
              f"recall {b['recall'][0]:.4f}")
        results["binary"] = b

    OUT.mkdir(parents=True, exist_ok=True)
    name = "attributor_final.json" if final else "attributor_baseline.json"
    (OUT / name).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nrisultati in {OUT/name}")
    print("\nSkill: 0 = come la media del training, 1 = predizione perfetta.")
    print("Il divario fra 'random' e 'interior' e' lo spazio per RQ2.")
    print("Se i tre modelli degradano allo stesso modo su 'interior', il")
    print("limite e' del problema e non del modello.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/qber_grid.csv")
    ap.add_argument("--models", default="rf,hgb,ridge")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    main(a.data, [m.strip() for m in a.models.split(",")], final=a.final)