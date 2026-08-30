"""
Baseline di ricampionamento: SMOTE e alternative classiche.

Completa la Fase 1-A. Senza queste misure il confronto con la QGAN e'
incompleto: un revisore chiedera' perche' serva un modello generativo
quando esistono metodi di ricampionamento che costano nulla.

METODI CONFRONTATI
  none              nessun intervento (dai risultati di baseline.py)
  class_weight      riponderazione, gia' misurata in baseline.py
  random_over       duplicazione casuale della minoritaria
  smote             interpolazione fra i k vicini piu' prossimi
  borderline_smote  SMOTE limitato ai campioni vicini alla frontiera
  adasyn            densita' di generazione adattiva
  smote_tomek       SMOTE seguito da pulizia dei link di Tomek

AVVERTENZA SU k
  A 1:1000 restano 53 campioni minoritari. SMOTE interpola fra i k vicini,
  con k=5 di default: con cosi' pochi punti va ridotto, altrimenti
  fallisce o produce campioni degeneri. Qui k e' adattato alla numerosita'.

METODO
  Valutazione su validation, cinque semi. Il test resta intatto.

Uso:
    python src/ids/resampling.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score)

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import (ADASYN, BorderlineSMOTE, RandomOverSampler,
                                    SMOTE)

DATA = Path("data/ids")
OUT = Path("results")

CLASSES = ["probe", "r2l"]
RATIOS = ["1_200", "1_500", "1_1000"]
MODE = "global"          # la variante PCA non e' il fuoco di questo confronto
SEEDS = [1, 2, 3, 4, 5]


def make_sampler(name, n_minority, seed):
    """k adattato: SMOTE richiede k_neighbors < numero di campioni."""
    k = max(1, min(5, n_minority - 1))
    if name == "random_over":
        return RandomOverSampler(random_state=seed)
    if name == "smote":
        return SMOTE(random_state=seed, k_neighbors=k)
    if name == "borderline_smote":
        return BorderlineSMOTE(random_state=seed, k_neighbors=k,
                               m_neighbors=max(1, min(10, n_minority - 1)))
    if name == "adasyn":
        return ADASYN(random_state=seed, n_neighbors=k)
    if name == "smote_tomek":
        return SMOTETomek(random_state=seed,
                          smote=SMOTE(random_state=seed, k_neighbors=k))
    raise ValueError(name)


def load(cls, ratio):
    tr = pd.read_csv(DATA / f"{cls}_train_{ratio}_{MODE}.csv")
    va = pd.read_csv(DATA / f"{cls}_val.csv")
    feat = [c for c in tr.columns if c != "y"]
    return tr[feat].values, tr["y"].values, va[feat].values, va["y"].values


def evaluate(Xtr, ytr, Xva, yva, seed):
    clf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                 n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xva)
    proba = clf.predict_proba(Xva)[:, 1]
    return {
        "f1": f1_score(yva, pred, zero_division=0),
        "precision": precision_score(yva, pred, zero_division=0),
        "recall": recall_score(yva, pred, zero_division=0),
        "pr_auc": average_precision_score(yva, proba),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    methods = ["none", "class_weight", "random_over", "smote",
               "borderline_smote", "adasyn", "smote_tomek"]
    rows = []

    for cls in CLASSES:
        print(f"\n=== {cls} ===")
        for ratio in RATIOS:
            Xtr, ytr, Xva, yva = load(cls, ratio)
            n_min = int(ytr.sum())
            print(f"\n  {ratio}  (minoritaria: {n_min})")
            print(f"  {'metodo':>17} {'F1':>15} {'recall':>15} "
                  f"{'PR-AUC':>15}  n_sint")

            for meth in methods:
                runs, n_gen = [], 0
                for s in SEEDS:
                    try:
                        if meth == "none":
                            Xs, ys = Xtr, ytr
                            clf_bal = None
                        elif meth == "class_weight":
                            Xs, ys = Xtr, ytr
                            clf_bal = "balanced"
                        else:
                            samp = make_sampler(meth, n_min, s)
                            Xs, ys = samp.fit_resample(Xtr, ytr)
                            clf_bal = None
                            n_gen = int(ys.sum()) - n_min

                        clf = RandomForestClassifier(
                            n_estimators=200, random_state=s, n_jobs=-1,
                            class_weight=clf_bal)
                        clf.fit(Xs, ys)
                        pred = clf.predict(Xva)
                        proba = clf.predict_proba(Xva)[:, 1]
                        runs.append({
                            "f1": f1_score(yva, pred, zero_division=0),
                            "precision": precision_score(yva, pred,
                                                         zero_division=0),
                            "recall": recall_score(yva, pred,
                                                   zero_division=0),
                            "pr_auc": average_precision_score(yva, proba),
                        })
                    except Exception as e:
                        print(f"  {meth:>17}  FALLITO: {type(e).__name__}")
                        break

                if not runs:
                    continue
                agg = {k: (float(np.mean([r[k] for r in runs])),
                           float(np.std([r[k] for r in runs])))
                       for k in runs[0]}
                print(f"  {meth:>17} "
                      f"{agg['f1'][0]:>8.4f}±{agg['f1'][1]:.3f} "
                      f"{agg['recall'][0]:>8.4f}±{agg['recall'][1]:.3f} "
                      f"{agg['pr_auc'][0]:>8.4f}±{agg['pr_auc'][1]:.3f} "
                      f" {n_gen:>6}")
                rows.append({"class": cls, "ratio": ratio, "method": meth,
                             "n_minority": n_min, "n_synthetic": n_gen,
                             **{f"{k}_mean": v[0] for k, v in agg.items()},
                             **{f"{k}_std": v[1] for k, v in agg.items()}})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "resampling.csv", index=False)

    print("\n\nGuadagno rispetto a 'none' (delta F1):")
    piv = df.pivot_table(index=["class", "ratio"], columns="method",
                         values="f1_mean")
    for m in [c for c in piv.columns if c != "none"]:
        piv[m] = piv[m] - piv["none"]
    print(piv.drop(columns="none").round(4).to_string())

    print(f"\nrisultati in {OUT/'resampling.csv'}")
    print("\nLa QGAN dovra' battere il migliore di questi metodi, non 'none'.")


if __name__ == "__main__":
    main()