"""
Baseline della Parte 1: classificatori addestrati sui dataset sbilanciati,
SENZA augmentation.

Sono i numeri di riferimento contro cui si misurera' il guadagno della
QGAN. Valutazione su validation; il test resta intatto fino ai risultati
finali.

COSA VIENE VARIATO
  - classe minoritaria : probe (facile), r2l (difficile)
  - rapporto           : 1:200, 1:500, 1:1000
  - variante PCA       : global (rappresentazione stabile), local (realistica)
  - classificatore     : Random Forest, Logistic Regression, Gradient Boosting
  - bilanciamento      : con e senza class_weight

METODO
  Cinque semi per configurazione, con media e deviazione standard: e' il
  primo dei quattro requisiti di rigore dichiarati nella proposta. La
  variabilita' fra semi e' essa stessa un risultato: nel regime di
  scarsita' e' alta, e ridurla e' un beneficio che l'augmentation puo'
  produrre oltre al guadagno medio.

Uso:
    python src/ids/baseline.py
    python src/ids/baseline.py --quick     # solo Random Forest, 3 semi
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

DATA = Path("data/ids")
OUT = Path("results")

CLASSES = ["probe", "r2l"]
RATIOS = ["1_200", "1_500", "1_1000"]
MODES = ["global", "local"]
SEEDS = [1, 2, 3, 4, 5]


def make_model(name, seed, balanced):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1,
            class_weight="balanced" if balanced else None)
    if name == "lr":
        return LogisticRegression(
            max_iter=2000, random_state=seed,
            class_weight="balanced" if balanced else None)
    if name == "gb":
        return HistGradientBoostingClassifier(
            random_state=seed,
            class_weight="balanced" if balanced else None)
    raise ValueError(name)


def load(cls, ratio, mode):
    """Restituisce (train, validation) coerenti con la variante di PCA."""
    tr = pd.read_csv(DATA / f"{cls}_train_{ratio}_{mode}.csv")
    if mode == "global":
        va = pd.read_csv(DATA / f"{cls}_val.csv")
    else:
        va = pd.read_csv(DATA / f"{cls}_val_{ratio}_local.csv")
    feat = [c for c in tr.columns if c != "y"]
    return tr[feat].values, tr["y"].values, va[feat].values, va["y"].values


def evaluate(model_name, Xtr, ytr, Xva, yva, seed, balanced):
    m = make_model(model_name, seed, balanced)
    m.fit(Xtr, ytr)

    pred = m.predict(Xva)
    proba = m.predict_proba(Xva)[:, 1]
    return {
        "f1": f1_score(yva, pred, zero_division=0),
        "precision": precision_score(yva, pred, zero_division=0),
        "recall": recall_score(yva, pred, zero_division=0),
        "pr_auc": average_precision_score(yva, proba),
        "roc_auc": roc_auc_score(yva, proba),
    }


def main(quick=False):
    OUT.mkdir(parents=True, exist_ok=True)
    models = ["rf"] if quick else ["rf", "lr", "gb"]
    seeds = SEEDS[:3] if quick else SEEDS
    balanced_opts = [True] if quick else [True, False]

    rows = []
    for cls in CLASSES:
        for ratio in RATIOS:
            for mode in MODES:
                Xtr, ytr, Xva, yva = load(cls, ratio, mode)
                n_min = int(ytr.sum())
                for mname in models:
                    for bal in balanced_opts:
                        runs = [evaluate(mname, Xtr, ytr, Xva, yva, s, bal)
                                for s in seeds]
                        agg = {}
                        for k in runs[0]:
                            v = [r[k] for r in runs]
                            agg[f"{k}_mean"] = float(np.mean(v))
                            agg[f"{k}_std"] = float(np.std(v))
                        rows.append({
                            "class": cls, "ratio": ratio, "pca": mode,
                            "model": mname, "balanced": bal,
                            "n_minority": n_min, "n_seeds": len(seeds),
                            **agg})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "baseline.csv", index=False)

    # ---- stampa leggibile ------------------------------------------
    print("BASELINE senza augmentation (validation)\n")
    for cls in CLASSES:
        print(f"=== {cls} ===")
        sub = df[(df["class"] == cls) & (df["balanced"])]
        print(f"{'ratio':>7} {'n_min':>6} {'pca':>7} {'model':>6} "
              f"{'F1':>15} {'recall':>15} {'PR-AUC':>15}")
        for _, r in sub.iterrows():
            print(f"{r['ratio']:>7} {r['n_minority']:>6} {r['pca']:>7} "
                  f"{r['model']:>6} "
                  f"{r['f1_mean']:>8.4f}±{r['f1_std']:.3f} "
                  f"{r['recall_mean']:>8.4f}±{r['recall_std']:.3f} "
                  f"{r['pr_auc_mean']:>8.4f}±{r['pr_auc_std']:.3f}")
        print()

    # ---- confronto global vs local: la rappresentazione e' un limite? --
    print("Effetto della variante PCA (differenza local - global, F1):")
    piv = df[df["balanced"]].pivot_table(
        index=["class", "ratio", "model"], columns="pca", values="f1_mean")
    piv["delta"] = piv["local"] - piv["global"]
    print(piv.round(4).to_string())
    print("\nSe delta e' vicino a zero, la rappresentazione non e' il collo")
    print("di bottiglia: il degrado viene dal classificatore, ed e' li' che")
    print("l'augmentation puo' agire.")

    # ---- il modello migliore per configurazione ---------------------
    best = (df[df["balanced"]]
            .sort_values("f1_mean", ascending=False)
            .groupby(["class", "ratio", "pca"]).first()
            .reset_index()[["class", "ratio", "pca", "model",
                            "f1_mean", "recall_mean"]])
    (OUT / "baseline_best.json").write_text(
        best.to_json(orient="records", indent=2))
    print(f"\nrisultati completi in {OUT/'baseline.csv'}")
    print(f"migliori per configurazione in {OUT/'baseline_best.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(quick=a.quick)