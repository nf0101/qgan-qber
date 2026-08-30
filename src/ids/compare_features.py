"""
Confronto fra strategie di riduzione a 6 feature.

METODO: la valutazione avviene su un VALIDATION set ritagliato dal training.
Il test non viene mai toccato: sara' usato una sola volta, alla fine, con
la strategia gia' scelta.

La selezione supervisionata e' calcolata sul regime di sbilanciamento piu'
severo (1:50), cosi' da non impiegare mai piu' informazione di quella
disponibile nel caso peggiore.

Strategie confrontate:
    1. PCA-6            direzioni di massima varianza
    2. PCA-20 -> MI-6   20 componenti, poi le 6 piu' informative su y
    3. MI-6 grezze      le 6 colonne originali piu' informative
    4. RF-6 grezze      le 6 colonne con maggiore importanza Random Forest
    rif. tutte le colonne codificate

Uso:
    python src/ids/compare_features.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nslkdd_common import SEED, load_splits, subsample_indices

K = 6
SELECTION_RATIO = 50          # regime piu' severo: 1:50
OUT = Path("data/ids")


def score(Xtr, ytr, Xva, yva):
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED,
                                 n_jobs=-1, class_weight="balanced")
    clf.fit(Xtr, ytr)
    p = clf.predict(Xva)
    proba = clf.predict_proba(Xva)[:, 1]
    return f1_score(yva, p), average_precision_score(yva, proba)


def main():
    d = load_splits()
    Xtr, ytr, Xva, yva = d["Xtr"], d["ytr"], d["Xva"], d["yva"]
    feat = d["features"]

    print(f"training {Xtr.shape}   validation {Xva.shape}   "
          f"(test non toccato)")
    print(f"probe: train {ytr.mean():.1%}   val {yva.mean():.1%}\n")

    # sottoinsieme di training nel regime piu' severo: e' su questo che
    # si calcolano tutte le decisioni supervisionate
    sel_idx = subsample_indices(ytr, SELECTION_RATIO)
    Xsel, ysel = Xtr.iloc[sel_idx], ytr.iloc[sel_idx]
    print(f"selezione calcolata su regime 1:{SELECTION_RATIO} -> "
          f"{len(Xsel):,} righe, probe {ysel.mean():.2%}\n")

    sc = StandardScaler().fit(Xsel)
    Zsel, Ztr, Zva = sc.transform(Xsel), sc.transform(Xtr), sc.transform(Xva)

    results, choice = {}, {}
    print(f"Strategie di riduzione a {K} feature (valutate su validation):\n")

    # --- 1. PCA-6 -----------------------------------------------------
    pca = PCA(n_components=K, random_state=SEED).fit(Zsel)
    mm = MinMaxScaler().fit(pca.transform(Zsel))
    f1, ap = score(mm.transform(pca.transform(Ztr)), ytr,
                   np.clip(mm.transform(pca.transform(Zva)), 0, 1), yva)
    results["pca6"] = (f1, ap)
    print(f"  1. PCA-6            F1={f1:.4f}  PR-AUC={ap:.4f}  "
          f"var={pca.explained_variance_ratio_.sum():.1%}")

    # --- 2. PCA-20 poi MI ---------------------------------------------
    pca20 = PCA(n_components=20, random_state=SEED).fit(Zsel)
    P_sel, P_tr, P_va = (pca20.transform(Zsel), pca20.transform(Ztr),
                         pca20.transform(Zva))
    mi = mutual_info_classif(P_sel, ysel, random_state=SEED)
    idx = np.sort(np.argsort(mi)[::-1][:K])
    mm2 = MinMaxScaler().fit(P_sel[:, idx])
    f1, ap = score(mm2.transform(P_tr[:, idx]), ytr,
                   np.clip(mm2.transform(P_va[:, idx]), 0, 1), yva)
    results["pca20_mi6"] = (f1, ap)
    choice["components"] = [int(i) + 1 for i in idx]
    print(f"  2. PCA-20 -> MI-6   F1={f1:.4f}  PR-AUC={ap:.4f}  "
          f"componenti {choice['components']}")

    # --- 3. MI sulle colonne grezze -----------------------------------
    mi_raw = mutual_info_classif(Xsel, ysel, random_state=SEED)
    s3 = np.argsort(mi_raw)[::-1][:K]
    mm3 = MinMaxScaler().fit(Xsel.iloc[:, s3])
    f1, ap = score(mm3.transform(Xtr.iloc[:, s3]), ytr,
                   np.clip(mm3.transform(Xva.iloc[:, s3]), 0, 1), yva)
    results["mi6_raw"] = (f1, ap)
    print(f"  3. MI-6 grezze      F1={f1:.4f}  PR-AUC={ap:.4f}")
    print(f"     -> {', '.join(feat[i] for i in s3)}")

    # --- 4. Importanza Random Forest ----------------------------------
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED,
                                n_jobs=-1).fit(Xsel, ysel)
    s4 = np.argsort(rf.feature_importances_)[::-1][:K]
    mm4 = MinMaxScaler().fit(Xsel.iloc[:, s4])
    f1, ap = score(mm4.transform(Xtr.iloc[:, s4]), ytr,
                   np.clip(mm4.transform(Xva.iloc[:, s4]), 0, 1), yva)
    results["rf6_raw"] = (f1, ap)
    print(f"  4. RF-6 grezze      F1={f1:.4f}  PR-AUC={ap:.4f}")
    print(f"     -> {', '.join(feat[i] for i in s4)}")

    # --- riferimento ---------------------------------------------------
    f1, ap = score(Xtr, ytr, Xva, yva)
    results["all_features"] = (f1, ap)
    print(f"\n  rif. tutte le {len(feat)}   F1={f1:.4f}  PR-AUC={ap:.4f}")

    best = max((k for k in results if k != "all_features"),
               key=lambda k: results[k][0])
    print(f"\nmigliore su validation: {best}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "feature_selection.json").write_text(json.dumps({
        "k": K,
        "selection_ratio": SELECTION_RATIO,
        "evaluated_on": "validation (20% del training, stratificato)",
        "test_used": False,
        "results_f1_prauc": {k: [float(a), float(b)]
                             for k, (a, b) in results.items()},
        "best": best,
        "pca20_mi6_components": choice.get("components"),
        "mi6_raw_features": [feat[i] for i in s3],
        "rf6_raw_features": [feat[i] for i in s4],
        "seed": SEED,
    }, indent=2))
    print(f"risultati salvati in {OUT/'feature_selection.json'}")


if __name__ == "__main__":
    main()