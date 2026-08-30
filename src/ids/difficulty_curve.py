"""
Curva di difficolta': dove la baseline comincia a degradare?

Il compito Probe-vs-normal a 1:50 e' risultato troppo facile (F1 ~ 0.99):
in quel regime l'augmentation non ha spazio per migliorare nulla. Questo
script cerca il regime in cui il problema diventa non banale, esplorando
due leve:

  - il RAPPORTO di sbilanciamento, da 1:10 fino a 1:1000
  - la CLASSE minoritaria: probe (facile) contro r2l (notoriamente difficile)

Valutazione su validation. Il test non viene toccato.

Uso:
    python src/ids/difficulty_curve.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nslkdd_common import SEED, _read, _to_binary, CATEGORICAL, TOP_SERVICES
from sklearn.model_selection import train_test_split

K = 6
RATIOS = [10, 50, 100, 200, 500, 1000]
SEEDS = [1, 2, 3]
OUT = Path("")

CLASSES = {
    "probe": {"satan", "ipsweep", "portsweep", "nmap", "mscan", "saint"},
    "r2l": {"ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
            "warezclient", "warezmaster", "sendmail", "named",
            "snmpgetattack", "snmpguess", "xlock", "xsnoop", "worm"},
}


def build(minority):
    """Carica, filtra su normal vs classe scelta, codifica, splitta."""
    def prep(split):
        d = _read(split)
        keep = d["class"].isin(minority) | (d["class"] == "normal")
        d = d[keep].copy()
        d["y"] = d["class"].isin(minority).astype(int)
        return d.drop(columns=["class"])

    tr_full, te = prep("Train"), prep("Test")
    tr, va = train_test_split(tr_full, test_size=0.2,
                              stratify=tr_full["y"], random_state=SEED)
    top = tr["service"].value_counts().head(TOP_SERVICES).index.tolist()

    def enc(d):
        d = d.copy()
        d["service"] = d["service"].where(d["service"].isin(top), "other")
        return pd.get_dummies(d, columns=CATEGORICAL)

    tr_e = enc(tr)
    va_e = enc(va).reindex(columns=tr_e.columns, fill_value=0)
    feat = [c for c in tr_e.columns if c != "y"]
    return tr_e[feat], tr_e["y"].values, va_e[feat], va_e["y"].values


def subsample(y, ratio, rng):
    normal = np.flatnonzero(y == 0)
    minor = np.flatnonzero(y == 1)
    n = len(normal) // ratio
    if n < 5:
        return None
    if n > len(minor):
        return None
    keep = rng.choice(minor, size=n, replace=False)
    return np.concatenate([normal, keep])


def run(Xtr, ytr, Xva, yva, ratio, seed):
    rng = np.random.RandomState(seed)
    idx = subsample(ytr, ratio, rng)
    if idx is None:
        return None
    Xs, ys = Xtr.iloc[idx], ytr[idx]

    sc = StandardScaler().fit(Xs)
    pca = PCA(n_components=K, random_state=seed).fit(sc.transform(Xs))
    mm = MinMaxScaler().fit(pca.transform(sc.transform(Xs)))

    def proj(X):
        return np.clip(mm.transform(pca.transform(sc.transform(X))), 0, 1)

    clf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                 n_jobs=-1, class_weight="balanced")
    clf.fit(proj(Xs), ys)
    P = proj(Xva)
    pred = clf.predict(P)
    proba = clf.predict_proba(P)[:, 1]
    return {
        "n_minor": int(ys.sum()),
        "f1": f1_score(yva, pred, zero_division=0),
        "recall": recall_score(yva, pred, zero_division=0),
        "pr_auc": average_precision_score(yva, proba),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []

    for cls_name, labels in CLASSES.items():
        Xtr, ytr, Xva, yva = build(labels)
        print(f"\n=== classe minoritaria: {cls_name} ===")
        print(f"training {Xtr.shape}, minoritaria {ytr.sum():,} "
              f"({ytr.mean():.2%})   validation {Xva.shape}")
        print(f"\n{'ratio':>7} {'n_min':>7} {'F1':>16} {'recall':>16} "
              f"{'PR-AUC':>16}")

        for ratio in RATIOS:
            runs = [run(Xtr, ytr, Xva, yva, ratio, s) for s in SEEDS]
            runs = [r for r in runs if r is not None]
            if not runs:
                print(f"{ratio:>7} {'--':>7}  (troppo pochi campioni)")
                continue
            agg = {k: (np.mean([r[k] for r in runs]),
                       np.std([r[k] for r in runs]))
                   for k in ("f1", "recall", "pr_auc")}
            n = int(np.mean([r["n_minor"] for r in runs]))
            print(f"{ratio:>7} {n:>7} "
                  f"{agg['f1'][0]:>9.4f}±{agg['f1'][1]:.3f} "
                  f"{agg['recall'][0]:>9.4f}±{agg['recall'][1]:.3f} "
                  f"{agg['pr_auc'][0]:>9.4f}±{agg['pr_auc'][1]:.3f}")
            results.append({"class": cls_name, "ratio": ratio, "n_minor": n,
                            **{f"{k}_mean": float(v[0]) for k, v in agg.items()},
                            **{f"{k}_std": float(v[1]) for k, v in agg.items()}})

    pd.DataFrame(results).to_csv(OUT / "difficulty_curve.csv", index=False)
    print(f"\nrisultati in {OUT/'difficulty_curve.csv'}")
    print("\nCercare il regime in cui F1 scende sotto ~0.85: e' li' che")
    print("l'augmentation ha spazio per dimostrare un guadagno.")


if __name__ == "__main__":
    main()