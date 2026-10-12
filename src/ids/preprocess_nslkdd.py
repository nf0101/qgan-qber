"""
Preprocessing finale di NSL-KDD per la Parte 1.

Produce sei dataset di training: due classi minoritarie (probe, r2l) per
tre rapporti di sbilanciamento (1:200, 1:500, 1:1000), piu' validation e
test per ciascuna classe.

SCELTE DI PROGETTO

  Riduzione: PCA a 6 componenti. Il confronto con tre alternative
  supervisionate (compare_features.py, valutato su validation) ha mostrato
  differenze entro il rumore: F1 fra 0.981 e 0.993. Si sceglie quindi la
  PCA semplice, che non guarda le etichette ed elimina ogni questione di
  fuga di informazione. Le 6 componenti allineano il dominio IDS a quello
  QKD, dove le osservabili sono sei, cosi' che O1 sia misurabile.

  Classi: probe (facile) e r2l (difficile). A parita' di numerosita' della
  minoritaria danno prestazioni diverse, il che isola l'effetto della
  separabilita' intrinseca da quello della scarsita' di dati.

  Rapporti: 1:200, 1:500, 1:1000. Scelti dalla curva di difficolta'
  (difficulty_curve.py): a rapporti piu' bassi la baseline supera F1 0.97
  e l'augmentation non avrebbe spazio per mostrare alcun guadagno.

  DUE VARIANTI DI PCA, per un'ablazione prevista:
    'local'  PCA adattata sul sottocampione sbilanciato (realistico)
    'global' PCA adattata sul training completo (isola l'effetto)
  A poche decine di campioni la PCA locale e' instabile: confrontare le
  due varianti separa il degrado dovuto alla rappresentazione da quello
  dovuto al classificatore.

METODO
  Il test non viene mai usato per decisioni. Scaler, PCA e lista dei
  servizi frequenti sono adattati solo su dati di training.

Uso:
    python src/ids/preprocess_nslkdd.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nslkdd_common import (CATEGORICAL, SEED, TOP_SERVICES, VAL_FRACTION,
                           _read)

OUT = Path("data/ids")
K = 5
RATIOS = {"1_200": 200, "1_500": 500, "1_1000": 1000}
PCA_MODES = ["local", "global"]

CLASSES = {
    "probe": {"satan", "ipsweep", "portsweep", "nmap", "mscan", "saint"},
    "r2l": {"ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
            "warezclient", "warezmaster", "sendmail", "named",
            "snmpgetattack", "snmpguess", "xlock", "xsnoop", "worm"},
}


def build(minority, seed=SEED):
    """Carica, filtra su normal vs classe scelta, codifica, splitta."""
    def prep(split):
        d = _read(split)
        keep = d["class"].isin(minority) | (d["class"] == "normal")
        d = d[keep].copy()
        d["y"] = d["class"].isin(minority).astype(int)
        return d.drop(columns=["class"])

    tr_full, te = prep("Train"), prep("Test")
    tr, va = train_test_split(tr_full, test_size=VAL_FRACTION,
                              stratify=tr_full["y"], random_state=seed)
    top = tr["service"].value_counts().head(TOP_SERVICES).index.tolist()

    def enc(d):
        d = d.copy()
        d["service"] = d["service"].where(d["service"].isin(top), "other")
        return pd.get_dummies(d, columns=CATEGORICAL)

    tr_e = enc(tr)
    va_e = enc(va).reindex(columns=tr_e.columns, fill_value=0)
    te_e = enc(te).reindex(columns=tr_e.columns, fill_value=0)
    feat = [c for c in tr_e.columns if c != "y"]
    return {
        "Xtr": tr_e[feat].reset_index(drop=True),
        "ytr": tr_e["y"].values,
        "Xva": va_e[feat].reset_index(drop=True),
        "yva": va_e["y"].values,
        "Xte": te_e[feat].reset_index(drop=True),
        "yte": te_e["y"].values,
        "features": feat, "top_services": top,
    }


def subsample(y, ratio, seed):
    rng = np.random.RandomState(seed)
    normal = np.flatnonzero(y == 0)
    minor = np.flatnonzero(y == 1)
    n = len(normal) // ratio
    if n > len(minor):
        raise ValueError(f"servono {n} campioni minoritari, "
                         f"disponibili {len(minor)}")
    keep = rng.choice(minor, size=n, replace=False)
    out = np.concatenate([normal, keep])
    rng.shuffle(out)
    return out


def fit_projection(X, seed):
    """Scaler + PCA-6 + MinMax, adattati su X."""
    sc = StandardScaler().fit(X)
    pca = PCA(n_components=K, random_state=seed).fit(sc.transform(X))
    mm = MinMaxScaler().fit(pca.transform(sc.transform(X)))

    def project(Z):
        return np.clip(mm.transform(pca.transform(sc.transform(Z))), 0, 1)

    return project, pca.explained_variance_ratio_


def frame(P, y):
    f = pd.DataFrame(P, columns=[f"z{i+1}" for i in range(K)])
    f["y"] = np.asarray(y)
    return f


def main():
    import argparse
    global OUT, SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(OUT),
                    help="directory di uscita; una per seme")
    a = ap.parse_args()
    SEED, OUT = a.seed, Path(a.out)
    print(f"seme {SEED}   uscita {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"k": K, "strategy": f"PCA-{K}", "ratios": RATIOS,
            "pca_modes": PCA_MODES, "seed": SEED, "classes": {},
            "note": ("Test mai usato per decisioni. Variante 'local': PCA "
                     "adattata sul sottocampione sbilanciato. Variante "
                     "'global': PCA adattata sul training completo, per "
                     "isolare l'effetto della rappresentazione.")}

    for cls, labels in CLASSES.items():
        d = build(labels, SEED)
        ytr, Xtr = d["ytr"], d["Xtr"]
        print(f"\n=== {cls} ===")
        print(f"train {Xtr.shape}  minoritaria {ytr.sum():,} "
            f"({ytr.mean():.2%})  val {d['Xva'].shape}  test {d['Xte'].shape}")

        # proiezione globale: usata per val, test e per la variante 'global'
        proj_g, var_g = fit_projection(Xtr, SEED)
        print(f"PCA globale: varianza cumulata {var_g.sum():.1%}")

        frame(proj_g(d["Xva"]), d["yva"]).to_csv(
            OUT / f"{cls}_val.csv", index=False)
        frame(proj_g(d["Xte"]), d["yte"]).to_csv(
            OUT / f"{cls}_test.csv", index=False)

        info = {"n_minority_train": int(ytr.sum()),
                "explained_variance_global": float(var_g.sum()),
                "top_services": d["top_services"],
                "n_features_onehot": len(d["features"]),
                "minority_labels": sorted(labels),
                "subsets": {}}

        for name, ratio in RATIOS.items():
            idx = subsample(ytr, ratio, SEED)
            Xs, ys = Xtr.iloc[idx], ytr[idx]

            # variante 'global': proiezione adattata sul training completo
            frame(proj_g(Xs), ys).to_csv(
                OUT / f"{cls}_train_{name}_global.csv", index=False)

            # variante 'local': proiezione adattata solo sul sottocampione
            proj_l, var_l = fit_projection(Xs, SEED)
            frame(proj_l(Xs), ys).to_csv(
                OUT / f"{cls}_train_{name}_local.csv", index=False)
            # val e test devono essere proiettati con la STESSA trasformazione
            frame(proj_l(d["Xva"]), d["yva"]).to_csv(
                OUT / f"{cls}_val_{name}_local.csv", index=False)
            frame(proj_l(d["Xte"]), d["yte"]).to_csv(
                OUT / f"{cls}_test_{name}_local.csv", index=False)

            print(f"  {name:>7}: {len(idx):,} righe, minoritaria "
                  f"{int(ys.sum()):>4} ({ys.mean():.3%})  "
                  f"var locale {var_l.sum():.1%}")

            info["subsets"][name] = {
                "ratio": ratio, "n_rows": int(len(idx)),
                "n_minority": int(ys.sum()),
                "explained_variance_local": float(var_l.sum()),
            }

        meta["classes"][cls] = info

    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nmetadati in {OUT/'meta.json'}")
    print("\nFile per la variante 'global': {cls}_train_{ratio}_global.csv"
          "  con {cls}_val.csv e {cls}_test.csv")
    print("File per la variante 'local' : {cls}_train_{ratio}_local.csv"
          "   con {cls}_val_{ratio}_local.csv e test corrispondente")


if __name__ == "__main__":
    main()