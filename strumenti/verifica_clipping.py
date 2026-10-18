"""Quanta informazione perde KDDTest+ nel troncamento a [0,1]?

IL PROBLEMA. `project` in preprocess_nslkdd.py termina con

    np.clip(mm.transform(pca.transform(sc.transform(Z))), 0, 1)

e viene applicata anche a validazione e test. Il min-max e' adattato sul
solo addestramento — cosa corretta — ma di conseguenza ogni osservazione
di test che cade fuori dall'intervallo appreso viene SCHIACCIATA sul
bordo. E KDDTest+ contiene per costruzione attacchi di tipo non visto in
addestramento: se sono proprio quelli a finire fuori intervallo, il
troncamento ne cancella la parte distintiva prima che qualunque metodo
possa sfruttarla.

Non e' un errore dimostrato: e' una possibile distorsione da quantificare,
e si quantifica senza riaddestrare niente.

COSA MISURA. Rifa' la proiezione con gli stessi semi e confronta il
risultato PRIMA e DOPO il troncamento:

  - per ciascuna delle K componenti, la frazione di valori sotto 0 e
    sopra 1, e di quanto sforano;
  - la frazione di righe con almeno una componente troncata;
  - gli stessi numeri separati per CLASSE, che e' la domanda vera: se la
    minoritaria viene troncata molto piu' della maggioritaria, allora il
    troncamento agisce selettivamente proprio sugli attacchi;
  - gli stessi numeri separati per insieme (validazione e test), perche'
    il validation viene dallo stesso split del training e dovrebbe
    sforare molto meno.

COME SI LEGGE. Se la minoritaria di test e' troncata in misura simile
alla maggioritaria e le frazioni sono piccole, la distorsione e'
trascurabile e basta dichiararlo. Se la minoritaria e' troncata molto di
piu', il confronto fuori distribuzione va rifatto su una variante senza
troncamento sul test, tenendo identiche tutte le altre condizioni.

    cd ~/PycharmProjects/Quantum/qgan-qber
    python verifica_clipping.py
    python verifica_clipping.py --classe r2l --ratio 1_1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

RADICE = Path(__file__).resolve().parents[1]
for sotto in ("src/ids", "src"):
    sys.path.insert(0, str(RADICE / sotto))


def carica_modulo():
    """`build`, `fit_projection` e `subsample` dal pre-processing."""
    errori = []
    for nome in ("preprocess_nslkdd", "preprocess"):
        try:
            m = __import__(nome)
        except Exception as e:                    # noqa: BLE001
            errori.append(f"  {nome}: {type(e).__name__}: {e}")
            continue
        mancanti = [f for f in ("build", "fit_projection", "subsample")
                    if not hasattr(m, f)]
        if mancanti:
            errori.append(f"  {nome}: importato, mancano {mancanti}")
            continue
        return m, nome
    raise SystemExit(
        "non trovo il modulo di pre-processing.\n" + "\n".join(errori)
        + "\nLanciare dalla radice del repo, col venv attivo."
    )


def senza_taglio(m, X, sc, pca, mm):
    """La proiezione fermata PRIMA del np.clip finale."""
    return mm.transform(pca.transform(sc.transform(X)))


def riporta(nome, P, y, K):
    sotto = (P < 0.0)
    sopra = (P > 1.0)
    fuori = sotto | sopra
    righe = fuori.any(axis=1)
    print(f"\n  --- {nome}: {len(P)} righe ---")
    print(f"  righe con almeno una componente troncata  "
          f"{righe.mean():7.3%}  ({righe.sum()})")
    print(f"  {'comp.':<7}{'sotto 0':>10}{'sopra 1':>10}"
          f"{'sforo max':>12}{'sforo medio':>14}")
    for k in range(K):
        col = P[:, k]
        ecc = np.where(col < 0, -col, np.where(col > 1, col - 1, 0.0))
        n_e = ecc > 0
        print(f"  z{k+1:<6}{sotto[:, k].mean():>10.3%}"
              f"{sopra[:, k].mean():>10.3%}{ecc.max():>12.4f}"
              f"{(ecc[n_e].mean() if n_e.any() else 0.0):>14.4f}")
    if y is not None and len(np.unique(y)) > 1:
        print("  per classe, righe con almeno una componente troncata:")
        for cl in sorted(np.unique(y)):
            s = righe[y == cl]
            et = "minoritaria (attacchi)" if cl == 1 else "maggioritaria"
            print(f"    {et:<24} {s.mean():7.3%}  ({s.sum()} su {len(s)})")
        a = float(righe[y == 1].mean())
        b = float(righe[y == 0].mean())
        if b > 0:
            print(f"    rapporto minoritaria/maggioritaria  {a/b:.2f}")
        elif a > 0:
            print("    rapporto minoritaria/maggioritaria  infinito "
                  "(la maggioritaria non sfora mai)")
        selettivo = (a > 0 and (b == 0 or a / b > 2))
        if selettivo:
            print("    => il troncamento agisce SELETTIVAMENTE sugli")
            print("       attacchi: cancella la parte che li distingue,")
            print("       e la distorsione va quantificata a valle — non")
            print("       basta dichiararla.")
        elif a + b == 0:
            print("    => nessun troncamento: nulla da dichiarare.")
        else:
            print("    => il troncamento non distingue le classi in modo")
            print("       marcato: dichiararlo e' sufficiente.")
    return righe.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classe", default="r2l")
    ap.add_argument("--ratio", default="1_1000",
                    help="chiave di RATIOS, es. 1_1000")
    ap.add_argument("--seme", type=int, default=None,
                    help="default: il SEED del modulo")
    a = ap.parse_args()

    m, nome = carica_modulo()
    seme = a.seme if a.seme is not None else getattr(m, "SEED", 42)
    K = getattr(m, "K", 5)
    print(f"modulo {nome}   seme {seme}   K {K}")

    # Il dizionario delle classi ha nomi diversi da progetto a progetto:
    # invece di indovinarlo, si cerca fra gli attributi del modulo il
    # primo dizionario che mappi stringhe a collezioni di etichette e che
    # contenga la classe richiesta.
    def candidati():
        for nome_attr, v in vars(m).items():
            if nome_attr.startswith("_") or not isinstance(v, dict) or not v:
                continue
            if all(isinstance(k, str) for k in v) and all(
                    isinstance(x, (list, tuple, set, frozenset))
                    for x in v.values()):
                yield nome_attr, v

    etichette = nome_dizionario = None
    trovati = list(candidati())
    for nome_attr, v in trovati:
        if a.classe in v:
            etichette, nome_dizionario = v[a.classe], nome_attr
            break
    if etichette is None:
        print(f"\nnon trovo la classe {a.classe!r}. Dizionari di classi "
              "visti nel modulo:")
        for nome_attr, v in trovati:
            print(f"  {nome_attr}: {sorted(v)[:8]}")
        if not trovati:
            print("  nessuno. Attributi di tipo dict presenti:")
            for nome_attr, v in vars(m).items():
                if isinstance(v, dict) and not nome_attr.startswith("_"):
                    print(f"  {nome_attr}: {list(v)[:8]}")
        raise SystemExit("\nRilanciare con --classe fra quelle elencate.")
    print(f"classe {a.classe!r} da {nome_dizionario} "
          f"({len(etichette)} etichette)")

    d = m.build(etichette, seme)
    ytr, Xtr = d["ytr"], d["Xtr"]

    ratios = getattr(m, "RATIOS", {})
    if a.ratio not in ratios:
        raise SystemExit(f"ratio {a.ratio!r} non in RATIOS: {list(ratios)}")
    idx = m.subsample(ytr, ratios[a.ratio], seme)
    Xs = Xtr.iloc[idx]

    # stessa catena di fit_projection, ma trattenuta prima del clip
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    sc = StandardScaler().fit(Xs)
    pca = PCA(n_components=K, random_state=seme).fit(sc.transform(Xs))
    mm = MinMaxScaler().fit(pca.transform(sc.transform(Xs)))

    print(f"proiezione adattata su {len(Xs)} righe di addestramento "
          f"(variante locale, {a.ratio})")
    print("=" * 62)
    print("TRONCAMENTO A [0,1]: quanto perde ciascun insieme")
    print("=" * 62)

    riporta("addestramento (atteso ~0)",
            senza_taglio(m, Xs, sc, pca, mm), ytr[idx], K)
    riporta("validazione",
            senza_taglio(m, d["Xva"], sc, pca, mm), d["yva"], K)
    q = riporta("TEST (KDDTest+)",
                senza_taglio(m, d["Xte"], sc, pca, mm), d["yte"], K)

    print()
    print("  L'addestramento deve sforare pochissimo per costruzione: il")
    print("  min-max e' adattato proprio su quelle righe. Cio' che conta e'")
    print("  quanto sforano test e validazione, e se le due classi sforano")
    print("  in modo diverso.")
    return 0 if q < 0.5 else 0


if __name__ == "__main__":
    raise SystemExit(main())