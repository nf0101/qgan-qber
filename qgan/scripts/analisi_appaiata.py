"""Confronto appaiato per cella, come prescritto da SPEC-FASE2 §11.

Lo skill riportato per seme ha deviazione 0,273 su una media di 0,185: l'errore
copre il 150 % del risultato. La componente dominante non e' rumore di stima ma
la selezione delle celle escluse — il MAE del riferimento banale varia di un
fattore 1,7 fra semi, perche' quando le celle capitano vicine alla media della
griglia predire la media diventa insolitamente buono.

§11 della specifica prescrive per questo il confronto appaiato sulle stesse
celle. Applicato qui significa tre cose:

  1. SKILL MESSO IN COMUNE. Un solo rapporto sulle MAE aggregate di tutte le
     celle di tutti i semi, invece di una media di rapporti per seme. Con 5
     semi x 4 celle sono 20 osservazioni invece di 5.

  2. TEST APPAIATO CONTRO IL BANALE. Per ciascuna cella, la differenza
     `MAE_banale - MAE_condizionato`. Il segno dice se il generatore batte la
     media della griglia su quella cella; il test dice se il risultato regge.
     E' molto piu' potente del test sui 5 skill per seme, perche' elimina la
     variabilita' del denominatore.

  3. TEST APPAIATO CONTRO L'INCONDIZIONATO. La stessa cosa con
     `MAE_incondizionato - MAE_condizionato`: isola il contributo del
     condizionamento, che e' cio' che RQ2 misura.

Il terzo e' il confronto che conta per RQ2; il secondo dice se il generatore ha
valore assoluto; il primo e' il numero da riportare.

    python scripts/analisi_appaiata.py results/rq2_qkd_unico_5.jsonl
    python scripts/analisi_appaiata.py a.jsonl b.jsonl --etichette unico,per_feature
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import stats


def carica(path):
    righe = [json.loads(l) for l in open(path) if l.strip()]
    celle = []
    for r in righe:
        for p in r.get("per_cella", []):
            celle.append({
                "seed": r["seed"],
                "cella": p["cella"],
                "c": tuple(p["c"]),
                "cond": p["mae_condizionato"],
                "incond": p["mae_incondizionato"],
                "banale": p["mae_banale"],
            })
    return righe, celle


def appaiato(d, nome):
    """Test appaiato su differenze gia' calcolate. Restituisce una riga."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        return f"{nome}: meno di due celle, test non applicabile"
    sem = d.std(ddof=1) / np.sqrt(n)
    t = d.mean() / sem if sem > 0 else np.nan
    p = float(stats.ttest_1samp(d, 0.0).pvalue) if sem > 0 else np.nan
    # test dei segni, non parametrico: non assume normalita' delle differenze
    pos = int((d > 0).sum())
    p_segni = float(stats.binomtest(pos, n, 0.5).pvalue)
    return (f"{nome:<34} {d.mean():+9.5f} {sem:9.5f} {t:7.2f} {p:9.4f} "
            f"{pos:>3}/{n:<3} {p_segni:9.4f}")


def analizza(path, etichetta):
    righe, celle = carica(path)
    if not celle:
        print(f"{etichetta}: nessuna cella nel file")
        return
    cond = np.array([c["cond"] for c in celle])
    inc = np.array([c["incond"] for c in celle])
    ban = np.array([c["banale"] for c in celle])
    semi = sorted({c["seed"] for c in celle})

    print(f"\n{'=' * 92}")
    print(f"{etichetta}   {len(celle)} celle su {len(semi)} semi {semi}")
    print("=" * 92)

    # --- skill per seme, per confronto con quanto gia' riportato ---------
    per_seme = []
    for s in semi:
        m = np.array([c["seed"] for c in celle]) == s
        per_seme.append(1 - cond[m].mean() / ban[m].mean())
    per_seme = np.array(per_seme)
    print(f"skill per seme        {per_seme.mean():+.3f} +- "
          f"{per_seme.std(ddof=1):.3f}   ({(per_seme > 0).sum()}/{len(per_seme)})")

    # --- skill messo in comune -------------------------------------------
    skill_pool = 1 - cond.mean() / ban.mean()
    # errore standard per bootstrap sulle celle: il rapporto non ha una
    # formula chiusa affidabile con 20 osservazioni
    rng = np.random.default_rng(0)
    bs = [1 - cond[i].mean() / ban[i].mean()
          for i in (rng.integers(0, len(celle), len(celle)) for _ in range(2000))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"skill messo in comune {skill_pool:+.3f}   "
          f"intervallo bootstrap 95% [{lo:+.3f}, {hi:+.3f}]")
    print(f"  MAE condizionato {cond.mean():.5f}   banale {ban.mean():.5f}   "
          f"incondizionato {inc.mean():.5f}")

    # --- test appaiati ----------------------------------------------------
    print(f"\n{'confronto':<34} {'media':>9} {'errore':>9} {'t':>7} "
          f"{'p':>9} {'segni':>7} {'p segni':>9}")
    print("-" * 92)
    print(appaiato(ban - cond, "banale - condizionato"))
    print(appaiato(inc - cond, "incondizionato - condizionato"))
    print(appaiato(ban - inc, "banale - incondizionato"))

    return {"etichetta": etichetta, "n_celle": len(celle),
            "skill_per_seme": float(per_seme.mean()),
            "skill_messo_in_comune": float(skill_pool),
            "bootstrap_95": [float(lo), float(hi)],
            "cond": cond, "incond": inc, "banale": ban,
            "chiavi": [(c["seed"], c["c"]) for c in celle]}


def confronta(a, b):
    """Confronto appaiato fra due configurazioni sulle stesse celle."""
    ka = {k: i for i, k in enumerate(a["chiavi"])}
    comuni = [k for k in b["chiavi"] if k in ka]
    if not comuni:
        print("\nnessuna cella in comune fra i due file: confronto non "
              "appaiabile")
        return
    ia = [ka[k] for k in comuni]
    kb = {k: i for i, k in enumerate(b["chiavi"])}
    ib = [kb[k] for k in comuni]
    d = a["cond"][ia] - b["cond"][ib]   # positivo = b migliore
    print(f"\n{'=' * 92}")
    print(f"CONFRONTO APPAIATO   {a['etichetta']} contro {b['etichetta']}   "
          f"{len(comuni)} celle in comune")
    print("=" * 92)
    print(f"{'confronto':<34} {'media':>9} {'errore':>9} {'t':>7} "
          f"{'p':>9} {'segni':>7} {'p segni':>9}")
    print("-" * 92)
    print(appaiato(d, f"MAE {a['etichetta']} - {b['etichetta']}"))
    print(f"\nmedia positiva = {b['etichetta']} ha MAE minore, cioe' e' "
          f"migliore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--etichette", default=None)
    a = ap.parse_args()

    et = (a.etichette.split(",") if a.etichette
          else [f.split("/")[-1].replace(".jsonl", "") for f in a.files])
    res = [analizza(f, e) for f, e in zip(a.files, et)]
    res = [r for r in res if r]
    if len(res) == 2:
        confronta(res[0], res[1])

    print("\nLo skill messo in comune e' il numero da riportare: usa 20")
    print("osservazioni invece di 5 e non risente della variabilita' del")
    print("denominatore fra semi. Il test 'incondizionato - condizionato' e'")
    print("quello che misura RQ2.")


if __name__ == "__main__":
    main()