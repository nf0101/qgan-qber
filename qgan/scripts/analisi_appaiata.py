"""Inferenza con DUE raggruppamenti incrociati: corse e celle.

Chiude il difetto 4 dell'audit. La versione precedente concatenava tutte le
righe per cella e applicava t-test e bootstrap ordinario come se ciascuna
fosse una replica indipendente. Non lo sono: le quattro celle di una corsa
condividono generatore, addestramento e checkpoint, e le stesse celle fisiche
ricompaiono fra corse. Venti righe erano cinque corse per quattro celle.

Qui l'unita' di inferenza e' DICHIARATA, e se ne riportano tre, perche'
rispondono a domande diverse:

  per corsa    generalizza su nuove ESECUZIONI, celle fissate. E' la domanda
               "se rilancio l'addestramento, ottengo lo stesso risultato?".
  per cella    generalizza su nuove CELLE, esecuzioni fissate. "il risultato
               vale su regioni del dominio diverse da queste?"
  incrociata   generalizza su entrambe. E' l'intervallo onesto da riportare,
               e il piu' largo dei tre.

Il bootstrap incrociato e' quello a nido di piccione (Owen): si ricampionano
corse e celle INDIPENDENTEMENTE e si prendono le osservazioni alle
intersezioni. E' la procedura corretta per un disegno incrociato, dove un
bootstrap su un solo asse sottostima la varianza.

AVVERTENZA sulla potenza. La griglia sintetica ha 8 celle interne e ogni
corsa ne esclude 4: le celle distinte sono poche e l'intervallo incrociato
sara' largo. Non e' un difetto della procedura, e' quanta informazione c'e'.
Dichiararlo e' il punto.

    python scripts/analisi_appaiata.py --dir results/n_P32
    python scripts/analisi_appaiata.py --dir results/raggi_unico --quantita mae
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def carica(percorso):
    """Righe dai JSONL, deduplicate per (firma, seme)."""
    file = sorted(glob.glob(os.path.join(percorso, "*.jsonl")))
    if not file:
        file = [percorso] if os.path.isfile(percorso) else []
    viste, righe = set(), []
    for f in file:
        for riga in open(f):
            riga = riga.strip()
            if not riga:
                continue
            r = json.loads(riga)
            chiave = (r.get("firma_config"), r.get("seed"))
            if chiave in viste:
                continue
            viste.add(chiave)
            righe.append(r)
    firme = {r.get("firma_config") for r in righe}
    if len(firme) > 1:
        raise SystemExit(
            f"configurazioni diverse nello stesso insieme: {sorted(firme)}. "
            "L'analisi appaiata richiede corse omogenee."
        )
    return sorted(righe, key=lambda r: r["seed"])


def estrai(righe, quantita):
    """{(corsa, cella): valore} piu' gli elenchi di corse e celle.

    `per_cella` e' la struttura prodotta da cond_validation.py. Se i nomi dei
    campi non corrispondono, lo script elenca quelli disponibili invece di
    indovinare.
    """
    oss, corse, celle = {}, [], set()
    for r in righe:
        s = r["seed"]
        corse.append(s)
        pc = r.get("per_cella")
        ce = r.get("celle_escluse")
        if not pc or not ce:
            raise SystemExit("righe senza 'per_cella' o 'celle_escluse'")
        for cella, d in zip(ce, pc):
            if not isinstance(d, dict):
                raise SystemExit(
                    "'per_cella' non contiene dizionari; struttura trovata: "
                    f"{type(d).__name__}"
                )
            v = valore(d, quantita)
            oss[(s, int(cella))] = v
            celle.add(int(cella))
    return oss, corse, sorted(celle)


def valore(d, quantita):
    """Il valore della cella, con i nomi di campo tollerati."""
    candidati = {
        "skill": ["skill", "skill_condizionato", "skill_cond"],
        "mae": ["mae_condizionato", "mae_cond"],
        "guadagno": ["guadagno", "guadagno_mmd"],
        "divario": None,     # mae_banale - mae_condizionato
    }
    if quantita == "divario":
        a = primo(d, ["mae_banale", "mae_trivial", "mae_riferimento"])
        b = primo(d, ["mae_condizionato", "mae_cond"])
        return float(a - b)
    nomi = candidati.get(quantita)
    if nomi is None:
        raise SystemExit(f"quantita' non riconosciuta: {quantita}")
    return float(primo(d, nomi))


def primo(d, nomi):
    for n in nomi:
        if n in d:
            return d[n]
    raise SystemExit(
        f"nessuno dei campi {nomi} e' presente. Campi disponibili nella "
        f"cella: {sorted(d)}"
    )


def media_su(oss, corse, celle):
    v = [oss[(s, c)] for s in corse for c in celle if (s, c) in oss]
    return float(np.mean(v)) if v else np.nan


def bootstrap(oss, corse, celle, modo, repliche=20000, seed=0):
    """Intervallo al 95% ricampionando corse, celle, o entrambe."""
    rng = np.random.default_rng(seed)
    nc, nk = len(corse), len(celle)
    out = []
    for _ in range(repliche):
        cs = [corse[i] for i in rng.integers(0, nc, nc)] if modo in \
            ("corsa", "incrociata") else corse
        ks = [celle[i] for i in rng.integers(0, nk, nk)] if modo in \
            ("cella", "incrociata") else celle
        m = media_su(oss, cs, ks)
        if not np.isnan(m):
            out.append(m)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def componenti(oss, corse, celle):
    """Quanta varianza sta fra corse, fra celle, e nel residuo."""
    M = np.full((len(corse), len(celle)), np.nan)
    for i, s in enumerate(corse):
        for j, c in enumerate(celle):
            if (s, c) in oss:
                M[i, j] = oss[(s, c)]
    gran = np.nanmean(M)
    v_corsa = np.nanvar(np.nanmean(M, axis=1), ddof=1)
    v_cella = np.nanvar(np.nanmean(M, axis=0), ddof=1)
    res = M - np.nanmean(M, axis=1, keepdims=True) \
            - np.nanmean(M, axis=0, keepdims=True) + gran
    v_res = np.nanvar(res, ddof=1)
    tot = v_corsa + v_cella + v_res
    return v_corsa, v_cella, v_res, tot, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="directory con i JSONL, o un singolo file")
    ap.add_argument("--quantita", default="skill",
                    choices=["skill", "mae", "guadagno", "divario"])
    ap.add_argument("--repliche", type=int, default=20000)
    ap.add_argument("--contro", type=float, default=0.0,
                    help="valore di riferimento per il test dei segni")
    a = ap.parse_args()

    righe = carica(a.dir)
    oss, corse, celle = estrai(righe, a.quantita)
    v_corsa, v_cella, v_res, tot, M = componenti(oss, corse, celle)

    print(f"{len(corse)} corse, {len(celle)} celle distinte, "
          f"{len(oss)} osservazioni   quantita': {a.quantita}")
    print(f"celle: {celle}")
    conteggi = {c: sum(1 for s in corse if (s, c) in oss) for c in celle}
    print(f"ricorrenze per cella: min {min(conteggi.values())} "
          f"max {max(conteggi.values())}")
    print()

    print("DECOMPOSIZIONE DELLA VARIANZA")
    for et, v in (("fra corse", v_corsa), ("fra celle", v_cella),
                  ("residuo", v_res)):
        print(f"  {et:12s} {v:.6f}   {v/tot:6.1%}")
    print("  La componente dominante dice su cosa conviene spendere repliche.")
    print()

    media = float(np.nanmean(M))
    print(f"MEDIA {media:+.4f}")
    print("UNITA' DI INFERENZA          intervallo 95%        ampiezza")
    for modo, et in (("corsa", "per corsa  (nuove esecuzioni)"),
                     ("cella", "per cella  (nuove celle)"),
                     ("incrociata", "incrociata (entrambe)")):
        lo, hi = bootstrap(oss, corse, celle, modo, a.repliche)
        segno = " " if lo > a.contro or hi < a.contro else "  contiene il riferimento"
        print(f"  {et:30s} [{lo:+.4f}, {hi:+.4f}]  {hi-lo:.4f}{segno}")
    print()

    # Test dei segni sulle MEDIE PER CORSA: l'unita' e' la corsa, non la riga.
    per_corsa = np.nanmean(M, axis=1)
    seg = int((per_corsa > a.contro).sum())
    n = int(np.isfinite(per_corsa).sum())
    from scipy import stats
    print(f"TEST DEI SEGNI sulle medie per corsa, contro {a.contro:+.4f}")
    print(f"  {seg}/{n} corse sopra il riferimento   "
          f"p = {stats.binomtest(seg, n, 0.5).pvalue:.5f}")
    if n <= 20:
        print(f"  p minimo ottenibile con {n} corse: "
              f"{2 * 0.5 ** n:.5f} — e' il pavimento del test, non "
              "l'ampiezza dell'effetto.")
    print()
    print("L'intervallo INCROCIATO e' quello da riportare: gli altri due "
          "tengono\nfisso un asse e sottostimano l'incertezza di una "
          "conclusione generale.")


if __name__ == "__main__":
    main()