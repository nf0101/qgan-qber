"""Contrasto appaiato fra DUE CAMPI della stessa cella.

`analisi_appaiata.py` sa leggere una grandezza per cella e confrontarla con
zero; `contrasto_gruppi.py` sa confrontare due gruppi di corse. Qui serve la
terza forma: dentro la stessa cella convivono piu' stimatori — condizionato,
incondizionato, banale, vicino, lineare — e la domanda e' quale vince.

CONVENZIONE DI SEGNO. Per MAE e MMD il valore piu' basso e' migliore, quindi
il contrasto e' calcolato come

    altro - riferimento

e un valore POSITIVO significa che il RIFERIMENTO (il condizionato) e'
migliore dell'altro stimatore. La convenzione e' stampata a ogni corsa per
non lasciarla alla memoria.

L'appaiamento e' per (corsa, cella): la differenza elimina l'effetto della
cella, che e' la componente dominante di varianza, e l'inferenza riusa il
bootstrap incrociato di analisi_appaiata.

    python contrasto_campi.py --dir qgan/results/rq2_reale_v3.jsonl
    python contrasto_campi.py --dir qgan/results/rq2_reale_v3.jsonl \\
        --famiglie mmd --riferimento condizionato
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

RADICE = Path(__file__).resolve().parent
sys.path.insert(0, str(RADICE / "qgan" / "scripts"))
from analisi_appaiata import bootstrap, componenti  # noqa: E402


def righe(percorso):
    file = sorted(glob.glob(os.path.join(percorso, "*.jsonl")))
    if not file:
        file = [percorso] if os.path.isfile(percorso) else []
    if not file:
        raise SystemExit(f"nessun .jsonl in {percorso}")
    viste, out = set(), []
    for f in file:
        for riga in open(f, encoding="utf-8", errors="replace"):
            riga = riga.strip()
            if not riga:
                continue
            r = json.loads(riga)
            ch = (r.get("firma_config"), r.get("seed"))
            if ch in viste:
                continue
            viste.add(ch)
            out.append(r)
    return out


def etichetta_cella(d, indice, escluse):
    """Identificatore della cella, dal campo proprio o dall'indice."""
    c = d.get("cella")
    if c is None and escluse and indice < len(escluse):
        c = escluse[indice]
    cond = d.get("condizione")
    if cond is None:
        return str(c)
    return f"{c}|{cond}"


def estrai(rr):
    """{(corsa, cella): dict della cella}."""
    oss = {}
    for r in rr:
        s = r.get("seed")
        pc = r.get("per_cella")
        if not pc:
            continue
        esc = r.get("celle_escluse") or []
        for i, d in enumerate(pc):
            if isinstance(d, dict):
                oss[(s, etichetta_cella(d, i, esc))] = d
    if not oss:
        raise SystemExit("nessuna cella leggibile: manca `per_cella`")
    return oss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory o singolo JSONL")
    ap.add_argument("--riferimento", default="condizionato",
                    help="suffisso dello stimatore di riferimento")
    ap.add_argument("--famiglie", default="mae,mmd",
                    help="prefissi dei campi, separati da virgola")
    ap.add_argument("--repliche-bs", type=int, default=20000)
    a = ap.parse_args()

    oss = estrai(righe(a.dir))
    corse = sorted({k[0] for k in oss})
    celle = sorted({k[1] for k in oss})
    campi = sorted(set().union(*(set(d) for d in oss.values())))

    print(f"{len(corse)} corse, {len(celle)} celle, {len(oss)} osservazioni")
    print(f"campi disponibili: {', '.join(campi)}")
    print()

    from scipy import stats
    for fam in [f.strip() for f in a.famiglie.split(",") if f.strip()]:
        rif = f"{fam}_{a.riferimento}"
        altri = sorted(c for c in campi
                       if c.startswith(fam + "_") and c != rif)
        if rif not in campi:
            print(f"[{fam}] riferimento {rif} assente, salto")
            continue
        if not altri:
            print(f"[{fam}] nessun altro stimatore, salto")
            continue

        print("=" * 68)
        print(f"FAMIGLIA {fam.upper()}   riferimento {rif}")
        print("=" * 68)
        print(f"  livelli medi:  {rif} {np.mean([d[rif] for d in oss.values() if rif in d]):.5f}")
        for c in altri:
            v = [d[c] for d in oss.values() if c in d]
            print(f"                 {c} {np.mean(v):.5f}")
        print()
        print(f"  Segno: POSITIVO = {rif} migliore (per {fam} il minore"
              " vince).")

        riepilogo = []
        for c in altri:
            chiavi = [k for k, d in oss.items() if rif in d and c in d]
            if not chiavi:
                continue
            diff = {k: float(oss[k][c] - oss[k][rif]) for k in chiavi}
            cs = sorted({k[0] for k in chiavi})
            ks = sorted({k[1] for k in chiavi})
            v_c, v_k, v_r, tot, M = componenti(diff, cs, ks)
            media = float(np.nanmean(M))
            per_corsa = np.nanmean(M, axis=1)
            print(f"\n  {c} - {rif}   ({len(chiavi)} coppie)")
            print(f"    media {media:+.5f}   varianza: fra corse "
                  f"{v_c/tot:5.1%}  fra celle {v_k/tot:5.1%}  "
                  f"residuo {v_r/tot:5.1%}")
            esiti = {}
            for modo, et in (("corsa", "per corsa"), ("cella", "per cella"),
                             ("incrociata", "incrociata")):
                lo, hi = bootstrap(diff, cs, ks, modo, a.repliche_bs)
                esclude = lo > 0 or hi < 0
                esiti[modo] = (lo, hi, esclude)
                marca = "" if esclude else "   contiene lo zero"
                print(f"      {et:14s} [{lo:+.5f}, {hi:+.5f}]{marca}")
            n = int(np.isfinite(per_corsa).sum())
            seg = int((per_corsa > 0).sum())
            p = stats.binomtest(seg, n, 0.5).pvalue if n else float("nan")
            print(f"    segni per corsa  {seg}/{n}   p = {p:.5f}")
            lo, hi, esclude = esiti["incrociata"]
            riepilogo.append((c, media, lo, hi, seg, n, p, esclude))

        print()
        print(f"  RIEPILOGO {fam}   positivo = {rif} vince")
        print(f"    {'stimatore':<22} {'media':>10}  "
              f"{'intervallo 95%':>24} {'segni':>8}")
        for c, m, lo, hi, seg, n, p, esclude in sorted(
                riepilogo, key=lambda x: -x[1]):
            marca = "" if esclude else "  *"
            print(f"    {c:<22} {m:+10.5f}  [{lo:+9.5f}, {hi:+9.5f}] "
                  f"{seg:>4}/{n:<3}{marca}")
        print("    * intervallo incrociato che contiene lo zero: su nuove")
        print("      corse e nuove celle i due stimatori non si distinguono.")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())