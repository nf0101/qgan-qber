"""Contrasto appaiato fra gruppi di corse, con inferenza incrociata.

`analisi_appaiata.py` lavora su una directory sola: aggrega i semi di UNA
configurazione e si rifiuta di mescolarne due, che e' il controllo giusto.
Ma le ablazioni del progetto stanno fra directory — o1_qkd_h8 contro
o1_qkd_h32_16, raggi_perfeature contro raggi_unico — e quel confronto
nessuno script lo faceva.

Qui le osservazioni vengono appaiate su (corsa, cella): i gruppi
condividono seed_base e seed_test, quindi la stessa coppia identifica la
stessa condizione sperimentale nei due gruppi, e la differenza elimina
l'effetto della cella, che e' la componente dominante di varianza.

L'inferenza riusa `componenti` e `bootstrap` di analisi_appaiata, gia'
verificati su dati con struttura nota: le corse e le celle sono assi
incrociati e si ricampionano entrambi. Il test dei segni opera sulle medie
per corsa.

    python contrasto_gruppi.py --dir qgan/results/o1_qkd_h8 \\
        qgan/results/o1_qkd_h16 qgan/results/o1_qkd_h64_32 \\
        --contro qgan/results/o1_qkd_h32_16

    python contrasto_gruppi.py --dir qgan/results/raggi_perfeature \\
        qgan/results/raggi_misto --contro qgan/results/raggi_unico
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "qgan" / "scripts"))
from analisi_appaiata import (bootstrap, carica, componenti,  # noqa: E402
                              estrai)


def osservazioni(percorso, quantita):
    """{(corsa, cella): valore} per una directory omogenea."""
    oss, corse, celle = estrai(carica(percorso), quantita)
    return oss, sorted(set(corse)), celle


def riporta(diff, corse, celle, etichetta, repliche_bs):
    from scipy import stats

    v_c, v_k, v_r, tot, M = componenti(diff, corse, celle)
    media = float(np.nanmean(M))
    per_corsa = np.nanmean(M, axis=1)

    print(f"  media su tutte le osservazioni   {media:+.4f}")
    print(f"  varianza della differenza: fra corse {v_c/tot:5.1%}   "
          f"fra celle {v_k/tot:5.1%}   residuo {v_r/tot:5.1%}")
    print("  unita' di inferenza            intervallo 95%       ampiezza")
    esiti = {}
    for modo, et in (("corsa", "per corsa"), ("cella", "per cella"),
                     ("incrociata", "incrociata")):
        lo, hi = bootstrap(diff, corse, celle, modo, repliche_bs)
        esclude = lo > 0 or hi < 0
        esiti[modo] = (lo, hi, esclude)
        marca = "" if esclude else "   contiene lo zero"
        print(f"    {et:28s} [{lo:+.4f}, {hi:+.4f}]  {hi-lo:.4f}{marca}")

    n = int(np.isfinite(per_corsa).sum())
    seg = int((per_corsa > 0).sum())
    p = stats.binomtest(seg, n, 0.5).pvalue if n else float("nan")
    print(f"  segni sulle medie per corsa      {seg}/{n}   p = {p:.5f}")
    return media, esiti["incrociata"], (seg, n, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", required=True,
                    help="gruppi da confrontare col riferimento")
    ap.add_argument("--contro", required=True,
                    help="gruppo di riferimento")
    ap.add_argument("--quantita", default="skill",
                    choices=["skill", "mae", "guadagno", "divario"])
    ap.add_argument("--repliche-bs", type=int, default=20000)
    a = ap.parse_args()

    rif_oss, rif_corse, rif_celle = osservazioni(a.contro, a.quantita)
    nome_rif = os.path.basename(os.path.normpath(a.contro))

    print(f"quantita' {a.quantita}   riferimento {nome_rif}")
    print(f"  {len(rif_corse)} corse, {len(rif_celle)} celle, "
          f"{len(rif_oss)} osservazioni")
    print()
    print("LIVELLI  media su tutte le osservazioni del gruppo")
    gruppi = {}
    for d in [a.contro] + list(a.dir):
        nome = os.path.basename(os.path.normpath(d))
        if nome in gruppi:
            continue
        oss, corse, celle = osservazioni(d, a.quantita)
        gruppi[nome] = (oss, corse, celle)
        print(f"  {nome:<24} {np.mean(list(oss.values())):+.4f}   "
              f"{len(corse)} corse, {len(celle)} celle")

    print()
    print("=" * 66)
    print(f"CONTRASTI APPAIATI   ogni gruppo meno {nome_rif}")
    print("=" * 66)
    riepilogo = []
    for d in a.dir:
        nome = os.path.basename(os.path.normpath(d))
        if nome == nome_rif:
            continue
        oss = gruppi[nome][0]
        comuni = sorted(set(oss) & set(rif_oss))
        if not comuni:
            print(f"\n{nome} - {nome_rif}: nessuna coppia (corsa, cella) in "
                  "comune, confronto impossibile")
            continue
        solo_a = len(set(oss) - set(rif_oss))
        solo_b = len(set(rif_oss) - set(oss))
        diff = {k: oss[k] - rif_oss[k] for k in comuni}
        corse = sorted({k[0] for k in comuni})
        celle = sorted({k[1] for k in comuni})
        print(f"\n{nome} - {nome_rif}")
        print(f"  {len(comuni)} coppie appaiate su {len(corse)} corse e "
              f"{len(celle)} celle")
        if solo_a or solo_b:
            print(f"  scartate: {solo_a} solo in {nome}, "
                  f"{solo_b} solo in {nome_rif}")
        m, (lo, hi, esclude), (seg, n, p) = riporta(
            diff, corse, celle, nome, a.repliche_bs)
        riepilogo.append((nome, m, lo, hi, seg, n, p, esclude))

    print()
    print("=" * 66)
    print(f"RIEPILOGO   differenza da {nome_rif}, intervallo incrociato")
    print("=" * 66)
    print(f"  {'gruppo':<24} {'media':>9}  {'intervallo 95%':>22} "
          f"{'segni':>8} {'p':>9}")
    for nome, m, lo, hi, seg, n, p, esclude in sorted(
            riepilogo, key=lambda x: -x[1]):
        marca = "" if esclude else "  *"
        print(f"  {nome:<24} {m:+9.4f}  [{lo:+8.4f}, {hi:+8.4f}] "
              f"{seg:>4}/{n:<3} {p:9.5f}{marca}")
    print()
    print("  * l'intervallo incrociato contiene lo zero: su nuove corse e")
    print("    nuove celle la differenza non e' distinguibile.")
    print()
    print("  La differenza elimina l'effetto della cella, che nei livelli e'")
    print("  la componente dominante: per questo i contrasti appaiati sono")
    print("  molto piu' precisi del confronto fra le medie dei gruppi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())