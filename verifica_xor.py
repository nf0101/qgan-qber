"""Verifica l'identita' XOR dell'attribuzione, e documenta i casi limite.

LA DOMANDA. Il paper afferma che ogni errore osservato e' la composizione
XOR dei tre indicatori a causa singola,

    e_oss = e_eve XOR e_can XOR e_riv,

e che l'identita' e' esatta. Un revisore ha giustamente osservato che la
verifica su 299 posizioni comuni riguarda l'ACCOPPIAMENTO dei flussi
casuali fra condizioni diverse, non questa identita'.

COME SI VERIFICA MEGLIO DI UN CONTEGGIO. `attribute()` e' una funzione
pura: il suo esito dipende da (alice_bit, basis, intercettazione, i due
Pauli di canale, l'inversione del rivelatore, l'esito grezzo di Bob) e da
nient'altro. Lo spazio di quegli ingressi e' FINITO e piccolo — due bit,
due basi, quattro Pauli per segmento, due stati di intercettazione, due
esiti — quindi invece di contare violazioni su un campione lo si enumera
per intero. Una enumerazione esaustiva senza violazioni non e' un test:
e' una dimostrazione, sul codice effettivamente eseguito.

Si noti che la questione della corrispondenza fra posizioni, che il
revisore solleva, qui non si pone: i tre indicatori sono calcolati sulla
MEDESIMA posizione setacciata della MEDESIMA corsa, disattivando le altre
cause nel calcolo e non in una simulazione separata. Non c'e' nessun
sifting divergente da riconciliare. E' invece una questione reale per
l'accoppiamento fra condizioni, che e' l'altra verifica.

IL SECONDO CONTROLLO. Sul file di griglia si misurano due cose che il
paper dichiara ma non quantifica:

  - le finestre a denominatore nullo, dove nessuna causa produce errori e
    il codice assegna convenzionalmente y = (0,0,0), che non giace sul
    simplesso. La convenzione non e' neutra: nelle celle a rumore basso
    la probabilita' che 200 bit setacciati non contengano errori non e'
    trascurabile, e quelle finestre entrano nell'addestramento con un
    bersaglio fuori dal simplesso;

  - il residuo di additivita' sui tassi, somma dei tre tassi a causa
    singola meno il tasso osservato. La composizione XOR non e' additiva
    sui tassi, quindi il residuo deve essere POSITIVO e di ordine 2*p_i*p_j:
    il paper lo stima a ~1e-3 e qui lo si misura.

    cd ~/PycharmProjects/Quantum/qgan-qber
    python verifica_xor.py
    python verifica_xor.py --dati data/qber_grid.csv
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RADICE = Path(__file__).resolve().parent
for sotto in ("src/qkd", "qgan/scripts", "src"):
    sys.path.insert(0, str(RADICE / sotto))


def carica_attribute():
    """`attribute` e `flips` dal modulo BB84 del progetto.

    Riporta l'errore di OGNI tentativo, non solo dell'ultimo: un modulo
    che fallisce per una ragione sua (NetSquid assente, un simbolo non
    definito) produrrebbe altrimenti il messaggio fuorviante "modulo non
    trovato", che manda a cercare nel posto sbagliato.
    """
    errori = []
    for mod in ("BB84_crn", "bb84_crn", "BB84_main", "bb84"):
        try:
            m = __import__(mod)
        except Exception as e:          # noqa: BLE001
            errori.append(f"  {mod}: {type(e).__name__}: {e}")
            continue
        if hasattr(m, "attribute") and hasattr(m, "flips"):
            return m.attribute, m.flips, getattr(m, "PAULI", None), mod
        errori.append(f"  {mod}: importato, ma senza `attribute`/`flips`")
    raise SystemExit(
        "non trovo un modulo con `attribute` e `flips`.\n"
        + "\n".join(errori)
        + "\nLanciare dalla radice del repo con il venv attivo. Se il file"
          "\nBB84 sta altrove, aggiungere la sua cartella ai sys.path in"
          "\ntesta a questo script. L'import carica NetSquid, quindi serve"
          "\nl'ambiente in cui gira la simulazione."
    )


def enumera(attribute, flips, pauli_tab):
    """Tutti gli ingressi possibili di `attribute`, uno per uno."""
    pauli = list(pauli_tab.values()) if pauli_tab else [None, "X", "Y", "Z"]
    casi = itertools.product(
        (0, 1),            # alice_bit
        (0, 1),            # basis
        (False, True),     # eve intercepted
        pauli,             # p1
        pauli,             # p2
        (False, True),     # det_flip
        (0, 1),            # bob_raw
    )
    tot = viol = 0
    esempi = []
    for a, b, inter, p1, p2, det, braw in casi:
        r = attribute(0, a, b, {"intercepted": inter}, p1, p2, det, braw)
        sinistra = bool(r["error"])
        destra = bool(r["cf_eve"]) ^ bool(r["cf_canale"]) ^ bool(r["cf_rivelatore"])
        tot += 1
        if sinistra != destra:
            viol += 1
            if len(esempi) < 5:
                esempi.append(dict(alice=a, basis=b, intercettato=inter,
                                   p1=p1, p2=p2, det_flip=det, bob_raw=braw,
                                   osservato=sinistra, xor=destra))
    return tot, viol, esempi


def su_griglia(percorso):
    d = pd.read_csv(percorso)
    f = ["f_eve", "f_canale", "f_rivelatore"]
    cf = ["qber_cf_eve", "qber_cf_canale", "qber_cf_rivelatore"]
    manca = [c for c in f + cf + ["qber"] if c not in d.columns]
    if manca:
        print(f"  colonne assenti, salto: {manca}")
        return

    somma = d[f].sum(axis=1)
    nulle = (somma < 1e-12).to_numpy()
    print(f"  finestre totali            {len(d)}")
    print(f"  a denominatore nullo       {nulle.sum()} "
          f"({nulle.mean():.3%})  -> y = (0,0,0), fuori dal simplesso")
    if nulle.any() and "intercept_rate" in d.columns:
        g = d[nulle].groupby(["intercept_rate", "depolar_total"]).size()
        print("  dove si concentrano (prime 5 celle):")
        for (ir, dp), k in g.sort_values(ascending=False).head(5).items():
            tot_cella = ((d.intercept_rate == ir) &
                         (d.depolar_total == dp)).sum()
            print(f"    I_e={ir:<5} D_tot={dp:<6} {k:>5} su {tot_cella} "
                  f"({k/tot_cella:.1%})")
    altre = ~nulle
    if altre.any():
        s = (somma[altre] - 1.0).abs()
        print(f"  sulle restanti, |somma(f)-1| max {s.max():.2e} "
              f"(deve essere ~0: le f sono normalizzate per costruzione)")

    res = d[cf].sum(axis=1) - d["qber"]
    print(f"  residuo di additivita' sui tassi, somma(cf) - qber:")
    print(f"    mediana {res.median():+.5f}   media {res.mean():+.5f}   "
          f"max {res.max():+.5f}")
    print(f"    frazione positiva {(res > 0).mean():.3%}")
    print("    Atteso POSITIVO e di ordine 2*p_i*p_j: la composizione e'")
    print("    XOR, e due inversioni che si annullano danno un bit")
    print("    corretto, quindi la somma dei tassi sovrastima il QBER.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dati", default="data/qber_grid.csv")
    ap.add_argument("--salta-griglia", action="store_true")
    a = ap.parse_args()

    attribute, flips, pauli_tab, nome = carica_attribute()
    print(f"modulo {nome}")
    print()
    print("=" * 62)
    print("1. IDENTITA' XOR, enumerazione esaustiva degli ingressi")
    print("=" * 62)
    tot, viol, esempi = enumera(attribute, flips, pauli_tab)
    print(f"  combinazioni esaminate {tot}")
    print(f"  violazioni             {viol}")
    if viol:
        print("  ESITO: l'identita' NON vale. Primi controesempi:")
        for e in esempi:
            print(f"    {e}")
    else:
        print("  ESITO: nessuna violazione su alcun ingresso possibile.")
        print("  Essendo l'enumerazione esaustiva e `attribute` una")
        print("  funzione pura, questo non e' un test campionario ma una")
        print("  verifica completa sul codice eseguito.")
    print()

    if not a.salta_griglia:
        print("=" * 62)
        print("2. CASI LIMITE E ADDITIVITA' SUL FILE DI GRIGLIA")
        print("=" * 62)
        if not Path(a.dati).is_file():
            print(f"  {a.dati} non trovato, salto")
        else:
            su_griglia(a.dati)
    return 1 if viol else 0


if __name__ == "__main__":
    raise SystemExit(main())