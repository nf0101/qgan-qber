"""Prova del framework CRN: le tre proprieta' su cui poggia O3.

Il modulo dichiara tre cose. Nessuna e' mai stata verificata.

  1. STABILITA'. Gli stessi parametri nominali identificano gli stessi flussi
     casuali fra esecuzioni.
  2. COERENZA. La catena modellata e' quella che la simulazione esegue
     davvero: l'errore analitico coincide con quello osservato dove la
     catena e' deterministica.
  3. ACCOPPIAMENTO. Condizioni sperimentali diverse restano accoppiate
     posizione per posizione. E' la proprieta' da cui discendono i
     controfattuali analitici, cioe' l'intero obiettivo O3.

La terza e' la decisiva e la sola che non si puo' dedurre leggendo il codice.
Le intercettazioni sono nidificate — la soglia cambia, il flusso no — quindi
una posizione intercettata a r=0,25 lo e' anche a r=0,50. Su quelle posizioni
comuni l'esito di Eve, il Pauli e l'esito di Bob devono essere IDENTICI. Se le
misure non sono accoppiate, divergeranno su circa meta'.

    python src/prova_crn.py                 # con accoppiamento (default)
    python src/prova_crn.py --senza-accoppiamento

Eseguire ENTRAMBE le varianti: il confronto fra le due dice quanto
l'accoppiamento delle misure stia effettivamente facendo.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np

import BB84_crn as B


# =====================================================================
# 1. Stabilita' della derivazione del seed
# =====================================================================
def prova_stabilita(seed=1):
    print("=" * 66)
    print("1. STABILITA' DEI FLUSSI FRA PROCESSI")
    print("=" * 66)

    casi = [("eve", 10), ("det", 10), ("depol_s1", 7), ("bob_meas", 3)]
    miei = {f"{t}:{i}": B.seme_derivato(seed, t, i) for t, i in casi}

    codice = (
        "import sys; sys.path.insert(0, %r)\n"
        "import BB84_crn as B\n"
        "print(repr({f'{t}:{i}': B.seme_derivato(%d, t, i) "
        "for t, i in %r}))" % (sys.path[0], seed, casi)
    )
    altro = eval(subprocess.check_output(
        [sys.executable, "-c", codice]).decode())

    for k in miei:
        stato = "uguale" if miei[k] == altro[k] else "DIVERSO"
        print(f"  {k:16s} {miei[k]:>12d}  {altro[k]:>12d}   {stato}")

    ok = miei == altro
    print(f"\n  esito: {'STABILE' if ok else 'INSTABILE fra processi'}")
    return ok


# =====================================================================
# 2. Coerenza fra catena modellata e catena eseguita
# =====================================================================
def prova_coerenza(num_bits=4000, seed=1):
    print()
    print("=" * 66)
    print("2. COERENZA FRA ERRORE ANALITICO E OSSERVATO")
    print("=" * 66)
    print("  Dove Eve non intercetta la catena e' deterministica dati i Pauli")
    print("  e det_flip: le discordanze devono essere ESATTAMENTE zero.\n")

    ok = True
    for r in (0.0, 0.25, 0.5, 1.0):
        s, _ = B.run_bb84(intercept_rate=r, num_bits=num_bits, seed=seed,
                          verbose=False)
        n, d = s["n_non_intercettati"], s["discordanze"]
        quota = d / n if n else float("nan")
        print(f"  r={r:<5.2f} QBER {s['qber']:>7.2%}   discordanze "
              f"{d:>5d}/{n:<6d}  ({quota:.2%})")
        if d:
            ok = False

    print(f"\n  esito: {'COERENTE' if ok else 'INCOERENTE'}")
    if not ok:
        print("  Discordanze gia' a r=0,00 indicano un problema a monte")
        print("  (misura di Bob o modello del canale). Discordanze che")
        print("  crescono con r indicano le misure di Eve.")
    return ok


# =====================================================================
# 3. Accoppiamento fra condizioni — la prova decisiva
# =====================================================================
def prova_accoppiamento(num_bits=4000, seed=1, ra=0.25, rb=0.5):
    print()
    print("=" * 66)
    print("3. ACCOPPIAMENTO FRA CONDIZIONI")
    print("=" * 66)
    print(f"  Due corse, r={ra} e r={rb}. Le intercettazioni sono nidificate:")
    print("  sulle posizioni intercettate in ENTRAMBE, esito di Eve, Pauli e")
    print("  esito di Bob devono coincidere. Se le misure non sono")
    print("  accoppiate divergeranno su circa meta'.\n")

    _, A = B.run_bb84(intercept_rate=ra, num_bits=num_bits, seed=seed,
                      verbose=False)
    _, C = B.run_bb84(intercept_rate=rb, num_bits=num_bits, seed=seed,
                      verbose=False)
    a = {r["index"]: r for r in A}
    c = {r["index"]: r for r in C}

    comuni = sorted(set(a) & set(c))
    print(f"  posizioni setacciate in comune: {len(comuni)}")
    if not comuni:
        print("  nessuna posizione comune: prova non conclusiva")
        return False

    # Il canale e il rivelatore sono gia' per posizione: se divergono, il
    # problema e' piu' grave dell'accoppiamento delle misure.
    div_p = sum(1 for i in comuni
                if (a[i]["p1"], a[i]["p2"]) != (c[i]["p1"], c[i]["p2"]))
    div_d = sum(1 for i in comuni if a[i]["det_flip"] != c[i]["det_flip"])
    print(f"  Pauli del canale divergenti : {div_p:>5d}/{len(comuni)}")
    print(f"  flip del rivelatore diverg. : {div_d:>5d}/{len(comuni)}")

    # Il cuore: le posizioni intercettate in entrambe le corse.
    entrambe = [i for i in comuni
                if a[i]["eve_intercepted"] and c[i]["eve_intercepted"]]
    solo_b = [i for i in comuni
              if c[i]["eve_intercepted"] and not a[i]["eve_intercepted"]]
    print(f"\n  intercettate in ENTRAMBE     : {len(entrambe)}")
    print(f"  intercettate solo a r={rb}    : {len(solo_b)}  "
          "(nidificazione: atteso > 0)")

    if not entrambe:
        print("  nessuna posizione intercettata in entrambe: alzare num_bits")
        return False

    div_b = sum(1 for i in entrambe if a[i]["bob_bit"] != c[i]["bob_bit"])
    quota = div_b / len(entrambe)
    print(f"\n  esito di Bob divergente      : {div_b:>5d}/{len(entrambe)}"
          f"  ({quota:.1%})")

    ok = div_p == 0 and div_d == 0 and div_b == 0
    print()
    if ok:
        print("  esito: ACCOPPIATO. Le condizioni restano appaiate posizione")
        print("  per posizione e i controfattuali analitici sono fondati.")
    elif quota > 0.2:
        print("  esito: NON ACCOPPIATO. La divergenza vicina al 50% e' la")
        print("  firma di una misura che pesca da un flusso non controllato:")
        print("  le due corse non condividono i numeri casuali dove il QBER")
        print("  da intercettazione si genera. I controfattuali di O3 non")
        print("  poggiano sulla proprieta' che il modulo dichiara.")
    else:
        print("  esito: ACCOPPIAMENTO PARZIALE. Da capire prima di procedere:")
        print("  una divergenza piccola ma non nulla non e' spiegata dalla")
        print("  sola casualita' delle misure.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-bits", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--senza-accoppiamento", action="store_true",
                    help="disattiva l'accoppiamento delle misure, per "
                         "misurare quanto stia effettivamente facendo")
    a = ap.parse_args()

    if a.senza_accoppiamento:
        B.ACCOPPIA_MISURE = False
    stato = "attivo" if B.ACCOPPIA_MISURE else "DISATTIVATO"
    api = "trovata" if B._SET_RS is not None else "ASSENTE"
    print(f"accoppiamento delle misure: {stato}")
    print(f"API ns.set_random_state: {api}")
    if B._SET_RS is None:
        print("  Senza quell'API l'accoppiamento non puo' essere applicato:")
        print("  la prova 3 misurera' il comportamento non accoppiato.")
    print()

    e = [prova_stabilita(a.seed),
         prova_coerenza(a.num_bits, a.seed),
         prova_accoppiamento(a.num_bits, a.seed)]

    print()
    print("=" * 66)
    nomi = ["stabilita'", "coerenza", "accoppiamento"]
    for n, v in zip(nomi, e):
        print(f"  {n:16s} {'OK' if v else 'FALLITA'}")
    print("=" * 66)
    if all(e):
        print("Il framework CRN regge: O3 poggia su una proprieta' verificata.")
    else:
        print("Almeno una proprieta' non regge. Va deciso con Alfonso se")
        print("investire sull'accoppiamento delle misure o se separare O3")
        print("dalla rivendicazione principale.")
    return 0 if all(e) else 1


if __name__ == "__main__":
    sys.exit(main())