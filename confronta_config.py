"""Dice quale ablazione contiene ciascun gruppo di risultati.

`corsa.py` scrive una configurazione per file, quindi cercare campi che
variano DENTRO un file non trova niente: le ablazioni stanno fra gruppi.
Questo script raggruppa per directory, legge la configurazione di ogni
gruppo e stampa soltanto i campi che DIFFERISCONO fra i gruppi — cioe'
esattamente le variabili manipolate, senza le decine di parametri comuni.

Tre cose che segnala a parte:
  - gruppi senza campo `config`: sono le corse scritte prima che venisse
    aggiunta la provenienza, e non si possono attribuire a un'ablazione;
  - gruppi con configurazioni discordi al proprio interno, che non
    dovrebbero esistere;
  - quanti file e quanti semi distinti contiene ogni gruppo, perche'
    e' cio' che distingue una corsa completa da una di prova.

    python confronta_config.py
    python confronta_config.py --risultati qgan/results
    python confronta_config.py --campo hidden
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

RADICE = Path(__file__).resolve().parent


def prima_riga(percorso):
    try:
        with open(percorso, encoding="utf-8", errors="replace") as f:
            for riga in f:
                riga = riga.strip()
                if riga:
                    return json.loads(riga)
    except (OSError, json.JSONDecodeError):
        pass
    return None


def chiave(v):
    return json.dumps(v, sort_keys=True, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--risultati", default=None,
                    help="albero da ispezionare; default qgan/results")
    ap.add_argument("--campo", default=None,
                    help="mostra solo le righe in cui compare questo campo")
    ap.add_argument("--largo", type=int, default=22,
                    help="larghezza massima di un valore in tabella")
    a = ap.parse_args()

    base = Path(a.risultati) if a.risultati else RADICE / "qgan" / "results"
    if not base.is_dir():
        raise SystemExit(f"non trovo {base}")
    file = sorted(glob.glob(str(base / "**" / "*.jsonl"), recursive=True))
    if not file:
        raise SystemExit(f"nessun .jsonl sotto {base}")

    gruppi, semi, conteggi, discordi, senza = {}, {}, {}, [], []
    for f in file:
        g = os.path.relpath(os.path.dirname(f) or ".", base)
        r = prima_riga(f)
        conteggi[g] = conteggi.get(g, 0) + 1
        if r is None:
            continue
        semi.setdefault(g, set()).add(r.get("seed"))
        c = r.get("config")
        c = c if isinstance(c, dict) else None
        if c is None:
            if g not in senza:
                senza.append(g)
            continue
        if g in gruppi and chiave(gruppi[g]) != chiave(c):
            if g not in discordi:
                discordi.append(g)
        gruppi.setdefault(g, c)

    print(f"radice {base}")
    print(f"{len(file)} file in {len(conteggi)} gruppi, "
          f"{len(gruppi)} con configurazione\n")

    if not gruppi:
        print("Nessun gruppo ha il campo `config`: tutte le corse sono")
        print("anteriori all'introduzione della provenienza.")
        return 1

    tutte = set().union(*(set(c) for c in gruppi.values()))
    varia = sorted(k for k in tutte
                   if len({chiave(c.get(k)) for c in gruppi.values()}) > 1)
    comuni = sorted(tutte - set(varia))

    if a.campo:
        varia = [k for k in varia if a.campo in k] or varia

    print("CAMPI CHE DIFFERISCONO FRA I GRUPPI")
    if not varia:
        print("  nessuno: tutti i gruppi hanno la stessa configurazione")
    else:
        print(f"  {', '.join(varia)}\n")
        intest = f"  {'gruppo':<26} {'file':>5} {'semi':>5}  " + "  ".join(
            f"{k[:a.largo]:>{min(len(k), a.largo)}}" for k in varia)
        print(intest)
        print("  " + "-" * (len(intest) - 2))
        for g in sorted(gruppi):
            vals = []
            for k in varia:
                t = str(gruppi[g].get(k, "--"))
                vals.append(f"{t[:a.largo]:>{min(len(k), a.largo)}}")
            n_semi = len([s for s in semi.get(g, ()) if s is not None])
            print(f"  {g:<26} {conteggi[g]:>5} {n_semi:>5}  " + "  ".join(vals))

    print(f"\n{len(comuni)} campi uguali in tutti i gruppi"
          + (f": {', '.join(comuni[:12])}" if comuni else ""))
    if len(comuni) > 12:
        print(f"  ... e altri {len(comuni) - 12}")

    if senza:
        print("\nGRUPPI SENZA `config` — provenienza assente, non")
        print("attribuibili a un'ablazione:")
        for g in senza:
            print(f"  {g}  ({conteggi[g]} file)")
    if discordi:
        print("\nATTENZIONE: configurazioni discordi dentro lo stesso gruppo:")
        for g in discordi:
            print(f"  {g}")
        print("  Un gruppo dovrebbe contenere una sola configurazione.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())