"""Controlla che le repliche del pre-processing siano davvero repliche.

Un esperimento su dieci repliche e' vacuo se il seme non muove lo split. Qui
si verifica il contrario, prima di spendere venti addestramenti:

  1. PRESENZA. Ogni directory contiene i CSV attesi, con lo stesso numero di
     righe (il rapporto di squilibrio e' lo stesso, cambia solo quali righe).
  2. DISTINZIONE. I file di training sono a due a due DIVERSI. Se due repliche
     coincidono, il seme non e' arrivato dove deve.
  3. ANCORAGGIO. La replica del seme 42 e' identica a data/ids, bit per bit:
     e' il ponte con tutti i risultati gia' raccolti.
  4. PCA RIADATTATA. La varianza spiegata varia fra repliche. Se fosse
     identica a molte cifre, la PCA sarebbe stata riusata e non riadattata:
     le repliche condividerebbero la proiezione e sarebbero meno
     indipendenti di quanto si dichiara.

Nessuna scrittura, nessuna modifica. Esce 1 se un controllo fallisce.

    python src/ids/verifica_repliche.py --semi 42,101,202,303,404,505,606,707,808,909
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

QUI = Path(__file__).resolve().parent
RADICE = QUI.parent.parent


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blocco in iter(lambda: f.read(1 << 20), b""):
            h.update(blocco)
    return h.hexdigest()


def righe(p: Path) -> int:
    with open(p) as f:
        return sum(1 for _ in f) - 1          # meno l'intestazione


def numeri(d, prefisso=""):
    """Campi numerici di un JSON annidato, appiattiti."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(numeri(v, f"{prefisso}{k}."))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefisso.rstrip(".")] = float(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semi", default="42,101,202,303,404,505,606,707,808,909")
    ap.add_argument("--classe", default="r2l")
    ap.add_argument("--ratio", default="1_1000")
    ap.add_argument("--pca", default="local")
    ap.add_argument("--radice", default=str(RADICE / "data"))
    a = ap.parse_args()

    semi = [int(x) for x in a.semi.split(",")]
    base = Path(a.radice)
    treno = f"{a.classe}_train_{a.ratio}_{a.pca}.csv"
    val = (f"{a.classe}_val.csv" if a.pca == "global"
           else f"{a.classe}_val_{a.ratio}_local.csv")
    attesi = [treno, val]

    esiti = []

    # ---------------------------------------------------------- 1. presenza
    print("=" * 68)
    print("1. PRESENZA E DIMENSIONI")
    print("=" * 68)
    dirs, mancanti = {}, []
    for s in semi:
        d = base / f"ids_s{s}"
        if not d.is_dir():
            mancanti.append(str(d))
            continue
        assenti = [n for n in attesi if not (d / n).exists()]
        if assenti:
            mancanti.append(f"{d}: manca {', '.join(assenti)}")
            continue
        dirs[s] = d
    for m in mancanti:
        print(f"  ASSENTE  {m}")
    if not dirs:
        print("\n  Nessuna replica trovata: generale prima con")
        print("  preprocess_nslkdd.py --seed <s> --out data/ids_s<s>")
        return 1

    conteggi = {s: tuple(righe(dirs[s] / n) for n in attesi) for s in dirs}
    distinti = set(conteggi.values())
    for s in sorted(dirs):
        print(f"  seme {s:<5d} {dirs[s].name:<14s} "
              + "  ".join(f"{n}={c}" for n, c in zip(attesi, conteggi[s])))
    ok1 = not mancanti and len(distinti) == 1
    if len(distinti) > 1:
        print("\n  ATTENZIONE: numerosita' diverse fra repliche. Con lo stesso")
        print("  --ratio dovrebbero coincidere; una differenza indica che il")
        print("  sottocampionamento non e' stato applicato allo stesso modo.")
    print(f"\n  esito: {'OK' if ok1 else 'DA CONTROLLARE'}")
    esiti.append(("presenza e dimensioni", ok1))

    # -------------------------------------------------------- 2. distinzione
    print()
    print("=" * 68)
    print("2. LE REPLICHE SONO DIVERSE FRA LORO")
    print("=" * 68)
    impronte = {s: md5(dirs[s] / treno) for s in sorted(dirs)}
    for s, h in impronte.items():
        print(f"  seme {s:<5d} {h}  {treno}")
    coppie_uguali = [(x, y) for x, y in itertools.combinations(sorted(dirs), 2)
                     if impronte[x] == impronte[y]]
    n_dist = len(set(impronte.values()))
    print(f"\n  impronte distinte: {n_dist}/{len(impronte)}")
    for x, y in coppie_uguali:
        print(f"  IDENTICHE: seme {x} e seme {y}")
    ok2 = not coppie_uguali
    if not ok2:
        print("\n  Due repliche identiche significano che il seme non governa")
        print("  lo split: l'esperimento misurerebbe meno di quanto dichiara.")
    print(f"\n  esito: {'OK' if ok2 else 'FALLITO'}")
    esiti.append(("repliche distinte", ok2))

    # --------------------------------------------------------- 3. ancoraggio
    print()
    print("=" * 68)
    print("3. ANCORAGGIO AI RISULTATI ESISTENTI")
    print("=" * 68)
    orig = base / "ids" / treno
    if 42 in dirs and orig.exists():
        h42, ho = impronte[42], md5(orig)
        ok3 = h42 == ho
        print(f"  data/ids_s42/{treno}  {h42}")
        print(f"  data/ids/{treno}      {ho}")
        print(f"\n  {'IDENTICI' if ok3 else 'DIVERSI'}")
        if ok3:
            print("  La replica 42 riproduce i dati su cui poggiano i risultati")
            print("  gia' raccolti: le sue righe devono coincidere con quelle")
            print("  della corsa a cinque semi. E' una regressione gratuita.")
        else:
            print("  Il seme 42 non riproduce data/ids: le repliche non sono")
            print("  confrontabili con il lavoro precedente, e va capito")
            print("  perche' prima di interpretare qualunque differenza.")
    else:
        ok3 = True
        print("  salto: serve la replica 42 e data/ids presenti")
    esiti.append(("ancoraggio al seme 42", ok3))

    # ------------------------------------------------------ 4. PCA riadattata
    print()
    print("=" * 68)
    print("4. LA PCA E' STATA RIADATTATA")
    print("=" * 68)
    meta = {}
    for s in sorted(dirs):
        p = dirs[s] / "meta.json"
        if p.exists():
            try:
                meta[s] = numeri(json.loads(p.read_text()))
            except json.JSONDecodeError:
                pass
    if len(meta) < 2:
        ok4 = True
        print("  salto: meta.json assente o non leggibile in troppe repliche")
    else:
        campi = sorted(set.intersection(*(set(m) for m in meta.values())))
        # Il seme varia per costruzione: se lo contassimo come prova che la
        # PCA e' stata riadattata, il controllo passerebbe sempre.
        identif = [c for c in campi
                   if any(t in c.lower() for t in ("seed", "seme"))]
        campi = [c for c in campi if c not in identif]
        for c in identif:
            print(f"  ignorato  {c:<34s} (identifica la replica, non la PCA)")
        variabili, costanti = [], []
        for c in campi:
            vals = [meta[s][c] for s in sorted(meta)]
            (variabili if max(vals) - min(vals) > 1e-9 else costanti).append(c)
        for c in variabili:
            vals = [meta[s][c] for s in sorted(meta)]
            print(f"  VARIA     {c:<34s} min {min(vals):.6f}  "
                  f"max {max(vals):.6f}")
        for c in costanti:
            print(f"  costante  {c:<34s} {meta[sorted(meta)[0]][c]:.6f}")
        ok4 = len(variabili) > 0
        print()
        if ok4:
            print("  Almeno una quantita' varia fra repliche: la proiezione e'")
            print("  stata ricalcolata su ogni split, come serve.")
        else:
            print("  Nessuna quantita' varia. Se fra queste c'e' la varianza")
            print("  spiegata, la PCA e' la stessa in tutte le repliche: lo")
            print("  split varia ma la proiezione no, e le repliche sono meno")
            print("  indipendenti di quanto si dichiarerebbe nel capitolo.")
    esiti.append(("PCA riadattata", ok4))

    print()
    print("=" * 68)
    for n, v in esiti:
        print(f"  {n:26s} {'OK' if v else 'FALLITO'}")
    print("=" * 68)
    tutti = all(v for _, v in esiti)
    print("Le repliche sono utilizzabili." if tutti else
          "Almeno un controllo non regge: non lanciare le venti corse prima\n"
          "di averlo chiarito, perche' l'inferenza sull'asse replica ne\n"
          "dipende interamente.")
    return 0 if tutti else 1


if __name__ == "__main__":
    raise SystemExit(main())