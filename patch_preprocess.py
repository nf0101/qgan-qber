"""Parametrizza preprocess_nslkdd.py sul seme.

Tre modifiche:
  1. `main()` accetta --seed e --out, riassegnando le globali SEED e OUT.
  2. `strategy` diventa f"PCA-{K}" invece della costante "PCA-6", che era
     sbagliata (K vale 5) e che ogni rigenerazione avrebbe reintrodotto.
  3. `build(labels)` riceve il seme esplicitamente: lo prendeva come default
     di funzione, quindi ignorava la riassegnazione.

Il seme governa split, sottocampione dei minoritari e PCA. Variarlo produce
repliche INDIPENDENTI della scelta dei 53 attacchi, che e' cio' che serve per
generalizzare oltre un singolo split. Il test viene da KDDTest+.txt e non
dallo split, quindi resta lo stesso in ogni replica.

Non modifica nulla se le stringhe attese non corrispondono: in quel caso
stampa cosa non ha trovato e si ferma.

    cd ~/PycharmProjects/Quantum/qgan-qber
    python patch_preprocess.py src/ids/preprocess_nslkdd.py
"""

import io
import shutil
import sys

AGGIUNTA = '''    import argparse
    global OUT, SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(OUT),
                    help="directory di uscita; una per seme")
    a = ap.parse_args()
    SEED, OUT = a.seed, Path(a.out)
    print(f"seme {SEED}   uscita {OUT}")
'''

MODIFICHE = [
    ("def main():\n", "def main():\n" + AGGIUNTA),
    ('"strategy": "PCA-6"', '"strategy": f"PCA-{K}"'),
    ("d = build(labels)", "d = build(labels, SEED)"),
]


def main():
    percorso = sys.argv[1] if len(sys.argv) > 1 else "src/ids/preprocess_nslkdd.py"
    s = io.open(percorso, encoding="utf-8").read()

    if "--seed" in s and "global OUT, SEED" in s:
        print("gia' applicato: lo script accetta gia' --seed")
        return 0

    mancanti = [v for v, _ in MODIFICHE if s.count(v) != 1]
    if mancanti:
        print("ATTENZIONE: queste stringhe non compaiono esattamente una volta")
        for v in mancanti:
            print(f"  {s.count(v)} occorrenze: {v!r}")
        print("\nIl file NON e' stato modificato. Incollami le righe attorno")
        print("a 'def main():' e a 'strategy' e adeguo lo script.")
        return 1

    shutil.copy(percorso, percorso + ".bak")
    for v, n in MODIFICHE:
        s = s.replace(v, n)
    io.open(percorso, "w", encoding="utf-8").write(s)

    import ast
    ast.parse(s)
    print(f"ok: {len(MODIFICHE)} modifiche applicate, copia in {percorso}.bak")
    print("\nVerifica che il seme 42 riproduca i dati attuali:")
    print("  python src/ids/preprocess_nslkdd.py --seed 42 --out /tmp/ids_check")
    print("  md5sum /tmp/ids_check/r2l_train_1_1000_local.csv \\")
    print("         data/ids/r2l_train_1_1000_local.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())