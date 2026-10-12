"""Aggiunge --dati-dir a ids_augmentation.py.

Serve per leggere da una replica del pre-processing invece che dalla sola
data/ids. Quattro modifiche:

  1. `main()` dichiara `global DATA`: `carica` legge quella globale, quindi
     riassegnarla in main basta e non tocca il resto del file.
  2. Nuovo `--dati-dir`, con controllo di esistenza della directory E del CSV
     atteso per --classe/--ratio/--pca: senza, un nome sbagliato arriva fino a
     un traceback di pandas a meta' corsa.
  3. Il NOME della directory entra in `config`, quindi nella firma. Senza
     questo, dieci repliche con gli stessi semi avrebbero la stessa firma e
     `gia_presenti` salterebbe le nove successive come duplicati. Entra il
     nome e non il percorso assoluto, cosi' la firma resta la stessa sulla
     macchina di Alfonso.
  4. Ogni riga di risultato porta un campo `dati` di primo livello, cosi'
     l'analisi puo' raggruppare per replica senza scavare in config.

Non modifica nulla se le stringhe attese non corrispondono esattamente una
volta: in quel caso stampa cosa non ha trovato e si ferma.

    cd ~/PycharmProjects/Quantum/qgan-qber
    python patch_augmentation.py src/ids/ids_augmentation.py
"""

import io
import shutil
import sys

VECCHIO_MAIN = "def main():\n    ap = argparse.ArgumentParser()\n"
NUOVO_MAIN = "def main():\n    global DATA\n    ap = argparse.ArgumentParser()\n"

VECCHIO_ARG = (
    '    ap.add_argument("--out", default="results/ids_augmentation.jsonl")\n'
    '    a = ap.parse_args()\n'
)
NUOVO_ARG = '''    ap.add_argument("--dati-dir", default=None,
                    help="directory dei CSV, una per replica del "
                         "pre-processing (data/ids_s42, data/ids_s101...). "
                         "Il nome entra nella firma: repliche diverse non "
                         "si scambiano per duplicati.")
    ap.add_argument("--out", default="results/ids_augmentation.jsonl")
    a = ap.parse_args()

    if a.dati_dir:
        DATA = Path(a.dati_dir).resolve()
    if not DATA.is_dir():
        raise SystemExit(f"directory dati inesistente: {DATA}")
    atteso = DATA / f"{a.classe}_train_{a.ratio}_{a.pca}.csv"
    if not atteso.exists():
        raise SystemExit(
            f"manca {atteso.name} in {DATA}\\n"
            "La replica non contiene questa combinazione di --classe, "
            "--ratio e --pca: rigenerala con preprocess_nslkdd.py --seed."
        )
    print(f"dati {DATA}")
'''

VECCHIO_CONF = "    config = dict(FROZEN, classe=a.classe, ratio=a.ratio, pca=a.pca,\n"
NUOVO_CONF = ("    config = dict(FROZEN, dati=DATA.name, classe=a.classe, "
              "ratio=a.ratio,\n                  pca=a.pca,\n")

VECCHIO_RIGA = "        r.update(corsa.intestazione(config))\n"
NUOVO_RIGA = ('        r["dati"] = DATA.name\n'
              "        r.update(corsa.intestazione(config))\n")

MODIFICHE = [
    (VECCHIO_MAIN, NUOVO_MAIN),
    (VECCHIO_ARG, NUOVO_ARG),
    (VECCHIO_CONF, NUOVO_CONF),
    (VECCHIO_RIGA, NUOVO_RIGA),
]


def main():
    percorso = sys.argv[1] if len(sys.argv) > 1 else "src/ids/ids_augmentation.py"
    s = io.open(percorso, encoding="utf-8").read()

    if "--dati-dir" in s and "global DATA" in s:
        print("gia' applicato: lo script accetta gia' --dati-dir")
        return 0

    mancanti = [v for v, _ in MODIFICHE if s.count(v) != 1]
    if mancanti:
        print("ATTENZIONE: queste stringhe non compaiono esattamente una volta")
        for v in mancanti:
            print(f"  {s.count(v)} occorrenze: {v!r}")
        print("\nIl file NON e' stato modificato. Incollami le righe attorno")
        print("a 'def main():', a '--out' e a 'config = dict(FROZEN'.")
        return 1

    shutil.copy(percorso, percorso + ".bak")
    for v, n in MODIFICHE:
        s = s.replace(v, n)
    io.open(percorso, "w", encoding="utf-8").write(s)

    import ast
    ast.parse(s)
    print(f"ok: {len(MODIFICHE)} modifiche applicate, copia in {percorso}.bak")
    print()
    print("La firma cambia: le righe vecchie in results/ non vengono piu'")
    print("riconosciute come duplicati. Scrivi le repliche in un file nuovo.")
    print()
    print("Verifica che senza --dati-dir il comportamento sia invariato:")
    print("  python src/ids/ids_augmentation.py --help | grep dati-dir")
    return 0


if __name__ == "__main__":
    sys.exit(main())