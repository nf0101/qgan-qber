"""Sposta in archivio/ le corse IDS prodotte con protocolli superati.

Il problema: in results/ convivono corse fatte con protocolli diversi, e i
numeri si assomigliano abbastanza da poter essere citati per sbaglio. Le
corse su reps, per esempio, misurano F1 a soglia 0.5 sul validation, senza
gaussiana di controllo e senza test: oggi quei numeri non significano quello
che sembrano.

Cosa resta vivo: SOLO results/ids_finale. Le tre passate intermedie sono
strettamente contenute in quella, perche' le patch sono state additive e i
semi non sono cambiati — l'identita' delle MMD e degli scarti di covarianza
fra le passate lo conferma. Quindi archiviarle non perde nessun dato.

Non indovina: classifica per nome secondo l'elenco qui sotto e di tutto
cio' che non riconosce si limita a dare notizia, senza toccarlo.

    python archivia_risultati.py                 # dice cosa farebbe
    python archivia_risultati.py --esegui
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

VIVI = ["ids_finale"]

GRUPPI = [
    (
        "passate intermedie delle repliche",
        ["ids_repliche", "ids_repliche2", "ids_repliche3"],
        "Le tre passate che hanno portato a ids_finale, in ordine: dieci\n"
        "repliche con F1@0.5 e PR-AUC sul validation; poi con la soglia\n"
        "scelta fuori campione; poi con gaussiana, gaussiana_cov e le due\n"
        "scale di rumore. Ogni campo che contengono e' presente anche in\n"
        "ids_finale, con gli stessi valori: le patch erano additive e i semi\n"
        "non sono cambiati, come conferma l'identita' delle MMD e degli\n"
        "scarti di covarianza fra le passate. Si tengono come registro\n"
        "della progressione, non perche' servano.",
    ),
    (
        "ablazione sulla profondita' dell'ansatz",
        ["ids_reps4", "ids_reps6", "ids_reps8", "ids_reps10", "ids_reps12"],
        "PROTOCOLLO SUPERATO. F1 a soglia fissa 0.5 sul validation, un solo\n"
        "split, nessuna gaussiana di controllo, nessuna misura sul test.\n"
        "NON citare questi numeri. L'ablazione sulla profondita' va rifatta\n"
        "sul protocollo finale, che e' l'unico modo di dire se la\n"
        "conclusione «il circuito non aggiunge nulla oltre i primi due\n"
        "momenti» valga anche a profondita' maggiore di reps=4.",
    ),
    (
        "confronti su singolo split",
        ["ids_1_1000", "ids_1_200", "ids_aug_1_200", "ids_5semi",
         "ids_ablazione5", "ids_augmentation.jsonl"],
        "PROTOCOLLO SUPERATO. Cinque semi su UN solo campione di 53 attacchi\n"
        "(ids_1_1000) o 269 (ids_1_200), quello fissato dal seme 42. Da qui\n"
        "veniva il divario di -0.196 della QGAN, che su dieci split\n"
        "indipendenti vale -0.112: il seme 42 era la replica piu' ostile\n"
        "delle dieci, con -0.206 contro -0.049 della migliore. Si tengono\n"
        "perche' sono l'origine del numero corretto.",
    ),
    (
        "calibrazione del raggio e ordinamento delle feature",
        ["ids_aug_calibrato", "ids_aug_feature", "ids_ord_nat",
         "ids_ord_cri", "ids_nat_feat", "ids_cri_feat", "ids_ord_cri_feat"],
        "PROTOCOLLO SUPERATO, ma attenzione: da qui vengono due elementi di\n"
        "O1 che il capitolo cita ancora. E' un disegno 2x2 — ordinamento\n"
        "naturale o per criterio, raggio scalare 0.9327 o per feature\n"
        "[1.3989, 1.2907, 0.7322, 0.5930, 0.3959] — con UNA corsa per\n"
        "cella, F1 a soglia 0.5 sul validation di un solo split.\n"
        "\n"
        "Il raggio 0.9327 resta valido: fu calibrato sulla dispersione del\n"
        "generatore all'inizializzazione contro quella dei 53 reali\n"
        "(0.0647), che e' una proprieta' del generatore e non dipende dal\n"
        "protocollo di valutazione.\n"
        "\n"
        "La scelta dell'ordinamento naturale e il rigetto dei raggi per\n"
        "feature furono invece giudicati a valle, una volta sola, sulla\n"
        "metrica che poi si e' rivelata invertirsi fra validation e test.\n"
        "Va dichiarato come limite. Rimisurarlo non e' necessario: se il\n"
        "circuito non aggiunge nulla oltre i primi due momenti, la\n"
        "struttura su cui l'ordinamento agisce non porta segnale a valle,\n"
        "quindi che l'ordinamento non conti e' previsto dal risultato\n"
        "principale invece di essere una lacuna.",
    ),
    (
        "prima ablazione causale",
        ["ids_ablazione.jsonl"],
        "PROTOCOLLO SUPERATO. La prima corsa con qgan_sd e smote_distorta,\n"
        "un seme solo: e' da qui che veniva l'ipotesi sulle dispersioni per\n"
        "feature (F1 0.783 -> 0.566 distorcendo SMOTE). Su dieci repliche e\n"
        "sul test quell'ipotesi non regge in nessuno dei due versi. Si tiene\n"
        "perche' documenta come l'ipotesi e' nata, non come conferma.",
    ),
]

# Tutto e' ancorato alla posizione DELLO SCRIPT, non alla directory
# corrente: lo script vive nella radice del repo, quindi funziona anche
# lanciato da fuori. Senza questo, `python qgan-qber/archivia.py` da una
# directory sopra cercherebbe un results/ che non c'e'.
RADICE = Path(__file__).resolve().parent
DEST = RADICE / "archivio" / "ids_protocolli_precedenti"


def git_mv(sorgente: Path, destinazione: Path) -> str:
    """git mv, cosi' la storia segue il file. Ripiega su uno spostamento."""
    try:
        subprocess.run(["git", "mv", str(sorgente), str(destinazione)],
                       check=True, capture_output=True, cwd=RADICE)
        return "git mv"
    except (subprocess.CalledProcessError, FileNotFoundError):
        shutil.move(str(sorgente), str(destinazione))
        return "spostato (non tracciato da git)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--risultati", default=None,
                    help="per default results/ accanto a questo script")
    ap.add_argument("--esegui", action="store_true",
                    help="senza questo non tocca niente")
    a = ap.parse_args()

    base = Path(a.risultati) if a.risultati else RADICE / "results"
    if not base.is_dir():
        raise SystemExit(
            f"non trovo {base}\n"
            f"Questo script si aspetta di stare nella radice del repo, con\n"
            f"results/ accanto. Se l'hai copiato altrove, passa --risultati."
        )
    print(f"repo {RADICE}")

    presenti = sorted(p.name for p in base.iterdir())
    # Nell'elenco i nomi stanno senza estensione: una voce puo' essere una
    # directory (ids_reps8) o un file (ids_ablazione.jsonl), e scriverli in
    # due forme sarebbe un invito a dimenticarne una.
    def risolvi(nome):
        for c in (nome, nome + ".jsonl"):
            if c in presenti:
                return c
        return None

    noti = set(VIVI)
    for _, nomi, _ in GRUPPI:
        for x in nomi:
            noti.update({x, x + ".jsonl"})

    da_spostare = []
    print("=" * 70)
    print(f"{len(presenti)} voci in {base}")
    print("=" * 70)
    for etichetta, nomi, _ in GRUPPI:
        trovati = [c for c in (risolvi(n) for n in nomi) if c]
        print(f"\n{etichetta}")
        if not trovati:
            print("  nessuna presente")
        for n in trovati:
            print(f"  -> archivio   {n}")
            da_spostare.append(n)

    print("\nrestano in results/")
    for n in VIVI:
        print(f"  {'presente' if n in presenti else 'ASSENTE'}   {n}")

    ignoti = [n for n in presenti if n not in noti]
    if ignoti:
        print("\nnon riconosciuti, lasciati dove sono:")
        for n in ignoti:
            print(f"  {n}")
        print("  Se qualcuno di questi e' di un protocollo superato, dimmelo")
        print("  e lo aggiungo all'elenco invece di indovinare.")

    if not da_spostare:
        print("\nniente da spostare.")
        return 0
    if not a.esegui:
        print(f"\n{len(da_spostare)} voci da spostare. Niente e' stato")
        print("toccato: rilancia con --esegui.")
        return 0

    DEST.mkdir(parents=True, exist_ok=True)
    for n in da_spostare:
        come = git_mv(base / n, DEST / n)
        print(f"  {n}  ({come})")

    righe = [
        "# Risultati IDS di protocolli precedenti",
        "",
        "Spostati qui perche' i loro numeri non sono confrontabili con quelli",
        "del capitolo. L'unica directory viva e' `results/ids_finale`.",
        "",
    ]
    for etichetta, nomi, nota in GRUPPI:
        trovati = [n for n in nomi if n in da_spostare]
        if not trovati:
            continue
        righe += [f"## {etichetta}", "",
                  "".join(f"- `{n}`\n" for n in trovati), nota, ""]
    righe += [
        "## Protocollo finale, per confronto",
        "",
        "Dieci repliche indipendenti dello split (semi 42, 101, 202, 303,",
        "404, 505, 606, 707, 808, 909) x due semi di addestramento; tredici",
        "varianti; soglia scelta fuori campione sul training e, per il test,",
        "sul validation; conferma su KDDTest+ mai usato per nessuna scelta;",
        "inferenza con bootstrap incrociato su repliche x semi e unita' di",
        "riferimento la replica. Metrica dichiarata prima della misura:",
        "F1@0.5 sul validation; sul test si riporta la PR-AUC, perche' con",
        "prior 1:1000 in addestramento e 1:5 nel test una soglia fissa",
        "misura lo scarto fra i prior e non la qualita' del metodo.",
        "",
    ]
    (DEST / "README.md").write_text("\n".join(righe), encoding="utf-8")
    print(f"\nscritto {DEST / 'README.md'}")
    print("Controlla con  git status --short  prima di committare.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())