"""Aggiunge il controllo di sola forma: il vicino sulla media vera.

Richiede patch_ricentraggio.py gia' applicata.

IL DIFETTO DA CHIUDERE. La corsa con --ricentra mostra che il 99% del
deficit di MMD del condizionato era errore di media, e che la nuvola
ricentrata (0.00060 medio) sta SOTTO `lineare` (0.00283) su tutti e
cinque i semi. Ma il confronto non e' equo: al generatore viene iniettata
la media VERA, mentre `lineare` usa la media INTERPOLATA, che un errore
ce l'ha ancora — 0.00305 sul seme 0, fino a 0.00820 sul seme 4. Parte del
vantaggio del generatore e' semplicemente che gli e' stata passata la
risposta.

IL CONTROLLO. `vicino_ricentrato` prende le stesse finestre reali della
cella adiacente e le trasla sulla media vera, esattamente come si fa col
campione generato. Entrambi ricevono la stessa iniezione, quindi la
differenza che resta e' SOLO forma: marginali, correlazioni, rumore.

E' anche, letto al contrario, `lineare` con l'errore di interpolazione
azzerato — cioe' il miglior stimatore geometrico ottenibile su questa
griglia. Se il condizionato ricentrato lo batte ancora, il circuito
produce finestre piu' simili a quelle vere di quanto non lo siano
finestre vere prese a una cella di distanza, ed e' un'affermazione forte
che regge da sola. Se lo pareggia, la forma del generatore e' equivalente
a quella del dato reale vicino, che resta un buon risultato ma va detto
cosi'. Se perde, il vantaggio osservato era tutto nell'iniezione.

UNA NOTA SUL TRONCAMENTO. Entrambe le nuvole vengono troncate a [0, 1]
dopo la traslazione, come gia' fa `X_lin`. Lo spostamento applicato al
campione generato e' mediamente dieci volte quello applicato al vicino,
quindi il troncamento morde di piu' sul generato: se la differenza fra i
due ricentrati fosse piccola, varrebbe la pena misurare quanta massa
viene troncata prima di leggerla.

NESSUN EFFETTO SUL FLUSSO CASUALE. Il controllo non estrae niente: usa
X_vic, gia' campionato, e mu_vero, gia' calcolato. Una corsa con questa
patch e senza --ricentra resta bit-identica; una con --ricentra riproduce
esattamente i numeri di rq2_ricentrato.jsonl, perche' l'unica aggiunta e'
un calcolo deterministico su dati esistenti. La firma non cambia: i nuovi
record si aggiungono a quelli vecchi senza separarsi, ed e' voluto.

    cd ~/PycharmProjects/Quantum/qgan-qber
    python patch_vicino_ricentrato.py qgan/scripts/rq2_reale.py

    python qgan/scripts/rq2_reale.py --dati data/qber_grid.csv \\
        --seeds 5 --ricentra --backend denso \\
        --out qgan/results/rq2_ricentrato2.jsonl

    python contrasto_campi.py --dir qgan/results/rq2_ricentrato2.jsonl \\
        --famiglie mmd --riferimento condizionato_ricentrato
"""

import shutil
import sys

# ---------------------------------------------------------------- 1
V_VOCE = '''                "mmd_incondizionato_ricentrato": float(metrics.mmd_rbf(
                    Xh, np.clip(X_unc + (mu_vero - mu_unc), 0.0, 1.0),
                    sigma=sig)),
            }'''
N_VOCE = '''                "mmd_incondizionato_ricentrato": float(metrics.mmd_rbf(
                    Xh, np.clip(X_unc + (mu_vero - mu_unc), 0.0, 1.0),
                    sigma=sig)),
                # Le finestre VERE della cella adiacente sulla stessa
                # media vera: l'unico confronto di sola forma, perche'
                # riceve l'identica iniezione del campione generato.
                # E' anche `lineare` con l'errore di interpolazione
                # azzerato, cioe' il miglior stimatore geometrico.
                "mmd_vicino_ricentrato": float(metrics.mmd_rbf(
                    Xh,
                    np.clip(X_vic + (mu_vero - X_vic.mean(axis=0)),
                            0.0, 1.0),
                    sigma=sig)),
            }'''

# ---------------------------------------------------------------- 2
V_AGG = '''            "mmd_incondizionato_ricentrato_m":
                agg("mmd_incondizionato_ricentrato") / len(per_cella),'''
N_AGG = '''            "mmd_incondizionato_ricentrato_m":
                agg("mmd_incondizionato_ricentrato") / len(per_cella),
            "mmd_vicino_ricentrato_m":
                agg("mmd_vicino_ricentrato") / len(per_cella),'''

# ---------------------------------------------------------------- 3
V_STAMPA = '''            print(f"  riferimento: lineare {r['mmd_lineare_m']:.5f}. Se la "
                  f"ricentrata gli si avvicina, la forma era giusta e il "
                  f"difetto e' la media.", flush=True)'''
N_STAMPA = '''            print(f"  SOLA FORMA  cond ricentrato "
                  f"{r['mmd_condizionato_ricentrato_m']:.5f}  contro "
                  f"vicino ricentrato {r['mmd_vicino_ricentrato_m']:.5f}"
                  f"   (stessa media iniettata: resta solo la forma)",
                  flush=True)
            print(f"  per scala: lineare {r['mmd_lineare_m']:.5f}, che e' il "
                  f"vicino sulla media INTERPOLATA — non confrontabile col "
                  f"ricentrato, che riceve quella vera.", flush=True)'''

MODIFICHE = [
    ("voce per cella", V_VOCE, N_VOCE),
    ("aggregato", V_AGG, N_AGG),
    ("stampa di sola forma", V_STAMPA, N_STAMPA),
]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    percorso = sys.argv[1]
    with open(percorso, encoding="utf-8") as f:
        testo = f.read()

    if "RICENTRA" not in testo:
        print("Serve prima patch_ricentraggio.py: `RICENTRA` non c'e'.")
        return 1
    if "mmd_vicino_ricentrato" in testo:
        print("La patch sembra gia' applicata.")
        return 1

    for nome, vecchio, nuovo in MODIFICHE:
        n = testo.count(vecchio)
        if n != 1:
            print(f"ABORT [{nome}]: {n} occorrenze, attesa 1. "
                  "Nessuna scrittura.")
            return 1
        testo = testo.replace(vecchio, nuovo, 1)

    compile(testo, percorso, "exec")

    shutil.copy2(percorso, percorso + ".prima_vicino")
    with open(percorso, "w", encoding="utf-8") as f:
        f.write(testo)

    print(f"{len(MODIFICHE)} modifiche applicate a {percorso}")
    print(f"copia dell'originale in {percorso}.prima_vicino")
    print()
    print("Il controllo non estrae numeri casuali, quindi la corsa deve")
    print("riprodurre esattamente rq2_ricentrato.jsonl sulle quantita' che")
    print("gia' c'erano — ed e' questa la prova che la patch non tocca")
    print("niente. La firma NON cambia, ma `gia_presenti` legge solo il")
    print("file di uscita: scrivendo su un file nuovo non serve --forza.")
    print()
    print("  python qgan/scripts/rq2_reale.py --dati data/qber_grid.csv \\")
    print("      --seeds 5 --ricentra --backend denso \\")
    print("      --out qgan/results/rq2_ricentrato2.jsonl")
    print()
    print("Atteso sul seme 0: cond ricentrato 0.00068, come prima.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())