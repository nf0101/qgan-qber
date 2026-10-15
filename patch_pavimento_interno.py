"""Porta il pavimento MMD dentro la corsa, sulle stesse celle.

Richiede patch_ricentraggio.py e patch_vicino_ricentrato.py applicate.

IL DIFETTO DA CHIUDERE. Il confronto fra l'errore di forma del generatore
e il segnale di forma disponibile poggia oggi su due misure diverse:

  errore del generatore   mmd_condizionato_ricentrato - mmd_vicino_ricentrato
                          dentro la corsa, 12 celle per seme, media
  segnale disponibile     mmd_vicino_ricentrato - pavimento
                          dalla diagnosi sul CSV, 24 celle, mediana

Ciascuna e' appaiata dentro la propria misura, quindi immune al bias del
ricentraggio; ma il loro RAPPORTO mescola insiemi di celle e aggregazioni
diversi, e resta percio' un'indicazione di scala anziche' una stima.

IL RIMEDIO. Il pavimento non ha bisogno di una misura separata: si calcola
dentro la corsa, sulla cella esclusa stessa, come MMD fra due meta' delle
sue finestre reali con la seconda ricentrata sulla prima. Subisce lo
stesso bias del pareggiamento delle medie, sta sulle stesse celle, entra
nello stesso record. Da quel momento le due quantita'

    eccesso_generatore = mmd_cond_ric   - mmd_interno_ric
    eccesso_vicino     = mmd_vicino_ric - mmd_interno_ric

sono differenze appaiate per (corsa, cella) sulla medesima base, e il
rapporto diventa una stima con intervallo, calcolabile dal bootstrap
incrociato di contrasto_campi.py.

NESSUN EFFETTO SUL FLUSSO CASUALE. La permutazione delle finestre usa un
generatore PROPRIO, seminato in modo deterministico da (seme, indice di
cella), e non tocca `rng_diag`. Una corsa con questa patch riproduce
percio' esattamente rq2_ricentrato2.jsonl su tutte le quantita' che gia'
c'erano: e' il controllo da fare per primo.

    cd ~/PycharmProjects/Quantum/qgan-qber
    python patch_pavimento_interno.py qgan/scripts/rq2_reale.py

    python qgan/scripts/rq2_reale.py --dati data/qber_grid.csv \\
        --seeds 5 --ricentra --backend denso \\
        --out qgan/results/rq2_omogeneo.jsonl

    python contrasto_campi.py --dir qgan/results/rq2_omogeneo.jsonl \\
        --famiglie mmd --riferimento interno_ricentrato
"""

import shutil
import sys

# ---------------------------------------------------------------- 1
V_VOCE = '''                "mmd_vicino_ricentrato": float(metrics.mmd_rbf(
                    Xh,
                    np.clip(X_vic + (mu_vero - X_vic.mean(axis=0)),
                            0.0, 1.0),
                    sigma=sig)),
            }'''
N_VOCE = '''                "mmd_vicino_ricentrato": float(metrics.mmd_rbf(
                    Xh,
                    np.clip(X_vic + (mu_vero - X_vic.mean(axis=0)),
                            0.0, 1.0),
                    sigma=sig)),
                # PAVIMENTO sulla cella stessa: due meta' delle sue
                # finestre reali, la seconda ricentrata sulla prima.
                # Subisce lo stesso bias del pareggiamento delle medie,
                # quindi e' il riferimento corretto per le due righe
                # sopra — e sta sulle stesse celle, nello stesso record,
                # con la stessa aggregazione.
                "mmd_interno_ricentrato": float(_pavimento(
                    Xh, n_gen, sig, metrics.mmd_rbf, seed, i)),
            }'''

# ---------------------------------------------------------------- 2
V_FUN = '''def interne(d):'''
N_FUN = '''def _pavimento(Xh, n_gen, sig, mmd, seme, cella):
    """MMD fra due meta' della stessa cella, la seconda ricentrata.

    Il generatore casuale e' PROPRIO e deterministico in (seme, cella):
    non tocca `rng_diag`, quindi aggiungere questa misura non sposta il
    flusso delle altre e la corsa resta riproducibile.
    """
    import numpy as _np
    r = _np.random.default_rng((int(seme), int(cella), 0xBADA55))
    meta = min(n_gen, len(Xh) // 2)
    if meta < 20:                      # troppo poche per una stima sensata
        return float("nan")
    perm = r.permutation(len(Xh))
    A, B = Xh[perm[:meta]], Xh[perm[meta:2 * meta]]
    B_ric = _np.clip(B + (A.mean(axis=0) - B.mean(axis=0)), 0.0, 1.0)
    return float(mmd(A, B_ric, sigma=sig))


def interne(d):'''

# ---------------------------------------------------------------- 3
V_AGG = '''            "mmd_vicino_ricentrato_m":
                agg("mmd_vicino_ricentrato") / len(per_cella),'''
N_AGG = '''            "mmd_vicino_ricentrato_m":
                agg("mmd_vicino_ricentrato") / len(per_cella),
            "mmd_interno_ricentrato_m":
                agg("mmd_interno_ricentrato") / len(per_cella),
            # Le due quantita' del confronto, ora sulla stessa base.
            "eccesso_generatore": (agg("mmd_condizionato_ricentrato")
                                   - agg("mmd_interno_ricentrato"))
                                  / len(per_cella),
            "eccesso_vicino": (agg("mmd_vicino_ricentrato")
                               - agg("mmd_interno_ricentrato"))
                              / len(per_cella),'''

# ---------------------------------------------------------------- 4
V_STAMPA = '''            print(f"  SOLA FORMA  cond ricentrato "'''
N_STAMPA = '''            eg, ev = r["eccesso_generatore"], r["eccesso_vicino"]
            print(f"  OMOGENEO    pavimento interno "
                  f"{r['mmd_interno_ricentrato_m']:+.5f}   eccesso: "
                  f"generatore {eg:+.5f}  vicino {ev:+.5f}   "
                  f"rapporto {eg/ev if ev else float('nan'):.1f}",
                  flush=True)
            print(f"  SOLA FORMA  cond ricentrato "'''

MODIFICHE = [
    ("funzione del pavimento", V_FUN, N_FUN),
    ("voce per cella", V_VOCE, N_VOCE),
    ("aggregati e due eccessi", V_AGG, N_AGG),
    ("stampa omogenea", V_STAMPA, N_STAMPA),
]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    percorso = sys.argv[1]
    with open(percorso, encoding="utf-8") as f:
        testo = f.read()

    if "mmd_vicino_ricentrato" not in testo:
        print("Servono prima patch_ricentraggio.py e "
              "patch_vicino_ricentrato.py.")
        return 1
    if "mmd_interno_ricentrato" in testo:
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

    shutil.copy2(percorso, percorso + ".prima_pavimento")
    with open(percorso, "w", encoding="utf-8") as f:
        f.write(testo)

    print(f"{len(MODIFICHE)} modifiche applicate a {percorso}")
    print(f"copia dell'originale in {percorso}.prima_pavimento")
    print()
    print("Il pavimento usa un flusso casuale proprio, quindi la corsa")
    print("deve riprodurre rq2_ricentrato2.jsonl su tutto il resto.")
    print("Atteso sul seme 0: cond ricentrato 0.00068, vicino -0.00011.")
    print()
    print("  python qgan/scripts/rq2_reale.py --dati data/qber_grid.csv \\")
    print("      --seeds 5 --ricentra --backend denso \\")
    print("      --out qgan/results/rq2_omogeneo.jsonl")
    print()
    print("  python contrasto_campi.py --dir qgan/results/rq2_omogeneo.jsonl \\")
    print("      --famiglie mmd --riferimento interno_ricentrato")
    print()
    print("Nel riepilogo cercare le due righe mmd_condizionato_ricentrato e")
    print("mmd_vicino_ricentrato: sono i due eccessi sul pavimento, appaiati")
    print("per (corsa, cella), ciascuno col proprio intervallo incrociato.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())