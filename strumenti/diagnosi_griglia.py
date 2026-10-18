"""Dice se una griglia ha struttura condizionale oltre la media.

PERCHE'. Sulla griglia stazionaria il controllo di sola forma ha dato un
risultato che chiude RQ2 prima ancora di addestrare: le finestre reali di
una cella adiacente, corretta la SOLA media, sono indistinguibili da
quelle della cella esclusa — MMD stimata nulla. Se conoscere la media
basta a riprodurre la distribuzione, predire la cella esclusa e' un
problema di regressione della media, e nessun modello generativo puo'
guadagnarci: l'interpolazione lineare vince perche' risolve il problema
vero.

Quel controllo e' costato una corsa di QGAN. Non serviva: e' una
proprieta' dei DATI, non del modello, e si misura sul CSV in pochi
secondi. Questo script lo fa, per qualunque griglia, PRIMA di spendere
addestramenti.

COSA MISURA. Per ogni cella interna k:

  grezza        MMD fra le finestre di k e quelle della cella piu' vicina j
  ricentrata    le stesse finestre di j traslate sulla media di k
  interna       MMD fra due meta' casuali delle finestre di k stesse
  interna_ric   le stesse due meta', ma la seconda ricentrata sulla prima

IL PAVIMENTO GIUSTO E' `interna_ric`, NON `interna` NE' LO ZERO. Ricentrare
impone alle due nuvole la stessa media campionaria, che due estrazioni
indipendenti dalla stessa distribuzione NON hanno: la U-statistica non
distorta ne risente con un bias negativo sistematico. Confrontare la
ricentrata con lo zero, o con il pavimento non ricentrato, fa apparire
identiche due distribuzioni che non lo sono. Verificato: su una griglia
sintetica in cui forma e correlazione variano per costruzione, il
confronto con lo zero dichiara comunque "nessuna differenza".

La quantita' da leggere e' quindi l'ECCESSO

    eccesso = ricentrata - interna_ric

cioe' quanta discrepanza sopravvive al pareggiamento della media oltre
quella che il pareggiamento stesso produce su campioni identici.

COME SI LEGGE.

  eccesso ~ 0        la forma non varia fra celle adiacenti: tutta la
                     struttura condizionale sta nella media. RQ2 su questa
                     griglia non discrimina un generatore da un regressore
                     della media. NON vale la pena correrci.

  eccesso > 0        resta discrepanza di forma dopo aver pareggiato la
                     media: la griglia ha struttura che un modello
                     distribuzionale puo' catturare e l'interpolazione no.
                     E' il banco giusto per RQ2.

La scala si legge su `quota_forma = eccesso / grezza`: la frazione del
divario fra celle adiacenti che NON e' spiegata dalla media.

GRIGLIE NON STAZIONARIE. Se il CSV ha una colonna `profilo`, le finestre
di profili diversi hanno distribuzioni diverse per costruzione e
mescolarle gonfia ogni MMD senza che questo significhi struttura FRA
CELLE. Con --per-profilo l'analisi viene ripetuta dentro ciascun profilo,
che e' l'unico confronto interpretabile finche' il profilo non entra nel
vettore di condizionamento.

    cd ~/PycharmProjects/Quantum/qgan-qber
    python diagnosi_griglia.py --dati data/qber_grid.csv
    python diagnosi_griglia.py --dati data/qber_grid_nonstaz.csv --per-profilo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RADICE = Path(__file__).resolve().parents[1]

# Il pacchetto vero sta in qgan/src/qgan/, non in qgan/: alla radice del
# repo esiste anche un qgan/__init__.py che lo maschera se si lascia la
# radice davanti in sys.path. rq2_reale.py risolve inserendo
# qgan/scripts/../src, e qui si fa lo stesso — ma DOPO scripts, perche'
# insert(0, ...) mette davanti l'ultimo inserito.
sys.path.insert(0, str(RADICE / "qgan" / "scripts"))
sys.path.insert(0, str(RADICE / "qgan" / "src"))

OSSERVABILI = ["qber", "qber_rect", "qber_diag", "sift_rate", "loss_rate"]
CONDIZIONI = ["intercept_rate", "length_km", "depolar_total"]


def carica_metriche():
    """La MMD del progetto, non una reimplementazione."""
    try:
        from qgan import metrics
        from qgan.metrics import median_bandwidth
        return metrics.mmd_rbf, median_bandwidth
    except ImportError as e:
        atteso = RADICE / "qgan" / "src" / "qgan" / "metrics.py"
        raise SystemExit(
            f"non trovo qgan.metrics ({e}).\n"
            f"cercato il pacchetto in {RADICE / 'qgan' / 'src'}\n"
            f"  {atteso} "
            f"{'esiste' if atteso.exists() else 'NON esiste'}\n"
            "Lanciare dalla radice del repo, col venv attivo. Se il "
            "pacchetto sta altrove, correggere i due sys.path in testa."
        )


def assi_variabili(d):
    """Assi di condizionamento con almeno TRE valori distinti.

    Un asse con uno o due valori non ha interno: ogni cella vi sta a un
    estremo. Usarlo per escludere azzera le celle interne anche quando
    gli altri assi ne hanno in abbondanza — e' cio' che succede sulle
    griglie ridotte, dove un parametro e' tenuto fisso. Quegli assi vanno
    trattati come costanti, non come dimensioni di interpolazione.
    """
    return [c for c in CONDIZIONI if d[c].nunique() >= 3]


def interne(d, assi=None):
    """Celle non al bordo di nessun asse interpolabile."""
    assi = assi if assi is not None else assi_variabili(d)
    masch = np.ones(len(d), bool)
    for c in assi:
        v = np.sort(d[c].unique())
        masch &= ~d[c].isin([v[0], v[-1]]).to_numpy()
    return masch


def scala_globale(X):
    """[0,1] sui minimi e sulle ampiezze dell'intera griglia.

    La corsa vera scala sul solo addestramento; qui non c'e' una
    partizione, e usare tutto e' l'unica scelta neutra. Sposta i valori
    assoluti di poco e non puo' creare differenze fra celle, perche' la
    trasformazione e' la stessa per tutte.
    """
    lo = X.min(axis=0)
    amp = X.max(axis=0) - lo
    return np.clip((X - lo) / np.where(amp > 0, amp, 1.0), 0.0, 1.0)


def analizza(d, nome, mmd, banda, n_gen, rng):
    X = scala_globale(d[OSSERVABILI].to_numpy(float))
    C_grezze = d[CONDIZIONI].to_numpy(float)
    # Le condizioni vanno normalizzate PRIMA di cercare la cella vicina,
    # come fa rq2_reale.py: in unita' native length_km copre 5-40 e
    # schiaccia gli altri due assi, cosi' che il "vicino" sarebbe sempre
    # la cella con la stessa lunghezza, indipendentemente dal resto.
    assi = assi_variabili(d)
    if not assi:
        print(f"[{nome}] nessun asse con almeno 3 valori, salto")
        return None
    fissi = [c for c in CONDIZIONI if c not in assi]
    # Solo gli assi interpolabili entrano nella distanza: un asse tenuto
    # fisso non puo' contribuire, e includerlo non cambierebbe nulla, ma
    # dichiararlo rende leggibile che cosa si sta misurando.
    C = scala_globale(C_grezze[:, [CONDIZIONI.index(c) for c in assi]])
    chiave = pd.Series(list(map(tuple, C_grezze)), index=d.index)
    masch_int = interne(d, assi)
    if fissi:
        print(f"[{nome}] assi interpolabili {assi}; "
              f"tenuti fissi {fissi} "
              + ", ".join(f"{c}={sorted(d[c].unique())}" for c in fissi))

    celle = sorted(set(chiave))
    centri = {k: C[(chiave == k).to_numpy()][0] for k in celle}
    interne_k = sorted({k for k in celle
                        if masch_int[(chiave == k).to_numpy()].any()})
    if not interne_k:
        print(f"[{nome}] nessuna cella interna, salto")
        return None

    sig = float(banda(X, rng=np.random.default_rng(0)))
    print(f"[{nome}] {len(d)} finestre, {len(celle)} celle, "
          f"{len(interne_k)} interne, sigma {sig:.4f}")

    def estrai(k, quante):
        Y = X[(chiave == k).to_numpy()]
        if len(Y) > quante:
            Y = Y[rng.choice(len(Y), quante, replace=False)]
        return Y

    righe = []
    for k in interne_k:
        Xh = estrai(k, n_gen)
        if len(Xh) < 40:
            continue
        altri = [j for j in celle if j != k]
        j = min(altri, key=lambda a: float(
            ((centri[a] - centri[k]) ** 2).sum()))
        Xv = estrai(j, n_gen)
        Xv_ric = np.clip(Xv + (Xh.mean(axis=0) - Xv.mean(axis=0)), 0.0, 1.0)

        # pavimento: due meta' della STESSA cella, anche ricentrate, per
        # subire lo stesso bias del pareggiamento di media
        tutte = X[(chiave == k).to_numpy()]
        perm = rng.permutation(len(tutte))
        meta = min(n_gen, len(tutte) // 2)
        A, B = tutte[perm[:meta]], tutte[perm[meta:2 * meta]]
        B_ric = np.clip(B + (A.mean(axis=0) - B.mean(axis=0)), 0.0, 1.0)

        righe.append({
            "cella": k,
            "grezza": float(mmd(Xh, Xv, sigma=sig)),
            "ricentrata": float(mmd(Xh, Xv_ric, sigma=sig)),
            "interna": float(mmd(A, B, sigma=sig)),
            "interna_ric": float(mmd(A, B_ric, sigma=sig)),
            "scarto_media": float(np.abs(
                Xh.mean(axis=0) - Xv.mean(axis=0)).mean()),
        })

    if not righe:
        print(f"[{nome}] nessuna cella con finestre sufficienti")
        return None

    r = pd.DataFrame(righe)
    # L'eccesso e' APPAIATO per cella: ogni cella fornisce la propria
    # ricentrata e il proprio pavimento, quindi la differenza elimina
    # l'effetto della cella, che e' la componente dominante di varianza.
    r["eccesso"] = r["ricentrata"] - r["interna_ric"]
    ecc = float(np.median(r["eccesso"]))
    grezza = float(np.median(r["grezza"]))
    quota_forma = ecc / grezza if grezza > 0 else float("nan")
    pos = int((r["eccesso"] > 0).sum())

    print(f"  {'quantita':<14} {'mediana':>10} {'min':>10} {'max':>10}")
    for c in ("grezza", "ricentrata", "interna", "interna_ric"):
        print(f"  {c:<14} {r[c].median():>10.5f} "
              f"{r[c].min():>10.5f} {r[c].max():>10.5f}")
    print(f"  scarto di media fra celle adiacenti: "
          f"{r['scarto_media'].median():.5f}")
    print()
    from scipy import stats
    pv = float(stats.binomtest(pos, len(r), 0.5).pvalue)

    print(f"  pavimento ricentrato (mediana) {float(np.median(r['interna_ric'])):+.5f}")
    print(f"  ricentrata (mediana)           {float(np.median(r['ricentrata'])):+.5f}")
    print(f"  ECCESSO (appaiato per cella)   {ecc:+.5f}")
    print(f"  segni                          {pos}/{len(r)} celle"
          f"   p = {pv:.4f}")
    print(f"  quota di forma sul divario     {quota_forma:.4f}"
          f"   (il resto e' media)")
    print()
    # Il verdetto sta sui SEGNI, non sulla quota. La quota e' sempre
    # piccola perche' `grezza` e' dominata dallo scostamento di media
    # anche quando la forma varia molto: su una griglia sintetica con
    # dispersione sestuplicata fra celle la quota resta 0.002, mentre i
    # segni danno 11 su 12. Un generatore puo' sfruttare un segnale di
    # forma anche debole, se la media e' gestita a parte; quello che non
    # puo' sfruttare e' un segnale ASSENTE.
    if not (ecc > 0 and pv < 0.05):
        print("  => DOMINATA DALLA MEDIA. Tolta la media, le celle")
        print("     adiacenti sono indistinguibili. Un generatore non ha")
        print("     nulla da catturare che un regressore della media non")
        print("     catturi gia': non vale la pena correrci RQ2.")
    elif pos < 0.8 * len(r):
        print("  => STRUTTURA DEBOLE ma presente. L'eccesso e' positivo in")
        print("     modo significativo, con alcune celle discordi: c'e'")
        print("     forma da imparare, ma non in tutte le regioni della")
        print("     griglia. Vale la pena correrci RQ2 guardando le celle")
        print("     una per una, non solo l'aggregato.")
    else:
        print("  => STRUTTURA PRESENTE. La forma varia fra celle adiacenti")
        print("     ben oltre il rumore: qui RQ2 e' una domanda con una")
        print("     risposta, e il confronto con l'interpolazione e' equo.")
    print()
    return quota_forma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dati", required=True)
    ap.add_argument("--per-profilo", action="store_true",
                    help="ripete l'analisi dentro ciascun profilo temporale")
    ap.add_argument("--n", type=int, default=512,
                    help="finestre per campione, come n_gen della corsa vera")
    ap.add_argument("--seme", type=int, default=0)
    a = ap.parse_args()

    mmd, banda = carica_metriche()
    d = pd.read_csv(a.dati)
    manca = [c for c in OSSERVABILI + CONDIZIONI if c not in d.columns]
    if manca:
        raise SystemExit(f"colonne assenti in {a.dati}: {manca}")

    rng = np.random.default_rng(a.seme)
    print(f"griglia {a.dati}")
    print()
    if a.per_profilo and "profilo" in d.columns:
        for p in sorted(d["profilo"].unique()):
            analizza(d[d["profilo"] == p].reset_index(drop=True),
                     f"profilo {p}", mmd, banda, a.n, rng)
    else:
        if "profilo" in d.columns and d["profilo"].nunique() > 1:
            print("ATTENZIONE: la griglia ha piu' profili temporali e li")
            print("sta mescolando dentro ogni cella. Il risultato gonfia la")
            print("forma senza che significhi struttura FRA celle. Usare")
            print("--per-profilo.")
            print()
        analizza(d, "tutta la griglia", mmd, banda, a.n, rng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())