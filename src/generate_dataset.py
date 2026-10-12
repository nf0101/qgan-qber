"""
Generazione della griglia di dataset QBER.

Produce un CSV con una riga per finestra temporale. Ogni riga contiene:
  - le feature OSSERVABILI (quelle che un operatore QKD misurerebbe davvero)
  - il vettore delle FRAZIONI CAUSALI (il target di O3, non osservabile)
  - il vettore di CONDIZIONAMENTO c = (attacco, condizione di canale)

Uso:
    python src/generate_dataset.py --benchmark     # stima i tempi
    python src/generate_dataset.py --smoke         # griglia ridotta
    python src/generate_dataset.py                 # griglia completa, parallela
    python src/generate_dataset.py --serial        # esecuzione seriale
    python src/generate_dataset.py --workers 4     # limita i processi
"""

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import BB84_crn as m
from BB84_crn import run_bb84, windows

# =====================================================================
# GRIGLIA
# =====================================================================
# La griglia e' generata PER INTERO: le celle held-out si scelgono in fase
# di addestramento, perche' per valutare RQ2 serve la verita' anche li'.

INTERCEPT_RATES = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
LENGTHS_KM = [5.0, 10.0, 20.0, 40.0]
DEPOLAR_TOTALS = [0.01, 0.02, 0.03, 0.045, 0.06]

SEEDS = [1, 2, 3, 4, 5]

# --- griglia NON STAZIONARIA (profili temporali) ---------------------
# Separata dalla principale, che resta congelata con la sua impronta.
#   deriva          AR(1) lento sulla depolarizzazione: autocorrelazione VERA
#                   fra finestre, cioe' qualcosa che w_{t-1} puo' predire.
#   intermittente   Eve che si accende e spegne con periodo piu' corto della
#                   traccia: ogni traccia contiene una transizione.
#   entrambi        i due insieme.
# La deriva sulla DEPOLARIZZAZIONE e' satura: quel parametro contribuisce al
# QBER con dep/2 <= 0,0225, mentre il rumore binomiale di stima su una
# finestra da 200 posizioni vale ~0,025. Misurato sulla prima griglia non
# stazionaria: autocorrelazione del QBER +0,036, e un modello analitico la
# ferma a +0,11 anche portando sigma a 1,0. Non e' un parametro sbagliato,
# e' il segnale che non puo' superare il proprio valore medio.
#
# La deriva sul TASSO DI EVE non ha quel limite: Eve contribuisce r/4, che a
# r=0,5 vale 0,125. Attesa +0,47 a r=0,25 e +0,66 a r=0,50 con sigma 0,5.
PROFILI = ["deriva_eve", "intermittente", "entrambi"]
SIGMA_DERIVA_EVE = 0.5
# Tassi nominali bassi: con m fino a 2,0 il prodotto r*m resta sotto 1 e il
# segnale non viene tagliato dal limite superiore.
NS_RATES = [0.25, 0.50]
NS_LENGTHS = [10.0, 20.0]
# QUINDICI semi, non cinque. profilo_deriva(seed) dipende SOLO dal seme,
# quindi le 8 configurazioni di uno stesso seme condividono la stessa
# realizzazione del moltiplicatore: il numero di semi E' il numero di forme
# di deriva distinte. Con 5 semi l'addestramento ne vedrebbe 3, e un esito
# negativo di §6 non si distinguerebbe da una diversita' insufficiente.
# Far dipendere la deriva anche dalla cella romperebbe l'accoppiamento CRN
# fra condizioni allo stesso seme, cioe' la proprieta' verificata con 0/299.
NS_SEEDS = list(range(1, 16))
# 3 profili x 2 tassi x 2 L x 2 dep x 15 semi = 360 celle
NS_DEPOLS = [0.02, 0.045]
OUT_NS = Path(__file__).resolve().parent.parent / "data" / "qber_grid_ns.csv"


def costruisci_profili(nome, seed):
    """(profilo_eve, profilo_depol) dal nome. 'costante' = stazionario."""
    cost = m.profilo_costante()
    if nome == "costante":
        return cost, cost
    if nome == "deriva":                       # storico: deriva su depolar
        return cost, m.profilo_deriva(seed)
    if nome == "deriva_eve":                   # memoria temporale VERA
        return m.profilo_deriva(seed, tag="deriva_eve",
                                sigma=SIGMA_DERIVA_EVE), cost
    if nome == "intermittente":                # cambio di regime
        return m.profilo_intermittente(fase=m.fase_da_seme(seed)), cost
    if nome == "entrambi":                     # regime E dinamica, su Eve
        return m.profilo_prodotto(
            m.profilo_intermittente(fase=m.fase_da_seme(seed)),
            m.profilo_deriva(seed, tag="deriva_eve",
                             sigma=SIGMA_DERIVA_EVE)), cost
    raise ValueError(f"profilo non riconosciuto: {nome!r}")



NUM_BITS = 100_000          # riferimento a 10 km
WINDOW = 200

P_DETECTOR_ERR = 0.01
P_LOSS_LENGTH = 0.2

# Ancorato alla posizione dello script, non alla directory corrente.
# Il valore precedente, "../data/qber_grid.csv", si risolveva in modo diverso
# a seconda di dove veniva lanciato: e' cosi' che sono finite in giro tre
# griglie con nomi che non le distinguono, piu' un qber_grid_small.csv da
# ZERO byte accanto a un omonimo da 3,4 MB in un'altra directory.
OUT = Path(__file__).resolve().parent.parent / "data" / "qber_grid.csv"

SMOKE = dict(rates=INTERCEPT_RATES[:2], lengths=LENGTHS_KM[:2],
             depols=DEPOLAR_TOTALS[:1], seeds=SEEDS[:1], scale=0.2)


def bits_for(length, scale=1.0):
    """Bit da trasmettere per ottenere un numero di finestre confrontabile
    a ogni distanza, compensando l'attenuazione in fibra."""
    trasmittanza = 10 ** (-P_LOSS_LENGTH * length / 10)
    rif = 10 ** (-P_LOSS_LENGTH * 10.0 / 10)      # trasmittanza a 10 km
    return int(NUM_BITS * scale * rif / trasmittanza)


def versione_codice():
    """Revisione git, con marcatore se il working tree e' sporco."""
    qui = Path(__file__).resolve().parent
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=qui, stderr=subprocess.DEVNULL,
                                      timeout=5).decode().strip()
        sporco = subprocess.check_output(["git", "status", "--porcelain"],
                                         cwd=qui, stderr=subprocess.DEVNULL,
                                         timeout=5).decode().strip()
        return rev + ("+modificato" if sporco else "")
    except Exception:
        return None


def metadati(smoke, workers, parallel, out):
    """Configurazione completa della generazione, scritta accanto al CSV.

    La griglia precedente non portava con se' ne' la configurazione ne' la
    versione del codice: stabilire quale delle tre griglie trovate in giro
    fosse la sorgente dei raggi di B.2 ha richiesto di rieseguire lo script
    di calibrazione su ciascuna e confrontare i numeri.
    """
    return {
        "intercept_rates": INTERCEPT_RATES, "lengths_km": LENGTHS_KM,
        "depolar_totals": DEPOLAR_TOTALS, "seeds": SEEDS,
        "ns_seeds": NS_SEEDS, "profili": PROFILI,
        "sigma_deriva_eve": SIGMA_DERIVA_EVE,
        "num_bits_riferimento": NUM_BITS, "window": WINDOW,
        "p_detector_err": P_DETECTOR_ERR, "p_loss_length": P_LOSS_LENGTH,
        "eve_position": m.EVE_POSITION,
        "accoppia_misure": m.ACCOPPIA_MISURE,
        "smoke": smoke, "parallelo": parallel, "workers": workers,
        "versione_codice": versione_codice(),
        "istante": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "out": str(out),
    }


def grid(smoke=False):
    if smoke:
        s = SMOKE
        cells = itertools.product(s["rates"], s["lengths"],
                                  s["depols"], s["seeds"])
        scale = s["scale"]
    else:
        cells = itertools.product(INTERCEPT_RATES, LENGTHS_KM,
                                  DEPOLAR_TOTALS, SEEDS)
        scale = 1.0
    return [("costante", r, L, d, sd, bits_for(L, scale), WINDOW)
            for r, L, d, sd in cells]


def semi_da_spec(spec):
    """'1-15' oppure '6,7,8' -> lista di interi."""
    out = []
    for pezzo in str(spec).split(","):
        pezzo = pezzo.strip()
        if "-" in pezzo:
            a, b = pezzo.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif pezzo:
            out.append(int(pezzo))
    return out


def grid_ns(semi=None):
    """Griglia non stazionaria: 3 profili x 2 tassi x 2 L x 2 dep x N semi."""
    return [(pr, r, L, d, sd, bits_for(L), WINDOW)
            for pr, r, L, d, sd in itertools.product(
                PROFILI, NS_RATES, NS_LENGTHS, NS_DEPOLS,
                semi if semi else NS_SEEDS)]


def configure(length, depolar, seed):
    m.L_TOTAL = length
    m.L1 = length * m.EVE_POSITION
    m.L2 = length * (1.0 - m.EVE_POSITION)
    m.DEPOLAR_TOTAL = depolar
    m.DEPOLAR_SEG = depolar / 2.0
    m.P_DETECTOR_ERR = P_DETECTOR_ERR
    m.P_LOSS_LENGTH = P_LOSS_LENGTH
    m.SEED = seed


def run_cell(profilo, rate, length, depolar, seed, num_bits, window):
    configure(length, depolar, seed)
    p_eve, p_dep = costruisci_profili(profilo, seed)
    t0 = time.time()
    stats, rec, ricevute = run_bb84(intercept_rate=rate, num_bits=num_bits,
                                    seed=seed, verbose=False,
                                    con_ricevute=True,
                                    profilo_eve=p_eve, profilo_depol=p_dep)
    ws = windows(rec, size=window, ricevute=ricevute)
    elapsed = time.time() - t0

    rows = [{
        # ---- condizionamento c ----
        "profilo": profilo,
        "intercept_rate": rate,
        "length_km": length,
        "depolar_total": depolar,
        "seed": seed,
        "window_idx": k,
        "num_bits": num_bits,
        # ---- osservabili ----
        "qber": w["qber"],
        "qber_rect": w["qber_rect"],
        "qber_diag": w["qber_diag"],
        "basis_asymmetry": w["basis_asymmetry"],
        "qber_stability": w["qber_stability"],
        "n_subblocks": w["n_subblocks"],
        # sift_rate e loss_rate vengono dalla FINESTRA. Prima erano
        # `len(rec)/num_bits` e `stats["loss_rate"]`, cioe' due scalari PER
        # CORSA replicati identici su tutte le finestre della cella: nella
        # griglia precedente hanno esattamente un valore distinto per cella,
        # su tutte e 600. Due qubit su cinque modellavano costanti gia'
        # determinate da c, la disparita' di dispersione fra osservabili
        # risultava 47x invece di ~2,6x, e in modalita' autoregressiva due
        # delle cinque componenti di w_{t-1} erano costanti lungo la traccia.
        "sift_rate": w["sift_rate"],
        "loss_rate": w["loss_rate"],
        "n_window": w["n"],
        "n_grezze": w["n_grezze"],
        "t_start": w["t_start"],
        # ---- target causale (NON osservabile) ----
        # ---- condizioni effettive nella finestra (target, non osservabili) ----
        "rate_eff": w["rate_eff"],
        "depol_eff": w["depol_eff"],
        "f_canale": w["f_canale"],
        "f_eve": w["f_eve"],
        "f_rivelatore": w["f_rivelatore"],
        "qber_cf_canale": w["qber_cf_canale"],
        "qber_cf_eve": w["qber_cf_eve"],
        "qber_cf_rivelatore": w["qber_cf_rivelatore"],
    } for k, w in enumerate(ws)]

    return rows, elapsed, stats


def _cell(args):
    """Wrapper per ProcessPoolExecutor: deve stare a livello di modulo."""
    rows, el, st = run_cell(*args)
    return rows, el, st, args


# =====================================================================
# BENCHMARK
# =====================================================================

def benchmark():
    """Misura il throughput e ne deriva il tempo della griglia completa.

    Si misura bit/secondo anziche' secondi/cella: con bits_for() le celle
    hanno carichi molto diversi e una media semplice sarebbe fuorviante.
    """
    print("Misura del throughput su tre celle agli estremi.\n")
    cases = [(0.0, 5.0, 0.01), (1.0, 40.0, 0.06), (0.5, 10.0, 0.03)]
    tot_bits = tot_time = 0.0

    for rate, L, dep in cases:
        nbits = bits_for(L)
        rows, el, st = run_cell("costante", rate, L, dep, 1, nbits, WINDOW)
        tot_bits += nbits
        tot_time += el
        print(f"r={rate:.2f} L={L:5.1f} dep={dep:.2f} | {nbits:>7d} bit "
              f"-> {el:6.1f}s | finestre={len(rows):3d} | "
              f"QBER={st['qber']:6.2%} perdita={st['loss_rate']:5.1%}")

    thr = tot_bits / tot_time
    cells = grid()
    grid_bits = sum(c[4] for c in cells)
    seriale = grid_bits / thr

    print(f"\nthroughput     : {thr:,.0f} bit/s")
    print(f"celle          : {len(cells)}")
    print(f"bit totali     : {grid_bits:,}")
    print(f"stima seriale  : {seriale/60:.0f} min ({seriale/3600:.1f} h)")
    for w in (4, 8, 12, 16):
        print(f"  con {w:2d} worker: ~{seriale/60/w:.0f} min")


# =====================================================================
# GENERAZIONE
# =====================================================================

def generate(smoke=False, workers=None, parallel=True, ns=False, semi=None):
    global OUT
    if ns:
        OUT = OUT_NS
    tasks = grid_ns(semi) if ns else grid(smoke=smoke)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    state = {"writer": None, "written": 0, "vuote": 0, "incoerenti": 0}

    meta = metadati(smoke, workers, parallel, OUT)
    print(f"codice {meta['versione_codice']}   accoppiamento misure "
          f"{'attivo' if m.ACCOPPIA_MISURE else 'DISATTIVATO'}")
    if not m.ACCOPPIA_MISURE:
        print("ATTENZIONE: senza accoppiamento le condizioni non restano "
              "appaiate posizione per posizione e i controfattuali non sono "
              "fondati. Interrompere.")

    print(f"celle: {len(tasks)}   modo: "
          f"{'parallelo' if parallel else 'seriale'}"
          f"{'  (smoke)' if smoke else ''}\n")

    def handle(fh, n, rows, el, st, args):
        prof, rate, L, dep, seed = args[0], args[1], args[2], args[3], args[4]
        # Coerenza fra catena modellata e catena eseguita, su OGNI cella.
        # Costa zero: e' gia' calcolata in run_bb84. Meglio accorgersene
        # alla cella 12 che a sei ore di simulazione finite.
        if st.get("discordanze"):
            state["incoerenti"] += 1
            print(f"[{n}/{len(tasks)}] INCOERENTE {prof} r={rate} L={L} "
                  f"dep={dep} seed={seed} -- {st['discordanze']}/"
                  f"{st['n_non_intercettati']} discordanze analitico/osservato")
        if not rows:
            state["vuote"] += 1
            print(f"[{n}/{len(tasks)}] VUOTA {prof} r={rate} L={L} "
                  f"dep={dep} seed={seed} -- perdita {st['loss_rate']:.1%}")
            return
        if state["writer"] is None:
            state["writer"] = csv.DictWriter(fh, fieldnames=list(rows[0]))
            state["writer"].writeheader()
        state["writer"].writerows(rows)
        state["written"] += len(rows)
        eta = (time.time() - t0) / n * (len(tasks) - n) / 60
        print(f"[{n}/{len(tasks)}] {prof:14s} r={rate:.2f} L={L:5.1f} "
              f"dep={dep:.2f} seed={seed} -> {len(rows):3d} finestre "
              f"({el:5.1f}s, ETA {eta:4.0f} min)")

    with OUT.open("w", newline="") as fh:
        if parallel:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for n, (rows, el, st, args) in enumerate(
                        ex.map(_cell, tasks), 1):
                    handle(fh, n, rows, el, st, args)
        else:
            for n, args in enumerate(tasks, 1):
                rows, el, st = run_cell(*args)
                handle(fh, n, rows, el, st, args)

    meta["finestre_scritte"] = state["written"]
    meta["celle_vuote"] = state["vuote"]
    meta["celle_incoerenti"] = state["incoerenti"]
    meta["durata_min"] = round((time.time() - t0) / 60, 2)
    meta["sha256"] = hashlib.sha256(OUT.read_bytes()).hexdigest()
    perc_meta = OUT.with_suffix(".meta.json")
    perc_meta.write_text(json.dumps(meta, indent=2, default=str))

    print(f"\nscritte {state['written']} finestre in {OUT}")
    print(f"impronta sha256: {meta['sha256']}")
    print(f"metadati in {perc_meta}")
    if state["vuote"]:
        print(f"ATTENZIONE: {state['vuote']} celle vuote (alza NUM_BITS)")
    if state["incoerenti"]:
        print(f"ATTENZIONE: {state['incoerenti']} celle con discordanze fra "
              "errore analitico e osservato. La catena modellata non e' "
              "quella eseguita: NON usare questa griglia.")
    print(f"tempo totale: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--semi", default=None,
                    help="semi da generare, es. '1-15' o '6,7,8'; solo con --ns")
    ap.add_argument("--ns", action="store_true",
                    help="genera la griglia NON STAZIONARIA (profili "
                         "temporali) in data/qber_grid_ns.csv")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="percorso del CSV; default data/qber_grid.csv "
                         "accanto alla radice del progetto")
    a = ap.parse_args()
    if a.out is not None:
        OUT = a.out

    if a.benchmark:
        benchmark()
    else:
        generate(smoke=a.smoke, workers=a.workers, parallel=not a.serial,
                 ns=a.ns, semi=semi_da_spec(a.semi) if a.semi else None)