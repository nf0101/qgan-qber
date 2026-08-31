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
import itertools
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
NUM_BITS = 100_000          # riferimento a 10 km
WINDOW = 200

P_DETECTOR_ERR = 0.01
P_LOSS_LENGTH = 0.2

OUT = Path("../data/qber_grid.csv")

SMOKE = dict(rates=INTERCEPT_RATES[:2], lengths=LENGTHS_KM[:2],
             depols=DEPOLAR_TOTALS[:1], seeds=SEEDS[:1], scale=0.2)


def bits_for(length, scale=1.0):
    """Bit da trasmettere per ottenere un numero di finestre confrontabile
    a ogni distanza, compensando l'attenuazione in fibra."""
    trasmittanza = 10 ** (-P_LOSS_LENGTH * length / 10)
    rif = 10 ** (-P_LOSS_LENGTH * 10.0 / 10)      # trasmittanza a 10 km
    return int(NUM_BITS * scale * rif / trasmittanza)


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
    return [(r, L, d, sd, bits_for(L, scale), WINDOW)
            for r, L, d, sd in cells]


def configure(length, depolar, seed):
    m.L_TOTAL = length
    m.L1 = length * m.EVE_POSITION
    m.L2 = length * (1.0 - m.EVE_POSITION)
    m.DEPOLAR_TOTAL = depolar
    m.DEPOLAR_SEG = depolar / 2.0
    m.P_DETECTOR_ERR = P_DETECTOR_ERR
    m.P_LOSS_LENGTH = P_LOSS_LENGTH
    m.SEED = seed


def run_cell(rate, length, depolar, seed, num_bits, window):
    configure(length, depolar, seed)
    t0 = time.time()
    stats, rec = run_bb84(intercept_rate=rate, num_bits=num_bits,
                          seed=seed, verbose=False)
    ws = windows(rec, size=window)
    elapsed = time.time() - t0

    rows = [{
        # ---- condizionamento c ----
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
        "basis_asymmetry": abs(w["qber_rect"] - w["qber_diag"]),
        "sift_rate": len(rec) / num_bits,
        "loss_rate": stats["loss_rate"],
        "n_window": w["n"],
        "t_start": w["t_start"],
        # ---- target causale (NON osservabile) ----
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
        rows, el, st = run_cell(rate, L, dep, 1, nbits, WINDOW)
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

def generate(smoke=False, workers=None, parallel=True):
    tasks = grid(smoke=smoke)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    state = {"writer": None, "written": 0, "vuote": 0}

    print(f"celle: {len(tasks)}   modo: "
          f"{'parallelo' if parallel else 'seriale'}"
          f"{'  (smoke)' if smoke else ''}\n")

    def handle(fh, n, rows, el, st, args):
        rate, L, dep, seed = args[0], args[1], args[2], args[3]
        if not rows:
            state["vuote"] += 1
            print(f"[{n}/{len(tasks)}] VUOTA r={rate} L={L} dep={dep} "
                  f"seed={seed} -- perdita {st['loss_rate']:.1%}")
            return
        if state["writer"] is None:
            state["writer"] = csv.DictWriter(fh, fieldnames=list(rows[0]))
            state["writer"].writeheader()
        state["writer"].writerows(rows)
        state["written"] += len(rows)
        eta = (time.time() - t0) / n * (len(tasks) - n) / 60
        print(f"[{n}/{len(tasks)}] r={rate:.2f} L={L:5.1f} dep={dep:.2f} "
              f"seed={seed} -> {len(rows):3d} finestre "
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

    print(f"\nscritte {state['written']} finestre in {OUT}")
    if state["vuote"]:
        print(f"ATTENZIONE: {state['vuote']} celle vuote (alza NUM_BITS)")
    print(f"tempo totale: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    if a.benchmark:
        benchmark()
    else:
        generate(smoke=a.smoke, workers=a.workers, parallel=not a.serial)