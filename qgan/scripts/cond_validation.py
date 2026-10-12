"""Efficacia del condizionamento — prerequisito di RQ2.

SPEC-FASE2 §5 richiede che la configurazione non condizionata sia una baseline
obbligatoria, perche' isola il contributo del condizionamento. Questo script
la costruisce e misura tre cose distinte, che e' facile confondere:

  1. RISPOSTA. I campioni generati si muovono nella direzione che `c` impone?
     E' il requisito minimo: senza questo il condizionamento e' decorativo.

  2. GENERALIZZAZIONE A CELLE NON VISTE. Addestrando su una griglia parziale,
     il generatore ricostruisce le celle escluse? E' l'analogo diretto del
     protocollo `interior` di Fase 1-B, ed e' RQ2 su dati sintetici.

  3. GUADAGNO SUL NON CONDIZIONATO. La MMD sulle celle escluse e' migliore di
     quella di un generatore addestrato sugli stessi dati ma senza `c`?
     Se non lo e', il condizionamento non aggiunge nulla e RQ2 non ha spazio.

La terza e' quella che conta. Un generatore condizionato puo' superare le
prime due e restare inutile: se la distribuzione complessiva e' gia' vicina a
quella di ogni singola cella, l'incondizionato la eguaglia senza sforzo.

Il bersaglio e' sintetico e imita la griglia QKD nella struttura — tre assi,
celle interne definite come non estreme su alcun asse — non nel contenuto.
Vale la stessa motivazione di §9.1: la calibrazione e la validazione
metodologica non toccano i dati reali.

    python scripts/conditional_validation.py --seeds 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import corsa  # noqa: E402
from qgan import QGAN, metrics  # noqa: E402
from qgan.metrics import median_bandwidth  # noqa: E402

# Configurazione congelata in Appendice A
FROZEN = dict(
    lr_g=0.002,
    lr_d=0.002,
    hidden=(32, 16),
    batch_size=16,
    max_steps=1350,
    eval_every=25,
    patience=20,
    mmd_smooth=3,
    val_fraction=0.0,
    n_monitor=256,
    monitor_repeats=1,
)

AXES = [np.linspace(0.0, 1.0, 4)] * 3  # 4 x 4 x 4 = 64 celle
PER_CELL = 32
# Finestre per cella usate in VALUTAZIONE. Era un LIMITE SUPERIORE applicato a
# un sottoinsieme della griglia di addestramento: con --per-cella 8 il test
# conteneva 8 finestre, non 32, e cambiava a ogni numerosita'. Ora e' il numero
# esatto di finestre di un test generato a parte, identico in ogni condizione.
PER_CELL_EVAL = 32

# Semi di base dei flussi. SEED_TEST e' separato e non dipende dal seme della
# corsa: il test di una cella e' una proprieta' della cella, non della corsa
# che la valuta, ed e' cio' che rende confrontabili i punti della curva n.
SEED_BASE = 3000
SEED_TEST = 90210


# Rumore condizionale per osservabile.
#
# `omogeneo` e' il bersaglio usato in FASE2 §12-§15: tutte le osservabili con
# la stessa dispersione. Non mette alla prova i raggi per feature, perche' un
# raggio unico gia' le soddisfa tutte.
#
# `qkd` riproduce le dispersioni misurate sulla griglia reale, entro
# configurazione e in unita' normalizzate min-max. E' il bersaglio su cui i
# raggi per feature vanno validati prima di applicarli ai dati veri.
#
# VALORI AGGIORNATI al 16/09/2026, griglia rigenerata con codice 8d71755.
# I valori precedenti, [0.05754, 0.07024, 0.07138, 0.00318, 0.00152],
# venivano da una griglia in cui sift_rate e loss_rate erano scalari PER CORSA
# replicati su tutte le finestre: le loro dispersioni erano varianza fra semi,
# non variabilita' entro condizione, e davano un fattore 47 fra osservabili
# che non era fisico. Sulla griglia corretta il fattore e' 3.
# Base nell'ordine di SPEC §1.2, prima di qualunque permutazione sui qubit.
NOMI_BASE = ["qber", "qber_rect", "qber_diag", "sift_rate", "loss_rate"]
SIGMA_BASE = np.array([0.05336, 0.07109, 0.07136, 0.03584, 0.02084])

# Assegnazione delle osservabili ai qubit. La questione e' aperta: B.6 la
# derivava da una misura all'inizializzazione, mentre il test end-to-end sul
# dominio IDS (scripts/ordine_qubit_ids.py) da' il risultato opposto — mettere
# la dispersione maggiore sul qubit terminale peggiora, perche' li' il
# generatore fatica a PRODURRE dispersione, non solo a comprimerla.
#
#   naturale     ordine di §1.2, nessuna permutazione
#   b6           dispersione maggiore sul qubit terminale (Appendice B.6)
#   decrescente  dispersione maggiore sul qubit 0, come l'ordine "naturale"
#                del test IDS che risultava migliore
ORDINI = {
    "naturale": [0, 1, 2, 3, 4],
    "b6": [3, 1, 0, 4, 2],
    "decrescente": [2, 1, 0, 3, 4],
}
# Ordine naturale: decisione corrente di B.6 e FASE2 §19. La variante "b6"
# resta disponibile come opzione esplicita, non come default.
ORDINE = "naturale"

# Simulatore del circuito. 'denso' e' equivalente ad Aer entro 1e-15 su
# forward e jacobiana (tests/test_correctness.py) ed e' cio' che rende
# sostenibili i 15-20 semi decisi per l'inferenza sulla corsa come unita'.
BACKEND = "denso"

SIGMA = {"omogeneo": np.full(5, 0.02), "qkd": SIGMA_BASE}
NOMI = {"omogeneo": ["f0", "f1", "f2", "f3", "f4"], "qkd": NOMI_BASE}
TARGET = "omogeneo"


def _perm():
    return np.asarray(ORDINI[ORDINE])


def cell_features(c, rng, n):
    """Osservabili in funzione di c.

    La struttura delle medie e' la stessa nei due bersagli; cambia solo il
    rumore condizionale, che e' cio' che i raggi per feature devono eguagliare.
    """
    q = 0.02 + 0.30 * c[0] + 0.05 * c[1] + 0.35 * c[2] ** 1.5
    asym = 0.04 * (c[0] - 0.5)
    sift = 0.50 - 0.08 * c[1] ** 2
    loss = 0.05 + 0.60 * c[1] ** 1.3
    mu = np.array([q, q + asym, q - asym, sift, loss])   # ordine di §1.2
    sig = SIGMA[TARGET]
    X = mu + rng.normal(0, 1, (n, 5)) * sig[None, :]
    if TARGET == "qkd":
        p = _perm()
        X, = (X[:, p],)
    return np.clip(X, 0, 1)


def build_grid(rng):
    cells = list(itertools.product(*AXES))
    X, C, cell_id = [], [], []
    for i, c in enumerate(cells):
        c = np.array(c)
        X.append(cell_features(c, rng, PER_CELL))
        C.append(np.repeat(c[None, :], PER_CELL, axis=0))
        cell_id.append(np.full(PER_CELL, i))
    return (np.vstack(X), np.vstack(C), np.concatenate(cell_id),
            [np.array(c) for c in cells])


def interior_mask(cells):
    """Celle non estreme su alcun asse, dedotte dai dati e non cablate."""
    out = []
    for c in cells:
        inner = all(ax.min() < v < ax.max() for v, ax in zip(c, AXES))
        out.append(inner)
    return np.array(out)


def run_seed(seed, steps, verbose=0, theta_init=0.1, latent_hi=np.pi,
             latent_hi_uncond=None, hidden=(32, 16), label_smoothing=0.0,
             weight_decay=0.0):
    # Tre flussi per tre ruoli. Con un flusso unico lo stato consumato da
    # build_grid dipendeva da PER_CELL, quindi cambiare la numerosita' del
    # training spostava anche le celle escluse: per il seme 0 si passava da
    # {22, 26, 42, 38} a {38, 26, 25, 41} solo aumentando --per-cella. La
    # curva rispetto a n non isolava la numerosita'.
    rng, rng_split, rng_diag = corsa.flussi(SEED_BASE, seed, 3)
    X, C, cid, cells = build_grid(rng)
    inner = interior_mask(cells)
    inner_idx = np.flatnonzero(inner)

    # meta' delle celle interne escluse dall'addestramento; dipende SOLO dal
    # seme, non da quanto ha consumato la generazione della griglia
    held = rng_split.choice(inner_idx, len(inner_idx) // 2, replace=False)
    train = ~np.isin(cid, held)

    kw = dict(FROZEN, max_steps=steps, seed=seed, verbose=verbose,
              backend=BACKEND,
              theta_init=theta_init, latent_hi=latent_hi,
              hidden=tuple(hidden), label_smoothing=label_smoothing,
              d_weight_decay=weight_decay)

    # Validazione: campione casuale su TUTTE le celle di addestramento. Le
    # prime righe della griglia appartengono a un solo angolo (c[0] = 0),
    # quindi uno slice iniziale monitorerebbe una fetta non rappresentativa.
    vi = rng_split.choice(np.flatnonzero(train), 256, replace=False)

    cond = QGAN(d_c=3, **kw)
    cond.fit(X[train], C[train], X_val=X[vi], c_val=C[vi])

    # Il ramo incondizionato riceve il PROPRIO raggio del latente. Passargli
    # quello del condizionato lo penalizza per costruzione: senza `c` il
    # latente e' la sua unica fonte di diversita', e a raggio stretto non
    # copre piu' la griglia. Il guadagno sulla MMD ne risulterebbe gonfiato,
    # che e' la stessa obiezione sollevata in Fase 1-B §5 sul confronto con
    # il modello piu' debole. Lo skill non ne e' affetto, perche' e' misurato
    # contro la media del training e non contro questo ramo.
    kw_u = dict(kw)
    kw_u["latent_hi"] = np.pi if latent_hi_uncond is None else latent_hi_uncond
    uncond = QGAN(d_c=0, **kw_u)
    uncond.fit(X[train], X_val=X[vi])

    # --- 1. risposta al condizionamento --------------------------------
    lo = cond.sample(400, np.tile([0.1, 0.5, 0.1], (400, 1)))
    hi = cond.sample(400, np.tile([0.9, 0.5, 0.9], (400, 1)))
    d_gen = hi.mean(0) - lo.mean(0)
    d_true = (cell_features(np.array([0.9, 0.5, 0.9]), rng_diag, 2000).mean(0)
              - cell_features(np.array([0.1, 0.5, 0.1]), rng_diag, 2000).mean(0))
    risposta = float(np.corrcoef(d_gen, d_true)[0, 1])

    # --- 2 e 3. celle escluse, confronto appaiato ----------------------
    # Banda UNICA, ricavata dall'intera griglia. Ricavarla dalla singola cella
    # darebbe circa 0,043 contro 0,330: con un kernel cosi' stretto qualunque
    # campione generato a piu' di pochi centesimi dalla cella da' similarita'
    # nulla, la MMD satura e non distingue piu' fra i due rami.
    sig = float(median_bandwidth(X, rng=np.random.default_rng(0)))

    # Riferimento banale per lo skill score: la media della griglia di
    # addestramento, cioe' la migliore predizione possibile ignorando `c`.
    mu_banale = X[train].mean(axis=0)

    per_cell = []
    for h in held:
        ch = cells[h]
        # Test generato a parte, da un flusso proprio della cella: sempre
        # PER_CELL_EVAL finestre, identiche per ogni valore di --per-cella e
        # per ogni seme che escluda questa cella. E' la condizione perche' i
        # punti della curva rispetto a n siano confrontabili fra loro.
        Xh = cell_features(ch, corsa.flusso_fisso(SEED_TEST, h), PER_CELL_EVAL)
        gc = cond.sample(len(Xh), np.tile(ch, (len(Xh), 1)))
        gu = uncond.sample(len(Xh))
        # Errore di posizione. Non soffre dell'asimmetria della MMD contro una
        # cella stretta, che penalizza una risposta sbagliata piu' di una
        # non-risposta e concede quindi una rendita al generatore ampio.
        mu_vero = Xh.mean(axis=0)
        mae_cond = float(np.abs(gc.mean(axis=0) - mu_vero).mean())
        mae_unc = float(np.abs(gu.mean(axis=0) - mu_vero).mean())
        mae_ban = float(np.abs(mu_banale - mu_vero).mean())

        per_cell.append({
            "cella": int(h),
            "c": ch.tolist(),
            "mmd_condizionato": metrics.mmd_rbf(Xh, gc, sigma=sig),
            "mmd_incondizionato": metrics.mmd_rbf(Xh, gu, sigma=sig),
            "detect_condizionato": metrics.detectability(Xh, gc, seed=seed),
            "mae_condizionato": mae_cond,
            "mae_incondizionato": mae_unc,
            "mae_banale": mae_ban,
            "skill_condizionato": 1.0 - mae_cond / mae_ban if mae_ban > 0 else float("nan"),
            "skill_incondizionato": 1.0 - mae_unc / mae_ban if mae_ban > 0 else float("nan"),
            "std_vera": Xh.std(axis=0).mean(),
            "std_condizionata": float(gc.std(axis=0).mean()),
            "std_per_feature_cella": gc.std(axis=0).tolist(),
        })

    # Diagnostico: il modello ha imparato qualcosa? Senza questo controllo un
    # guadagno nullo e' ambiguo fra "il condizionamento non serve" e "nessuno
    # dei due generatori ha appreso".
    # Sottocampionato: `mmd_rbf` costruisce una matrice kernel n x n, quindi
    # con l'intera griglia di addestramento (30.720 righe) sarebbero 7,5 GB
    # per processo. Il controllo non ha bisogno di tutte le righe.
    Xt_full = X[train]
    Xt = (Xt_full[rng_diag.choice(len(Xt_full), 1024, replace=False)]
          if len(Xt_full) > 1024 else Xt_full)
    c_sub = C[train][rng_diag.choice(train.sum(), 512)]
    # Modelli all'inizializzazione, uno per ramo. Il riferimento va preso sul
    # MEDESIMO ramo: confrontare il condizionato addestrato con un modello
    # iniziale NON condizionato mescola due differenze e puo' far apparire
    # appreso un generatore fermo.
    cond0 = QGAN(d_c=3, **kw)
    uncond0 = QGAN(d_c=0, **kw_u)
    sanity = {
        "n_train": int(len(Xt_full)),
        "mmd_train_condizionato": metrics.mmd_rbf(
            Xt, cond.sample(512, c_sub), sigma=sig),
        "mmd_train_incondizionato": metrics.mmd_rbf(
            Xt, uncond.sample(512), sigma=sig),
        "mmd_train_iniziale_cond": metrics.mmd_rbf(
            Xt, cond0.sample(512, c_sub), sigma=sig),
        "mmd_train_iniziale_incond": metrics.mmd_rbf(
            Xt, uncond0.sample(512), sigma=sig),
    }
    # Chiave storica, conservata per non rompere la lettura dei file gia'
    # salvati; il confronto corretto usa le due chiavi per ramo.
    sanity["mmd_train_iniziale"] = sanity["mmd_train_iniziale_incond"]

    # Le tre grandezze in conflitto, da leggere insieme:
    #  - c/z  : quanto l'uscita risponde al condizionamento
    #  - Var(grad) : indicatore di barren plateau, motivo per cui §7 prescrive
    #    l'inizializzazione vicina all'identita'
    #  - dispersione : meta' del guadagno di c/z ad ampiezza maggiore viene
    #    dalla riduzione del contributo del latente, che e' cio' che fornisce
    #    diversita' ai campioni. Un rapporto alto ottenuto stringendo l'uscita
    #    non e' un buon risultato.
    hc = cond.history
    gen_std = float(cond.sample(512, c_sub).std(0).mean())
    # c/z misurato sui MODELLI, non letto dalla storia: history[0] e' la prima
    # VALUTAZIONE (al passo eval_every, non al passo zero) e history[-1] e'
    # l'ultima valutazione, non il checkpoint effettivamente ripristinato.
    lh_vec = cond._latent_hi_vec.tolist()
    cz_init = cond0.gen.conditioning_sensitivity(
        cond0.theta, n=200, seed=seed, latent_hi=lh_vec)["rapporto"]
    cz_ckpt = cond.gen.conditioning_sensitivity(
        cond.theta, n=200, seed=seed, latent_hi=lh_vec)["rapporto"]
    # Risposta a ciascuna componente di c presa da sola: la media non
    # distingue un condizionamento che usa tutte le componenti da uno che ne
    # ignora una parte.
    sens_ingresso = cond.gen.input_sensitivity(
        cond.theta, n=200, seed=seed, latent_hi=lh_vec).tolist()
    conflitto = {
        "theta_init": theta_init,
        "n_train": int(train.sum()),
        "per_cella": PER_CELL,
        "per_cella_eval": PER_CELL_EVAL,
        "target": TARGET,
        "ordine": ORDINE,
        "latent_hi": np.atleast_1d(latent_hi).tolist(),
        "latent_hi_uncond": float(kw_u["latent_hi"]),
        "hidden": list(hidden),
        "label_smoothing": label_smoothing,
        "weight_decay": weight_decay,
        "c_su_z_iniziale": float(cz_init),
        "c_su_z_finale": float(cz_ckpt),
        "c_su_z_ultima_valutazione": hc["sensibilita_c"][-1],
        "sensibilita_per_ingresso": sens_ingresso,
        "best_step": int(cond.best_step_),
        "steps_run": int(cond.steps_run_),
        "monitor": cond.monitor_,
        "strati_assi": getattr(cond, "strati_assi_", None),
        "strati_utili": getattr(cond, "strati_utili_", None),
        "grad_var_iniziale": hc["grad_var"][0],
        "grad_var_minima": float(np.min(hc["grad_var"])),
        "theta_ampiezza_finale": hc["theta_ampiezza"][-1],
        "dispersione_generata": gen_std,
        "dispersione_reale": float(X[train].std(0).mean()),
        "loss_d_minima": float(np.min(hc["loss_d"])),
    }

    mc = np.array([p["mmd_condizionato"] for p in per_cell])
    mu = np.array([p["mmd_incondizionato"] for p in per_cell])
    return {
        "seed": seed,
        "steps": steps,
        "conflitto": conflitto,
        "sigma": sig,
        "sanity": sanity,
        "celle_escluse": [int(h) for h in held],
        "risposta_correlazione": risposta,
        "delta_generato": d_gen.tolist(),
        "delta_atteso": d_true.tolist(),
        "mmd_condizionato": float(mc.mean()),
        "mmd_incondizionato": float(mu.mean()),
        "guadagno": float((mu - mc).mean()),
        "guadagno_per_cella": (mu - mc).tolist(),
        # Skill sul MAE AGGREGATO, come in Fase 1-B §2: 1 - MAE_modello /
        # MAE_banale con le due MAE mediate sulle celle. Mediare invece gli
        # skill calcolati cella per cella e' una media di rapporti dove serve
        # un rapporto di medie: con quattro celle, una in cui il riferimento
        # banale e' casualmente ottimo produce un rapporto enorme che domina.
        # Sul bersaglio a latente pi/4 la differenza fra le due formule e'
        # +0,020 contro +0,292, con deviazione 0,169 contro 0,063.
        "skill_condizionato": float(
            1.0 - np.mean([p["mae_condizionato"] for p in per_cell])
            / np.mean([p["mae_banale"] for p in per_cell])),
        "skill_incondizionato": float(
            1.0 - np.mean([p["mae_incondizionato"] for p in per_cell])
            / np.mean([p["mae_banale"] for p in per_cell])),
        "skill_per_cella_rapporti": [p["skill_condizionato"] for p in per_cell],
        "mae_condizionato": float(np.mean([p["mae_condizionato"] for p in per_cell])),
        "mae_banale": float(np.mean([p["mae_banale"] for p in per_cell])),
        "std_vera": float(np.mean([p["std_vera"] for p in per_cell])),
        "std_per_feature": np.mean(
            [p["std_per_feature_cella"] for p in per_cell], axis=0).tolist(),
        "std_condizionata": float(np.mean([p["std_condizionata"] for p in per_cell])),
        "per_cella": per_cell,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1350)
    ap.add_argument("--per-cella", type=int, default=32,
                    help="finestre per cella nella griglia di addestramento; "
                         "cambia n senza cambiare il costo, che dipende solo "
                         "da max_steps (FASE2.md §18)")
    ap.add_argument("--ordine", default="naturale",
                    choices=list(ORDINI),
                    help="assegnazione delle osservabili ai qubit; solo per "
                         "--target qkd. Il default e' l'ordine naturale, "
                         "deciso in B.6 e FASE2 §19; 'b6' resta disponibile "
                         "come variante storica esplicita")
    ap.add_argument("--target", default="omogeneo",
                    choices=["omogeneo", "qkd"],
                    help="'qkd' usa le dispersioni misurate sulla griglia "
                         "reale, con fattore 3 fra osservabili")
    ap.add_argument("--hidden", default="32,16",
                    help="strati nascosti del discriminatore, separati da "
                         "virgola; (32,16) e' il valore dell'Appendice A")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--latent-hi", type=str, default="0.8200",
                    help="scalare, oppure 5 valori separati da virgola "
                         "(un raggio per qubit). "
                         "Estremo superiore del latente per il ramo "
                         "CONDIZIONATO. Il default 0,8200 e' il raggio che "
                         "eguaglia la dispersione della griglia rigenerata; "
                         "0,7342 era calcolato sulla griglia con sift_rate e "
                         "loss_rate costanti per corsa. Il valore di SPEC §3 "
                         "e' pi, che "
                         "pero' produce dispersioni cinque volte troppo "
                         "grandi e skill negativo (FASE2.md §15.2)")
    ap.add_argument("--latent-hi-uncond", type=float, default=None,
                    help="raggio del latente per il ramo INCONDIZIONATO; "
                         "default pi, che e' il suo ottimo. Non ereditare "
                         "--latent-hi: penalizzerebbe la baseline e "
                         "gonfierebbe il guadagno sulla MMD")
    ap.add_argument("--backend", default="denso", choices=["denso", "aer"],
                    help="simulatore del circuito. 'denso' e' vettorizzato sul batch, verificato contro Aer a 1e-15 su forward e jacobiana e molto piu' rapido; 'aer' e' il riferimento di SPEC §7.2")
    ap.add_argument("--theta-init", type=float, default=0.1,
                    help="ampiezza dell'inizializzazione di theta; 0,1 e' il "
                         "valore congelato in SPEC-FASE2 §7")
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--forza", action="store_true",
                    help="riesegue e riaccoda anche semi gia' presenti nel "
                         "file con la stessa configurazione")
    ap.add_argument("--out", default="results/conditional_validation.jsonl")
    a = ap.parse_args()

    global TARGET, PER_CELL, ORDINE, BACKEND
    TARGET = a.target
    ORDINE = a.ordine
    BACKEND = a.backend
    PER_CELL = a.per_cella
    hidden = tuple(int(x) for x in a.hidden.split(","))
    lh = [float(x) for x in str(a.latent_hi).split(",")]
    a.latent_hi = lh[0] if len(lh) == 1 else lh
    if abs(np.mean(lh) - np.pi) < 1e-6:
        print("ATTENZIONE: latente a pi, il valore di SPEC §3. Appendice B.2 "
              "lo sostituisce\n  con 0,8200: a pi le dispersioni escono troppo "
              "grandi e lo\n  skill e' negativo (FASE2.md §15.2). "
              "Usare --latent-hi 0.8200.\n", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    # Configurazione completa, firmata: identifica la corsa nei risultati e
    # permette di riconoscere un riaccodamento. Lo script accodava sempre, e
    # in n_P8_s0, n_P32_s0 e n_P128_s0 sono finite righe identiche.
    config = dict(
        FROZEN, steps=a.steps, per_cella=PER_CELL, per_cella_eval=PER_CELL_EVAL,
        backend=BACKEND,
        target=TARGET, ordine=ORDINE, hidden=list(hidden),
        latent_hi=np.atleast_1d(a.latent_hi).tolist(),
        latent_hi_uncond=a.latent_hi_uncond, theta_init=a.theta_init,
        label_smoothing=a.label_smoothing, weight_decay=a.weight_decay,
        seed_base=SEED_BASE, seed_test=SEED_TEST,
        assi=[ax.tolist() for ax in AXES],
        # Le dispersioni del bersaglio dipendono dalla griglia da cui sono
        # calibrate: due corse con SIGMA_BASE diverse devono avere firme
        # diverse, altrimenti il controllo duplicati le confonde.
        sigma_base=SIGMA_BASE.tolist(),
    )
    config["hidden"] = list(hidden)
    fr = corsa.firma(config)
    gia = corsa.gia_presenti(a.out)
    dup = corsa.duplicati(a.out)
    print(f"configurazione {fr}   codice {corsa.versione_codice()}")
    if dup:
        print(f"ATTENZIONE: {a.out} contiene gia' {dup} righe duplicate: "
              "vanno rimosse prima di qualunque analisi, perche' gonfiano la "
              "numerosita' e sottostimano gli errori standard.", flush=True)

    rows = []
    for s in range(a.seed_offset, a.seed_offset + a.seeds):
        if (fr, s) in gia and not a.forza:
            print(f"[seme {s}] gia' presente con questa configurazione, "
                  "saltato (--forza per rieseguire)", flush=True)
            continue
        print(f"[seme {s}]", flush=True)
        r = run_seed(s, a.steps, a.verbose, theta_init=a.theta_init,
                     latent_hi=a.latent_hi,
                     latent_hi_uncond=a.latent_hi_uncond, hidden=hidden,
                     label_smoothing=a.label_smoothing,
                     weight_decay=a.weight_decay)
        r.update(corsa.intestazione(config))
        corsa.scrivi(a.out, r)
        rows.append(r)
        sa = r["sanity"]
        print(f"  controllo: MMD sul training  condizionato "
              f"{sa['mmd_train_iniziale_cond']:.4f} -> "
              f"{sa['mmd_train_condizionato']:.4f}   incondizionato "
              f"{sa['mmd_train_iniziale_incond']:.4f} -> "
              f"{sa['mmd_train_incondizionato']:.4f}", flush=True)
        print(f"  risposta r={r['risposta_correlazione']:+.3f}   "
              f"MMD celle escluse: cond {r['mmd_condizionato']:.4f}  "
              f"incond {r['mmd_incondizionato']:.4f}  "
              f"guadagno {r['guadagno']:+.4f}", flush=True)
        print(f"  posizione: MAE cond {r['mae_condizionato']:.4f}  "
              f"banale {r['mae_banale']:.4f}  "
              f"SKILL {r['skill_condizionato']:+.3f}  "
              f"(incond {r['skill_incondizionato']:+.3f})", flush=True)
        print(f"  dispersione: vera {r['std_vera']:.4f}  "
              f"generata {r['std_condizionata']:.4f}", flush=True)
        if "std_per_feature" in r:
            p = _perm() if TARGET == "qkd" else np.arange(5)
            nomi = [NOMI[TARGET][j] for j in p]
            sig = SIGMA[TARGET][p]
            print("  per osservabile:  " + "   ".join(
                f"{nomi[i]} {r['std_per_feature'][i]:.4f}/{sig[i]:.4f}"
                for i in range(5)), flush=True)
        k = r["conflitto"]
        print(f"  conflitto:  theta_init {k['theta_init']:.2f}  "
              f"z<={np.round(k['latent_hi'],3)} "
              f"(incond {k['latent_hi_uncond']:.3f})  "
              f"c/z {k['c_su_z_iniziale']:.3f}->{k['c_su_z_finale']:.3f} "
              f"(checkpoint, passo {k['best_step']})  "
              f"Var(g) min {k['grad_var_minima']:.2e}  "
              f"dispersione gen {k['dispersione_generata']:.4f} "
              f"(reale {k['dispersione_reale']:.4f})", flush=True)

    if len(rows) < 2:
        return
    g = np.array([r["guadagno"] for r in rows])
    rr = np.array([r["risposta_correlazione"] for r in rows])
    print(f"\nrisposta   r = {rr.mean():+.3f} +- {rr.std(ddof=1):.3f}")
    print(f"guadagno   {g.mean():+.4f} +- {g.std(ddof=1):.4f}  "
          f"({(g>0).sum()}/{len(g)} semi favorevoli)")
    lu = rows[0]["conflitto"]["latent_hi_uncond"]
    lc = float(np.mean(rows[0]["conflitto"]["latent_hi"]))
    if abs(lu - lc) > 1e-9:
        print(f"           baseline incondizionata con z<={lu:.3f}, "
              f"il suo ottimo, non {lc:.3f}")
    sa = np.array([r["sanity"]["mmd_train_condizionato"] for r in rows])
    si = np.array([r["sanity"]["mmd_train_iniziale_cond"] for r in rows])
    if (sa >= si * 0.9).all():
        print("\nATTENZIONE: la MMD sul training non e' scesa rispetto")
        print("all'inizializzazione. I generatori non hanno appreso, e il")
        print("confronto fra i due rami non e' interpretabile.")
        return
    kk = [r["conflitto"] for r in rows]
    print(f"\ntheta_init {kk[0]['theta_init']:.2f}   "
          f"c/z finale {np.mean([k['c_su_z_finale'] for k in kk]):.3f}   "
          f"Var(grad) min {np.mean([k['grad_var_minima'] for k in kk]):.2e}   "
          f"dispersione {np.mean([k['dispersione_generata'] for k in kk]):.4f} "
          f"su {np.mean([k['dispersione_reale'] for k in kk]):.4f} reale")
    sk = np.array([r["skill_condizionato"] for r in rows])
    print(f"skill      {sk.mean():+.3f} +- {sk.std(ddof=1):.3f}  "
          f"({(sk>0).sum()}/{len(sk)} semi sopra il riferimento banale)")
    print("\nLo skill e' calcolato sul MAE aggregato sulle celle (Fase 1-B §2),")
    print("non come media degli skill per cella.")
    print("\nLo skill score e' la misura da leggere per prima: vale 1 se la")
    print("posizione della cella e' ricostruita esattamente, 0 se equivale a")
    print("ignorare `c`, negativo se peggio. A differenza della MMD contro una")
    print("cella stretta non concede rendita al generatore ampio.")
    if sk.mean() <= 0.0:
        print("\nIl condizionamento non trasporta informazione utilizzabile sulla")
        print("posizione delle celle escluse. RQ2 non ha spazio in questa forma:")
        print("va rivisto lo scope, o va rivista l'implementazione del")
        print("condizionamento, prima della Fase 4.")
    elif g.mean() <= 0:
        print("\nIl condizionamento colloca le celle meglio del riferimento banale")
        print("ma non abbastanza da superare il generatore ampio sulla MMD.")
        print("Le due misure divergono: riportarle entrambe.")


if __name__ == "__main__":
    main()