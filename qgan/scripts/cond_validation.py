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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
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
# Numero di finestre per cella usate in VALUTAZIONE, tenuto fisso al variare di
# PER_CELL: altrimenti aumentando la numerosita' migliorerebbe anche la stima
# del MAE sulle celle escluse, e i due effetti sarebbero confusi.
PER_CELL_EVAL = 32


# Rumore condizionale per osservabile.
#
# `omogeneo` e' il bersaglio usato in FASE2 §12-§15: tutte le osservabili con
# la stessa dispersione. Non mette alla prova i raggi per feature, perche' un
# raggio unico gia' le soddisfa tutte.
#
# `qkd` riproduce le dispersioni misurate sulla griglia reale (Appendice B.2),
# con un fattore 47 fra la piu' alta e la piu' bassa, e l'ordinamento sui
# qubit proposto in B.6. E' il bersaglio su cui i raggi per feature vanno
# validati prima di applicarli ai dati veri.
# Base nell'ordine di SPEC §1.2, prima di qualunque permutazione sui qubit.
NOMI_BASE = ["qber", "qber_rect", "qber_diag", "sift_rate", "loss_rate"]
SIGMA_BASE = np.array([0.05754, 0.07024, 0.07138, 0.00318, 0.00152])

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
ORDINE = "b6"

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
    rng = np.random.default_rng(3000 + seed)
    X, C, cid, cells = build_grid(rng)
    inner = interior_mask(cells)
    inner_idx = np.flatnonzero(inner)

    # meta' delle celle interne escluse dall'addestramento
    held = rng.choice(inner_idx, len(inner_idx) // 2, replace=False)
    train = ~np.isin(cid, held)

    kw = dict(FROZEN, max_steps=steps, seed=seed, verbose=verbose,
              theta_init=theta_init, latent_hi=latent_hi,
              hidden=tuple(hidden), label_smoothing=label_smoothing,
              d_weight_decay=weight_decay)

    # Validazione: campione casuale su TUTTE le celle di addestramento. Le
    # prime righe della griglia appartengono a un solo angolo (c[0] = 0),
    # quindi uno slice iniziale monitorerebbe una fetta non rappresentativa.
    vi = rng.choice(np.flatnonzero(train), 256, replace=False)

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
    d_true = (cell_features(np.array([0.9, 0.5, 0.9]), rng, 2000).mean(0)
              - cell_features(np.array([0.1, 0.5, 0.1]), rng, 2000).mean(0))
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
        Xh = X[cid == h]
        if len(Xh) > PER_CELL_EVAL:
            Xh = Xh[rng.choice(len(Xh), PER_CELL_EVAL, replace=False)]
        ch = cells[h]
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
    Xt = (Xt_full[rng.choice(len(Xt_full), 1024, replace=False)]
          if len(Xt_full) > 1024 else Xt_full)
    sanity = {
        "mmd_train_condizionato": metrics.mmd_rbf(
            Xt, cond.sample(512, C[train][rng.choice(train.sum(), 512)]), sigma=sig),
        "n_train": int(len(Xt_full)),
        "mmd_train_incondizionato": metrics.mmd_rbf(
            Xt, uncond.sample(512), sigma=sig),
        "mmd_train_iniziale": metrics.mmd_rbf(
            Xt, QGAN(d_c=0, **kw).sample(512), sigma=sig),
    }

    # Le tre grandezze in conflitto, da leggere insieme:
    #  - c/z  : quanto l'uscita risponde al condizionamento
    #  - Var(grad) : indicatore di barren plateau, motivo per cui §7 prescrive
    #    l'inizializzazione vicina all'identita'
    #  - dispersione : meta' del guadagno di c/z ad ampiezza maggiore viene
    #    dalla riduzione del contributo del latente, che e' cio' che fornisce
    #    diversita' ai campioni. Un rapporto alto ottenuto stringendo l'uscita
    #    non e' un buon risultato.
    hc = cond.history
    gen_std = float(cond.sample(512, C[train][rng.choice(train.sum(), 512)]).std(0).mean())
    conflitto = {
        "theta_init": theta_init,
        "n_train": int(train.sum()),
        "per_cella": PER_CELL,
        "ordine": ORDINE,
        "latent_hi": np.atleast_1d(latent_hi).tolist(),
        "latent_hi_uncond": float(kw_u["latent_hi"]),
        "hidden": list(hidden),
        "label_smoothing": label_smoothing,
        "weight_decay": weight_decay,
        "c_su_z_iniziale": hc["sensibilita_c"][0],
        "c_su_z_finale": hc["sensibilita_c"][-1],
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
    ap.add_argument("--ordine", default="b6",
                    choices=list(ORDINI),
                    help="assegnazione delle osservabili ai qubit; solo per "
                         "--target qkd")
    ap.add_argument("--target", default="omogeneo",
                    choices=["omogeneo", "qkd"],
                    help="'qkd' usa le dispersioni misurate sulla griglia "
                         "reale, con fattore 47 fra osservabili")
    ap.add_argument("--hidden", default="32,16",
                    help="strati nascosti del discriminatore, separati da "
                         "virgola; (32,16) e' il valore dell'Appendice A")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--latent-hi", type=str, default="0.7342",
                    help="scalare, oppure 5 valori separati da virgola "
                         "(un raggio per qubit). "
                         "Estremo superiore del latente per il ramo "
                         "CONDIZIONATO. Il default 0,7342 e' il valore deciso "
                         "in Appendice B.2; il valore di SPEC §3 e' pi, che "
                         "pero' produce dispersioni cinque volte troppo "
                         "grandi e skill negativo (FASE2.md §15.2)")
    ap.add_argument("--latent-hi-uncond", type=float, default=None,
                    help="raggio del latente per il ramo INCONDIZIONATO; "
                         "default pi, che e' il suo ottimo. Non ereditare "
                         "--latent-hi: penalizzerebbe la baseline e "
                         "gonfierebbe il guadagno sulla MMD")
    ap.add_argument("--theta-init", type=float, default=0.1,
                    help="ampiezza dell'inizializzazione di theta; 0,1 e' il "
                         "valore congelato in SPEC-FASE2 §7")
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--out", default="results/conditional_validation.jsonl")
    a = ap.parse_args()

    global TARGET, PER_CELL, ORDINE
    TARGET = a.target
    ORDINE = a.ordine
    PER_CELL = a.per_cella
    hidden = tuple(int(x) for x in a.hidden.split(","))
    lh = [float(x) for x in str(a.latent_hi).split(",")]
    a.latent_hi = lh[0] if len(lh) == 1 else lh
    if abs(np.mean(lh) - np.pi) < 1e-6:
        print("ATTENZIONE: latente a pi, il valore di SPEC §3. Appendice B.2 "
              "lo sostituisce\n  con 0,7342: a pi le dispersioni escono cinque "
              "volte troppo grandi e lo\n  skill e' negativo (FASE2.md §15.2). "
              "Usare --latent-hi 0.7342.\n", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = []
    for s in range(a.seed_offset, a.seed_offset + a.seeds):
        print(f"[seme {s}]", flush=True)
        r = run_seed(s, a.steps, a.verbose, theta_init=a.theta_init,
                     latent_hi=a.latent_hi,
                     latent_hi_uncond=a.latent_hi_uncond, hidden=hidden,
                     label_smoothing=a.label_smoothing,
                     weight_decay=a.weight_decay)
        with open(a.out, "a") as f:
            f.write(json.dumps(r, default=float) + "\n")
        rows.append(r)
        sa = r["sanity"]
        print(f"  controllo: MMD sul training  iniziale {sa['mmd_train_iniziale']:.4f}"
              f"  cond {sa['mmd_train_condizionato']:.4f}"
              f"  incond {sa['mmd_train_incondizionato']:.4f}", flush=True)
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
              f"c/z {k['c_su_z_iniziale']:.3f}->{k['c_su_z_finale']:.3f}  "
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
    si = np.array([r["sanity"]["mmd_train_iniziale"] for r in rows])
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