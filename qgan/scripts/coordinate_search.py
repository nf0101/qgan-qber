"""Ricerca per coordinate su learning rate e larghezza del discriminatore.

La calibrazione avviene esclusivamente su bersagli sintetici, mai su NSL-KDD
o sulla griglia QBER. In Fase 1 le baseline sono state lasciate deliberatamente
non ottimizzate, con la motivazione che un confronto fra modello ottimizzato e
baseline non ottimizzata ne falserebbe l'esito; ottimizzare la QGAN sui dati
reali, anche solo su validation, reintrodurrebbe la stessa asimmetria.

Due bersagli, costruiti per somigliare ai due domini nelle statistiche che
contano — numerosita', multimodalita', dimensione del condizionamento — e non
nel contenuto:

  ids   n = 53, quattro modalita' in [0,1]^5, non condizionato.
        Riproduce r2l a 1:1000: pochi campioni, struttura multimodale.
  qkd   n = 2000, condizionato con d_c = 3, relazione monotona ma non
        lineare fra c e le osservabili. Riproduce la griglia sottocampionata.

Due stadi:

  1. discriminatore fissato a (16, 8); lr_d in {0,001, 0,002, 0,005} per
     rapporto lr_g/lr_d in {0,5, 1}. Sei configurazioni.
  2. con i learning rate vincenti, tre larghezze: (32,16), (16,8), (8,).

Selezione a due stadi. Si scartano anzitutto le configurazioni in cui la
perdita del discriminatore scende sotto la soglia in un qualunque momento
della corsa: la divergenza della MMD e' preannunciata da quel crollo, e una
configurazione che raggiunge un buon valore e poi diverge non e' utilizzabile
perche' il checkpoint verrebbe scelto dal rumore. Fra le superstiti si prende
la MMD piu' bassa, misurata su un insieme di test indipendente.

Una configurazione deve superare il filtro su ENTRAMBI i bersagli. Se e'
stabile solo in uno, quello e' gia' un risultato di O1 e va riportato, non
aggirato scegliendo valori diversi per dominio.

Uso:

    python scripts/coordinate_search.py --stage 1 --steps 1500 --seeds 3
    python scripts/coordinate_search.py --stage 2 --steps 1500 --seeds 3
    python scripts/coordinate_search.py --analyze

I risultati sono accodati a un file JSONL, per cui la ricerca e' interrompibile
e riprendibile: le combinazioni gia' presenti vengono saltate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qgan import QGAN, metrics  # noqa: E402
from qgan.metrics import median_bandwidth  # noqa: E402

BATCH = 16
EVAL_EVERY = 25
# La perdita del discriminatore resta un indicatore diagnostico utile, ma NON
# e' un criterio di rifiuto: nel passaggio rapido due configurazioni con L_D
# sotto 0,3 hanno comunque ridotto la MMD del 45 %. Il criterio corretto e' la
# profondita' del minimo della MMD, misurata con un kernel fissato una volta
# sui soli dati reali.
LD_FLOOR = 0.45
N_PARAMS = 50

CENTERS = np.array(
    [
        [0.20, 0.20, 0.75, 0.30, 0.50],
        [0.75, 0.25, 0.25, 0.70, 0.20],
        [0.30, 0.80, 0.40, 0.20, 0.80],
        [0.70, 0.70, 0.70, 0.65, 0.55],
    ]
)


# --------------------------------------------------------------------- #
# Bersagli sintetici
# --------------------------------------------------------------------- #
def target_ids(n, rng):
    """Pochi campioni, quattro modalita', nessun condizionamento."""
    m = rng.integers(0, 4, n)
    X = np.clip(CENTERS[m] + rng.normal(0, 0.05, (n, 5)), 0, 1)
    return X, None, m


def target_qkd(n, rng):
    """Condizionato, relazione monotona non lineare, modalita' assenti.

    Le cinque colonne imitano qber, qber_rect, qber_diag, sift_rate e
    loss_rate: le prime tre correlate fra loro e crescenti con l'attacco e
    con il rumore, le ultime due dipendenti soprattutto dalla distanza.
    """
    c = rng.random((n, 3))  # intercept_rate, length_km, depolar_total
    q = 0.02 + 0.30 * c[:, 0] + 0.05 * c[:, 1] + 0.35 * c[:, 2] ** 1.5
    asym = 0.04 * (c[:, 0] - 0.5)
    X = np.stack(
        [
            q,
            q + asym,
            q - asym,
            0.50 - 0.08 * c[:, 1] ** 2,
            0.05 + 0.60 * c[:, 1] ** 1.3,
        ],
        axis=1,
    )
    X = np.clip(X + rng.normal(0, 0.02, X.shape), 0, 1)
    return X, c, None


TARGETS = {"ids": (target_ids, 53, 0), "qkd": (target_qkd, 2000, 3)}


def make_target(name, seed):
    fn, n, d_c = TARGETS[name]
    rng = np.random.default_rng(1000 + seed)
    Xtr, ctr, modes = fn(n, rng)
    Xva, cva, _ = fn(256, rng)
    Xte, cte, _ = fn(512, rng)
    return dict(
        X=Xtr, c=ctr, modes=modes, X_val=Xva, c_val=cva,
        X_test=Xte, c_test=cte, d_c=d_c,
    )


# --------------------------------------------------------------------- #
# Configurazioni
# --------------------------------------------------------------------- #
def stage1_configs():
    out = []
    for lr_d in (0.001, 0.002, 0.005):
        for ratio in (0.5, 1.0):
            out.append(
                {
                    "name": f"lrd{lr_d}_r{ratio}",
                    "lr_d": lr_d,
                    "lr_g": round(lr_d * ratio, 6),
                    "hidden": (16, 8),
                }
            )
    return out


def quick_configs():
    """Passaggio di rifiuto: sei configurazioni, un solo bersaglio, un seme.

    La larghezza del discriminatore e' il fattore dominante misurato in
    §4.2, per cui entra nella griglia fin da subito; sui learning rate si
    tengono solo due punti, quelli che nel confronto di §4.2 non erano
    instabili. Lo scopo non e' selezionare la configurazione migliore ma
    scoprire se ne esiste una che non diverge: e' un rifiuto, e per
    rifiutare basta un seme e una corsa breve, perche' il crollo della
    perdita del discriminatore e' un fenomeno grossolano che si manifesta
    entro il passo 150.
    """
    out = []
    for h in [(32, 16), (16, 8), (8,)]:
        for lr_g, lr_d in [(0.002, 0.002), (0.001, 0.005)]:
            out.append(
                {
                    "name": f"h{'x'.join(map(str,h))}_g{lr_g}_d{lr_d}",
                    "lr_g": lr_g,
                    "lr_d": lr_d,
                    "hidden": h,
                }
            )
    return out


def stage2_configs(lr_g, lr_d):
    return [
        {
            "name": f"h{'x'.join(map(str,h))}",
            "lr_d": lr_d,
            "lr_g": lr_g,
            "hidden": h,
        }
        for h in [(32, 16), (16, 8), (8,)]
    ]


# --------------------------------------------------------------------- #
def run_one(cfg, target_name, seed, steps, verbose=0):
    t = make_target(target_name, seed)
    q = QGAN(
        d_c=t["d_c"],
        lr_g=cfg["lr_g"],
        lr_d=cfg["lr_d"],
        hidden=tuple(cfg["hidden"]),
        batch_size=BATCH,
        max_steps=steps,
        eval_every=EVAL_EVERY,
        patience=10**9,  # nessun arresto: interessa la traiettoria intera
        val_fraction=0.0,
        n_monitor=256,
        monitor_repeats=1,
        mmd_smooth=3,
        seed=seed,
        verbose=verbose,
    )
    t0 = time.time()
    # Un solo kernel per tutta la corsa, ricavato dai soli dati reali di test.
    sigma = float(median_bandwidth(t["X_test"], rng=np.random.default_rng(0)))
    mmd_init = metrics.mmd_rbf(
        t["X_test"], q.sample(len(t["X_test"]), t["c_test"]), sigma=sigma
    )
    q.fit(t["X"], t["c"], X_val=t["X_val"], c_val=t["c_val"], modes=t["modes"])
    Xg = q.sample(len(t["X_test"]), t["c_test"])

    ld = np.array(q.history["loss_d"])
    sm = np.array(q.history["mmd_smooth"])
    return {
        "config": cfg["name"],
        "target": target_name,
        "seed": seed,
        "lr_g": cfg["lr_g"],
        "lr_d": cfg["lr_d"],
        "hidden": list(cfg["hidden"]),
        "steps": steps,
        "mmd_init": float(mmd_init),
        "mmd_sigma": sigma,
        "mmd_test_finale": float(metrics.mmd_rbf(t["X_test"], Xg, sigma=sigma)),
        "mmd_smooth_min": float(sm.min()),
        "best_step": int(q.best_step_),
        "loss_d_min": float(ld.min()),
        "stabile": bool(ld.min() >= LD_FLOOR),
        "plateau_step": plateau_step(q.history["step"], sm),
        "detectability": float(metrics.detectability(t["X_test"], Xg, seed=seed)),
        "history": {k: q.history[k] for k in
                    ("step", "loss_d", "loss_g", "mmd_val", "mmd_smooth", "grad_var")},
        "secondi": time.time() - t0,
    }


def plateau_step(steps, smoothed, tol=0.02):
    """Primo passo in cui il minimo corrente arriva entro `tol` dal minimo finale.

    E' la lettura del budget dalle curve: oltre questo punto la MMD non
    migliora piu' in modo apprezzabile.
    """
    run_min = np.minimum.accumulate(np.asarray(smoothed, dtype=float))
    final = run_min[-1]
    if not np.isfinite(final):
        return None
    thr = final + tol * abs(final)
    idx = int(np.argmax(run_min <= thr))
    return int(steps[idx])


# --------------------------------------------------------------------- #
def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def key(r):
    return (r["config"], r["target"], r["seed"])


def cost_estimate(n_runs, steps):
    circ = n_runs * steps * BATCH * (1 + 2 * N_PARAMS)
    return circ, circ * 0.00028 / 3600  # ~0,28 ms/circuito raggruppato


# --------------------------------------------------------------------- #
def analyze_quick(rows):
    """Esito del passaggio di rifiuto: si va avanti o si cambia strada."""
    rows = [r for r in rows if r["target"] == "ids"]
    if not rows:
        print("nessun risultato")
        return
    print(f"\ncriterio: riduzione della MMD > 10 % al checkpoint migliore\n")
    print(f"{'configurazione':24s} {'L_D min':>8s} {'MMD in':>8s} "
          f"{'MMD best':>8s} {'riduz.':>8s} {'esito':>9s}")
    print("L_D min e' diagnostico, non un filtro: vedere FASE2.md §9.4")
    print("-" * 72)
    surv = []
    for r in sorted(rows, key=lambda x: x["mmd_test_finale"]):
        red = 1.0 - r["mmd_test_finale"] / r["mmd_init"]
        ok = red > 0.10  # riduzione apprezzabile della MMD al checkpoint migliore
        if ok:
            surv.append(r)
        print(f"{r['config']:24s} {r['loss_d_min']:8.3f} {r['mmd_init']:8.4f} "
              f"{r['mmd_test_finale']:8.4f} {red:+8.1%} "
              f"{('ok' if ok else 'no'):>9s}")

    print()
    if not surv:
        print("NESSUNA configurazione sopravvive.")
        print("Il problema non e' la calibrazione ma la funzione di costo:")
        print("l'entropia incrociata su 53 campioni non produce un segnale")
        print("utilizzabile. La via documentata e' la perdita a kernel sulla")
        print("MMD, che in §4.1 raggiunge 0,010 senza instabilita'. Comporta")
        print("una modifica a §7 e va discussa con il relatore.")
        return
    best = min(surv, key=lambda r: r["mmd_test_finale"])
    print(f"sopravvivono: {', '.join(r['config'] for r in surv)}")
    print(f"migliore: {best['config']}  "
          f"lr_g={best['lr_g']} lr_d={best['lr_d']} "
          f"D={'x'.join(map(str,best['hidden']))}")
    print("\nPasso successivo: conferma dei sopravvissuti sul bersaglio qkd,")
    print("un seme, poi corsa lunga del vincitore su entrambi i bersagli.")
    print(f"\n  python scripts/coordinate_search.py --quick --targets qkd \\")
    print(f"      --configs {','.join(r['config'] for r in surv)}")


def analyze(rows):
    if not rows:
        print("nessun risultato")
        return
    cfgs = sorted({r["config"] for r in rows})
    targets = sorted({r["target"] for r in rows})

    print(f"\nsoglia di stabilita': L_D >= {LD_FLOOR}\n")
    header = f"{'configurazione':16s} {'lr_g':>7s} {'lr_d':>7s} {'D':>8s}"
    for t in targets:
        header += f" | {t+' L_D':>10s} {t+' MMD':>10s} {'stab':>5s}"
    print(header)
    print("-" * len(header))

    summary = {}
    for cfg in cfgs:
        sub = [r for r in rows if r["config"] == cfg]
        line = (f"{cfg:16s} {sub[0]['lr_g']:7.4f} {sub[0]['lr_d']:7.4f} "
                f"{'x'.join(map(str,sub[0]['hidden'])):>8s}")
        ok_all, mmds = True, {}
        for t in targets:
            st = [r for r in sub if r["target"] == t]
            if not st:
                line += f" | {'-':>10s} {'-':>10s} {'-':>5s}"
                ok_all = False
                continue
            ld = np.mean([r["loss_d_min"] for r in st])
            mm = np.mean([r["mmd_test_finale"] for r in st])
            ok = all(r["stabile"] for r in st)
            ok_all &= ok
            mmds[t] = mm
            line += f" | {ld:10.3f} {mm:10.4f} {'si' if ok else 'NO':>5s}"
        summary[cfg] = {"stabile": ok_all, "mmd": mmds}
        print(line)

    surv = [c for c, v in summary.items() if v["stabile"] and v["mmd"]]
    print(f"\nsuperano il filtro su tutti i bersagli: "
          f"{', '.join(surv) if surv else 'nessuna'}")
    if not surv:
        print("Nessuna configurazione e' stabile su entrambi i bersagli. "
              "Non scegliere valori diversi per dominio: e' un risultato di O1 "
              "e va riportato.")
        return

    best = min(surv, key=lambda c: max(summary[c]["mmd"].values()))
    sub = [r for r in rows if r["config"] == best]
    print(f"\nconfigurazione scelta: {best}  "
          f"lr_g={sub[0]['lr_g']}  lr_d={sub[0]['lr_d']}  "
          f"D={'x'.join(map(str,sub[0]['hidden']))}")
    print("   criterio: la peggiore MMD fra i due bersagli e' la piu' bassa "
          "fra le configurazioni stabili")

    pl = {t: [r["plateau_step"] for r in sub if r["target"] == t
              and r["plateau_step"]] for t in targets}
    print("\n  altopiano della MMD, per bersaglio:")
    meds = []
    for t, v in pl.items():
        if v:
            meds.append(float(np.median(v)))
            print(f"    {t:5s} mediana {np.median(v):7.0f} passi   "
                  f"(semi: {sorted(v)})")
    if meds:
        rec = int(np.ceil(1.5 * max(meds) / 50.0) * 50)
        print(f"\n  max_steps consigliato: {rec}")
        print(f"    = 1,5 x il massimo fra le mediane ({max(meds):.0f}), "
              f"arrotondato a 50")
        print(f"    si prende il massimo fra i domini perche' il rischio e' "
              f"asimmetrico:\n    un budget eccessivo costa solo calcolo, "
              f"uno insufficiente tronca l'addestramento")
        print(f"\n  da congelare: batch {BATCH}, eval_every {EVAL_EVERY}, "
              f"mmd_smooth 3, patience 20 valutazioni ({20*EVAL_EVERY} passi)")


# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2])
    ap.add_argument("--quick", action="store_true",
                    help="passaggio di rifiuto: 6 corse su ids, 1 seme, 600 passi")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--targets", default="ids,qkd")
    ap.add_argument("--lr-g", type=float, help="stadio 2: lr_g vincente")
    ap.add_argument("--lr-d", type=float, help="stadio 2: lr_d vincente")
    ap.add_argument("--configs", default=None,
                    help="sottoinsieme di configurazioni, separate da virgola")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--out", default="results/coordinate_search.jsonl")
    a = ap.parse_args()

    if a.analyze:
        analyze(load(a.out))
        return

    if a.quick:
        cfgs = quick_configs()
        if a.targets == "ids,qkd":  # non sovrascrive una scelta esplicita
            a.targets = "ids"
        if a.seeds == 3:
            a.seeds = 1
        if a.steps == 1500:
            a.steps = 600
    elif a.stage == 1:
        cfgs = stage1_configs()
    else:
        if a.stage is None:
            ap.error("indicare --quick, --stage 1, --stage 2 oppure --analyze")
        if a.lr_g is None or a.lr_d is None:
            ap.error("lo stadio 2 richiede --lr-g e --lr-d dallo stadio 1")
        cfgs = stage2_configs(a.lr_g, a.lr_d)

    if a.configs:
        keep = set(a.configs.split(","))
        cfgs = [c for c in cfgs if c["name"] in keep]
        if not cfgs:
            ap.error(f"nessuna configurazione fra {keep}")
    targets = a.targets.split(",")
    done = {key(r) for r in load(a.out)}
    todo = [
        (c, t, s)
        for c in cfgs
        for t in targets
        for s in range(a.seeds)
        if (c["name"], t, s) not in done
    ]
    circ, ore = cost_estimate(len(todo), a.steps)
    etichetta = "rifiuto rapido" if a.quick else f"stadio {a.stage}"
    print(f"{etichetta}: {len(cfgs)} configurazioni x {len(targets)} "
          f"bersagli x {a.seeds} semi = {len(cfgs)*len(targets)*a.seeds} corse, "
          f"{len(todo)} da eseguire")
    print(f"costo stimato: {circ/1e6:.1f}M circuiti, circa {ore:.1f} ore")
    if a.dry_run:
        return

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for i, (c, t, s) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {c['name']} / {t} / seme {s}", flush=True)
        r = run_one(c, t, s, a.steps, verbose=a.verbose)
        with open(a.out, "a") as f:
            f.write(json.dumps(r, default=float) + "\n")
        print(f"    L_D min {r['loss_d_min']:.3f}  "
              f"MMD {r['mmd_init']:.4f} -> {r['mmd_test_finale']:.4f}  "
              f"altopiano {r['plateau_step']}  "
              f"{'stabile' if r['stabile'] else 'DIVERGE'}  "
              f"{r['secondi']:.0f}s", flush=True)

    if a.quick:
        analyze_quick(load(a.out))
    else:
        analyze(load(a.out))


if __name__ == "__main__":
    main()
