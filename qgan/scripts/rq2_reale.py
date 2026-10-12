"""RQ2 sui dati QKD reali — la pipeline dichiarata da costruire in B.7.

Finora il condizionamento e' stato validato solo su bersagli sintetici,
deliberatamente. Questo script esegue lo stesso disegno sulla griglia QBER
vera: configurazioni interne escluse dall'addestramento, generatore
condizionato contro incondizionato contro riferimento banale.

DIFFERENZE RISPETTO AL SINTETICO, tutte a favore del reale

  24 configurazioni interne contro 8. Sul sintetico la varianza fra celle
  valeva il 60% del totale e limitava la generalizzazione piu' delle
  repliche; con 24 l'intervallo incrociato dovrebbe stringersi.

  Nessun sottocampionamento di valutazione. Le configurazioni escluse non
  entrano in addestramento in alcun modo, e si valutano su TUTTE le loro
  finestre: circa 785, uguali per costruzione in ogni condizione. Il difetto
  3 — numerosita' di valutazione che cambia con quella di addestramento —
  qui non puo' presentarsi.

  La relazione fra c e osservabili non e' monotona ne' liscia. Fase 1-B
  documenta f_canale con skill 0,088 sull'interpolazione: un esito positivo
  qui vale molto piu' di uno sul sintetico.

ESCLUSIONE PER CONFIGURAZIONE, NON PER CELLA. I cinque semi di una stessa
configurazione sono repliche dello STESSO c: escluderne alcuni e tenerne
altri metterebbe in addestramento il condizionamento che si vuole valutare.
Si escludono quindi configurazioni intere, con tutti i loro semi.

    python scripts/rq2_reale.py --dati ../data/qber_grid.csv --seeds 1
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import corsa  # noqa: E402
from qgan import QGAN, metrics  # noqa: E402
from qgan.metrics import median_bandwidth  # noqa: E402

OSSERVABILI = ["qber", "qber_rect", "qber_diag", "sift_rate", "loss_rate"]
CONDIZIONI = ["intercept_rate", "length_km", "depolar_total"]

SEED_BASE = 4100
LATENT_HI = 0.8200          # B.2, ricalibrato sulla griglia rigenerata
LATENT_HI_UNCOND = np.pi    # B.5: il ramo non condizionato usa il proprio

FROZEN = dict(
    lr_g=0.002, lr_d=0.002, hidden=(32, 16), batch_size=16,
    eval_every=25, patience=20, mmd_smooth=3, val_fraction=0.0,
    n_monitor=256, monitor_repeats=1,
)


def interne(d):
    """Configurazioni non al bordo di nessun asse del condizionamento.

    Sul bordo il generatore dovrebbe estrapolare fuori dall'intervallo visto,
    che e' una domanda diversa da RQ2: qui si misura l'interpolazione.
    """
    masch = np.ones(len(d), bool)
    for c in CONDIZIONI:
        v = np.sort(d[c].unique())
        masch &= ~d[c].isin([v[0], v[-1]]).to_numpy()
    return masch


def scala(v, lo, amp):
    return np.clip((v - lo) / np.where(amp > 0, amp, 1.0), 0.0, 1.0)


def run_seed(seed, dati, steps, verbose, backend, quota_escluse=0.5,
             n_conf_train=None):
    d = pd.read_csv(dati)
    if "profilo" in d.columns:
        raise SystemExit("questa e' la griglia non stazionaria; RQ2 usa "
                         "qber_grid.csv")

    conf = d[CONDIZIONI].drop_duplicates().reset_index(drop=True)
    masch_int = interne(conf)
    idx_int = np.flatnonzero(masch_int)

    rng, rng_split, rng_diag, rng_sub = corsa.flussi(SEED_BASE, seed, 4)
    n_escl = max(1, int(round(len(idx_int) * quota_escluse)))
    escluse = np.sort(rng_split.choice(idx_int, n_escl, replace=False))
    conf_escl = conf.iloc[escluse]

    chiave = d[CONDIZIONI].apply(tuple, axis=1)
    chiavi_escl = set(conf_escl.apply(tuple, axis=1))
    m_te = chiave.isin(chiavi_escl).to_numpy()
    m_tr = ~m_te

    # Densita' della griglia di ADDESTRAMENTO. Sui dati reali un
    # interpolatore lineare sulle 108 configurazioni batte la QGAN di un
    # fattore 14 sul MAE e 40 sulla MMD: con vicini fitti la distribuzione
    # condizionale varia poco e non resta spazio per un modello generativo.
    # La domanda non e' SE la QGAN batta l'interpolazione, e' QUANDO —
    # cioe' a quale densita' i vicini smettono di bastare. E' il regime del
    # dominio IDS, dove la classe minoritaria ha 53 campioni e vicini non
    # ce ne sono. Le configurazioni ESCLUSE restano le stesse: varia solo
    # quante ne vede l'addestramento, da un flusso proprio.
    chiavi_tr = sorted(set(chiave[m_tr]))
    if n_conf_train is not None and n_conf_train < len(chiavi_tr):
        sel = {chiavi_tr[i] for i in
               rng_sub.choice(len(chiavi_tr), n_conf_train, replace=False)}
        m_tr = m_tr & chiave.isin(sel).to_numpy()

    X_all = d[OSSERVABILI].to_numpy(float)
    C_all = d[CONDIZIONI].to_numpy(float)

    # Riscalamento min-max sulle sole righe di ADDESTRAMENTO, riusato
    # identico in valutazione: le escluse non devono influenzarlo.
    lo, hi = X_all[m_tr].min(0), X_all[m_tr].max(0)
    amp = hi - lo
    Xs = scala(X_all, lo, amp)
    c_lo = C_all[m_tr].min(0)
    c_amp = C_all[m_tr].max(0) - c_lo
    Cs = scala(C_all, c_lo, c_amp)

    Xt, Ct = Xs[m_tr], Cs[m_tr]
    vi = rng_split.choice(len(Xt), min(1024, len(Xt)), replace=False)
    sig = float(median_bandwidth(Xt, rng=np.random.default_rng(0)))

    kw = dict(FROZEN, max_steps=steps, seed=seed, verbose=verbose,
              backend=backend, latent_hi=LATENT_HI)
    kw_u = dict(kw, latent_hi=float(LATENT_HI_UNCOND))

    cond = QGAN(d_c=3, **kw)
    cond.fit(Xt, Ct, X_val=Xt[vi], c_val=Ct[vi])
    uncond = QGAN(d_c=0, **kw_u)
    uncond.fit(Xt, X_val=Xt[vi])

    # --- RIFERIMENTI ---------------------------------------------------
    # `banale` e' la media globale: non usa c ed e' il minimo che un
    # condizionamento deve battere. Su una griglia densa e' pero' un
    # avversario facile, e battere una costante non dimostra granche'.
    #
    # Si aggiungono due riferimenti che USANO c, costruiti dalle sole
    # configurazioni di addestramento:
    #   vicino   la configurazione piu' vicina in c normalizzato
    #   lineare  interpolazione lineare sulle 108 configurazioni di
    #            addestramento, che su 3 assi e griglia regolare e'
    #            l'avversario vero
    # La differenza fra "batte la media globale" e "batte un interpolatore"
    # separa un risultato pubblicabile da uno ovvio.
    mu_banale = Xt.mean(axis=0)

    # Il confronto va fatto sulla Series pandas: un array NumPy di oggetti
    # confrontato con una tupla finisce in broadcast elemento per elemento.
    k_tr = sorted(set(chiave[m_tr]))
    masch_tr = {k: (chiave == k).to_numpy() for k in k_tr}
    mu_tr = np.array([Xs[masch_tr[k] & m_tr].mean(axis=0) for k in k_tr])
    c_tr = np.array([Cs[masch_tr[k]][0] for k in k_tr])
    interp_lin = LinearNDInterpolator(c_tr, mu_tr)
    interp_vic = NearestNDInterpolator(c_tr, mu_tr)

    per_cella, n_gen, fuori = [], 512, 0
    for i, k in zip(escluse, conf_escl.apply(tuple, axis=1)):
        m = (chiave == k).to_numpy()
        Xh = Xs[m]
        ch = Cs[m][0]
        mu_vero = Xh.mean(axis=0)
        mu_cond = cond.sample(n_gen, ch).mean(axis=0)
        mu_unc = uncond.sample(n_gen).mean(axis=0)
        mu_lin = np.asarray(interp_lin(ch[None, :]))[0]
        if not np.all(np.isfinite(mu_lin)):      # fuori dall'inviluppo
            mu_lin = np.asarray(interp_vic(ch[None, :]))[0]
            fuori += 1
        mu_vic = np.asarray(interp_vic(ch[None, :]))[0]
        # --- riferimenti DISTRIBUZIONALI -------------------------------
        # L'interpolatore produce un punto, la QGAN una distribuzione: il
        # confronto sul MAE delle medie misura il primo compito. Qui si
        # costruiscono riferimenti che producono anch'essi finestre.
        #   vicino   le finestre vere della configurazione piu' vicina
        #   lineare  le stesse finestre TRASLATE sulla media interpolata:
        #            ha la media giusta e una forma realistica, ed e'
        #            l'avversario serio
        #   banale   un campione casuale dell'addestramento
        j = int(np.argmin(((c_tr - ch) ** 2).sum(axis=1)))
        X_vic = Xs[masch_tr[k_tr[j]] & m_tr]
        if len(X_vic) > n_gen:
            X_vic = X_vic[rng_diag.choice(len(X_vic), n_gen, replace=False)]
        X_lin = np.clip(X_vic + (mu_lin - X_vic.mean(axis=0)), 0.0, 1.0)
        X_ban = Xt[rng_diag.choice(len(Xt), n_gen, replace=False)]

        per_cella.append({
            "cella": int(i),
            "condizione": [float(x) for x in k],
            "n_finestre": int(m.sum()),
            "mae_condizionato": float(np.abs(mu_cond - mu_vero).mean()),
            "mae_incondizionato": float(np.abs(mu_unc - mu_vero).mean()),
            "mae_banale": float(np.abs(mu_banale - mu_vero).mean()),
            "mae_vicino": float(np.abs(mu_vic - mu_vero).mean()),
            "mae_lineare": float(np.abs(mu_lin - mu_vero).mean()),
            "mmd_condizionato": float(metrics.mmd_rbf(
                Xh, cond.sample(n_gen, ch), sigma=sig)),
            "mmd_incondizionato": float(metrics.mmd_rbf(
                Xh, uncond.sample(n_gen), sigma=sig)),
            "mmd_vicino": float(metrics.mmd_rbf(Xh, X_vic, sigma=sig)),
            "mmd_lineare": float(metrics.mmd_rbf(Xh, X_lin, sigma=sig)),
            "mmd_banale": float(metrics.mmd_rbf(Xh, X_ban, sigma=sig)),
        })

    def agg(campo):
        return float(np.sum([p[campo] for p in per_cella]))

    # Skill su MAE AGGREGATE (B.4), non media di rapporti.
    def skill(contro):
        return 1.0 - agg("mae_condizionato") / max(agg(contro), 1e-12)
    sk_c = skill("mae_banale")
    sk_u = 1.0 - agg("mae_incondizionato") / max(agg("mae_banale"), 1e-12)

    lh = cond._latent_hi_vec.tolist()
    return {
        "seed": seed, "steps": steps, "dati": os.path.basename(dati),
        "configurazioni_totali": int(len(conf)),
        "configurazioni_interne": int(masch_int.sum()),
        "celle_escluse": [int(x) for x in escluse],
        "n_train": int(m_tr.sum()), "n_test": int(m_te.sum()),
        "sigma_mmd": sig,
        "scala_min": lo.tolist(), "scala_ampiezza": amp.tolist(),
        "skill_condizionato": float(sk_c),
        "skill_incondizionato": float(sk_u),
        "skill_vs_vicino": float(skill("mae_vicino")),
        "skill_vs_lineare": float(skill("mae_lineare")),
        # Sul compito DISTRIBUZIONALE, che e' quello che un generatore fa
        "mmd_condizionato_m": agg("mmd_condizionato") / len(per_cella),
        "mmd_vicino_m": agg("mmd_vicino") / len(per_cella),
        "mmd_lineare_m": agg("mmd_lineare") / len(per_cella),
        "mmd_banale_m": agg("mmd_banale") / len(per_cella),
        "skill_mmd_vs_banale": float(
            1.0 - agg("mmd_condizionato") / max(agg("mmd_banale"), 1e-12)),
        "skill_mmd_vs_vicino": float(
            1.0 - agg("mmd_condizionato") / max(agg("mmd_vicino"), 1e-12)),
        "skill_mmd_vs_lineare": float(
            1.0 - agg("mmd_condizionato") / max(agg("mmd_lineare"), 1e-12)),
        "mae_vicino": agg("mae_vicino") / len(per_cella),
        "mae_lineare": agg("mae_lineare") / len(per_cella),
        "n_configurazioni_train": int(len(k_tr)),
        "configurazioni_richieste": n_conf_train,
        "escluse_fuori_inviluppo": int(fuori),
        "mae_condizionato": agg("mae_condizionato") / len(per_cella),
        "mae_incondizionato": agg("mae_incondizionato") / len(per_cella),
        "mae_banale": agg("mae_banale") / len(per_cella),
        "per_cella": per_cella,
        "conflitto": {
            "c_su_z_iniziale": float(QGAN(d_c=3, **kw).gen
                                     .conditioning_sensitivity(
                QGAN(d_c=3, **kw).theta, n=200, seed=seed,
                latent_hi=lh)["rapporto"]),
            "c_su_z_finale": float(cond.gen.conditioning_sensitivity(
                cond.theta, n=200, seed=seed, latent_hi=lh)["rapporto"]),
            "sensibilita_per_ingresso": cond.gen.input_sensitivity(
                cond.theta, n=200, seed=seed, latent_hi=lh).tolist(),
            "best_step": int(cond.best_step_),
            "steps_run": int(cond.steps_run_),
            "monitor": cond.monitor_,
            "strati_assi": getattr(cond, "strati_assi_", None),
            "latent_hi": lh,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dati", default="../data/qber_grid.csv")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5400)
    ap.add_argument("--quota-escluse", type=float, default=0.5)
    ap.add_argument("--configurazioni-train", type=int, default=None,
                    help="quante configurazioni di addestramento tenere "
                         "(default: tutte). Le escluse non cambiano: serve "
                         "a trovare la densita' alla quale la QGAN supera "
                         "l'interpolazione")
    ap.add_argument("--backend", default="denso", choices=["denso", "aer"])
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--forza", action="store_true")
    ap.add_argument("--out", default="results/rq2_reale.jsonl")
    a = ap.parse_args()

    config = dict(FROZEN, steps=a.steps, dati=a.dati,
                  quota_escluse=a.quota_escluse, backend=a.backend,
                  configurazioni_train=a.configurazioni_train,
                  latent_hi=LATENT_HI, latent_hi_uncond=float(LATENT_HI_UNCOND),
                  osservabili=OSSERVABILI, condizioni=CONDIZIONI,
                  seed_base=SEED_BASE)
    fr = corsa.firma(config)
    gia = corsa.gia_presenti(a.out)
    print(f"configurazione {fr}   codice {corsa.versione_codice()}")

    for s in range(a.seed_offset, a.seed_offset + a.seeds):
        if (fr, s) in gia and not a.forza:
            print(f"[seme {s}] gia' presente, saltato", flush=True)
            continue
        print(f"[seme {s}]", flush=True)
        r = run_seed(s, a.dati, a.steps, a.verbose, a.backend,
                     a.quota_escluse, a.configurazioni_train)
        r.update(corsa.intestazione(config))
        corsa.scrivi(a.out, r)
        k = r["conflitto"]
        print(f"  configurazioni {r['configurazioni_interne']} interne su "
              f"{r['configurazioni_totali']}, escluse {len(r['celle_escluse'])}"
              f", in addestramento {r['n_configurazioni_train']}"
              f"   train {r['n_train']}  test {r['n_test']}"
              f"   escluse fuori inviluppo {r['escluse_fuori_inviluppo']}",
              flush=True)
        print(f"  MAE  cond {r['mae_condizionato']:.5f}  "
              f"incond {r['mae_incondizionato']:.5f}  "
              f"banale {r['mae_banale']:.5f}  "
              f"vicino {r['mae_vicino']:.5f}  "
              f"lineare {r['mae_lineare']:.5f}", flush=True)
        print(f"  SKILL vs banale {r['skill_condizionato']:+.4f}   "
              f"vs vicino {r['skill_vs_vicino']:+.4f}   "
              f"vs lineare {r['skill_vs_lineare']:+.4f}", flush=True)
        print(f"  MMD  cond {r['mmd_condizionato_m']:.5f}  "
              f"banale {r['mmd_banale_m']:.5f}  "
              f"vicino {r['mmd_vicino_m']:.5f}  "
              f"lineare {r['mmd_lineare_m']:.5f}", flush=True)
        print(f"  SKILL MMD vs banale {r['skill_mmd_vs_banale']:+.4f}   "
              f"vs vicino {r['skill_mmd_vs_vicino']:+.4f}   "
              f"vs lineare {r['skill_mmd_vs_lineare']:+.4f}", flush=True)
        print(f"  incond {r['skill_incondizionato']:+.4f}   "
              f"c/z {k['c_su_z_iniziale']:.3f}->{k['c_su_z_finale']:.3f}   "
              f"passo {k['best_step']}/{a.steps}", flush=True)


if __name__ == "__main__":
    main()