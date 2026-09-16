"""Perche' il condizionamento e' sommerso, e si puo' rimediare con un parametro?

La validazione di RQ2 su dati sintetici da' skill −2,10 ± 0,97, con tutti e tre
i semi sotto il riferimento banale. La causa misurata e' che l'uscita risponde
a `c` circa dieci volte meno di quanto risponda al latente `z`: il
discriminatore riceve `c` in chiaro e verifica la coerenza con `x`, mentre il
generatore ha un canale troppo debole per rispondere.

Due ipotesi sul meccanismo sono gia' state formulate e SMENTITE:

  - condizionamento con RY anziche' RZ, perche' RY agisce direttamente sulle
    popolazioni: 0,038 contro 0,052, nessun miglioramento;
  - cancellazione fra i due strati di data re-uploading: uno strato da' 0,0303,
    due danno 0,0283, differenza trascurabile.

Restano da provare l'ampiezza della rotazione e la scala del latente. Se la
sensibilita' cresce con l'ampiezza, la correzione e' di un parametro e tocca
solo §5 della specifica. Se non cresce, il limite e' nella lettura a
probabilita' marginale e la conversazione da fare e' diversa.

Lo script misura, non decide. L'esito va portato ad Alfonso e al relatore
prima di modificare la specifica.

    python scripts/conditioning_sweep.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qgan.circuit import QuantumGenerator  # noqa: E402


def measure(d_c, cond_scale, latent_hi, theta_amp, n=400, seed=0):
    rng = np.random.default_rng(seed)
    g = QuantumGenerator(d_c=d_c, cond_scale=cond_scale)
    th = rng.uniform(-theta_amp, theta_amp, g.n_params)
    Z = rng.uniform(0.0, latent_hi, (n, g.n_qubits))
    lo = g.forward(th, Z, np.zeros((n, d_c)))
    hi = g.forward(th, Z, np.ones((n, d_c)))
    sens = np.abs(hi.mean(0) - lo.mean(0))
    disp = lo.std(0)
    return {
        "cond_scale": cond_scale,
        "latent_hi": latent_hi,
        "theta_amp": theta_amp,
        "sensibilita": float(sens.mean()),
        "sensibilita_min": float(sens.min()),
        "dispersione": float(disp.mean()),
        "rapporto": float(sens.mean() / max(disp.mean(), 1e-12)),
    }


def table(rows, titolo, chiave):
    print(f"\n{titolo}")
    print(f"{chiave:>12s} {'sensibilita':>12s} {'min':>9s} "
          f"{'dispersione':>12s} {'rapporto':>9s}")
    print("-" * 58)
    for r in rows:
        print(f"{r[chiave]:12.3f} {r['sensibilita']:12.4f} "
              f"{r['sensibilita_min']:9.4f} {r['dispersione']:12.4f} "
              f"{r['rapporto']:9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-c", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/conditioning_sweep.json")
    a = ap.parse_args()

    def avg(**kw):
        rs = [measure(a.d_c, seed=s, **kw) for s in range(a.seeds)]
        out = dict(rs[0])
        for k in ("sensibilita", "sensibilita_min", "dispersione", "rapporto"):
            out[k] = float(np.mean([r[k] for r in rs]))
        return out

    PI = np.pi
    amp = [avg(cond_scale=c, latent_hi=PI, theta_amp=0.1)
           for c in (PI / 2, PI, 2 * PI, 4 * PI, 8 * PI)]
    table(amp, "Ampiezza del condizionamento (theta +-0,1, latente in [0,pi])",
          "cond_scale")

    lat = [avg(cond_scale=PI, latent_hi=h, theta_amp=0.1)
           for h in (PI / 4, PI / 2, PI, 2 * PI)]
    table(lat, "Scala del latente (cond_scale = pi, theta +-0,1)", "latent_hi")

    th = [avg(cond_scale=PI, latent_hi=PI, theta_amp=t)
          for t in (0.1, 0.3, 0.5, 1.0, 1.5)]
    table(th, "Ampiezza di theta (cond_scale = pi, latente in [0,pi])",
          "theta_amp")

    best = max(amp + lat + th, key=lambda r: r["rapporto"])
    print(f"\nrapporto migliore osservato: {best['rapporto']:.3f} "
          f"(cond_scale {best['cond_scale']:.3f}, latente {best['latent_hi']:.3f}, "
          f"theta {best['theta_amp']:.2f})")
    print("Riferimento: alla configurazione della specifica il rapporto vale "
          "circa 0,09.")
    if best["rapporto"] < 0.25:
        print("\nNessuna combinazione porta il condizionamento a una frazione")
        print("apprezzabile della variabilita' del latente. Il limite non e'")
        print("un parametro da tarare: e' nella lettura a probabilita'")
        print("marginale, che comprime cinque qubit in cinque numeri e lascia")
        print("poco spazio a un secondo ingresso. Va discusso con il relatore.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"ampiezza": amp, "latente": lat, "theta": th},
              open(a.out, "w"), indent=2, default=float)
    print(f"\nscritto {a.out}")


if __name__ == "__main__":
    main()