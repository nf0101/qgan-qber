"""Dice sotto quale protocollo e' stata prodotta ogni corsa in results/.

Serve a classificare senza indovinare. Per ogni voce stampa quello che
identifica il protocollo: numero di righe, varianti presenti, metriche
registrate, e i campi di `config` che contano (rapporto, pca, passi,
profondita', raggio del latente, directory dei dati, fold della soglia).

Tre cose si leggono a vista:
  - se fra le metriche NON c'e' f1_soglia, la corsa e' a soglia fissa 0.5;
  - se non c'e' f1_test_soglia_val, non ha mai visto il test;
  - se fra le varianti non c'e' gaussiana_cov, non ha il controllo che
    distingue il contributo del circuito da quello dei momenti.

    python ispeziona_risultati.py
    python ispeziona_risultati.py --filtro 'ids_ord*'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RADICE = Path(__file__).resolve().parent
CHIAVI = ["classe", "ratio", "pca", "n_finale", "steps", "reps",
          "latent_hi", "ordine", "dati", "soglia_fold"]
SPIE = ["f1", "f1_soglia", "f1_test", "f1_test_soglia_val",
        "f1_test_oracolo", "pr_auc", "pr_auc_test"]


def righe_jsonl(p: Path):
    out = []
    try:
        for riga in p.open(encoding="utf-8", errors="replace"):
            riga = riga.strip()
            if riga:
                try:
                    out.append(json.loads(riga))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return out


def descrivi(nome: str, righe: list) -> None:
    if not righe:
        print(f"  {nome:28s} nessuna riga JSON leggibile")
        return
    r = righe[0]
    var = sorted((r.get("varianti") or {}))
    conf = r.get("config") or {}
    modelli = sorted(k for k, v in
                     ((r.get("varianti") or {}).get(var[0], {}) if var
                      else {}).items() if isinstance(v, dict))
    metriche = sorted((r["varianti"][var[0]][modelli[0]])
                      if var and modelli else [])

    print(f"  {nome:28s} {len(righe):3d} righe   "
          f"semi {sorted({x.get('seed') for x in righe})}")
    if conf:
        pezzi = [f"{k}={conf[k]}" for k in CHIAVI if k in conf]
        print(f"    config   {'  '.join(pezzi) if pezzi else '(vuota)'}")
    else:
        print("    config   ASSENTE — riga senza provenienza (difetto 9)")
    print(f"    varianti {', '.join(var) if var else '(nessuna)'}")
    if modelli:
        print(f"    modelli  {', '.join(modelli)}")
    presenti = [s for s in SPIE if s in metriche]
    print(f"    metriche {', '.join(presenti) if presenti else sorted(metriche)}")

    giudizi = []
    if "f1_soglia" not in metriche:
        giudizi.append("soglia fissa 0.5")
    if "f1_test_soglia_val" not in metriche:
        giudizi.append("mai valutata sul test")
    if "gaussiana_cov" not in var:
        giudizi.append("senza controllo gaussiano")
    if "dati" not in conf:
        giudizi.append("split unico (nessun campo dati)")
    if giudizi:
        print(f"    PROTOCOLLO: {'; '.join(giudizi)}")
    else:
        print("    PROTOCOLLO: finale")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtro", default="ids_*",
                    help="glob sui nomi dentro results/")
    ap.add_argument("--risultati", default=None)
    a = ap.parse_args()

    base = Path(a.risultati) if a.risultati else RADICE / "results"
    if not base.is_dir():
        raise SystemExit(f"non trovo {base}")
    print(f"repo {RADICE}\n")

    voci = sorted(base.glob(a.filtro))
    if not voci:
        raise SystemExit(f"nessuna voce corrisponde a {a.filtro!r}")

    for v in voci:
        if v.is_dir():
            file = sorted(v.glob("*.jsonl"))
            righe = [r for f in file for r in righe_jsonl(f)]
            print(f"{v.name}/   ({len(file)} file)")
            descrivi(v.name, righe)
        elif v.suffix == ".jsonl":
            print(f"{v.name}")
            descrivi(v.name, righe_jsonl(v))
        else:
            print(f"{v.name}   ({v.suffix or 'senza estensione'}, "
                  f"{v.stat().st_size} byte) — non e' un JSONL, non lo apro")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())