"""Lancia le dieci repliche indipendenti dello split IDS.

Tre fasi, riprendibili: quello che e' già fatto viene saltato, quindi
rilanciare dopo un'interruzione non ripete nulla.

  pre      preprocess_nslkdd.py --seed S --out data/ids_sS, in sequenza
  aug      ids_augmentation.py --dati-dir data/ids_sS, in parallelo
  analisi  analisi_repliche.py sull'insieme

In Python e non in fish perche' la sintassi fish non la posso verificare
prima di dartela, e tre errori di ieri erano esattamente quello. Questo gira
uguale su qualunque shell e sulla macchina di Alfonso.

Un file di uscita per replica, mai condiviso: con l'append in parallelo su un
unico JSONL due righe lunghe possono intrecciarsi, e le righe contengono
elenchi di dispersioni e feature che superano i 4 KB.

    python lancia_repliche.py --dry-run          # cosa farebbe
    python lancia_repliche.py                    # tutto
    python lancia_repliche.py --solo aug --parallele 5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SEMI = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]


def ora(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def controlla_patch(radice: Path) -> list[str]:
    """Le due patch devono essere applicate, o le fasi non hanno i flag."""
    guai = []
    pre = radice / "src" / "ids" / "preprocess_nslkdd.py"
    aug = radice / "src" / "ids" / "ids_augmentation.py"
    for p, marcatori, patch in (
        (pre, ("--seed", "global OUT, SEED"), "patch_preprocess.py"),
        (aug, ("--dati-dir", "global DATA"), "patch_augmentation.py"),
    ):
        if not p.exists():
            guai.append(f"manca {p}")
            continue
        testo = p.read_text(encoding="utf-8")
        assenti = [m for m in marcatori if m not in testo]
        if assenti:
            guai.append(f"{p.name} non e' stato modificato (manca "
                        f"{assenti[0]!r}): esegui  python {patch} {p}")
    return guai


def esegui(cmd, log=None, env=None, radice=None):
    """Avvia un processo. Con `log`, in background redirigendo su file."""
    if log is None:
        return subprocess.run(cmd, cwd=radice, env=env).returncode
    log.parent.mkdir(parents=True, exist_ok=True)
    f = open(log, "w")
    p = subprocess.Popen(cmd, cwd=radice, env=env, stdout=f,
                         stderr=subprocess.STDOUT)
    p._log, p._file = log, f
    return p


def coda(path, n=12):
    try:
        righe = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return "(nessun log)"
    return "\n".join("      " + r for r in righe[-n:])


# =====================================================================
def fase_pre(a, radice, semi):
    print("=" * 70)
    print("FASE 1  pre-processing: una replica per seme, in sequenza")
    print("=" * 70)
    atteso = f"{a.classe}_train_{a.ratio}_{a.pca}.csv"
    fatti, nuovi = [], []
    for s in semi:
        d = radice / "data" / f"ids_s{s}"
        if (d / atteso).exists():
            fatti.append(s)
            continue
        nuovi.append(s)
    if fatti:
        print(f"  gia' presenti: {fatti}")
    if not nuovi:
        print("  niente da generare")
        return 0
    print(f"  da generare: {nuovi}")
    if a.dry_run:
        for s in nuovi:
            print(f"  [dry] python src/ids/preprocess_nslkdd.py --seed {s} "
                  f"--out data/ids_s{s}")
        return 0
    for s in nuovi:
        t0 = time.time()
        print(f"  seme {s} ...", end=" ", flush=True)
        rc = esegui([sys.executable, "src/ids/preprocess_nslkdd.py",
                     "--seed", str(s), "--out", f"data/ids_s{s}"],
                    radice=radice)
        if rc:
            print(f"\n  FALLITO con codice {rc}: mi fermo")
            return rc
        print(f"fatto in {ora(time.time() - t0)}")
    return 0


def fase_verifica(a, radice, semi):
    print()
    print("=" * 70)
    print("FASE 2  verifica che le repliche siano davvero repliche")
    print("=" * 70)
    cmd = [sys.executable, "src/ids/verifica_repliche.py",
           "--semi", ",".join(str(s) for s in semi),
           "--classe", a.classe, "--ratio", a.ratio, "--pca", a.pca]
    if a.dry_run:
        print("  [dry] " + " ".join(cmd[1:]))
        return 0
    return esegui(cmd, radice=radice)


def fase_aug(a, radice, semi):
    print()
    print("=" * 70)
    print(f"FASE 3  augmentation: {a.semi_qgan} semi per replica, "
          f"{a.parallele} in parallelo")
    print("=" * 70)
    usc = radice / a.out
    usc.mkdir(parents=True, exist_ok=True)
    (radice / "logs").mkdir(exist_ok=True)

    da_fare = []
    for s in semi:
        f = usc / f"s{s}.jsonl"
        if f.exists() and sum(1 for _ in open(f)) >= a.semi_qgan:
            continue
        da_fare.append(s)
    pronti = [s for s in semi if s not in da_fare]
    if pronti:
        print(f"  gia' complete: {pronti}")
    if not da_fare:
        print("  niente da eseguire")
        return 0
    print(f"  da eseguire: {da_fare}")

    def comando(s):
        return [sys.executable, "src/ids/ids_augmentation.py",
                "--dati-dir", f"data/ids_s{s}",
                "--classe", a.classe, "--ratio", a.ratio, "--pca", a.pca,
                "--n-finale", str(a.n_finale), "--seeds", str(a.semi_qgan),
                "--steps", str(a.steps), "--backend", a.backend,
                "--out", f"{a.out}/s{s}.jsonl"]

    if a.dry_run:
        for s in da_fare:
            print("  [dry] " + " ".join(comando(s)[1:]))
        print(f"  [dry] {len(da_fare)} corse x {a.semi_qgan} semi = "
              f"{len(da_fare) * a.semi_qgan} addestramenti QGAN")
        return 0

    # Ogni processo usa pochi thread: la parallelizzazione sta fuori, e
    # lasciare a BLAS tutti i core con cinque processi insieme li fa
    # litigare invece di sommarsi.
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[v] = str(a.thread)

    t_inizio = time.time()
    attivi, falliti, finite = [], [], 0
    coda_semi = list(da_fare)
    while coda_semi or attivi:
        while coda_semi and len(attivi) < a.parallele:
            s = coda_semi.pop(0)
            log = radice / "logs" / f"repliche_s{s}.log"
            p = esegui(comando(s), log=log, env=env, radice=radice)
            p._seme, p._t0 = s, time.time()
            attivi.append(p)
            print(f"  avviata replica {s}   (log logs/{log.name})",
                  flush=True)
        time.sleep(2.0)
        for p in list(attivi):
            if p.poll() is None:
                continue
            attivi.remove(p)
            p._file.close()
            finite += 1
            dt = time.time() - p._t0
            if p.returncode:
                falliti.append(p._seme)
                print(f"  replica {p._seme} FALLITA (codice {p.returncode}) "
                      f"dopo {ora(dt)}; ultime righe del log:")
                print(coda(p._log))
            else:
                rimaste = len(coda_semi) + len(attivi)
                medio = (time.time() - t_inizio) / finite
                print(f"  replica {p._seme} finita in {ora(dt)}   "
                      f"restano {rimaste}   stima "
                      f"{ora(medio * rimaste / max(a.parallele, 1))}",
                      flush=True)

    print(f"\n  totale {ora(time.time() - t_inizio)}   "
          f"complete {len(da_fare) - len(falliti)}/{len(da_fare)}")
    if falliti:
        print(f"  FALLITE: {falliti}")
        print("  Rilancia lo stesso comando: le complete vengono saltate.")
        return 1
    return 0


def fase_analisi(a, radice, semi):
    print()
    print("=" * 70)
    print("FASE 4  analisi")
    print("=" * 70)
    rc = 0
    for metrica in a.metriche.split(","):
        cmd = [sys.executable, "src/ids/analisi_repliche.py",
               "--dir", a.out, "--metrica", metrica.strip(),
               "--modello", a.modello]
        if a.dry_run:
            print("  [dry] " + " ".join(cmd[1:]))
            continue
        print()
        print("#" * 70)
        print(f"# metrica {metrica.strip()}")
        print("#" * 70)
        rc = esegui(cmd, radice=radice) or rc
    return rc


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radice", default=".", help="radice del repo")
    ap.add_argument("--semi", default=",".join(str(s) for s in SEMI),
                    help="semi delle repliche del pre-processing")
    ap.add_argument("--classe", default="r2l")
    ap.add_argument("--ratio", default="1_1000")
    ap.add_argument("--pca", default="local")
    ap.add_argument("--n-finale", type=int, default=269)
    ap.add_argument("--semi-qgan", type=int, default=2,
                    help="semi di addestramento PER replica. Due bastano a "
                         "separare la componente fra semi da quella fra "
                         "repliche; le repliche in piu' valgono piu' dei "
                         "semi in piu', perche' l'asse replica e' quello "
                         "che generalizza")
    ap.add_argument("--steps", type=int, default=5400)
    ap.add_argument("--backend", default="denso")
    ap.add_argument("--parallele", type=int, default=5)
    ap.add_argument("--thread", type=int, default=2,
                    help="thread BLAS per processo")
    ap.add_argument("--out", default="results/ids_repliche")
    ap.add_argument("--metriche", default="f1,pr_auc")
    ap.add_argument("--modello", default="rf")
    ap.add_argument("--solo", default="tutto",
                    choices=["tutto", "pre", "verifica", "aug", "analisi"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--forza-senza-verifica", action="store_true",
                    help="prosegue anche se la verifica delle repliche "
                         "fallisce. Da usare solo dopo aver capito perche'")
    a = ap.parse_args()

    radice = Path(a.radice).resolve()
    if not (radice / "src" / "ids").is_dir():
        raise SystemExit(f"{radice} non sembra la radice del repo: "
                         "manca src/ids. Passa --radice.")
    semi = [int(x) for x in a.semi.split(",")]

    guai = controlla_patch(radice)
    if guai:
        print("Le patch non sono applicate:")
        for g in guai:
            print(f"  {g}")
        return 1

    print(f"radice {radice}")
    print(f"{len(semi)} repliche x {a.semi_qgan} semi = "
          f"{len(semi) * a.semi_qgan} corse   "
          f"{a.classe} {a.ratio} {a.pca}   steps {a.steps}")
    libero = shutil.disk_usage(radice).free / 2**30
    print(f"spazio libero {libero:.1f} GiB")
    print()

    if a.solo in ("tutto", "pre"):
        if fase_pre(a, radice, semi):
            return 1
    if a.solo in ("tutto", "verifica", "pre"):
        rc = fase_verifica(a, radice, semi)
        if rc and not a.forza_senza_verifica:
            print()
            print("La verifica non regge: mi fermo prima di spendere "
                  f"{len(semi) * a.semi_qgan} addestramenti.")
            print("Se sai perche' e vuoi procedere: --forza-senza-verifica")
            return 1
    if a.solo in ("tutto", "aug"):
        if fase_aug(a, radice, semi) and not a.dry_run:
            return 1
    if a.solo in ("tutto", "analisi"):
        fase_analisi(a, radice, semi)

    if not a.dry_run and a.solo in ("tutto", "analisi"):
        print()
        print("Da leggere per primo: il RIEPILOGO, colonna 'intervallo 95%'")
        print("incrociato, e la riga 'qgan'. Se contiene lo zero, il divario")
        print("della QGAN non e' distinguibile da zero su nuove repliche; se")
        print("resta tutto negativo, la conclusione passa da 'su questo")
        print("split' a 'su dieci split indipendenti'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())