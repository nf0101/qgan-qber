"""Identita' delle corse, provenienza dei risultati e flussi RNG per ruolo.

Modulo condiviso dagli script sperimentali. Risponde a due difetti distinti.

FLUSSI RNG (difetto 3). Con un flusso unico per corsa, lo stato consumato da
una fase dipende dalle dimensioni della precedente: cambiare la numerosita'
del training spostava anche le celle escluse, l'insieme di validazione e i
campioni di valutazione. La curva rispetto a n confrontava quindi condizioni
con test diversi, in un progetto che identifica proprio la scelta delle celle
come fonte dominante di variabilita'. ``flussi`` separa i ruoli; il consumo
dell'uno non sposta piu' l'altro.

IDENTITA' E PROVENIENZA (difetto 9). Gli script accodavano sempre, senza
controllare duplicati ne' omogeneita' della configurazione: in
``n_P8_s0.jsonl``, ``n_P32_s0.jsonl`` e ``n_P128_s0.jsonl`` sono finite righe
esattamente identiche, che nell'analisi gonfiano la numerosita' e
sottostimano gli errori standard. Qui ogni riga porta un identificatore di
corsa, la firma della configurazione, la versione del codice e l'istante di
esecuzione, e ``gia_presenti`` permette di rifiutare un riaccodamento.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid

import numpy as np


# ---------------------------------------------------------------- flussi RNG
def flussi(base: int, seed: int, n: int = 3) -> list[np.random.Generator]:
    """``n`` generatori indipendenti per la corsa ``seed``.

    Gli stream derivano da una SeedSequence e non da hash di stringhe: sono
    stabili fra processi, a differenza di ``hash((seed, tag, idx))``, che
    dipende dalla randomizzazione dell'hash di Python e cambia a ogni
    avvio dell'interprete.
    """
    ss = np.random.SeedSequence([int(base), int(seed)])
    return [np.random.default_rng(s) for s in ss.spawn(int(n))]


def flusso_fisso(base: int, chiave: int) -> np.random.Generator:
    """Flusso deterministico indicizzato da ``chiave``, indipendente dal seme.

    Serve per gli insiemi che devono restare IDENTICI fra condizioni: il test
    di una cella e' una proprieta' della cella, non della corsa che la valuta.
    """
    return np.random.default_rng(np.random.SeedSequence([int(base), int(chiave)]))


# ------------------------------------------------------- identita' e origine
def versione_codice() -> str | None:
    """Revisione git corrente, con marcatore se il working tree e' sporco."""
    qui = os.path.dirname(os.path.abspath(__file__))
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=qui, stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        sporco = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=qui, stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        return rev + ("+modificato" if sporco else "")
    except Exception:
        return None


def firma(config: dict) -> str:
    """Identificatore della CONFIGURAZIONE, stabile fra processi.

    Due corse con la stessa firma e lo stesso seme dovrebbero produrre lo
    stesso risultato: e' il criterio con cui si riconosce un duplicato.
    """
    testo = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(testo.encode("utf-8")).hexdigest()[:12]


def intestazione(config: dict) -> dict:
    """Campi di provenienza da unire a ogni riga di risultato."""
    return {
        "id_corsa": uuid.uuid4().hex[:12],
        "firma_config": firma(config),
        "config": config,
        "versione_codice": versione_codice(),
        "istante": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def gia_presenti(path: str) -> set[tuple]:
    """Coppie (firma_config, seme) gia' registrate nel file dei risultati."""
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path) as f:
        for riga in f:
            riga = riga.strip()
            if not riga:
                continue
            try:
                r = json.loads(riga)
            except json.JSONDecodeError:
                continue
            if "firma_config" in r:
                out.add((r["firma_config"], r.get("seed")))
    return out


def duplicati(path: str) -> int:
    """Numero di righe esattamente identiche a una precedente."""
    if not os.path.exists(path):
        return 0
    viste, n = set(), 0
    with open(path) as f:
        for riga in f:
            riga = riga.strip()
            if not riga:
                continue
            if riga in viste:
                n += 1
            viste.add(riga)
    return n


def scrivi(path: str, riga: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(riga, default=float) + "\n")