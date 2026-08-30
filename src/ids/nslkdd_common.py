"""
Caricamento e codifica di NSL-KDD, condivisi fra selezione delle feature
e preprocessing finale.

REGOLE METODOLOGICHE applicate qui e da non violare altrove:
  - il TEST non viene mai usato per alcuna decisione: solo per i numeri finali
  - scaler, PCA, lista dei servizi frequenti e selezione supervisionata
    sono adattati esclusivamente su dati di training
  - la selezione supervisionata usa il regime di sbilanciamento PIU' SEVERO,
    cosi' da non impiegare mai piu' informazione di quella disponibile
    nel caso peggiore
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SRC = Path("data/nslkdd")

SEED = 42
TOP_SERVICES = 15
VAL_FRACTION = 0.2

# Attacchi della categoria Probe. mscan e saint compaiono solo nel test:
# NSL-KDD include per progetto attacchi assenti dal training.
PROBE = {"satan", "ipsweep", "portsweep", "nmap", "mscan", "saint"}

CATEGORICAL = ["protocol_type", "service", "flag"]


def _read(split):
    cols = (SRC / "columns.txt").read_text().split()
    d = pd.read_csv(SRC / f"KDD{split}+.txt", names=cols)
    return d.drop(columns=["difficulty"])


def _to_binary(d):
    """normal -> 0, probe -> 1. Le altre categorie sono scartate."""
    keep = d["class"].isin(PROBE) | (d["class"] == "normal")
    out = d[keep].copy()
    out["y"] = out["class"].isin(PROBE).astype(int)
    return out.drop(columns=["class"])


def load_splits(seed=SEED, val_fraction=VAL_FRACTION):
    """Restituisce train, validation e test gia' codificati one-hot.

    Il validation e' ritagliato dal training in modo stratificato ed e'
    l'unico insieme su cui si prendono decisioni di modello.
    """
    tr_full = _to_binary(_read("Train"))
    te = _to_binary(_read("Test"))

    tr, va = train_test_split(tr_full, test_size=val_fraction,
                              stratify=tr_full["y"], random_state=seed)

    # i servizi frequenti si scelgono solo sul training vero e proprio
    top = tr["service"].value_counts().head(TOP_SERVICES).index.tolist()

    def encode(d):
        d = d.copy()
        d["service"] = d["service"].where(d["service"].isin(top), "other")
        return pd.get_dummies(d, columns=CATEGORICAL)

    tr_e = encode(tr)
    va_e = encode(va).reindex(columns=tr_e.columns, fill_value=0)
    te_e = encode(te).reindex(columns=tr_e.columns, fill_value=0)

    feat = [c for c in tr_e.columns if c != "y"]
    return {
        "Xtr": tr_e[feat], "ytr": tr_e["y"].reset_index(drop=True),
        "Xva": va_e[feat], "yva": va_e["y"].reset_index(drop=True),
        "Xte": te_e[feat], "yte": te_e["y"].reset_index(drop=True),
        "features": feat, "top_services": top,
    }


def subsample_indices(y, ratio, seed=SEED):
    """Indici per ottenere normal:probe = ratio:1 sottocampionando la
    minoritaria. Restituisce indici posizionali."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    normal = np.flatnonzero(y == 0)
    probe = np.flatnonzero(y == 1)
    n_probe = len(normal) // ratio
    if n_probe > len(probe):
        raise ValueError(f"servono {n_probe} campioni probe, "
                         f"disponibili {len(probe)}")
    keep = rng.choice(probe, size=n_probe, replace=False)
    out = np.concatenate([normal, keep])
    rng.shuffle(out)
    return out