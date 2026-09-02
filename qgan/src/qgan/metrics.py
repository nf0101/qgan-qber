"""Protocollo di valutazione comune ai due domini (SPEC-FASE2 §9).

Qualita' distribuzionale: MMD, Wasserstein sulle marginali, Kolmogorov-Smirnov,
detectability come AUC-ROC di un classificatore reale-contro-sintetico.
Diagnostica: copertura delle modalita'.

Le metriche non fanno alcuna assunzione sul dominio: prendono due matrici
(n, d) e nient'altro. E' la condizione perche' i risultati delle due parti
siano confrontabili, come richiesto da O1.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------- #
# MMD
# --------------------------------------------------------------------- #
def median_bandwidth(X: np.ndarray, max_n: int = 500, rng=None) -> float:
    """Euristica della mediana per la larghezza di banda del kernel RBF."""
    rng = rng or np.random.default_rng(0)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    iu = np.triu_indices(len(X), k=1)
    med = np.median(d2[iu]) if iu[0].size else 1.0
    return float(np.sqrt(max(med, 1e-12) / 2.0))


def mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> float:
    """MMD^2 non distorta con kernel gaussiano.

    Puo' assumere valori lievemente negativi: e' una proprieta' dello
    stimatore non distorto, non un errore.

    ATTENZIONE. Con ``sigma=None`` la larghezza di banda e' ricavata
    dall'unione dei due campioni, quindi cambia al cambiare di ``Y``. Due
    valori calcolati cosi' NON sono confrontabili fra loro: usano kernel
    diversi. Per monitorare l'addestramento o per confrontare
    configurazioni si deve fissare ``sigma`` una volta sui soli dati reali
    e riusarlo sempre.
    """
    if sigma is None:
        sigma = median_bandwidth(np.vstack([X, Y]))
    g = 1.0 / (2.0 * sigma**2)

    def k(A, B):
        d2 = np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :] - 2 * A @ B.T
        return np.exp(-g * np.maximum(d2, 0.0))

    n, m = len(X), len(Y)
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(
        Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2.0 * Kxy.mean()
    )


# --------------------------------------------------------------------- #
# Marginali
# --------------------------------------------------------------------- #
def marginal_wasserstein(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.array(
        [wasserstein_distance(X[:, j], Y[:, j]) for j in range(X.shape[1])]
    )


def marginal_ks(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = [ks_2samp(X[:, j], Y[:, j]) for j in range(X.shape[1])]
    return (
        np.array([o.statistic for o in out]),
        np.array([o.pvalue for o in out]),
    )


# --------------------------------------------------------------------- #
# Detectability
# --------------------------------------------------------------------- #
def detectability(
    X_real: np.ndarray, X_fake: np.ndarray, seed: int = 0, n_splits: int = 5
) -> float:
    """AUC-ROC di un classificatore addestrato a separare reale da sintetico.

    0,5 = indistinguibili; 1,0 = perfettamente separabili. Valutata in
    cross-validation stratificata per evitare che il valore dipenda da un
    singolo split, cosa rilevante con minoritarie di poche decine di campioni.
    """
    X = np.vstack([X_real, X_fake])
    y = np.concatenate([np.ones(len(X_real)), np.zeros(len(X_fake))])
    n_splits = min(n_splits, int(min(np.bincount(y.astype(int)))))
    if n_splits < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, te in cv.split(X, y):
        clf = RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1
        ).fit(X[tr], y[tr])
        scores.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(scores))


# --------------------------------------------------------------------- #
# Copertura delle modalita'
# --------------------------------------------------------------------- #
def mode_coverage(
    X_real: np.ndarray,
    modes: np.ndarray,
    X_fake: np.ndarray,
    min_share: float | None = None,
) -> dict:
    """Diagnosi di mode collapse per assegnazione al centroide piu' vicino.

    ``modes`` e' l'etichetta di modalita' dei campioni reali: nel dominio IDS
    i quattro attacchi distinti che compongono la classe probe (SPEC §8).

    Restituisce la frazione di modalita' coperte, l'entropia normalizzata
    della distribuzione delle assegnazioni e le quote per modalita'.
    Una modalita' e' coperta se riceve almeno ``min_share`` dei campioni
    generati; il default e' meta' della quota uniforme.
    """
    labels = np.unique(modes)
    K = len(labels)
    if min_share is None:
        min_share = 0.5 / K
    cent = np.array([X_real[modes == l].mean(axis=0) for l in labels])
    d2 = np.sum((X_fake[:, None, :] - cent[None, :, :]) ** 2, axis=-1)
    assign = np.argmin(d2, axis=1)
    share = np.bincount(assign, minlength=K) / len(X_fake)
    p = share[share > 0]
    ent = float(-(p * np.log(p)).sum() / np.log(K)) if K > 1 else 1.0
    return {
        "n_modes": int(K),
        "coverage": float((share >= min_share).mean()),
        "entropy": ent,
        "shares": share.tolist(),
        "labels": labels.tolist(),
    }


# --------------------------------------------------------------------- #
def evaluate(
    X_real: np.ndarray,
    X_fake: np.ndarray,
    modes: np.ndarray | None = None,
    seed: int = 0,
    with_detectability: bool = True,
    sigma: float | None = None,
) -> dict:
    """Blocco completo di qualita' distribuzionale."""
    ks_stat, ks_p = marginal_ks(X_real, X_fake)
    out = {
        "mmd": mmd_rbf(X_real, X_fake, sigma=sigma),
        "wasserstein_mean": float(marginal_wasserstein(X_real, X_fake).mean()),
        "wasserstein_per_feature": marginal_wasserstein(X_real, X_fake).tolist(),
        "ks_mean": float(ks_stat.mean()),
        "ks_per_feature": ks_stat.tolist(),
        "ks_pvalue_min": float(ks_p.min()),
    }
    if with_detectability:
        out["detectability_auc"] = detectability(X_real, X_fake, seed=seed)
    if modes is not None:
        out["mode_coverage"] = mode_coverage(X_real, modes, X_fake)
    return out
