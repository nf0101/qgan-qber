"""Generazione autoregressiva di tracce (SPEC-FASE2 §6, solo dominio QKD).

Una mappatura diretta traccia -> stato richiederebbe n_feature x T qubit.
Il circuito genera invece una finestra alla volta, condizionando su ``c`` e
sulla finestra precedente; i parametri theta sono condivisi fra le
invocazioni.

    c_eff = concat(c, w_{t-1}),  d_c = 3 + 5 = 8,  w_0 = 0,  T = 16

Il modulo non introduce una classe nuova: costruisce il dataset appaiato che
la QGAN standard consuma (teacher forcing) e implementa il campionamento
ricorsivo. La classe QGAN resta quella di §2, senza rami sul dominio.
"""

from __future__ import annotations

import numpy as np

T_DEFAULT = 16


def build_ar_dataset(
    traces: np.ndarray, cond: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Dataset per l'addestramento con teacher forcing.

    Parameters
    ----------
    traces : (n_tracce, T, n_feature), valori in [0, 1]
    cond   : (n_tracce, d_c0), condizionamento statico della traccia
             (es. intercept_rate, length_km, depolar_total normalizzati)

    Returns
    -------
    X     : (n_tracce * T, n_feature)  la finestra da generare
    c_eff : (n_tracce * T, d_c0 + n_feature)  condizionamento efficace,
            concatenazione di ``c`` e della finestra precedente REALE.
    """
    traces = np.asarray(traces, dtype=float)
    cond = np.asarray(cond, dtype=float)
    if traces.ndim != 3:
        raise ValueError("traces deve avere shape (n_tracce, T, n_feature)")
    n, T, F = traces.shape
    if len(cond) != n:
        raise ValueError("cond deve avere una riga per traccia")

    prev = np.concatenate(
        [np.zeros((n, 1, F)), traces[:, :-1, :]], axis=1
    )  # w_0 = 0
    X = traces.reshape(n * T, F)
    c_stat = np.repeat(cond, T, axis=0)
    c_eff = np.hstack([c_stat, prev.reshape(n * T, F)])
    return X, c_eff


def sample_traces(
    qgan, cond: np.ndarray, T: int = T_DEFAULT, batch: bool = True
) -> np.ndarray:
    """Genera tracce ricorsivamente condizionando sulla finestra generata.

    Parameters
    ----------
    qgan : istanza addestrata con il dataset di ``build_ar_dataset``
    cond : (n_tracce, d_c0)
    T    : lunghezza della traccia

    Returns
    -------
    (n_tracce, T, n_feature)
    """
    cond = np.atleast_2d(np.asarray(cond, dtype=float))
    n, F = len(cond), qgan.n_features
    if cond.shape[1] + F != qgan.d_c:
        raise ValueError(
            f"d_c del modello e' {qgan.d_c}, atteso {cond.shape[1]} + {F}"
        )
    out = np.empty((n, T, F))
    prev = np.zeros((n, F))
    for t in range(T):
        # Tutte le tracce avanzano di un passo insieme: n circuiti per passo,
        # non n*T chiamate separate.
        c_eff = np.hstack([cond, prev])
        prev = qgan.sample(n, c_eff)
        out[:, t, :] = prev
    return out


def autocorrelation(traces: np.ndarray, max_lag: int = 8) -> np.ndarray:
    """Autocorrelazione media per feature, lag 1..max_lag.

    E' la misura del divario fra teacher forcing e generazione libera
    indicata in §6: si confronta l'autocorrelazione delle tracce sintetiche
    con quella del simulatore.

    Returns
    -------
    (max_lag, n_feature)
    """
    traces = np.asarray(traces, dtype=float)
    n, T, F = traces.shape
    out = np.zeros((max_lag, F))
    for lag in range(1, max_lag + 1):
        a = traces[:, :-lag, :].reshape(-1, F)
        b = traces[:, lag:, :].reshape(-1, F)
        am, bm = a.mean(0), b.mean(0)
        sa, sb = a.std(0), b.std(0)
        denom = np.where((sa * sb) > 1e-12, sa * sb, np.nan)
        out[lag - 1] = ((a - am) * (b - bm)).mean(0) / denom
    return out


def drift_gap(real_traces: np.ndarray, fake_traces: np.ndarray, max_lag: int = 8) -> dict:
    """Divario di autocorrelazione fra tracce reali e sintetiche."""
    ar = autocorrelation(real_traces, max_lag)
    af = autocorrelation(fake_traces, max_lag)
    d = np.abs(ar - af)
    return {
        "acf_real": ar.tolist(),
        "acf_fake": af.tolist(),
        "mae_acf": float(np.nanmean(d)),
        "mae_acf_lag1": float(np.nanmean(d[0])),
    }
