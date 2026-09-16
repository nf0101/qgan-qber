"""Generazione autoregressiva di tracce (SPEC-FASE2 §6, solo dominio QKD).

Una mappatura diretta traccia -> stato richiederebbe n_feature x T qubit.
Il circuito genera invece una finestra alla volta, condizionando su ``c`` e
sulla finestra precedente; i parametri theta sono condivisi fra le
invocazioni.

    c_eff = concat(c, w_{t-1}),  d_c = 3 + 5 = 8,  w_0 = 0,  T = 16

Lo stato iniziale w_0 va calcolato con ``valore_w0`` e passato allo STESSO
valore a ``build_ar_dataset`` e a ``sample_traces``: sono due fasi distinte
e nulla, nel codice, garantisce da solo che partano dallo stesso stato.

Il modulo non introduce una classe nuova: costruisce il dataset appaiato che
la QGAN standard consuma (teacher forcing) e implementa il campionamento
ricorsivo. La classe QGAN resta quella di §2, senza rami sul dominio.
"""

from __future__ import annotations

import numpy as np

T_DEFAULT = 16


def valore_w0(traces, w0: str, w0_value=None) -> np.ndarray | None:
    """Stato iniziale w_0, calcolato UNA VOLTA per addestramento e generazione.

    Va invocato dal chiamante e il risultato passato sia a
    ``build_ar_dataset`` sia a ``sample_traces``.

    Nella versione precedente ``build_ar_dataset(w0='media')`` calcolava la
    media delle finestre, mentre ``sample_traces(w0='media')`` senza
    ``w0_value`` ripiegava in silenzio su zero. Addestramento e generazione
    libera partivano quindi da stati iniziali diversi, e il divario che ne
    risultava finiva letto come deriva del modello. I risultati in
    ``w0_media_s*.jsonl`` sono prodotti da questo disallineamento.

    Returns
    -------
    ``None`` per 'zero' e 'scarta' (nessun valore da trasportare: la finestra
    iniziale e' di zeri in entrambe le fasi), il vettore medio per 'media'.
    """
    if w0 in ("zero", "scarta"):
        return None
    if w0 != "media":
        raise ValueError("w0: 'zero', 'media' o 'scarta'")
    if w0_value is not None:
        return np.asarray(w0_value, dtype=float)
    traces = np.asarray(traces, dtype=float)
    if traces.ndim != 3:
        raise ValueError("traces deve avere shape (n_tracce, T, n_feature)")
    return traces.reshape(-1, traces.shape[-1]).mean(axis=0)


def build_ar_dataset(
    traces: np.ndarray, cond: np.ndarray, w0: str = "zero",
    w0_value: np.ndarray | None = None,
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

    # Trattamento della prima finestra. §6 prescrive w_0 = 0, ma misurato
    # (FASE2.md §17.2) la MMD al primo passo e' sedici volte peggiore delle
    # successive: il generatore deve produrre una finestra plausibile a partire
    # da un condizionamento che non corrisponde ad alcuno stato reale.
    #
    #   zero    w_0 = 0, come da specifica
    #   media   w_0 = media delle finestre di addestramento: uno stato
    #           plausibile, anche se non specifico della traccia
    #   scarta  la prima finestra non entra nell'addestramento; in generazione
    #           si genera una finestra in piu' e si scarta
    if w0 == "zero":
        first = np.zeros((n, 1, F))
    elif w0 == "media":
        m = valore_w0(traces, "media", w0_value)
        first = np.repeat(np.asarray(m, dtype=float)[None, None, :], n, axis=0)
    elif w0 == "scarta":
        first = None
    else:
        raise ValueError("w0: 'zero', 'media' o 'scarta'")

    if first is None:
        prev = traces[:, :-1, :]
        X = traces[:, 1:, :].reshape(n * (T - 1), F)
        c_stat = np.repeat(cond, T - 1, axis=0)
        return X, np.hstack([c_stat, prev.reshape(n * (T - 1), F)])

    prev = np.concatenate([first, traces[:, :-1, :]], axis=1)
    X = traces.reshape(n * T, F)
    c_stat = np.repeat(cond, T, axis=0)
    c_eff = np.hstack([c_stat, prev.reshape(n * T, F)])
    return X, c_eff


def sample_traces(
    qgan, cond: np.ndarray, T: int = T_DEFAULT, batch: bool = True,
    w0: str = "zero", w0_value: np.ndarray | None = None,
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
    # Deve corrispondere a build_ar_dataset: con `scarta` si genera una
    # finestra in piu' e si butta la prima, che e' l'unica prodotta da uno
    # stato iniziale arbitrario.
    scarta = w0 == "scarta"
    T_gen = T + 1 if scarta else T
    out = np.empty((n, T_gen, F))
    if w0 == "media":
        if w0_value is None:
            raise ValueError(
                "w0='media' richiede w0_value. Calcolarlo una volta con "
                "valore_w0(traces, 'media') e passare lo STESSO valore a "
                "build_ar_dataset e a sample_traces. Il ripiego silenzioso "
                "su zero faceva partire la generazione da uno stato mai "
                "visto in addestramento, e il divario risultante veniva "
                "attribuito al modello."
            )
        prev = np.repeat(np.asarray(w0_value, dtype=float)[None, :], n, axis=0)
    elif w0 in ("zero", "scarta"):
        # Con 'scarta' la prima finestra generata viene buttata, quindi lo
        # stato iniziale non entra nel risultato. Nota pero' che la SECONDA
        # finestra e' condizionata sulla prima, generata a partire da uno
        # stato che non compare in addestramento: lo scarto attenua il
        # problema, non lo elimina.
        prev = np.zeros((n, F))
    else:
        raise ValueError("w0: 'zero', 'media' o 'scarta'")
    for t in range(T_gen):
        # Tutte le tracce avanzano di un passo insieme: n circuiti per passo,
        # non n*T chiamate separate.
        c_eff = np.hstack([cond, prev])
        prev = qgan.sample(n, c_eff)
        out[:, t, :] = prev
    return out[:, 1:, :] if scarta else out


def autocorrelation(
    traces: np.ndarray, max_lag: int = 8, *, mode: str, livello=None
) -> np.ndarray:
    """Autocorrelazione media per feature, lag 1..max_lag.

    ``mode`` e' obbligatorio e senza default: le tre varianti misurano cose
    diverse e mescolarle senza dichiararlo e' esattamente il modo in cui il
    problema e' passato inosservato.

    aggregata
        Coppie temporali concatenate fra TUTTE le tracce, centratura sulla
        media globale. E' la versione precedente, conservata identica per
        confrontabilita' con i numeri gia' riportati. **Non isola la memoria
        temporale**: quando il livello della traccia dipende da ``c``, due
        finestre della stessa traccia condividono quel livello anche se il
        rumore e' indipendente nel tempo, e la varianza fra tracce domina il
        rapporto. Mille tracce con livello casuale costante e rumore
        temporalmente indipendente di deviazione 0,01 danno lag 1 ~ 0,997.

    intra
        Ogni traccia e' centrata sulla PROPRIA media: resta solo la
        covarianza temporale interna. E' la misura che §6 intende.
        Attenzione al bias di centratura su media empirica: con T finito il
        valore atteso in assenza di memoria non e' 0 ma circa -1/(T-1)
        (~-0,067 per T=16). Va letto contro ``autocorrelation_nulla``, non
        contro lo zero.

    residua
        Centratura su un livello fornito dal chiamante (``livello``, shape
        (n_tracce, n_feature) oppure (n_tracce, 1, n_feature)). Sul toy il
        livello condizionale e' noto e questa e' la variante corretta,
        perche' non paga il bias di centratura.

    Returns
    -------
    (max_lag, n_feature)
    """
    traces = np.asarray(traces, dtype=float)
    if traces.ndim != 3:
        raise ValueError("traces deve avere shape (n_tracce, T, n_feature)")
    n, T, F = traces.shape
    if max_lag >= T:
        raise ValueError(f"max_lag={max_lag} non compatibile con T={T}")
    out = np.zeros((max_lag, F))

    if mode == "aggregata":
        for lag in range(1, max_lag + 1):
            a = traces[:, :-lag, :].reshape(-1, F)
            b = traces[:, lag:, :].reshape(-1, F)
            am, bm = a.mean(0), b.mean(0)
            sa, sb = a.std(0), b.std(0)
            denom = np.where((sa * sb) > 1e-12, sa * sb, np.nan)
            out[lag - 1] = ((a - am) * (b - bm)).mean(0) / denom
        return out

    if mode == "intra":
        centro = traces.mean(axis=1, keepdims=True)
    elif mode == "residua":
        if livello is None:
            raise ValueError("mode='residua' richiede livello")
        centro = np.asarray(livello, dtype=float)
        if centro.ndim == 2:
            centro = centro[:, None, :]
        if centro.shape not in ((n, 1, F), (n, T, F)):
            raise ValueError(
                f"livello ha shape {centro.shape}, attesa {(n, 1, F)}"
            )
    else:
        raise ValueError("mode: 'aggregata', 'intra' o 'residua'")

    r = traces - centro
    for lag in range(1, max_lag + 1):
        a = r[:, :-lag, :].reshape(-1, F)
        b = r[:, lag:, :].reshape(-1, F)
        denom = np.sqrt((a**2).mean(0) * (b**2).mean(0))
        denom = np.where(denom > 1e-12, denom, np.nan)
        out[lag - 1] = (a * b).mean(0) / denom
    return out


def autocorrelation_nulla(
    traces: np.ndarray, max_lag: int = 8, *, mode: str, livello=None,
    seed: int = 0, repliche: int = 20,
) -> np.ndarray:
    """Valore della metrica quando la memoria temporale NON c'e'.

    Permuta l'asse del tempo dentro ciascuna traccia: il livello per traccia
    e le marginali restano intatti, la dipendenza temporale sparisce. E' la
    baseline richiesta per leggere l'autocorrelazione misurata, e con
    ``mode='aggregata'`` e' anche la dimostrazione del problema — resta alta
    quanto l'originale ogni volta che il livello dipende da ``c``.
    """
    traces = np.asarray(traces, dtype=float)
    rng = np.random.default_rng(seed)
    acc = []
    for _ in range(repliche):
        p = np.empty_like(traces)
        for i in range(len(traces)):
            p[i] = traces[i][rng.permutation(traces.shape[1])]
        acc.append(autocorrelation(p, max_lag, mode=mode, livello=livello))
    return np.nanmean(np.stack(acc), axis=0)


def drift_gap(
    real_traces: np.ndarray, fake_traces: np.ndarray, max_lag: int = 8, *,
    nulla: bool = True, seed: int = 0,
) -> dict:
    """Divario di autocorrelazione fra tracce reali e sintetiche.

    Riporta entrambe le varianti. Le chiavi senza suffisso restano quelle di
    prima (aggregata), cosi' i risultati gia' salvati restano leggibili; le
    chiavi ``_intra`` sono quelle da usare per concludere sulla dinamica
    temporale, lette contro le baseline ``acf_nulla_*``.
    """
    ris: dict = {}
    for mode, suf in (("aggregata", ""), ("intra", "_intra")):
        ar = autocorrelation(real_traces, max_lag, mode=mode)
        af = autocorrelation(fake_traces, max_lag, mode=mode)
        d = np.abs(ar - af)
        ris[f"acf_real{suf}"] = ar.tolist()
        ris[f"acf_fake{suf}"] = af.tolist()
        ris[f"mae_acf{suf}"] = float(np.nanmean(d))
        ris[f"mae_acf_lag1{suf}"] = float(np.nanmean(d[0]))
    if nulla:
        for mode, suf in (("aggregata", ""), ("intra", "_intra")):
            ris[f"acf_nulla_real{suf}"] = autocorrelation_nulla(
                real_traces, max_lag, mode=mode, seed=seed).tolist()
            ris[f"acf_nulla_fake{suf}"] = autocorrelation_nulla(
                fake_traces, max_lag, mode=mode, seed=seed).tolist()
    return ris