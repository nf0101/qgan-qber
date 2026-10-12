"""Augmentation della minoritaria IDS con la QGAN — Fase 3, dominio IDS.

E' il caso che giustifica un modello generativo. Su QKD, con una griglia
regolare a tre assi, un'interpolazione lineare sulle configurazioni di
addestramento batte la QGAN di un fattore 14 sul MAE delle medie e 40 sulla
MMD: con vicini fitti la distribuzione condizionale varia poco e non resta
spazio. Qui i vicini non ci sono. La minoritaria r2l a 1:1000 ha 53 campioni
in cinque dimensioni, non condizionati: "prendi il vicino e trasla" non e'
nemmeno definibile.

LA DOMANDA. Aggiungere campioni sintetici al solo addestramento migliora il
classificatore a valle? Misurata su validation, come la baseline di Fase 1;
il test resta intatto.

I RIFERIMENTI, tutti alla STESSA numerosita' finale

  nessuna        53 campioni, class_weight assente. Il pavimento.
  bilanciato     53 campioni, class_weight="balanced". E' l'augmentation
                 implicita, e costa zero: va battuta.
  duplicazione   ricampionamento con reimmissione dei 53. Se la QGAN non
                 batte la duplicazione non sta aggiungendo informazione,
                 sta solo ripesando.
  rumore         duplicazione piu' rumore gaussiano. Il minimo sindacale di
                 un metodo generativo.
  smote          interpolazione fra vicini nella minoritaria. E' lo standard
                 per questo problema ed e' l'avversario serio.
  qgan           campioni dal generatore quantistico, d_c=0 come in O1.

IL SOFFITTO. Portare la minoritaria da 53 a 269 campioni la riporta al
livello 1:200, che in `results/baseline.csv` e' gia' misurato con campioni
VERI. Il confronto dice quindi non solo se l'augmentation aiuta, ma quanto
dista dall'avere davvero quei dati.

AVVERTENZA. Le cinque feature sono componenti PCA che spiegano il 34% della
varianza (data/ids/meta.json). L'augmentation opera in uno spazio che perde
due terzi dell'informazione: e' il vincolo dei cinque qubit di SPEC §12, e
va dichiarato insieme al risultato.

    python src/ids/ids_augmentation.py --seeds 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, str(QUI.parent.parent / "qgan" / "scripts"))
sys.path.insert(0, str(QUI.parent.parent / "qgan" / "src"))
import corsa  # noqa: E402
from qgan import QGAN, metrics  # noqa: E402
from qgan.metrics import median_bandwidth  # noqa: E402

DATA = QUI.parent.parent / "data" / "ids"

# Raggio del latente CALIBRATO SULLA MINORITARIA IDS. B.5 prescrive pi per il
# ramo non condizionato, ma quel valore e' corretto per il dominio QKD e non
# era mai stato misurato qui: le cinque componenti PCA della minoritaria r2l
# hanno dispersione [0.128, 0.113, 0.042, 0.028, 0.013], media 0.0647, mentre
# a raggio pi il generatore all'inizializzazione ne produce 0.331 — cinque
# volte troppo. Con pi la MMD dei sintetici contro i 53 reali vale 0.47
# contro il ~0 di duplicazione, rumore e SMOTE.
#
# Il raggio e' quindi un parametro PER DOMINIO, ed e' un elemento di non
# trasferibilita' di O1 che si aggiunge alla larghezza del discriminatore.
LATENT_HI_IDS = 0.9327      # pi/3.37, eguaglia 0.0647
# Per feature, dalla stessa curva. La disparita' fra componenti e' 9.8x,
# contro i 3x del dominio QKD dove i raggi per feature si sono rivelati
# inutili: qui potrebbero servire, ed e' da misurare end-to-end.
LATENT_HI_IDS_FEATURE = [1.3989, 1.2907, 0.7322, 0.5930, 0.3959]

# ORDINE DELLE FEATURE SUI QUBIT. L'accoppiamento qubit-latente misurato e'
# q0 +1.00  q1 +0.99  q2 +0.99  q3 +1.00  q4 +0.21: il qubit 4 e' il meno
# pilotabile, e il criterio di latent_radius_qkd.py prescrive di assegnargli
# l'osservabile che richiede PIU' dispersione, perche' li' il raggio non la
# comprime comunque.
#
# L'ordine PCA naturale fa l'opposto: mette su q4 la componente z5, che
# richiede la dispersione MINIMA (0.013). Misurato all'inizializzazione, a
# reps=4 l'ordine naturale raggiunge una disparita' di 4.0x contro le 9.8x
# richieste, con errore massimo 3.11x sulla feature piu' stretta — ed e'
# esattamente il 3.65x osservato sui campioni generati. Con l'ordine per
# dispersione crescente la stessa architettura riproduce il bersaglio
# esattamente, errore 1.00x, senza toccare reps ne' altro di SPEC §12.
#
# B.6 decide "nessuna permutazione" ed e' corretto per QKD, dove la
# disparita' e' 3x. Qui non lo e': l'ordine e' un parametro PER DOMINIO.
ORDINI = ["naturale", "criterio"]

# ABLAZIONE SU reps. SPEC §12 congela reps=4; questa e' una misura, non una
# modifica. Il raggio calibrato dipende da reps — la curva dispersione-raggio
# cambia con la profondita' dell'ansatz — quindi ogni valore ha il proprio,
# tarato sulla stessa dispersione media dei 53 reali (0.0647). Senza questo,
# il confronto fra reps confonderebbe profondita' e calibrazione.
RAGGIO_PER_REPS = {4: 0.9300, 6: 0.8939, 8: 0.9738, 10: 0.8860, 12: 0.9527}


def permuta(nome, Xmin):
    """Indici delle colonne nell'ordine scelto, e l'inversa per tornare.

    'criterio': dispersione crescente, cosi' la feature piu' larga finisce
    sul qubit 4, il meno pilotabile.
    """
    if nome == "naturale":
        p = np.arange(Xmin.shape[1])
    elif nome == "criterio":
        p = np.argsort(Xmin.std(axis=0))
    else:
        raise ValueError(nome)
    inv = np.argsort(p)
    return p, inv

# Fold della cross-validation con cui si sceglie la soglia fuori campione.
# Zero disattiva il calcolo: la soglia resta 0.5 e `f1_soglia` non c'e'.
SOGLIA_FOLD = 3

FROZEN = dict(
    lr_g=0.002, lr_d=0.002, hidden=(32, 16), batch_size=16,
    eval_every=25, patience=20, mmd_smooth=3, val_fraction=0.0,
    n_monitor=256, monitor_repeats=1,
)
# Le tre ultime sono ABLAZIONI CAUSALI, non metodi proponibili. Verso
# atteso dichiarato prima della misura; fra parentesi l'esito su dieci
# repliche indipendenti dello split, differenza di F1 e intervallo
# incrociato al 95%.
#   smote_distorta  campioni SMOTE con applicati i rapporti di dispersione
#                   della QGAN. Atteso NEGATIVO: se l'F1 crolla, la
#                   distorsione delle dispersioni fa danno.
#                   (-0.129 [-0.160,-0.097], 0/10 — CONFERMATO)
#   qgan_sd         campioni QGAN con le dispersioni per feature riportate a
#                   quelle reali, media invariata. Atteso POSITIVO: se l'F1
#                   recupera, quella distorsione e' il difetto vincolante.
#                   (-0.008 [-0.027,+0.011], 2/10 — NON confermato)
#   qgan_cov        campioni QGAN portati su media E covarianza reali. Le
#                   prime due dicono che la dispersione per feature e'
#                   sufficiente a fare danno ma non e' cio' che frena la
#                   QGAN; `riscala` non tocca le correlazioni, quindi
#                   restava quella strada aperta. Atteso POSITIVO.
# Non formano una dimostrazione: la prima regge, la seconda no. Dimostrano
# che distorcere le dispersioni e' sufficiente a fare danno, NON che
# spieghi il divario della QGAN.
# Le tre ultime sono IN CODA di proposito: duplicazione, rumore e smote
# consumano `rng` in quest'ordine e qgan no (usa il proprio seme), quindi
# aggiungere dopo qgan lascia le precedenti bit-identiche fra corse.
#   gaussiana      N(mu, Sigma) dei 53 reali: il controllo di qgan_cov, ha
#                  gli stessi momenti e nessun contributo generativo.
#   rumore_lieve   come rumore ma a 0.03x la dispersione media (0.10x)
#   rumore_forte   come rumore ma a 0.30x
SINTETICHE = ["duplicazione", "rumore", "smote", "qgan",
              "gaussiana", "rumore_lieve", "rumore_forte"]
SCALE_RUMORE = {"rumore": 0.10, "rumore_lieve": 0.03, "rumore_forte": 0.30}
DERIVATE = ["qgan_sd", "smote_distorta", "qgan_cov", "gaussiana_cov"]
VARIANTI = (["nessuna", "bilanciato"] + SINTETICHE + DERIVATE)


def riscala(S, Xmin, rapporti=None):
    """Porta le dispersioni per feature a quelle reali, o le distorce.

    `rapporti=None` riporta ogni feature alla deviazione standard reale;
    altrimenti al suo multiplo. La media dei sintetici resta invariata: si
    manipola SOLO la dispersione. Il ritaglio in [0,1] puo' alterare
    leggermente il risultato, quindi i rapporti ottenuti vengono riportati.
    """
    mu, sd_s = S.mean(axis=0), S.std(axis=0)
    obiettivo = Xmin.std(axis=0) * (1.0 if rapporti is None else rapporti)
    fatt = np.where(sd_s > 1e-12, obiettivo / np.maximum(sd_s, 1e-12), 1.0)
    return np.clip(mu + (S - mu) * fatt, 0.0, 1.0)


def ricolora(S, Xmin):
    """Mappa affine che porta S su media E covarianza di Xmin.

    `riscala` agisce su una dispersione per volta e lascia le correlazioni
    fra feature come sono. Qui si sbianca S con la propria covarianza e si
    ricolora con quella reale:

        Z = (S - mu_S) C_S^(-1/2),      S' = Z C_X^(1/2) + mu_X

    Le radici sono simmetriche, per decomposizione agli autovalori con
    ritaglio di quelli non positivi: la covarianza reale con 53 campioni in
    5 dimensioni e' stimabile, ma quella dei sintetici puo' essere quasi
    singolare se il generatore collassa su una direzione, e in quel caso
    l'inversa va regolarizzata invece di esplodere.

    Il ritaglio finale in [0,1] perturba un poco la covarianza ottenuta:
    `errore_cov` nei risultati dice quanto.
    """
    def radice(C, inversa=False):
        C = np.atleast_2d(C)
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-12, None)
        r = np.sqrt(1.0 / w) if inversa else np.sqrt(w)
        return (V * r) @ V.T

    mu_s, mu_x = S.mean(axis=0), Xmin.mean(axis=0)
    Z = (S - mu_s) @ radice(np.cov(S, rowvar=False), inversa=True)
    return np.clip(Z @ radice(np.cov(Xmin, rowvar=False)) + mu_x, 0.0, 1.0)


def scarto_cov(S, Xmin):
    """Scarto relativo di covarianza, in norma di Frobenius."""
    Cx = np.atleast_2d(np.cov(Xmin, rowvar=False))
    Cs = np.atleast_2d(np.cov(S, rowvar=False))
    return float(np.linalg.norm(Cs - Cx) / max(np.linalg.norm(Cx), 1e-12))


def carica(cls, ratio, mode):
    """Training, validation e TEST.

    Il suffisso del rapporto su val e test riguarda la sola PCA: lo
    squilibrio e' applicato al training. I tre file val (1_200, 1_500,
    1_1000) hanno la stessa composizione, 199 positivi su 13.668; il test ne
    ha 2.754 su 12.465. I prior sono quindi 1:1000 in addestramento, 1:69 in
    validazione e 1:5 nel test, ed e' questa distanza che rende la
    collocazione della soglia la variabile dominante.
    """
    tr = pd.read_csv(DATA / f"{cls}_train_{ratio}_{mode}.csv")
    va = (pd.read_csv(DATA / f"{cls}_val.csv") if mode == "global"
          else pd.read_csv(DATA / f"{cls}_val_{ratio}_local.csv"))
    te = (pd.read_csv(DATA / f"{cls}_test.csv") if mode == "global"
          else pd.read_csv(DATA / f"{cls}_test_{ratio}_local.csv"))
    feat = [c for c in tr.columns if c != "y"]
    return (tr[feat].to_numpy(float), tr["y"].to_numpy(int),
            va[feat].to_numpy(float), va["y"].to_numpy(int),
            te[feat].to_numpy(float), te["y"].to_numpy(int), feat)


def smote(X, n, rng, k=5):
    """Interpolazione fra un campione e uno dei suoi k vicini."""
    k = min(k, len(X) - 1)
    _, idx = cKDTree(X).query(X, k=k + 1)      # la prima colonna e' se stesso
    i = rng.integers(0, len(X), n)
    j = idx[i, rng.integers(1, k + 1, n)]
    lam = rng.random((n, 1))
    return X[i] + lam * (X[j] - X[i])


def modello(nome, seed, bilanciato):
    if nome == "rf":
        return RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=1,
            class_weight="balanced" if bilanciato else None)
    if nome == "lr":
        return LogisticRegression(
            max_iter=2000, random_state=seed,
            class_weight="balanced" if bilanciato else None)
    if nome == "gb":
        return HistGradientBoostingClassifier(
            random_state=seed,
            class_weight="balanced" if bilanciato else None)
    raise ValueError(nome)


def soglia_fuori_campione(nome, Xtr, ytr, seed, bilanciato, fold):
    """Soglia che massimizza l'F1 su previsioni FUORI CAMPIONE del training.

    Scegliere la soglia sullo stesso insieme su cui si riporta l'F1 e'
    circolare: userebbe le etichette di valutazione. Qui le probabilita'
    vengono da cross_val_predict sul solo training, quindi ogni punto e'
    predetto da un modello che non l'ha visto, e il set di valutazione
    resta intatto. Restituisce la soglia e l'F1 fuori campione che le
    corrisponde — il secondo serve a capire se la soglia e' stata scelta
    su un segnale o sul rumore.
    """
    cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    p = cross_val_predict(modello(nome, seed, bilanciato), Xtr, ytr,
                          cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    prec, rec, sog = precision_recall_curve(ytr, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    # prec e rec hanno un elemento in piu' delle soglie: l'ultimo punto
    # (recall 0, precision 1) non corrisponde a nessuna soglia.
    k = int(np.nanargmax(f1[:-1]))
    return float(sog[k]), float(f1[k])


def soglia_massima(y, p):
    """Soglia che massimizza l'F1 su (y, p), e l'F1 che le corrisponde."""
    prec, rec, sog = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    k = int(np.nanargmax(f1[:-1]))
    return float(sog[k]), float(f1[k])


def valuta(nome, Xtr, ytr, Xva, yva, Xte, yte, seed, bilanciato):
    m = modello(nome, seed, bilanciato)
    m.fit(Xtr, ytr)
    pred = m.predict(Xva)
    proba = m.predict_proba(Xva)[:, 1]
    out = {
        "f1": float(f1_score(yva, pred, zero_division=0)),
        "precision": float(precision_score(yva, pred, zero_division=0)),
        "recall": float(recall_score(yva, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(yva, proba)),
        "roc_auc": float(roc_auc_score(yva, proba)),
    }
    if SOGLIA_FOLD:
        s, f1_oof = soglia_fuori_campione(nome, Xtr, ytr, seed, bilanciato,
                                          SOGLIA_FOLD)
        out["soglia"] = s
        out["f1_soglia_oof"] = f1_oof
        out["f1_soglia"] = float(
            f1_score(yva, (proba >= s).astype(int), zero_division=0))
        out["precision_soglia"] = float(
            precision_score(yva, (proba >= s).astype(int), zero_division=0))
        out["recall_soglia"] = float(
            recall_score(yva, (proba >= s).astype(int), zero_division=0))

    # CONFERMA sul test mai usato per nessuna scelta. La soglia viene dal
    # validation, la misura dal test: due insiemi distinti, e nessuno dei
    # due e' quello su cui il modello e' stato addestrato. E' il numero da
    # riportare in tesi.
    p_te = m.predict_proba(Xte)[:, 1]
    s_val, f1_val_max = soglia_massima(yva, proba)
    pred_te = (p_te >= s_val).astype(int)
    out["soglia_val"] = s_val
    out["f1_val_massimo"] = f1_val_max
    out["f1_test"] = float(f1_score(yte, (p_te >= 0.5).astype(int),
                                    zero_division=0))
    out["f1_test_soglia_val"] = float(f1_score(yte, pred_te, zero_division=0))
    out["precision_test_soglia_val"] = float(
        precision_score(yte, pred_te, zero_division=0))
    out["recall_test_soglia_val"] = float(
        recall_score(yte, pred_te, zero_division=0))
    out["pr_auc_test"] = float(average_precision_score(yte, p_te))
    out["roc_auc_test"] = float(roc_auc_score(yte, p_te))

    # DIAGNOSTICO, mai riportabile come prestazione di un metodo: la soglia
    # qui e' scelta guardando le etichette DEL TEST, quindi il valore e'
    # ottimistico per costruzione. Separa due cose che f1_test_soglia_val
    # confonde: la qualita' della curva precisione-richiamo, cioe' quanto il
    # classificatore ha imparato, e il trasferimento della soglia, cioe'
    # quanto bene una soglia scelta altrove si colloca qui. La differenza
    #     f1_test_oracolo - f1_test_soglia_val
    # e' la perdita da trasferimento, ed e' la quantita' su cui poggia tutta
    # la lettura del capitolo. Se e' quasi uguale fra le varianti, i metodi
    # differiscono per cio' che il classificatore impara; se varia molto,
    # differiscono per dove finisce l'operating point.
    out["f1_test_oracolo"] = soglia_massima(yte, p_te)[1]
    return out


def genera(variante, Xmin, n, rng, seed, steps, backend, verbose,
           latent_hi=LATENT_HI_IDS, ordine="naturale", reps=4):
    """Campioni sintetici della minoritaria, uno per variante."""
    if variante == "duplicazione":
        return Xmin[rng.integers(0, len(Xmin), n)], None
    if variante in SCALE_RUMORE:
        base = Xmin[rng.integers(0, len(Xmin), n)]
        sd = float(Xmin.std(axis=0).mean()) * SCALE_RUMORE[variante]
        return np.clip(base + rng.normal(0, sd, base.shape), 0.0, 1.0), None
    if variante == "gaussiana":
        # Il modello parametrico piu' semplice che usa i 53 punti: media e
        # covarianza campionarie, nient'altro. Sigma e' semidefinita
        # positiva per costruzione, ma con 53 campioni in 5 dimensioni puo'
        # essere mal condizionata, quindi si simmetrizza esplicitamente.
        C = np.atleast_2d(np.cov(Xmin, rowvar=False))
        C = 0.5 * (C + C.T)
        S = rng.multivariate_normal(Xmin.mean(axis=0), C, size=n)
        return np.clip(S, 0.0, 1.0), None
    if variante == "smote":
        return np.clip(smote(Xmin, n, rng), 0.0, 1.0), None
    if variante == "qgan":
        # d_c=0: la minoritaria non e' condizionata, com'e' il dominio ids
        # in O1. Con 53 righe non esiste uno split di validazione: il
        # criterio di arresto usa gli stessi dati, e il modello lo segnala.
        # Si addestra sulle colonne PERMUTATE e si rimettono in ordine i
        # campioni generati: il classificatore a valle non si accorge di
        # nulla, cambia solo quale feature sta su quale qubit.
        p, inv = permuta(ordine, Xmin)
        # I raggi per feature sono indicizzati sulle COLONNE, i qubit sulle
        # colonne PERMUTATE: vanno permutati insieme, altrimenti ogni qubit
        # riceve il raggio di un'altra feature. Un raggio scalare non ne
        # risente.
        lh = latent_hi
        if not np.isscalar(lh):
            lh = [float(np.asarray(lh, float)[j]) for j in p]
        q = QGAN(d_c=0, reps=reps, max_steps=steps, seed=seed,
                 verbose=verbose, backend=backend, latent_hi=lh, **FROZEN)
        q.fit(Xmin[:, p])
        return np.clip(q.sample(n)[:, inv], 0.0, 1.0), q
    raise ValueError(variante)


def run_seed(seed, cls, ratio, mode, n_finale, steps, backend, verbose,
             modelli, latent_hi=LATENT_HI_IDS, ordine="naturale", reps=4):
    Xtr, ytr, Xva, yva, Xte, yte, feat = carica(cls, ratio, mode)
    Xmin = Xtr[ytr == 1]
    if n_finale <= len(Xmin):
        raise SystemExit(
            f"--n-finale {n_finale} non supera la minoritaria esistente "
            f"({len(Xmin)}): le varianti sintetiche riceverebbero zero "
            "campioni e la MMD non sarebbe definita. Con --ratio 1_200 "
            "serve almeno --n-finale 270."
        )
    n_agg = n_finale - len(Xmin)
    rng = np.random.default_rng(corsa.flusso_fisso(9100, seed).integers(2**31))

    ris = {
        "seed": seed, "classe": cls, "ratio": ratio, "pca": mode,
        "n_minoritaria": int(len(Xmin)), "n_aggiunti": int(n_agg),
        "n_finale": int(len(Xmin) + n_agg), "n_train": int(len(Xtr)),
        "n_val": int(len(yva)), "n_val_pos": int(yva.sum()),
        "n_test": int(len(yte)), "n_test_pos": int(yte.sum()),
        "feature": feat, "steps": steps,
        "latent_hi": latent_hi if np.isscalar(latent_hi) else list(latent_hi),
        "dispersione_reale": Xmin.std(axis=0).tolist(),
        "ordine": ordine, "reps": reps,
        "permutazione": permuta(ordine, Xmin)[0].tolist(),
        "varianti": {},
    }
    sig = float(median_bandwidth(Xmin, rng=np.random.default_rng(0)))

    sintetici, modello_q = {}, None
    for v in SINTETICHE:
        S, q = genera(v, Xmin, n_agg, rng, seed, steps, backend,
                      verbose, latent_hi, ordine, reps)
        sintetici[v] = S
        if q is not None:
            modello_q = q
    # Derivate: stessa nuvola, sola dispersione manipolata.
    rap_q = (sintetici["qgan"].std(axis=0)
             / np.maximum(Xmin.std(axis=0), 1e-9))
    sintetici["qgan_sd"] = riscala(sintetici["qgan"], Xmin)
    sintetici["smote_distorta"] = riscala(sintetici["smote"], Xmin, rap_q)
    # Terza ablazione: media E covarianza reali, nessun riaddestramento.
    sintetici["qgan_cov"] = ricolora(sintetici["qgan"], Xmin)
    # Controllo appaiato ESATTAMENTE: la gaussiana passata per la stessa
    # mappa ha gli stessi momenti di qgan_cov cifra per cifra, quindi la
    # differenza fra le due e' solo struttura oltre il second'ordine.
    # Senza questo, il confronto con la gaussiana campionata
    # attribuirebbe al generatore anche il vantaggio di avere la
    # covarianza esatta invece di una sua stima rumorosa.
    sintetici["gaussiana_cov"] = ricolora(sintetici["gaussiana"], Xmin)

    for v in VARIANTI:
        d = ris["varianti"].setdefault(v, {})
        if v in ("nessuna", "bilanciato"):
            X, y = Xtr, ytr
        else:
            S = sintetici[v]
            X = np.vstack([Xtr, S])
            y = np.concatenate([ytr, np.ones(len(S), int)])
            # Quanto i sintetici somigliano ai reali. ATTENZIONE: per
            # duplicazione e SMOTE il confronto e' contro gli stessi punti da
            # cui derivano, quindi ~0 per costruzione e non per qualita'.
            d["mmd_vs_reali"] = float(metrics.mmd_rbf(Xmin, S, sigma=sig))
            d["errore_cov"] = scarto_cov(S, Xmin)
            d["dispersione_generata"] = S.std(axis=0).tolist()
            d["rapporto_dispersione"] = (
                S.std(axis=0) / np.maximum(Xmin.std(axis=0), 1e-9)).tolist()
            if v == "qgan" and modello_q is not None:
                d["best_step"] = int(modello_q.best_step_)
                d["steps_run"] = int(modello_q.steps_run_)
        d["n_minoritaria_finale"] = int(y.sum())
        for m in modelli:
            d[m] = valuta(m, X, y, Xva, yva, Xte, yte, seed,
                          v == "bilanciato")
    return ris


def main():
    global DATA, SOGLIA_FOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--classe", default="r2l", choices=["r2l", "probe"])
    ap.add_argument("--ratio", default="1_1000",
                    choices=["1_200", "1_500", "1_1000"])
    ap.add_argument("--pca", default="local", choices=["local", "global"])
    ap.add_argument("--n-finale", type=int, default=269,
                    help="numerosita' finale della minoritaria; 269 e' il "
                         "livello 1:200, gia' misurato con campioni VERI in "
                         "results/baseline.csv")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5400)
    ap.add_argument("--latent-hi", type=str, default=str(LATENT_HI_IDS),
                    help="raggio del latente. Il default e' calibrato sulla "
                         "minoritaria IDS; 'pi' usa il valore di B.5, che e' "
                         "corretto per QKD ma cinque volte troppo disperso "
                         "qui; 'feature' usa i raggi per componente")
    ap.add_argument("--backend", default="denso", choices=["denso", "aer"])
    ap.add_argument("--ordine", default="naturale", choices=ORDINI,
                    help="assegnazione delle feature ai qubit. 'criterio' "
                         "ordina per dispersione crescente, cosi' la feature "
                         "piu' larga finisce sul qubit meno pilotabile")
    ap.add_argument("--reps", type=int, default=4,
                    help="ripetizioni dell'ansatz. SPEC §12 congela 4; "
                         "valori diversi sono un'ABLAZIONE, non una scelta. "
                         "Con --latent-hi auto il raggio e' quello calibrato "
                         "per quel reps")
    ap.add_argument("--modelli", default="rf,lr,gb")
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--forza", action="store_true")
    ap.add_argument("--dati-dir", default=None,
                    help="directory dei CSV, una per replica del "
                         "pre-processing (data/ids_s42, data/ids_s101...). "
                         "Il nome entra nella firma: repliche diverse non "
                         "si scambiano per duplicati.")
    ap.add_argument("--soglia-fold", type=int, default=SOGLIA_FOLD,
                    help="fold della cross-validation per la soglia "
                         "fuori campione; 0 disattiva e lascia 0.5")
    ap.add_argument("--out", default="results/ids_augmentation.jsonl")
    a = ap.parse_args()

    if a.dati_dir:
        DATA = Path(a.dati_dir).resolve()
    if not DATA.is_dir():
        raise SystemExit(f"directory dati inesistente: {DATA}")
    atteso = DATA / f"{a.classe}_train_{a.ratio}_{a.pca}.csv"
    if not atteso.exists():
        raise SystemExit(
            f"manca {atteso.name} in {DATA}\n"
            "La replica non contiene questa combinazione di --classe, "
            "--ratio e --pca: rigenerala con preprocess_nslkdd.py --seed."
        )
    SOGLIA_FOLD = a.soglia_fold
    print(f"dati {DATA}")
    print(f"soglia: scelta fuori campione su {SOGLIA_FOLD} fold"
          if SOGLIA_FOLD else "soglia: fissa a 0.5")

    modelli = [m.strip() for m in a.modelli.split(",") if m.strip()]
    if a.latent_hi == "auto":
        if a.reps not in RAGGIO_PER_REPS:
            raise SystemExit(f"nessun raggio calibrato per reps={a.reps}: "
                             f"disponibili {sorted(RAGGIO_PER_REPS)}")
        lh = RAGGIO_PER_REPS[a.reps]
    elif a.latent_hi == "pi":
        lh = float(np.pi)
    elif a.latent_hi == "feature":
        lh = list(LATENT_HI_IDS_FEATURE)
    else:
        v = [float(x) for x in a.latent_hi.split(",")]
        lh = v[0] if len(v) == 1 else v
    config = dict(FROZEN, dati=DATA.name, soglia_fold=SOGLIA_FOLD,
                  classe=a.classe, ratio=a.ratio,
                  pca=a.pca,
                  n_finale=a.n_finale, steps=a.steps, backend=a.backend,
                  latent_hi=lh, ordine=a.ordine, reps=a.reps,
                  modelli=modelli, varianti=VARIANTI)
    fr = corsa.firma(config)
    gia = corsa.gia_presenti(a.out)
    print(f"configurazione {fr}   codice {corsa.versione_codice()}")

    for s in range(a.seed_offset, a.seed_offset + a.seeds):
        if (fr, s) in gia and not a.forza:
            print(f"[seme {s}] gia' presente, saltato", flush=True)
            continue
        print(f"[seme {s}]", flush=True)
        r = run_seed(s, a.classe, a.ratio, a.pca, a.n_finale, a.steps,
                     a.backend, a.verbose, modelli, lh, a.ordine, a.reps)
        r["dati"] = DATA.name
        r.update(corsa.intestazione(config))
        corsa.scrivi(a.out, r)
        print(f"  minoritaria {r['n_minoritaria']} -> {r['n_finale']}  "
              f"su {r['n_train']} righe   reps {r['reps']}  ordine "
              f"{r['ordine']}  raggio {r['latent_hi']}", flush=True)
        m0 = modelli[0]
        print(f"  {'variante':>15} {'F1':>8} {'PR-AUC':>8} {'recall':>8}"
              f"   {'MMD':>8}   sd gen/reale   ({m0})", flush=True)
        for v in VARIANTI:
            d = r["varianti"][v]
            e = d[m0]
            extra = (f"   {d['mmd_vs_reali']:+8.4f}   "
                     f"{np.round(d['rapporto_dispersione'], 2)}"
                     if "mmd_vs_reali" in d else "")
            print(f"  {v:>15} {e['f1']:8.4f} {e['pr_auc']:8.4f} "
                  f"{e['recall']:8.4f}{extra}", flush=True)


if __name__ == "__main__":
    main()