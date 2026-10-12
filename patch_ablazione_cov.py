"""Aggiunge a ids_augmentation.py la terza ablazione e la soglia fuori campione.

Richiede patch_augmentation.py gia' applicata.

PERCHE' LA TERZA ABLAZIONE. `riscala` corregge la deviazione standard colonna
per colonna e lascia la media invariata; le CORRELAZIONI fra feature non le
tocca. Una nuvola con le cinque dispersioni giuste e la struttura di
covarianza sbagliata passa quel controllo intatta, ed e' il motivo per cui
`qgan_sd - qgan` puo' valere zero (misurato: -0.0078, intervallo
[-0.027,+0.011], 2/10 repliche) mentre `smote_distorta - smote` vale -0.129:
la dispersione per feature e' SUFFICIENTE a fare danno ma non e' il difetto
vincolante della QGAN.

  qgan_cov   mappa affine che porta la nuvola della QGAN su media E
             covarianza reali. Verso atteso POSITIVO. Se recupera, il
             difetto e' la struttura di second'ordine e il generatore e'
             calibrabile; se non recupera, nessuna trasformazione affine lo
             chiude e il problema e' il supporto — dove mette i punti —
             che rimanda al raggio del latente e all'ansatz.

Si ottiene dalla nuvola GIA' generata, senza riaddestrare: le righe `qgan`
della nuova corsa devono riprodursi identiche a quelle della vecchia.

PERCHE' LA SOGLIA FUORI CAMPIONE. Su F1@0.5 il divario `qgan - bilanciato`
vale -0.112; su PR-AUC, che non dipende dalla soglia, la stessa coppia da
+0.018. La differenza fra le due metriche e' collocazione della soglia, non
qualita' dell'ordinamento. `f1_soglia` misura quanto del divario sopravvive
quando la soglia viene scelta bene — e la scelta e' FUORI CAMPIONE, su
previsioni cross-validate del training, perche' tararla sullo stesso insieme
su cui si riporta l'F1 userebbe le etichette di valutazione.

Il campo `errore_cov` registra per ogni variante lo scarto relativo di
covarianza dai reali: e' la quantita' che la nuova ablazione manipola, utile
anche per le altre varianti.

    cd ~/PycharmProjects/Quantum/qgan-qber
    python patch_ablazione_cov.py src/ids/ids_augmentation.py
"""

import io
import shutil
import sys

# ------------------------------------------------------------------ import
V_IMP = """from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
"""
N_IMP = """from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
"""

# ------------------------------------------------------------- costante e varianti
V_FROZEN = "FROZEN = dict(\n"
N_FROZEN = '''# Fold della cross-validation con cui si sceglie la soglia fuori campione.
# Zero disattiva il calcolo: la soglia resta 0.5 e `f1_soglia` non c'e'.
SOGLIA_FOLD = 3

FROZEN = dict(
'''

V_DER = 'DERIVATE = ["qgan_sd", "smote_distorta"]\n'
N_DER = 'DERIVATE = ["qgan_sd", "smote_distorta", "qgan_cov"]\n'

# ------------------------------------------------------------------ ricolora
V_CARICA = "def carica(cls, ratio, mode):\n"
N_CARICA = '''def ricolora(S, Xmin):
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
'''

# -------------------------------------------------------- soglia e valutazione
V_VAL = """def valuta(nome, Xtr, ytr, Xva, yva, seed, bilanciato):
    m = modello(nome, seed, bilanciato)
    m.fit(Xtr, ytr)
    pred = m.predict(Xva)
    proba = m.predict_proba(Xva)[:, 1]
    return {
        "f1": float(f1_score(yva, pred, zero_division=0)),
        "precision": float(precision_score(yva, pred, zero_division=0)),
        "recall": float(recall_score(yva, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(yva, proba)),
        "roc_auc": float(roc_auc_score(yva, proba)),
    }
"""
N_VAL = '''def soglia_fuori_campione(nome, Xtr, ytr, seed, bilanciato, fold):
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


def valuta(nome, Xtr, ytr, Xva, yva, seed, bilanciato):
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
    return out
'''

# ------------------------------------------------------------- nuova variante
V_SD = '    sintetici["smote_distorta"] = riscala(sintetici["smote"], Xmin, rap_q)\n'
N_SD = ('    sintetici["smote_distorta"] = riscala(sintetici["smote"], Xmin, rap_q)\n'
        '    # Terza ablazione: media E covarianza reali, nessun riaddestramento.\n'
        '    sintetici["qgan_cov"] = ricolora(sintetici["qgan"], Xmin)\n')

V_MMD = '            d["mmd_vs_reali"] = float(metrics.mmd_rbf(Xmin, S, sigma=sig))\n'
N_MMD = ('            d["mmd_vs_reali"] = float(metrics.mmd_rbf(Xmin, S, sigma=sig))\n'
         '            d["errore_cov"] = scarto_cov(S, Xmin)\n')

# ----------------------------------------------------------------------- main
V_GLOB = "    global DATA\n"
N_GLOB = "    global DATA, SOGLIA_FOLD\n"

V_ARG = '    ap.add_argument("--out", default="results/ids_augmentation.jsonl")\n'
N_ARG = ('    ap.add_argument("--soglia-fold", type=int, default=SOGLIA_FOLD,\n'
         '                    help="fold della cross-validation per la soglia "\n'
         '                         "fuori campione; 0 disattiva e lascia 0.5")\n'
         '    ap.add_argument("--out", default="results/ids_augmentation.jsonl")\n')

V_SET = '    print(f"dati {DATA}")\n'
N_SET = ('    SOGLIA_FOLD = a.soglia_fold\n'
         '    print(f"dati {DATA}")\n'
         '    print(f"soglia: scelta fuori campione su {SOGLIA_FOLD} fold"\n'
         '          if SOGLIA_FOLD else "soglia: fissa a 0.5")\n')

V_CONF = "    config = dict(FROZEN, dati=DATA.name, classe=a.classe, ratio=a.ratio,\n"
N_CONF = ("    config = dict(FROZEN, dati=DATA.name, soglia_fold=SOGLIA_FOLD,\n"
          "                  classe=a.classe, ratio=a.ratio,\n")

V_COMM = """# Le due ultime sono ABLAZIONI CAUSALI, non metodi proponibili.
#   qgan_sd         campioni QGAN con le dispersioni per feature riportate a
#                   quelle reali, media invariata. Se l'F1 recupera, la
#                   distorsione delle dispersioni e' causa e non sintomo.
#   smote_distorta  campioni SMOTE con applicati i rapporti di dispersione
#                   della QGAN. Verso opposto: se l'F1 crolla ai livelli
#                   della QGAN, la causa e' confermata.
# Insieme formano una dimostrazione; separatamente sarebbero due indizi.
"""
N_COMM = """# Le tre ultime sono ABLAZIONI CAUSALI, non metodi proponibili. Verso
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
"""

MODIFICHE = [
    (V_IMP, N_IMP), (V_FROZEN, N_FROZEN), (V_DER, N_DER),
    (V_CARICA, N_CARICA), (V_VAL, N_VAL), (V_SD, N_SD), (V_MMD, N_MMD),
    (V_GLOB, N_GLOB), (V_ARG, N_ARG), (V_SET, N_SET), (V_CONF, N_CONF),
    (V_COMM, N_COMM),
]


def main():
    percorso = sys.argv[1] if len(sys.argv) > 1 else "src/ids/ids_augmentation.py"
    s = io.open(percorso, encoding="utf-8").read()

    if "qgan_cov" in s and "soglia_fuori_campione" in s:
        print("gia' applicato")
        return 0
    if "--dati-dir" not in s:
        print("ATTENZIONE: manca --dati-dir, quindi patch_augmentation.py non")
        print("e' stata applicata. Questa patch si innesta su quella:")
        print(f"  python patch_augmentation.py {percorso}")
        return 1

    mancanti = [v for v, _ in MODIFICHE if s.count(v) != 1]
    if mancanti:
        print("ATTENZIONE: queste stringhe non compaiono esattamente una volta")
        for v in mancanti:
            print(f"  {s.count(v)} occorrenze: {v[:70]!r}")
        print("\nIl file NON e' stato modificato.")
        return 1

    shutil.copy(percorso, percorso + ".bak2")
    for v, n in MODIFICHE:
        s = s.replace(v, n)
    io.open(percorso, "w", encoding="utf-8").write(s)

    import ast
    ast.parse(s)
    print(f"ok: {len(MODIFICHE)} modifiche applicate, copia in {percorso}.bak2")
    print()
    print("La firma cambia (soglia_fold entra in config e c'e' una variante")
    print("in piu'): scrivi in una directory nuova.")
    print()
    print("  python lancia_repliche.py --solo aug --out results/ids_repliche2 \\")
    print("      --parallele 5")
    print("  python lancia_repliche.py --solo analisi --out results/ids_repliche2 \\")
    print("      --metriche f1,f1_soglia,pr_auc")
    print()
    print("Regressione: le righe 'qgan' di ids_s42 ai semi 0 e 1 devono")
    print("riprodurre 0.5786 e 0.5580 come prima. La nuvola e' la stessa,")
    print("qgan_cov e' una trasformazione a valle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())