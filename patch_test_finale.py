"""Aggiunge la misura sul test mai toccato. Patch di CONFERMA, additiva.

Richiede le tre patch precedenti applicate.

PERCHE'. Tutte le misure finora stanno sul validation: 13.668 righe, 199
positivi, prior 1:69. Accanto c'e' `{cls}_test*.csv` — 12.465 righe, 2.754
positivi, 1:5 — che `carica` non ha mai aperto. Viene da KDDTest+ e contiene
tipi di attacco assenti dal training: e' generalizzazione vera, non un altro
taglio dello stesso campione.

COSA AGGIUNGE, senza togliere niente. I campi esistenti restano calcolati
esattamente come prima, sullo stesso validation e con la stessa soglia fuori
campione, quindi la regressione contro le corse precedenti continua a valere
cifra per cifra. In piu':

  f1_test              F1 sul test a soglia 0.5
  f1_test_oracolo      DIAGNOSTICO: F1 alla soglia ottima sul test stesso.
                       Ottimistico per costruzione, non riportabile come
                       prestazione. La sua differenza da f1_test_soglia_val
                       e' la perdita da trasferimento della soglia, che e'
                       la quantita' da cui dipende la lettura.
  f1_test_soglia_val   F1 sul test alla soglia che massimizza F1 sul
                       VALIDATION. E' il protocollo classico pulito: la
                       soglia si sceglie su un insieme, si misura su un
                       altro, e non tocca ne' i dati di addestramento (in
                       parte sintetici) ne' i dati su cui si riporta.
  soglia_val           quella soglia, per poterla confrontare con `soglia`
  precision/recall_test_soglia_val, pr_auc_test, roc_auc_test

COSTO: nessun addestramento in piu'. Lo stesso modello gia' addestrato fa una
predizione in piu'; l'onere e' una curva precisione-richiamo su 13.668 righe.

NOTA SULL'ASSE REPLICA. Il test viene da KDDTest+ e non dallo split, quindi
la sua COMPOSIZIONE e' la stessa in tutte le repliche: cambia solo la
proiezione PCA, riadattata per replica. L'inferenza incrociata sull'asse
replica misura allora il solo effetto del campione di addestramento, che e'
piu' pulito di prima, dove variava anche l'insieme di valutazione.

    cd ~/PycharmProjects/Quantum/qgan-qber
    python patch_test_finale.py src/ids/ids_augmentation.py
"""

import io
import shutil
import sys

V_CARICA = '''def carica(cls, ratio, mode):
    tr = pd.read_csv(DATA / f"{cls}_train_{ratio}_{mode}.csv")
    va = (pd.read_csv(DATA / f"{cls}_val.csv") if mode == "global"
          else pd.read_csv(DATA / f"{cls}_val_{ratio}_local.csv"))
    feat = [c for c in tr.columns if c != "y"]
    return (tr[feat].to_numpy(float), tr["y"].to_numpy(int),
            va[feat].to_numpy(float), va["y"].to_numpy(int), feat)
'''
N_CARICA = '''def carica(cls, ratio, mode):
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
'''

V_UNPACK = "    Xtr, ytr, Xva, yva, feat = carica(cls, ratio, mode)\n"
N_UNPACK = "    Xtr, ytr, Xva, yva, Xte, yte, feat = carica(cls, ratio, mode)\n"

V_CHIAM = "            d[m] = valuta(m, X, y, Xva, yva, seed, v == \"bilanciato\")\n"
N_CHIAM = ("            d[m] = valuta(m, X, y, Xva, yva, Xte, yte, seed,\n"
           "                          v == \"bilanciato\")\n")

V_NRIGHE = '        "n_finale": int(len(Xmin) + n_agg), "n_train": int(len(Xtr)),\n'
N_NRIGHE = ('        "n_finale": int(len(Xmin) + n_agg), "n_train": int(len(Xtr)),\n'
            '        "n_val": int(len(yva)), "n_val_pos": int(yva.sum()),\n'
            '        "n_test": int(len(yte)), "n_test_pos": int(yte.sum()),\n')

V_VALUTA = '''def valuta(nome, Xtr, ytr, Xva, yva, seed, bilanciato):
    m = modello(nome, seed, bilanciato)
    m.fit(Xtr, ytr)
    pred = m.predict(Xva)
    proba = m.predict_proba(Xva)[:, 1]
'''
N_VALUTA = '''def soglia_massima(y, p):
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
'''

V_FINE = '''        out["recall_soglia"] = float(
            recall_score(yva, (proba >= s).astype(int), zero_division=0))
    return out
'''
N_FINE = '''        out["recall_soglia"] = float(
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
'''

MODIFICHE = [
    (V_CARICA, N_CARICA), (V_UNPACK, N_UNPACK), (V_CHIAM, N_CHIAM),
    (V_NRIGHE, N_NRIGHE), (V_VALUTA, N_VALUTA), (V_FINE, N_FINE),
]


def main():
    percorso = sys.argv[1] if len(sys.argv) > 1 else "src/ids/ids_augmentation.py"
    s = io.open(percorso, encoding="utf-8").read()

    if "f1_test_soglia_val" in s:
        print("gia' applicato")
        return 0
    for marcatore, patch in (("--dati-dir", "patch_augmentation.py"),
                             ("qgan_cov", "patch_ablazione_cov.py"),
                             ("gaussiana_cov", "patch_baseline_gaussiana.py")):
        if marcatore not in s:
            print(f"ATTENZIONE: manca {marcatore!r}, quindi {patch} non e'")
            print("stata applicata. Le patch si innestano in ordine:")
            print(f"  python {patch} {percorso}")
            return 1

    mancanti = [v for v, _ in MODIFICHE if s.count(v) != 1]
    if mancanti:
        print("ATTENZIONE: queste stringhe non compaiono esattamente una volta")
        for v in mancanti:
            print(f"  {s.count(v)} occorrenze: {v[:70]!r}")
        print("\nIl file NON e' stato modificato.")
        return 1

    shutil.copy(percorso, percorso + ".bak4")
    for v, n in MODIFICHE:
        s = s.replace(v, n)
    io.open(percorso, "w", encoding="utf-8").write(s)

    import ast
    ast.parse(s)
    print(f"ok: {len(MODIFICHE)} modifiche applicate, copia in {percorso}.bak4")
    print()
    print("  python lancia_repliche.py --solo aug --out results/ids_finale \\")
    print("      --parallele 5")
    print("  python lancia_repliche.py --solo analisi --out results/ids_finale \\")
    print("      --metriche f1_test_soglia_val,f1_test_oracolo,pr_auc_test")
    print()
    print("Poi, per la continuita' con quanto gia' misurato:")
    print("  python src/ids/analisi_repliche.py --dir results/ids_finale \\")
    print("      --metrica f1_test_soglia_val --contro duplicazione")
    print()
    print("REGRESSIONE: i campi vecchi non cambiano. Su ids_s42 seme 0, f1")
    print("deve ancora dare nessuna 0.5368, bilanciato 0.7754, smote 0.7829,")
    print("qgan 0.5786. Se si muovono, qualcosa ha toccato il flusso casuale")
    print("e i numeri sul test non sono confrontabili con il resto.")
    return 0


if __name__ == "__main__":
    sys.exit(main())