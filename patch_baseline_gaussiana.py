"""Aggiunge la gaussiana di controllo e due scale di rumore.

Richiede patch_augmentation.py e patch_ablazione_cov.py gia' applicate.

PERCHE' LA GAUSSIANA. `ricolora` monta sulla nuvola della QGAN la media e la
covarianza dei 53 reali, e il guadagno misurato e' grande (+0.048 di F1,
10/10 repliche). Ma lo scarto di covarianza di `qgan_cov` vale 0.012, mentre
`duplicazione` — i punti reali ricampionati — sta a 0.260: quello e' il
rumore di campionamento di 216 estrazioni da 53 punti, cioe' il PAVIMENTO.
Avere 0.012 significa corrispondenza esatta dei momenti, piu' esatta di un
ricampionamento dei reali stessi. Quindi il guadagno puo' venire tutto dai
momenti iniettati e niente dal generatore.

  gaussiana   N(mu, Sigma) stimata sui 53 reali. Ha esattamente cio' che
              ricolora inietta e nessun contributo generativo. Se
              `qgan_cov - gaussiana` e' indistinguibile da zero, il
              generatore non aggiunge niente e il risultato va scritto
              cosi'; se e' positivo, porta struttura oltre il second'ordine.

PERCHE' DUE SCALE DI RUMORE. Col protocollo a soglia tarata la graduatoria e'
rumore 0.831, smote 0.813, qgan_cov 0.797, e in fondo duplicazione 0.661 —
che sono i punti reali esatti. Ordinare i metodi per fedelta' non funziona:
il migliore per covarianza (qgan_cov, 0.012) e' terzo, SMOTE vince con 0.461.
Il quadro e' che conta la perturbazione LOCALE dei punti reali, non
l'apprendimento della distribuzione. Con 0.03x e 0.30x accanto allo 0.10x
attuale si vede se c'e' un ottimo interno, e l'interpretazione
"regolarizzatore" diventa misurata invece che suggerita.

ORDINE DEL FLUSSO CASUALE. Le tre varianti vanno IN CODA a SINTETICHE.
`duplicazione`, `rumore` e `smote` consumano `rng` in quell'ordine; `qgan`
no, perche' usa il proprio seme. Inserire le nuove dopo `qgan` lascia le tre
precedenti bit-identiche, quindi la corsa nuova deve riprodurre i numeri
della vecchia su tutte le varianti esistenti.

    cd ~/PycharmProjects/Quantum/qgan-qber
    python patch_baseline_gaussiana.py src/ids/ids_augmentation.py
"""

import io
import shutil
import sys

V_SINT = '''SINTETICHE = ["duplicazione", "rumore", "smote", "qgan"]
'''
N_SINT = '''# Le tre ultime sono IN CODA di proposito: duplicazione, rumore e smote
# consumano `rng` in quest'ordine e qgan no (usa il proprio seme), quindi
# aggiungere dopo qgan lascia le precedenti bit-identiche fra corse.
#   gaussiana      N(mu, Sigma) dei 53 reali: il controllo di qgan_cov, ha
#                  gli stessi momenti e nessun contributo generativo.
#   rumore_lieve   come rumore ma a 0.03x la dispersione media (0.10x)
#   rumore_forte   come rumore ma a 0.30x
SINTETICHE = ["duplicazione", "rumore", "smote", "qgan",
              "gaussiana", "rumore_lieve", "rumore_forte"]
SCALE_RUMORE = {"rumore": 0.10, "rumore_lieve": 0.03, "rumore_forte": 0.30}
'''

V_GEN = '''    if variante == "rumore":
        base = Xmin[rng.integers(0, len(Xmin), n)]
        sd = float(Xmin.std(axis=0).mean()) * 0.1
        return np.clip(base + rng.normal(0, sd, base.shape), 0.0, 1.0), None
'''
N_GEN = '''    if variante in SCALE_RUMORE:
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
'''

V_DER = 'DERIVATE = ["qgan_sd", "smote_distorta", "qgan_cov"]\n'
N_DER = 'DERIVATE = ["qgan_sd", "smote_distorta", "qgan_cov", "gaussiana_cov"]\n'

V_RIC = ('    # Terza ablazione: media E covarianza reali, nessun riaddestramento.\n'
         '    sintetici["qgan_cov"] = ricolora(sintetici["qgan"], Xmin)\n')
N_RIC = ('    # Terza ablazione: media E covarianza reali, nessun riaddestramento.\n'
         '    sintetici["qgan_cov"] = ricolora(sintetici["qgan"], Xmin)\n'
         '    # Controllo appaiato ESATTAMENTE: la gaussiana passata per la stessa\n'
         '    # mappa ha gli stessi momenti di qgan_cov cifra per cifra, quindi la\n'
         '    # differenza fra le due e\' solo struttura oltre il second\'ordine.\n'
         '    # Senza questo, il confronto con la gaussiana campionata\n'
         '    # attribuirebbe al generatore anche il vantaggio di avere la\n'
         '    # covarianza esatta invece di una sua stima rumorosa.\n'
         '    sintetici["gaussiana_cov"] = ricolora(sintetici["gaussiana"], Xmin)\n')

MODIFICHE = [(V_SINT, N_SINT), (V_GEN, N_GEN), (V_DER, N_DER), (V_RIC, N_RIC)]


def main():
    percorso = sys.argv[1] if len(sys.argv) > 1 else "src/ids/ids_augmentation.py"
    s = io.open(percorso, encoding="utf-8").read()

    if "gaussiana_cov" in s and "SCALE_RUMORE" in s:
        print("gia' applicato")
        return 0
    for marcatore, patch in (("--dati-dir", "patch_augmentation.py"),
                             ("qgan_cov", "patch_ablazione_cov.py")):
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

    shutil.copy(percorso, percorso + ".bak3")
    for v, n in MODIFICHE:
        s = s.replace(v, n)
    io.open(percorso, "w", encoding="utf-8").write(s)

    import ast
    ast.parse(s)
    print(f"ok: {len(MODIFICHE)} modifiche applicate, copia in {percorso}.bak3")
    print()
    print("  python lancia_repliche.py --solo aug --out results/ids_repliche3 \\")
    print("      --parallele 5")
    print("  python lancia_repliche.py --solo analisi --out results/ids_repliche3 \\")
    print("      --metriche f1_soglia,f1,pr_auc")
    print()
    print("Regressione da controllare prima di leggere qualunque cosa: su")
    print("ids_s42 seme 0, f1 deve tornare nessuna 0.5368, bilanciato 0.7754,")
    print("smote 0.7829, qgan 0.5786. Le nuove varianti sono in coda e non")
    print("spostano il flusso casuale delle precedenti.")
    return 0


if __name__ == "__main__":
    sys.exit(main())