"""Inferenza sulle repliche del pre-processing IDS.

Due assi INCROCIATI, non annidati:

  REPLICA  uno split indipendente: altri 53 attacchi fra i 796 disponibili,
           PCA locale riadattata su quello split, insieme di validazione
           proprio. E' l'asse che generalizza alla popolazione IDS.
  SEME     sola inizializzazione: pesi della QGAN, campionamento della
           generazione, stato del classificatore. A dati fissi.

Fino a oggi l'asse replica aveva UN solo livello. Tutte le conclusioni IDS —
che nessuna augmentation sintetica batta class_weight, che la QGAN perda
0,196 di F1, che correggere le dispersioni non la salvi — valevano per il
campione fissato dal seme 42, e i cinque semi misuravano la stabilita'
dell'addestramento, non la generalita' della conclusione. E' il vincolo che
l'audit ha indicato come il piu' stringente.

Con dieci repliche l'unita' di inferenza diventa la replica. Da qui tre
conseguenze sulla lettura, tutte implementate sotto:

  - la media va fatta per replica prima di tutto il resto; le venti corse non
    sono venti osservazioni indipendenti;
  - l'intervallo da riportare e' quello INCROCIATO, che ricampiona entrambi
    gli assi: gli altri due tengono fisso un asse e sottostimano;
  - il test dei segni si fa sulle MEDIE PER REPLICA, n = numero di repliche.

Le due ABLAZIONI CAUSALI sono contrasti a se', con un verso atteso ciascuno:
`smote_distorta - smote` deve essere fortemente NEGATIVO e `qgan_sd - qgan`
POSITIVO se la distorsione delle dispersioni e' la causa. Vengono stampati
separatamente perche' rispondono a una domanda diversa dal confronto
operativo.

    python src/ids/analisi_repliche.py --dir results/ids_repliche
    python src/ids/analisi_repliche.py --dir results/ids_repliche --metrica pr_auc
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI.parent.parent / "qgan" / "scripts"))
from analisi_appaiata import bootstrap, componenti  # noqa: E402

# Contrasti sempre stampati, oltre al confronto di ogni variante con il
# riferimento operativo. Nome, verso atteso, motivo.
ABLAZIONI = [
    ("smote_distorta", "smote", "-",
     "SMOTE con le dispersioni della QGAN: se l'F1 crolla, la distorsione\n"
     "   e' causa e non sintomo"),
    ("qgan_sd", "qgan", "+",
     "QGAN con le dispersioni riportate a quelle reali: se l'F1 recupera,\n"
     "   la causa e' confermata dal verso opposto"),
    ("qgan_cov", "qgan", "+",
     "QGAN portata su media E covarianza reali. riscala corregge una\n"
     "   dispersione per volta e lascia intatte media e correlazioni,\n"
     "   quindi qgan_sd non era una correzione completa: questa lo e',\n"
     "   entro le trasformazioni affini"),
]

# Contrasti a DUE CODE: nessun verso atteso, la domanda e' quanto vale la
# differenza. Isolano quanto del guadagno di qgan_cov venga dai momenti che
# ricolora inietta e quanto dal generatore.
CONTROLLI = [
    ("qgan_cov", "gaussiana_cov",
     "il contrasto DECISIVO, e l'unico appaiato esattamente: le due nuvole\n"
     "   passano per la stessa mappa, quindi hanno media e covarianza\n"
     "   identiche cifra per cifra e differiscono SOLO per la struttura\n"
     "   oltre il second'ordine. Indistinguibile da zero = il circuito non\n"
     "   aggiunge niente a una gaussiana"),
    ("qgan_cov", "gaussiana",
     "la stessa domanda contro la gaussiana campionata, che ha i momenti\n"
     "   stimati e non esatti: qui un vantaggio di qgan_cov puo' venire\n"
     "   anche dall'esattezza dei momenti, non dal generatore"),
    ("qgan", "gaussiana",
     "la QGAN grezza contro il modello parametrico piu' semplice che usa\n"
     "   gli stessi 53 punti"),
    ("rumore_forte", "rumore",
     "0.30x contro 0.10x della dispersione media: con rumore_lieve nella\n"
     "   tabella dei livelli dice se c'e' un ottimo interno nella scala\n"
     "   della perturbazione"),
]

# La MMD^2 e' stimata con la U-statistica non distorta e va SOTTO ZERO
# quando la MMD vera e' zero. Dentro questa tolleranza il valore si legge
# come "indistinguibile da zero", e i rapporti fra quantita' del genere non
# significano niente: si confrontano per differenza.
TOLL_MMD = 0.02
# Divario minimo perche' una coppia valga come controesempio. Sotto questa
# soglia due varianti non "rendono in modo molto diverso a valle".
DIVARIO_MINIMO = 0.02


# ======================================================================
# Caricamento
# ======================================================================
def carica(percorso):
    """Righe dai JSONL, deduplicate per (firma, seme).

    A differenza dell'analisi appaiata QKD, qui le firme DEVONO essere
    diverse: una per replica, perche' il nome della directory dei dati entra
    nella configurazione. Si verifica invece che le configurazioni differiscano
    SOLO per quel campo: se cambia anche --steps o --reps, il confronto fra
    repliche e' confuso con un confronto fra protocolli.
    """
    file = sorted(glob.glob(os.path.join(percorso, "*.jsonl")))
    if not file:
        file = [percorso] if os.path.isfile(percorso) else []
    if not file:
        raise SystemExit(f"nessun .jsonl in {percorso}")

    viste, righe = set(), []
    for f in file:
        for riga in open(f):
            riga = riga.strip()
            if not riga:
                continue
            r = json.loads(riga)
            chiave = (r.get("firma_config"), r.get("seed"))
            if chiave in viste:
                continue
            viste.add(chiave)
            righe.append(r)

    for r in righe:
        if "dati" not in r:
            r["dati"] = (r.get("config") or {}).get("dati", "ids")

    # Le configurazioni devono coincidere tranne che per 'dati'.
    conf = {}
    for r in righe:
        c = dict(r.get("config") or {})
        c.pop("dati", None)
        conf.setdefault(json.dumps(c, sort_keys=True, default=str),
                        []).append(r["dati"])
    if len(conf) > 1:
        chiavi = [json.loads(k) for k in conf]
        tutte = set().union(*(set(k) for k in chiavi))
        diff = sorted(k for k in tutte
                      if len({json.dumps(c.get(k), sort_keys=True,
                                         default=str) for c in chiavi}) > 1)
        print("ATTENZIONE: le repliche non condividono la configurazione.")
        print(f"  campi che differiscono: {diff}")
        for k, v in conf.items():
            d = json.loads(k)
            print(f"  {sorted(set(v))}: " +
                  ", ".join(f"{x}={d.get(x)}" for x in diff))
        raise SystemExit(
            "Un confronto fra repliche richiede lo stesso protocollo in "
            "tutte.\nRilancia le repliche discordanti con gli stessi flag, "
            "o analizzale\nseparatamente passando --dir diverse."
        )
    return righe


def matrice(righe, variante, riferimento, metrica, modello):
    """{(replica, seme): differenza} per il contrasto richiesto."""
    oss = {}
    for r in righe:
        v = r.get("varianti") or {}
        if variante not in v or riferimento not in v:
            disponibili = sorted(v)
            raise SystemExit(
                f"variante assente: serve '{variante}' e '{riferimento}', "
                f"nel file ci sono {disponibili}"
            )
        for nome in (variante, riferimento):
            if modello not in v[nome]:
                raise SystemExit(
                    f"modello '{modello}' assente in '{nome}'; "
                    f"disponibili {sorted(k for k in v[nome] if isinstance(v[nome][k], dict))}"
                )
            if metrica not in v[nome][modello]:
                raise SystemExit(
                    f"metrica '{metrica}' assente; disponibili "
                    f"{sorted(v[nome][modello])}"
                )
        d = (v[variante][modello][metrica]
             - v[riferimento][modello][metrica])
        oss[(r["dati"], r["seed"])] = float(d)
    return oss


def livello(righe, variante, metrica, modello):
    """{(replica, seme): valore assoluto}, per leggere i livelli."""
    return {(r["dati"], r["seed"]):
            float(r["varianti"][variante][modello][metrica])
            for r in righe if variante in (r.get("varianti") or {})}


def scarto_medio(righe, variante):
    """Scarto relativo di covarianza medio, dove registrato."""
    v = [r["varianti"][variante]["errore_cov"] for r in righe
         if "errore_cov" in (r.get("varianti") or {}).get(variante, {})]
    return float(np.mean(v)) if v else float("nan")


def etichetta_mmd(x):
    return f"{x:+.4f}" + ("  (~0)" if abs(x) < TOLL_MMD else "")


def mmd_medio(righe, variante):
    v = [r["varianti"][variante]["mmd_vs_reali"] for r in righe
         if "mmd_vs_reali" in (r.get("varianti") or {}).get(variante, {})]
    return float(np.mean(v)) if v else float("nan")


# ======================================================================
# Stampa di un contrasto
# ======================================================================
def riporta(oss, repliche, semi, etichetta, contro=0.0, repliche_bs=20000,
            atteso=None):
    from scipy import stats

    v_r, v_s, v_res, tot, M = componenti(oss, repliche, semi)
    media = float(np.nanmean(M))
    per_replica = np.nanmean(M, axis=1)

    print(f"  media su tutte le corse        {media:+.4f}")
    # Le percentuali sono della varianza della DIFFERENZA, non del livello:
    # il livello dipende quasi tutto dalla replica, ma nella differenza
    # l'effetto della replica si cancella perche' le due varianti girano
    # sugli stessi dati. Una componente fra repliche grande QUI significa
    # che il metodo danneggia alcuni split piu' di altri.
    print(f"  varianza della differenza: fra repliche {v_r/tot:5.1%}   "
          f"fra semi {v_s/tot:5.1%}   residuo {v_res/tot:5.1%}")

    print("  unita' di inferenza            intervallo 95%       ampiezza")
    esiti = {}
    for modo, et in (("corsa", "per replica"),
                     ("cella", "per seme"),
                     ("incrociata", "incrociata")):
        lo, hi = bootstrap(oss, repliche, semi, modo, repliche_bs)
        esclude = lo > contro or hi < contro
        esiti[modo] = (lo, esclude, hi)
        marca = "" if esclude else "   contiene lo zero"
        print(f"    {et:28s} [{lo:+.4f}, {hi:+.4f}]  {hi-lo:.4f}{marca}")

    n = int(np.isfinite(per_replica).sum())
    seg = int((per_replica > contro).sum())
    p = stats.binomtest(seg, n, 0.5).pvalue if n else float("nan")
    print(f"  segni sulle medie per replica  {seg}/{n} sopra "
          f"{contro:+.3f}   p = {p:.4f}")
    if n and n <= 20:
        print(f"    (pavimento del test con {n} repliche: {2*0.5**n:.4f})")

    lo, _, hi = esiti["incrociata"]
    regge = None
    if atteso == "-":
        regge = hi < 0
        print(f"  verso atteso NEGATIVO: "
              f"{'confermato' if regge else 'NON confermato'}")
    elif atteso == "+":
        regge = lo > 0
        print(f"  verso atteso POSITIVO: "
              f"{'confermato' if regge else 'NON confermato'}")
    return media, esiti["incrociata"], (seg, n, p), regge


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="directory con i JSONL delle repliche, o un file")
    ap.add_argument("--metrica", default="f1",
                    help="f1, pr_auc, recall, precision, roc_auc")
    ap.add_argument("--modello", default="rf", help="rf, lr, gb")
    ap.add_argument("--contro", default="bilanciato",
                    help="variante di riferimento del confronto operativo")
    ap.add_argument("--repliche-bs", type=int, default=20000)
    a = ap.parse_args()

    righe = carica(a.dir)
    repliche = sorted({r["dati"] for r in righe})
    semi = sorted({r["seed"] for r in righe})
    varianti = sorted(set().union(*(set(r.get("varianti") or {})
                                    for r in righe)))

    # Metrica e modello si controllano SUBITO: senza, un nome sbagliato
    # esce come KeyError a meta' della prima tabella.
    campione = (righe[0].get("varianti") or {}).get(a.contro)
    if campione is None:
        raise SystemExit(
            f"variante di riferimento '{a.contro}' assente; "
            f"nel file ci sono {varianti}"
        )
    modelli = sorted(k for k, v in campione.items() if isinstance(v, dict))
    if a.modello not in modelli:
        raise SystemExit(f"modello '{a.modello}' assente; "
                         f"disponibili {modelli}")
    if a.metrica not in campione[a.modello]:
        raise SystemExit(f"metrica '{a.metrica}' assente; disponibili "
                         f"{sorted(campione[a.modello])}")

    print(f"{len(righe)} corse   {len(repliche)} repliche   "
          f"{len(semi)} semi   metrica {a.metrica} ({a.modello})")
    print(f"repliche: {', '.join(repliche)}")
    print(f"semi: {semi}")
    pieni = sum(1 for d in repliche for s in semi
                if any(r["dati"] == d and r["seed"] == s for r in righe))
    print(f"celle piene della griglia replica x seme: {pieni}/"
          f"{len(repliche)*len(semi)}")
    if len(repliche) < 2:
        print()
        print("Con una sola replica non c'e' nulla da ricampiarare sull'asse")
        print("che conta: il risultato resterebbe vincolato a uno split, che")
        print("e' esattamente il limite da superare.")
        return 1
    if len(semi) < 2:
        print()
        print("Con un solo seme la componente fra semi non e' stimabile e")
        print("l'intervallo incrociato coincide con quello per replica.")
    print()

    # ---------------------------------------------------- livelli assoluti
    print("=" * 70)
    print(f"LIVELLI  media sulle repliche, deviazione FRA repliche")
    print("=" * 70)
    print(f"  {'variante':>16} {'media':>9} {'sd fra repl.':>13} "
          f"{'min':>9} {'max':>9}")
    for v in varianti:
        L = livello(righe, v, a.metrica, a.modello)
        _, _, _, _, M = componenti(L, repliche, semi)
        pr = np.nanmean(M, axis=1)
        print(f"  {v:>16} {np.nanmean(M):9.4f} {np.nanstd(pr, ddof=1):13.4f} "
              f"{np.nanmin(pr):9.4f} {np.nanmax(pr):9.4f}")
    print()
    print("  La deviazione FRA repliche e' la quantita' che non si poteva")
    print("  misurare con un solo split: dice quanto di cio' che si leggeva")
    print("  come effetto del metodo era scelta dei 53 attacchi.")
    print()

    # ------------------------------------------------- confronto operativo
    print("=" * 70)
    print(f"CONFRONTO OPERATIVO   ogni variante meno '{a.contro}'")
    print("=" * 70)
    riepilogo = []
    for v in varianti:
        if v == a.contro:
            continue
        print(f"\n{v} - {a.contro}")
        oss = matrice(righe, v, a.contro, a.metrica, a.modello)
        m, (lo, _, hi), (seg, n, p), _ = riporta(
            oss, repliche, semi, v, 0.0, a.repliche_bs)
        riepilogo.append((v, m, lo, hi, seg, n, p))

    # --------------------------------------------------- ablazioni causali
    print()
    print("=" * 70)
    print("ABLAZIONI CAUSALI   verso atteso dichiarato prima della misura")
    print("=" * 70)
    verdetti = []
    for var, rif, verso, motivo in ABLAZIONI:
        if var not in varianti or rif not in varianti:
            print(f"\n{var} - {rif}: salto, variante assente")
            continue
        print(f"\n{var} - {rif}   atteso {verso}")
        print(f"   {motivo}")
        oss = matrice(righe, var, rif, a.metrica, a.modello)
        m, _, _, regge = riporta(oss, repliche, semi, var, 0.0,
                                 a.repliche_bs, atteso=verso)
        verdetti.append((var, rif, verso, m, regge))
        mv, mr = mmd_medio(righe, var), mmd_medio(righe, rif)
        if np.isfinite(mv) and np.isfinite(mr):
            print(f"  MMD media   {var} {etichetta_mmd(mv)}   "
                  f"{rif} {etichetta_mmd(mr)}   differenza {mv - mr:+.4f}")
            if abs(mv - mr) < TOLL_MMD and abs(m) > DIVARIO_MINIMO:
                print(f"  La MMD non separa le due nuvole; {a.metrica} si'.")
                print("  A questa scala la MMD non porta informazione su "
                      "cosa rende a valle.")
        ec_v, ec_r = scarto_medio(righe, var), scarto_medio(righe, rif)
        if np.isfinite(ec_v) and np.isfinite(ec_r):
            print(f"  scarto covarianza   {var} {ec_v:.4f}   "
                  f"{rif} {ec_r:.4f}")

    # Le due ablazioni sono una CONGIUNZIONE: la distorsione delle
    # dispersioni e' la causa solo se entrambi i versi reggono. Se regge solo
    # quello negativo, si e' dimostrato che distorcere le dispersioni fa
    # danno, non che sia LA causa del divario della QGAN: resta spazio per
    # un secondo difetto che la correzione non tocca.
    buoni = [v for v in verdetti if v[4]]
    nomi_ok = {v[0] for v in buoni}
    if verdetti:
        print()
        print("-" * 70)
        print(f"LETTURA  {len(buoni)}/{len(verdetti)} versi attesi confermati")
        for var, rif, verso, m, regge in verdetti:
            print(f"  {var:>16} - {rif:<6} {m:+.4f}  atteso {verso}  "
                  f"{'confermato' if regge else 'NO'}")
        print()
        # Le correzioni applicate alla nuvola della QGAN, in ordine di
        # completezza: qgan_sd tocca le sole dispersioni per feature,
        # qgan_cov anche media e correlazioni.
        correzioni = [v for v in verdetti if v[1] == "qgan"]
        recupera = [v for v in correzioni if v[4]]
        nomi_corr = {v[0] for v in correzioni}
        if "smote_distorta" not in nomi_ok:
            print("  Nemmeno il verso negativo regge: l'ipotesi sulle")
            print("  dispersioni non e' sostenuta da nessuna ablazione, e il")
            print("  divario della QGAN va cercato altrove.")
        elif recupera:
            print(f"  Recupera: {', '.join(v[0] for v in recupera)}.")
            print("  Il difetto sta nella struttura che quella correzione")
            print("  ripristina: il generatore e' calibrabile a valle, senza")
            print("  toccare l'architettura. E' l'esito piu' utile delle tre")
            print("  ablazioni, perche' indica un intervento.")
        elif "qgan_cov" in nomi_corr:
            print("  Distorcere le dispersioni fa danno, ma nessuna correzione")
            print("  recupera — nemmeno la mappa affine completa, che riporta")
            print("  media, dispersioni e covarianza sui reali. Allora il")
            print("  problema non e' di calibrazione: e' il SUPPORTO, cioe'")
            print("  dove il generatore mette i punti, e nessuna")
            print("  trasformazione affine lo sposta. Rimanda al raggio del")
            print("  latente e all'ansatz, cioe' a O1.")
        else:
            print("  Distorcere le dispersioni fa danno, ma la correzione")
            print("  provata e' parziale: riscala lascia intatte media e")
            print("  correlazioni. Prima di concludere serve qgan_cov.")
        print("-" * 70)

    # ------------------------------------------------------------ controlli
    disponibili = [c for c in CONTROLLI
                   if c[0] in varianti and c[1] in varianti]
    if disponibili:
        print()
        print("=" * 70)
        print("CONTROLLI   contrasti a due code, nessun verso atteso")
        print("=" * 70)
        for var, rif, motivo in disponibili:
            print(f"\n{var} - {rif}")
            print(f"   {motivo}")
            oss = matrice(righe, var, rif, a.metrica, a.modello)
            m, (lo, esclude, hi), _, _ = riporta(
                oss, repliche, semi, var, 0.0, a.repliche_bs)
            if var == "qgan_cov" and rif == "gaussiana_cov":
                print()
                if not esclude:
                    print("  LETTURA  il generatore non aggiunge nulla oltre i")
                    print("  momenti. Una gaussiana sui 53 reali fa lo stesso di")
                    print("  qgan_cov, quindi il +0.048 di qgan_cov su qgan e'")
                    print("  tutto dei momenti iniettati da ricolora e il")
                    print("  circuito non porta struttura utile. Va scritto")
                    print("  cosi': il risultato e' sulla calibrazione, non")
                    print("  sulla generazione.")
                elif m > 0:
                    print("  LETTURA  il generatore aggiunge struttura oltre il")
                    print("  second'ordine: a parita' di media e covarianza la")
                    print("  nuvola del circuito rende piu' di una gaussiana.")
                    print("  E' il primo risultato positivo attribuibile alla")
                    print("  QGAN e non alla correzione.")
                else:
                    print("  LETTURA  la gaussiana BATTE qgan_cov a parita' di")
                    print("  momenti: quello che il circuito aggiunge oltre il")
                    print("  second'ordine e' dannoso, non utile.")
        print()
        print("  Confronta questi con la tabella dei LIVELLI: se gaussiana sta")
        print("  vicino a smote, il regime non premia nessun modello")
        print("  generativo, e la sezione va impostata su questo.")

    # ------------------------------------------------------- MMD e utilita'
    print()
    print("=" * 70)
    print("MMD CONTRO UTILITA' A VALLE")
    print("=" * 70)
    coppie = []
    for r in righe:
        for v, d in (r.get("varianti") or {}).items():
            if "mmd_vs_reali" in d and a.modello in d:
                coppie.append((v, d["mmd_vs_reali"],
                               d[a.modello].get(a.metrica),
                               d.get("errore_cov")))
    coppie = [c for c in coppie if c[1] is not None and c[2] is not None]
    if len(coppie) < 6:
        print("  troppe poche coppie con MMD registrata")
    else:
        from scipy import stats
        mm = np.array([c[1] for c in coppie])
        ff = np.array([c[2] for c in coppie])
        rho = stats.spearmanr(mm, ff)
        print(f"  {len(coppie)} coppie (variante x corsa)")
        print(f"  Spearman MMD vs {a.metrica}: rho = {rho.statistic:+.3f}"
              f"   p = {rho.pvalue:.4f}")
        print("  La correlazione aggregata dice se la MMD ordina i REGIMI")
        print("  (nuvole da ricampionamento contro nuvole generate). Non dice")
        print("  se ha risoluzione dove serve scegliere: quello sta sotto.")
        print()

        nomi = sorted({c[0] for c in coppie})
        med = {v: (float(np.mean([c[1] for c in coppie if c[0] == v])),
                   float(np.mean([c[2] for c in coppie if c[0] == v])))
               for v in nomi}

        # 1. RISOLUZIONE. Le varianti per cui la MMD stimata e'
        # indistinguibile da zero sono, per la MMD, equivalenti. Se a valle
        # non lo sono, la MMD non puo' scegliere fra loro.
        zero = [v for v in nomi if abs(med[v][0]) < TOLL_MMD]
        if len(zero) >= 2:
            dm = max(med[v][0] for v in zero) - min(med[v][0] for v in zero)
            fv = [med[v][1] for v in zero]
            print(f"  RISOLUZIONE  {len(zero)} varianti con MMD "
                  f"indistinguibile da zero:")
            print(f"    {', '.join(zero)}")
            print(f"    escursione MMD {dm:.4f}   escursione {a.metrica} "
                  f"{max(fv) - min(fv):.4f}")
            if max(fv) - min(fv) >= DIVARIO_MINIMO:
                print("    Dentro questo gruppo la MMD e' cieca e la metrica")
                print("    a valle no: per scegliere fra metodi buoni non")
                print("    serve.")
            print()

        # 2. CONTROESEMPIO, con tolleranza ASSOLUTA: con stime che passano
        # per lo zero una tolleranza relativa non e' definita.
        peggiore, divario = None, 0.0
        for x, y in ((x, y) for x in nomi for y in nomi if x < y):
            if (abs(med[x][0] - med[y][0]) < TOLL_MMD
                    and abs(med[x][1] - med[y][1]) > divario):
                peggiore, divario = (x, y), abs(med[x][1] - med[y][1])
        if peggiore and divario >= DIVARIO_MINIMO:
            x, y = peggiore
            print(f"  CONTROESEMPIO  {x} e {y}")
            print(f"    MMD {etichetta_mmd(med[x][0])} contro "
                  f"{etichetta_mmd(med[y][0])}   differenza "
                  f"{med[x][0] - med[y][0]:+.4f}")
            print(f"    {a.metrica} {med[x][1]:.4f} contro {med[y][1]:.4f}"
                  f"   divario {divario:.4f}")
            print("    Due nuvole che la MMD giudica equivalenti rendono in")
            print("    modo diverso a valle. Il criterio di arresto della")
            print("    QGAN lavora su differenze di questa scala.")
        elif peggiore:
            x, y = peggiore
            print(f"  La coppia piu' vicina per MMD e' ({x}, {y}), e a valle")
            print(f"  differisce di {divario:.4f}: sotto la soglia di")
            print(f"  rilevanza ({DIVARIO_MINIMO}), quindi NON e' un")
            print("  controesempio. Su queste repliche la MMD non viene")
            print("  falsificata come criterio.")
        else:
            print("  Nessuna coppia con MMD entro la tolleranza: la MMD")
            print("  distingue tutte le varianti e il confronto non si pone.")
        print()
        print(f"  {'variante':>16} {'MMD media':>12} {'scarto cov':>11} "
              f"{a.metrica:>9}")
        for v in nomi:
            sub = [c for c in coppie if c[0] == v]
            ec = [c[3] for c in sub if c[3] is not None]
            ecs = f"{np.mean(ec):11.4f}" if ec else f"{'—':>11}"
            print(f"  {v:>16} {etichetta_mmd(med[v][0]):>12} {ecs} "
                  f"{med[v][1]:9.4f}")
        print()
        print("  'scarto cov' e' lo scarto relativo di covarianza dai 53")
        print("  reali, in norma di Frobenius: e' la quantita' che qgan_cov")
        print("  manipola e che le dispersioni per feature non catturano.")

    # ----------------------------------------------------------- riepilogo
    print()
    print("=" * 70)
    print(f"RIEPILOGO   differenza da '{a.contro}', intervallo incrociato")
    print("=" * 70)
    print(f"  {'variante':>16} {'media':>9}  {'intervallo 95%':>22} "
          f"{'segni':>8} {'p':>8}")
    for v, m, lo, hi, seg, n, p in sorted(riepilogo, key=lambda x: -x[1]):
        marca = "" if (lo > 0 or hi < 0) else "  *"
        print(f"  {v:>16} {m:+9.4f}  [{lo:+8.4f}, {hi:+8.4f}] "
              f"{seg:>4}/{n:<3} {p:8.4f}{marca}")
    print()
    print("  * l'intervallo incrociato contiene lo zero: su nuove repliche e")
    print("    nuove inizializzazioni la differenza non e' distinguibile.")
    print()
    print("  Da riportare nel capitolo: media, intervallo INCROCIATO e il")
    print(f"  numero di repliche ({len(repliche)}), non le {len(righe)} corse."
          "  I semi dentro una")
    print("  replica condividono i dati e non sono repliche indipendenti")
    print("  dello split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())