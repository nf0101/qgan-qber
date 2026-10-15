"""Separa l'errore di media dall'errore di forma sulla griglia QKD reale.

COSA DICE LA MISURA ATTUALE. Su cinque semi e 23 celle escluse il
condizionato batte vicino sulla MMD (+0.0885, intervallo incrociato che
esclude lo zero) ma pareggia sulla media (+0.0105, contiene lo zero); e
perde contro `lineare` su entrambi gli assi (-0.0459 MAE, -0.0628 MMD,
0/5). Nei livelli: prendere le finestre VERE della cella adiacente e
correggerne solo la media porta la MMD da 0.1541 a 0.0028, fattore 54.
Tutto il divario distribuzionale del vicino era uno scostamento di media.
Il condizionato sta a 0.0656, con errore di media 0.0513 contro 0.0054
dell'interpolante: dieci volte tanto.

L'IPOTESI DA FALSIFICARE. Se l'intero deficit di MMD del condizionato e'
attribuibile al suo errore di media, allora ricentrando il campione
generato sulla media vera la MMD deve crollare verso i 0.003
dell'interpolante. In quel caso forme marginali e correlazioni prodotte
dal circuito sono corrette e l'unico difetto e' la risposta della media
al condizionamento — enunciato molto piu' specifico, e che dice dove
intervenire. Se invece resta alta, il difetto e' distribuzionale e il
condizionamento non e' il problema principale.

E' la trasposizione al ramo QKD del controllo che su IDS ha separato
`qgan` da `qgan_cov`: la' si iniettavano media e covarianza dei reali,
qui si inietta la sola media vera della cella. Non serve riaddestrare
niente: il campione e' gia' generato.

IL DIFETTO CHE VA SISTEMATO PRIMA. Oggi `cond.sample(n_gen, ch)` viene
chiamata DUE volte per cella: alla riga 170 per la media e alla riga 203
per la MMD. Sono due estrazioni indipendenti, quindi `mae_condizionato` e
`mmd_condizionato` della stessa cella si riferiscono a campioni diversi.
Non e' sbagliato — entrambi stimano la stessa quantita' — ma aggiunge
varianza evitabile, e soprattutto rende il ricentraggio senza senso: non
si puo' ricentrare un campione e confrontarne la MMD con quella di un
altro. Il flag riusa un campione solo per media, MMD e MMD ricentrata.

ORDINE DEL FLUSSO CASUALE. Riusare il campione CAMBIA il numero di
estrazioni per cella, da due a una per ciascun ramo, quindi una corsa con
`--ricentra` NON riproduce `rq2_reale_5semi.jsonl`: i numeri divergono
dalla seconda cella. E' inevitabile e va trattato come misura separata —
`ricentra` entra in `config`, quindi la firma cambia, i record non
collidono e la provenienza resta leggibile. Senza il flag le due
estrazioni restano dove erano, nello stesso ordine relativo, e il file e'
bit-identico: la patch si puo' applicare senza invalidare nulla.

CONTROLLO GRATUITO. Con il flag attivo `mmd_condizionato` deve cambiare
solo per rumore di campionamento rispetto alla corsa senza flag. Se
cambia in modo sistematico, il riuso del campione ha toccato qualcosa che
non doveva e il risultato non va letto.

Il ramo non condizionato riceve lo stesso trattamento: costa una riga e
risponde a una domanda in piu'. Se `mmd_incondizionato_ricentrato` e
`mmd_condizionato_ricentrato` coincidono, il condizionamento non aggiunge
nulla oltre la media, che e' la lettura suggerita da tutto il resto. Se
il condizionato ricentrato e' piu' basso, il condizionamento migliora
anche la forma.

    cd ~/PycharmProjects/Quantum/qgan-qber
    python patch_ricentraggio.py qgan/scripts/rq2_reale.py

    # la corsa: file proprio, la firma cambia da sola
    python qgan/scripts/rq2_reale.py --dati data/qber_grid.csv \\
        --seeds 5 --ricentra --backend denso \\
        --out qgan/results/rq2_ricentrato.jsonl

    # e poi il contrasto fra i due campi dentro la stessa cella
    python contrasto_campi.py --dir qgan/results/rq2_ricentrato.jsonl \\
        --famiglie mmd --riferimento condizionato_ricentrato
"""

import shutil
import sys

# ---------------------------------------------------------------- 1
V_GLOB = '''LATENT_HI_UNCOND = np.pi'''
N_GLOB = '''# Riusa un campione solo per cella fra media, MMD e MMD ricentrata.
# Cambia il numero di estrazioni, quindi i numeri non sono confrontabili
# con una corsa senza il flag: sta in `config`, la firma lo distingue.
RICENTRA = False

LATENT_HI_UNCOND = np.pi'''

# ---------------------------------------------------------------- 2
V_DRAW = '''        mu_vero = Xh.mean(axis=0)
        mu_cond = cond.sample(n_gen, ch).mean(axis=0)
        mu_unc = uncond.sample(n_gen).mean(axis=0)
'''
N_DRAW = '''        mu_vero = Xh.mean(axis=0)
        if RICENTRA:
            # Un campione solo per ramo: media e MMD devono riferirsi allo
            # STESSO oggetto, altrimenti ricentrarlo non misura niente.
            X_cond = cond.sample(n_gen, ch)
            X_unc = uncond.sample(n_gen)
            mu_cond, mu_unc = X_cond.mean(axis=0), X_unc.mean(axis=0)
        else:
            # Due estrazioni, nell'ordine originale: file bit-identico.
            X_cond = X_unc = None
            mu_cond = cond.sample(n_gen, ch).mean(axis=0)
            mu_unc = uncond.sample(n_gen).mean(axis=0)
'''

# ---------------------------------------------------------------- 3
V_MMD = '''        X_ban = Xt[rng_diag.choice(len(Xt), n_gen, replace=False)]

        per_cella.append({'''
N_MMD = '''        X_ban = Xt[rng_diag.choice(len(Xt), n_gen, replace=False)]

        if RICENTRA:
            # Il ricentraggio inietta la sola media vera e lascia intatto
            # tutto il resto: se la MMD crolla, la forma era giusta.
            extra = {
                "mmd_condizionato_ricentrato": float(metrics.mmd_rbf(
                    Xh, np.clip(X_cond + (mu_vero - mu_cond), 0.0, 1.0),
                    sigma=sig)),
                "mmd_incondizionato_ricentrato": float(metrics.mmd_rbf(
                    Xh, np.clip(X_unc + (mu_vero - mu_unc), 0.0, 1.0),
                    sigma=sig)),
            }
        else:
            # Le due estrazioni della MMD restano QUI, dopo i prelievi di
            # rng_diag, come nell'originale: l'ordine e' preservato.
            X_cond = cond.sample(n_gen, ch)
            X_unc = uncond.sample(n_gen)
            extra = {}
        mmd_c = float(metrics.mmd_rbf(Xh, X_cond, sigma=sig))
        mmd_u = float(metrics.mmd_rbf(Xh, X_unc, sigma=sig))

        per_cella.append({'''

# ---------------------------------------------------------------- 4
V_VOCE = '''            "mmd_condizionato": float(metrics.mmd_rbf(
                Xh, cond.sample(n_gen, ch), sigma=sig)),
            "mmd_incondizionato": float(metrics.mmd_rbf(
                Xh, uncond.sample(n_gen), sigma=sig)),
            "mmd_vicino": float(metrics.mmd_rbf(Xh, X_vic, sigma=sig)),'''
N_VOCE = '''            "mmd_condizionato": mmd_c,
            "mmd_incondizionato": mmd_u,
            **extra,
            "mmd_vicino": float(metrics.mmd_rbf(Xh, X_vic, sigma=sig)),'''

# ---------------------------------------------------------------- 5
V_AGG = '''        "mmd_banale_m": agg("mmd_banale") / len(per_cella),'''
N_AGG = '''        "mmd_banale_m": agg("mmd_banale") / len(per_cella),
        **({
            "mmd_condizionato_ricentrato_m":
                agg("mmd_condizionato_ricentrato") / len(per_cella),
            "mmd_incondizionato_ricentrato_m":
                agg("mmd_incondizionato_ricentrato") / len(per_cella),
            # Quanto del deficit del condizionato era errore di media:
            # 0 = niente, 1 = tutto. NON e' limitato superiormente a 1.
            # La MMD^2 e' stimata con una U-statistica non distorta, che
            # diventa negativa quando la MMD vera e' zero: se il
            # ricentraggio porta la nuvola sul pavimento di rumore,
            # l'aggregato ricentrato puo' essere negativo e la quota
            # supera 1. Non e' un errore ed e' la conferma piu' netta
            # possibile dell'ipotesi; ma in quel caso il numero non si
            # legge come una frazione, si legge come "il ricentraggio ha
            # esaurito il divario" e si riporta la MMD ricentrata grezza
            # accanto a quella di `lineare`.
            "quota_media": float(
                1.0 - agg("mmd_condizionato_ricentrato")
                / max(agg("mmd_condizionato"), 1e-12)),
        } if RICENTRA else {}),'''

# ---------------------------------------------------------------- 6
V_FLAG = '''    ap.add_argument("--backend", default="denso", choices=["denso", "aer"])'''
N_FLAG = '''    ap.add_argument("--ricentra", action="store_true",
                    help="riusa un campione solo per cella e aggiunge la "
                         "MMD del campione ricentrato sulla media vera. "
                         "Separa errore di media ed errore di forma. "
                         "Cambia il flusso casuale: scrivere su un file "
                         "proprio, la firma lo distingue da sola")
    ap.add_argument("--backend", default="denso", choices=["denso", "aer"])'''

# ---------------------------------------------------------------- 7
V_CONF = '''    config = dict(FROZEN, steps=a.steps, dati=a.dati,'''
N_CONF = '''    global RICENTRA
    RICENTRA = a.ricentra

    config = dict(FROZEN, steps=a.steps, dati=a.dati,'''

# ---------------------------------------------------------------- 7b
V_CHIAVE = '''                  seed_base=SEED_BASE)
    fr = corsa.firma(config)'''
N_CHIAVE = '''                  seed_base=SEED_BASE)
    if a.ricentra:
        # Entra in `config` SOLO quando e' attivo. Cosi' una corsa senza
        # il flag conserva la firma delle corse precedenti — che e' il
        # punto: senza il flag non cambia niente, e i record non devono
        # separarsi per un parametro che non ha avuto effetto. Con il
        # flag la firma cambia, e i due protocolli non si mescolano.
        config["ricentra"] = True
    fr = corsa.firma(config)'''

# ---------------------------------------------------------------- 8
V_STAMPA = '''        print(f"  SKILL MMD vs banale {r['skill_mmd_vs_banale']:+.4f}   "
              f"vs vicino {r['skill_mmd_vs_vicino']:+.4f}   "
              f"vs lineare {r['skill_mmd_vs_lineare']:+.4f}", flush=True)'''
N_STAMPA = '''        print(f"  SKILL MMD vs banale {r['skill_mmd_vs_banale']:+.4f}   "
              f"vs vicino {r['skill_mmd_vs_vicino']:+.4f}   "
              f"vs lineare {r['skill_mmd_vs_lineare']:+.4f}", flush=True)
        if a.ricentra:
            print(f"  RICENTRATA  cond {r['mmd_condizionato_m']:.5f}"
                  f" -> {r['mmd_condizionato_ricentrato_m']:.5f}"
                  f"   incond {r['mmd_incondizionato_m']:.5f}"
                  f" -> {r['mmd_incondizionato_ricentrato_m']:.5f}"
                  f"   quota di media {r['quota_media']:.3f}", flush=True)
            print(f"  riferimento: lineare {r['mmd_lineare_m']:.5f}. Se la "
                  f"ricentrata gli si avvicina, la forma era giusta e il "
                  f"difetto e' la media.", flush=True)'''

# il campo stampato sopra non esiste ancora fra gli aggregati
V_INCM = '''        "mmd_vicino_m": agg("mmd_vicino") / len(per_cella),'''
N_INCM = '''        "mmd_incondizionato_m": agg("mmd_incondizionato") / len(per_cella),
        "mmd_vicino_m": agg("mmd_vicino") / len(per_cella),'''

MODIFICHE = [
    ("globale RICENTRA", V_GLOB, N_GLOB),
    ("estrazione dei campioni", V_DRAW, N_DRAW),
    ("calcolo delle MMD prima della voce", V_MMD, N_MMD),
    ("voce per cella", V_VOCE, N_VOCE),
    ("aggregati ricentrati", V_AGG, N_AGG),
    ("aggregato mmd_incondizionato_m", V_INCM, N_INCM),
    ("flag --ricentra", V_FLAG, N_FLAG),
    ("globale dal flag", V_CONF, N_CONF),
    ("ricentra in config solo se attivo", V_CHIAVE, N_CHIAVE),
    ("stampa del ricentraggio", V_STAMPA, N_STAMPA),
]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    percorso = sys.argv[1]
    with open(percorso, encoding="utf-8") as f:
        testo = f.read()

    if "RICENTRA" in testo:
        print("La patch sembra gia' applicata: `RICENTRA` e' presente.")
        return 1

    for nome, vecchio, nuovo in MODIFICHE:
        n = testo.count(vecchio)
        if n != 1:
            print(f"ABORT [{nome}]: {n} occorrenze, attesa 1. "
                  "Nessuna scrittura.")
            return 1
        testo = testo.replace(vecchio, nuovo, 1)

    compile(testo, percorso, "exec")   # sintassi, prima di scrivere

    shutil.copy2(percorso, percorso + ".prima_ricentraggio")
    with open(percorso, "w", encoding="utf-8") as f:
        f.write(testo)

    print(f"{len(MODIFICHE)} modifiche applicate a {percorso}")
    print(f"copia dell'originale in {percorso}.prima_ricentraggio")
    print()
    print("Senza --ricentra il flusso casuale e' invariato: la prima cosa")
    print("da fare e' una corsa di controllo su un seme, che deve")
    print("riprodurre i numeri di rq2_reale_5semi.jsonl:")
    print()
    print("  python qgan/scripts/rq2_reale.py --dati data/qber_grid.csv \\")
    print("      --seeds 1 --backend denso --out /tmp/controllo.jsonl")
    print("  # atteso: MAE cond 0.04227, come il seme 0 del file esistente")
    print()
    print("Se il controllo passa, la corsa vera:")
    print()
    print("  python qgan/scripts/rq2_reale.py --dati data/qber_grid.csv \\")
    print("      --seeds 5 --ricentra --backend denso \\")
    print("      --out qgan/results/rq2_ricentrato.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())