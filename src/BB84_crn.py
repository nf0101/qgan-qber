"""
BB84 con intercept-resend e attribuzione causale controfattuale.

IDEA CENTRALE
Ogni sorgente di stocasticita' pesca da un flusso indicizzato per posizione
(common random numbers). Di conseguenza:
  1. le condizioni sperimentali restano accoppiate posizione per posizione;
  2. il Pauli applicato dal canale e' noto, non solo stimato;
  3. i controfattuali sono ANALITICI: si calcola cosa sarebbe successo
     senza una data causa, senza rieseguire la simulazione.

La fisica resta quella di NetSquid: la formula di perdita e' quella di
FibreLossModel, e l'applicazione di Pauli con probabilita' (1-3p, p, p, p),
p = prob/4, e' esattamente cio' che fa qapi.depolarize.

STATO: non eseguito dall'autore. Le API di base (Message header, FibreLossModel,
lose_qubit, attributi custom sul qubit) sono verificate su NetSquid 1.1.8.
"""

import numpy as np

import netsquid as ns
from netsquid.components import QuantumChannel
from netsquid.components.component import Message
from netsquid.components.models import FibreLossModel, FibreDelayModel
from netsquid.components.models.qerrormodels import QuantumErrorModel
from netsquid.nodes import Node
from netsquid.protocols import NodeProtocol
from netsquid.qubits import create_qubits, operate, measure, Z
from netsquid.qubits.operators import H, X, Y
from netsquid.qubits.operators import Z as ZOP

# =====================================================================
# CONFIGURAZIONE
# =====================================================================

NUM_BITS = 4000
SEED = 42

L_TOTAL = 10.0            # km
EVE_POSITION = 0.5

P_LOSS_LENGTH = 0.2       # dB/km
DEPOLAR_TOTAL = 0.03      # probabilita' di depolarizzazione, intera tratta
P_DETECTOR_ERR = 0.01

INTERCEPT_RATE = 1.0
PULSE_PERIOD = 1e3        # ns

WINDOW = 50               # posizioni setacciate per finestra temporale

L1 = L_TOTAL * EVE_POSITION
L2 = L_TOTAL * (1.0 - EVE_POSITION)
DEPOLAR_SEG = DEPOLAR_TOTAL / 2.0

PAULI = {0: None, 1: "X", 2: "Y", 3: "Z"}


# Identificatori NUMERICI delle sorgenti. La versione precedente derivava il
# seed da `abs(hash((seed, tag, idx)))` con tag stringa: l'hash delle stringhe
# in Python e' randomizzato per processo, quindi (1, 'eve', 10) produceva
# 2600891174 in un processo e 3562695232 in un altro. Gli stessi parametri
# nominali non identificavano gli stessi flussi casuali fra esecuzioni, e la
# riproducibilita' promessa dal modulo CRN non c'era.
TAG_ID = {
    "loss_s1": 1, "loss_s2": 2, "depol_s1": 3, "depol_s2": 4,
    "eve": 5, "det": 6, "eve_meas": 7, "bob_meas": 8,
    "bits": 9, "a_bases": 10, "b_bases": 11,
    "deriva_eve": 12, "deriva_depol": 13,
}


def seme_derivato(seed, tag, idx):
    """Seed a 32 bit stabile fra processi, da SeedSequence e non da hash()."""
    if tag not in TAG_ID:
        raise KeyError(f"sorgente non registrata in TAG_ID: {tag!r}")
    ss = np.random.SeedSequence([int(seed), TAG_ID[tag], int(idx)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def stream(seed, tag, idx):
    """RNG indipendente per (esperimento, sorgente, posizione).

    Resta RandomState e non default_rng: lose_qubit e il resto del modulo
    usano l'API legacy (.random_sample()), quindi il tipo non va cambiato.
    """
    return np.random.RandomState(seme_derivato(seed, tag, idx))


# ---------------------------------------------------------------------
# Accoppiamento delle misure quantistiche
#
# measure() pesca dallo stato casuale GLOBALE di NetSquid, che non era
# inizializzato da nessuna parte e che sim_reset() non tocca. Due
# conseguenze: le corse non erano ripetibili nemmeno nello stesso processo,
# e — piu' grave — l'accoppiamento CRN saltava proprio dove serve. Cambiando
# intercept_rate cambia il numero di misure eseguite da Eve, quindi il flusso
# globale viene consumato in modo diverso e l'esito di Bob alla stessa
# posizione viene estratto da un punto diverso. La casualita' dell'esito di
# Bob quando Eve ha misurato nella base sbagliata E' il QBER da
# intercettazione: e' la grandezza su cui poggia tutto il progetto.
#
# Qui lo stato globale viene fissato dal flusso della POSIZIONE subito prima
# di ogni misura, cosi' la stessa posizione pesca lo stesso valore in ogni
# condizione. L'API va verificata sulla propria installazione: prova_crn.py
# misura se l'accoppiamento regge davvero.
ACCOPPIA_MISURE = True
_SET_RS = getattr(ns, "set_random_state", None)


def misura_accoppiata(q, i, seed, tag):
    """Misura in base Z con esito accoppiato alla posizione."""
    if ACCOPPIA_MISURE and _SET_RS is not None:
        _SET_RS(seed=seme_derivato(seed, tag, i) % (2 ** 31 - 1))
    m, _ = measure(q, observable=Z)
    return int(m)


# =====================================================================
# PROFILI TEMPORALI
#
# Le condizioni fisiche possono variare lungo la traccia. Un profilo e' un
# moltiplicatore in funzione della posizione: p(i, num_bits) -> float, con
# valore nominale moltiplicato per p.
#
# `costante` restituisce 1.0 ovunque ed e' il DEFAULT: con tutti i profili
# costanti il modulo si comporta esattamente come prima, stessa sequenza di
# estrazioni dai flussi per posizione. Il caso stazionario non e' una
# modalita' alternativa, e' un caso particolare — e c'e' una regressione che
# lo verifica contro la griglia gia' generata.
#
# ATTENZIONE alla differenza fra i due profili non costanti, perche'
# producono fenomeni diversi:
#
#   gradino  un PUNTO DI CAMBIAMENTO. Le finestre restano scambiabili dentro
#            ciascun regime: l'autocorrelazione che ne risulta e' differenza
#            di livello fra i due tratti, non memoria temporale. Serve a O3,
#            dove l'attribuzione diventa dipendente dal tempo.
#   deriva   un AR(1) lento sul moltiplicatore, quindi AUTOCORRELAZIONE VERA
#            fra finestre vicine. E' il profilo che da' a §6 qualcosa che
#            `w_{t-1}` puo' effettivamente predire.
# =====================================================================

def profilo_costante():
    """Moltiplicatore 1.0 ovunque: comportamento stazionario."""
    def p(i, num_bits):
        return 1.0
    p.nome = "costante"
    return p


def profilo_gradino(frazione=0.5, prima=0.0, dopo=1.0):
    """Punto di cambiamento a `frazione` della traccia."""
    def p(i, num_bits):
        return prima if i < frazione * num_bits else dopo
    p.nome = f"gradino@{frazione:g}:{prima:g}->{dopo:g}"
    return p


def fase_da_seme(seed):
    """Sfasamento in [0,1) del profilo intermittente, per cella."""
    return float(stream(seed, "deriva_eve", 1).random_sample())


def profilo_intermittente(periodo=0.125, duty=0.5, fase=0.0,
                          prima=0.0, dopo=1.0):
    """Onda quadra: Eve che si accende e si spegne lungo la traccia.

    `periodo` e' in frazione della corsa. Con circa 157 finestre per cella,
    0,125 corrisponde a un ciclo di ~20 finestre e a un blocco acceso di ~10:
    poiche' e' piu' corto della traccia di T=16 finestre, OGNI traccia
    contiene almeno una transizione. Un gradino unico a meta' corsa non lo
    garantirebbe — la transizione cadrebbe in una traccia su dieci.

    Rispetto al gradino e' anche piu' realistico per O3: l'attributore deve
    seguire un intercettatore che va e viene, non un cambio di regime unico.
    """
    def p(i, num_bits):
        u = ((i / max(num_bits, 1)) / periodo + fase) % 1.0
        return dopo if u < duty else prima
    p.nome = f"intermittente(T={periodo:g},duty={duty:g},fase={fase:.3f})"
    return p


def profilo_deriva(seed, tag="deriva_depol", phi=0.97, sigma=0.25,
                   blocchi=400, minimo=0.05, massimo=2.0):
    """AR(1) lento attorno a 1, costante entro blocco.

    Il moltiplicatore e' precalcolato da un flusso proprio, quindi e'
    deterministico e accoppiato fra condizioni come tutto il resto: due corse
    con parametri nominali diversi e lo stesso seme vedono la STESSA deriva.

    `blocchi` fissa la granularita' temporale: con 400 blocchi e finestre da
    circa 630 posizioni grezze su 100.000 bit, una finestra copre piu' blocchi
    e la deriva e' liscia alla scala della finestra.
    """
    rng = stream(seed, tag, 0)
    x = np.zeros(blocchi)
    innov = sigma * np.sqrt(1 - phi ** 2)
    for t in range(1, blocchi):
        x[t] = phi * x[t - 1] + rng.normal(0.0, innov)
    m = np.clip(1.0 + x, minimo, massimo)

    def p(i, num_bits):
        return float(m[min(int(i * blocchi / max(num_bits, 1)), blocchi - 1)])
    p.nome = f"deriva(phi={phi:g},sigma={sigma:g})"
    return p


def profilo_prodotto(*profili):
    """Composizione di profili: il moltiplicatore e' il prodotto."""
    def p(i, num_bits):
        v = 1.0
        for f in profili:
            v *= f(i, num_bits)
        return v
    p.nome = " x ".join(getattr(f, "nome", "?") for f in profili)
    return p


def flips(pauli, basis):
    """Il Pauli inverte l'esito della misura nella base data?

    basis 0 = rettilinea (autostati di Z): invertono X e Y.
    basis 1 = diagonale  (autostati di X): invertono Z e Y.
    """
    if pauli is None:
        return False
    if basis == 0:
        return pauli in ("X", "Y")
    return pauli in ("Z", "Y")


# =====================================================================
# MODELLI CON FLUSSO CASUALE PER POSIZIONE
# =====================================================================

class CRNFibreLoss(FibreLossModel):
    """Perdita in fibra con RNG per posizione. Formula identica all'originale."""

    def __init__(self, *a, seed=SEED, tag="loss", **kw):
        super().__init__(*a, **kw)
        self._seed, self._tag = seed, tag

    def error_operation(self, qubits, delta_time=0, **kwargs):
        for idx, q in enumerate(qubits):
            if q is None:
                continue
            prob_loss = 1 - (1 - self.p_loss_init) * np.power(
                10, -kwargs["length"] * self.p_loss_length / 10)
            rng = stream(self._seed, self._tag, getattr(q, "idx", idx))
            self.lose_qubit(qubits, idx, prob_loss, rng=rng)


class CRNDepolar(QuantumErrorModel):
    """Canale depolarizzante con Pauli esplicito, estratto per posizione.

    Equivalente a qapi.depolarize(q, prob): p = prob/4, distribuzione
    (1-3p, p, p, p) su (I, X, Y, Z). Registra quale Pauli e' stato applicato.
    """

    def __init__(self, prob, seed=SEED, tag="depol", log=None,
                 profilo=None, num_bits=None):
        super().__init__()
        self.prob, self._seed, self._tag = prob, seed, tag
        self.log = log if log is not None else {}
        self.profilo = profilo if profilo is not None else profilo_costante()
        self.num_bits = num_bits
        self.log_prob = {}          # probabilita' effettiva per posizione

    def error_operation(self, qubits, delta_time=0, **kwargs):
        for q in qubits:
            if q is None:
                continue
            i = getattr(q, "idx", None)
            prob_i = self.prob * self.profilo(i, self.num_bits)
            self.log_prob[i] = prob_i
            p = 0.25 * prob_i
            r = stream(self._seed, self._tag, i).random_sample()
            if r < p:
                op, name = X, "X"
            elif r < 2 * p:
                op, name = Y, "Y"
            elif r < 3 * p:
                op, name = ZOP, "Z"
            else:
                op, name = None, None
            if op is not None:
                operate(q, op)
            self.log[i] = name


# =====================================================================
# PROTOCOLLI
# =====================================================================

class Alice(NodeProtocol):
    def __init__(self, node, bits, bases):
        super().__init__(node)
        self.bits, self.bases = bits, bases

    def run(self):
        for i, (bit, basis) in enumerate(zip(self.bits, self.bases)):
            q = create_qubits(1)[0]
            q.idx = i                      # indice per i flussi casuali
            if bit == 1:
                operate(q, X)
            if basis == 1:
                operate(q, H)
            self.node.ports["qout"].tx_output(Message(q, header=i))
            yield self.await_timer(PULSE_PERIOD)


class Eve(NodeProtocol):
    def __init__(self, node, intercept_rate, seed, log,
                 profilo=None, num_bits=None):
        super().__init__(node)
        self.rate, self.seed, self.log = intercept_rate, seed, log
        self.profilo = profilo if profilo is not None else profilo_costante()
        self.num_bits = num_bits

    def run(self):
        while True:
            yield self.await_port_input(self.node.ports["qin"])
            msg = self.node.ports["qin"].rx_input()
            if msg is None:
                continue
            q = msg.items[0]
            i = msg.meta.get("header")

            rng = stream(self.seed, "eve", i)
            # Il tasso puo' variare lungo la traccia; il FLUSSO no. Con
            # profilo costante l'estrazione e' identica a prima, e la
            # nidificazione fra tassi diversi resta: cio' che rende possibile
            # la prova di accoppiamento di prova_crn.py.
            rate_i = self.rate * self.profilo(i, self.num_bits)
            intercepted = rng.random_sample() < rate_i
            basis = int(rng.random_sample() < 0.5)   # estratto sempre
            outcome = None

            if intercepted:
                if basis == 1:
                    operate(q, H)
                outcome = misura_accoppiata(q, i, self.seed, "eve_meas")
                q = create_qubits(1)[0]
                q.idx = i
                if outcome == 1:
                    operate(q, X)
                if basis == 1:
                    operate(q, H)

            self.log[i] = {"intercepted": intercepted, "basis": basis,
                           "outcome": outcome, "rate": rate_i}
            self.node.ports["qout"].tx_output(Message(q, header=i))


class Bob(NodeProtocol):
    def __init__(self, node, bases, results, seed, p_det):
        super().__init__(node)
        self.bases, self.results = bases, results
        self.seed, self.p_det = seed, p_det

    def run(self):
        while True:
            yield self.await_port_input(self.node.ports["qin"])
            msg = self.node.ports["qin"].rx_input()
            if msg is None:
                continue
            q = msg.items[0]
            i = msg.meta.get("header")

            if self.bases[i] == 1:
                operate(q, H)
            raw = misura_accoppiata(q, i, self.seed, "bob_meas")

            det_flip = stream(self.seed, "det", i).random_sample() < self.p_det
            self.results[i] = {"raw": raw,
                               "bit": raw ^ int(det_flip),
                               "det_flip": bool(det_flip),
                               "t": ns.sim_time()}


# =====================================================================
# ATTRIBUZIONE CONTROFATTUALE
# =====================================================================

def attribute(i, alice_bit, basis, eve, p1, p2, det_flip, bob_raw):
    """Errore osservato e controfattuali a causa singola.

    Restituisce un dizionario di booleani: l'errore che si osserverebbe
    disattivando tutte le cause tranne una, e viceversa.
    """
    f1, f2 = flips(p1, basis), flips(p2, basis)

    if not eve["intercepted"]:
        # catena deterministica: Alice -> P1 -> P2 -> Bob -> rivelatore
        err_full = f1 ^ f2 ^ det_flip
        err_solo_canale = f1 ^ f2
        err_solo_eve = False
        err_solo_riv = det_flip
    else:
        # con Eve l'esito di Bob e' quello osservato: lo si usa direttamente
        err_full = (alice_bit != (bob_raw ^ int(det_flip)))
        # senza Eve, la catena tornerebbe deterministica
        err_solo_canale = f1 ^ f2
        # senza canale ne' rivelatore, l'errore residuo e' quello di Eve:
        # se la base di Eve coincide con quella di Alice non introduce errore
        err_solo_eve = (alice_bit != bob_raw) ^ f1 ^ f2
        err_solo_riv = det_flip

    return {
        "error": bool(err_full),
        "cf_canale": bool(err_solo_canale),
        "cf_eve": bool(err_solo_eve),
        "cf_rivelatore": bool(err_solo_riv),
    }


# =====================================================================
# SIMULAZIONE
# =====================================================================

def run_bb84(intercept_rate=INTERCEPT_RATE, num_bits=NUM_BITS, seed=SEED,
             verbose=True, con_ricevute=False,
             profilo_eve=None, profilo_depol=None):
    ns.sim_reset()
    if _SET_RS is not None:
        _SET_RS(seed=int(seed))   # ripetibilita' di base, indipendente dall'accoppiamento

    alice_n = Node("Alice", port_names=["qout"])
    eve_n = Node("Eve", port_names=["qin", "qout"])
    bob_n = Node("Bob", port_names=["qin"])

    profilo_eve = profilo_eve if profilo_eve is not None else profilo_costante()
    profilo_depol = (profilo_depol if profilo_depol is not None
                     else profilo_costante())

    depol1, depol2 = {}, {}
    modelli_depol = []

    def channel(name, length, dlog, tag):
        md = CRNDepolar(DEPOLAR_SEG, seed=seed, tag=f"depol_{tag}", log=dlog,
                        profilo=profilo_depol, num_bits=num_bits)
        modelli_depol.append(md)
        return QuantumChannel(name, length=length, models={
            "delay_model": FibreDelayModel(c=2e5),
            "quantum_loss_model": CRNFibreLoss(
                p_loss_init=0.0, p_loss_length=P_LOSS_LENGTH,
                seed=seed, tag=f"loss_{tag}"),
            "quantum_noise_model": md,
        })

    ch1 = channel("ch_AE", L1, depol1, "s1")
    ch2 = channel("ch_EB", L2, depol2, "s2")

    alice_n.ports["qout"].connect(ch1.ports["send"])
    eve_n.ports["qin"].connect(ch1.ports["recv"])
    eve_n.ports["qout"].connect(ch2.ports["send"])
    bob_n.ports["qin"].connect(ch2.ports["recv"])

    # Un flusso per ruolo. Prima bit e basi venivano estratti in sequenza da
    # un unico random.Random(seed): cambiare num_bits cambiava anche le basi,
    # stesso difetto della curva rispetto a n nel pacchetto QGAN. Con flussi
    # separati num_bits tronca la stessa sequenza invece di spostarla.
    bits    = np.random.RandomState(seme_derivato(seed, "bits", 0)
                                    ).randint(0, 2, num_bits).tolist()
    a_bases = np.random.RandomState(seme_derivato(seed, "a_bases", 0)
                                    ).randint(0, 2, num_bits).tolist()
    b_bases = np.random.RandomState(seme_derivato(seed, "b_bases", 0)
                                    ).randint(0, 2, num_bits).tolist()

    eve_log, results = {}, {}
    Bob(bob_n, b_bases, results, seed, P_DETECTOR_ERR).start()
    Eve(eve_n, intercept_rate, seed, eve_log,
        profilo=profilo_eve, num_bits=num_bits).start()
    Alice(alice_n, bits, a_bases).start()
    ns.sim_run()

    sifted = [i for i in range(num_bits)
              if a_bases[i] == b_bases[i] and i in results]

    records = []
    for i in sifted:
        e = eve_log.get(i, {"intercepted": False, "basis": None,
                            "outcome": None})
        att = attribute(i, bits[i], a_bases[i], e,
                        depol1.get(i), depol2.get(i),
                        results[i]["det_flip"], results[i]["raw"])
        records.append({
            # condizioni EFFETTIVE alla posizione: non sono osservabili da un
            # operatore, stanno con i target causali
            "rate_eff": e.get("rate", intercept_rate),
            "depol_eff": sum(md.log_prob.get(i, DEPOLAR_SEG)
                             for md in modelli_depol),
            "index": i, "t": results[i]["t"], "basis": a_bases[i],
            "alice_bit": bits[i], "bob_bit": results[i]["bit"],
            "eve_intercepted": e["intercepted"],
            "p1": depol1.get(i), "p2": depol2.get(i),
            "det_flip": results[i]["det_flip"], **att,
        })

    # ---- coerenza fra errore analitico e osservato ----
    # Sulle posizioni setacciate e NON intercettate la catena
    # Alice -> P1 -> P2 -> Bob -> rivelatore e' deterministica dati i Pauli e
    # det_flip: err_full e l'errore osservato devono coincidere posizione per
    # posizione, senza eccezioni. Una discordanza significa che entra nella
    # catena qualcosa di non modellato — ed e' il controllo che avrebbe fatto
    # emergere subito la casualita' non controllata delle misure.
    #
    # Copre anche un caso che non si vede: depol.get(i) restituisce None sia
    # quando il Pauli applicato e' l'identita', sia quando il modello non ha
    # registrato nulla per quella posizione. flips(None, basis) da' False in
    # entrambi i casi, quindi un log mancante verrebbe trattato in silenzio
    # come "nessun errore".
    puliti = [r for r in records if not r["eve_intercepted"]]
    n_disc = sum(1 for r in puliti
                 if r["error"] != (r["alice_bit"] != r["bob_bit"]))

    qber = sum(r["error"] for r in records) / len(records) if records else float("nan")
    stats = {"intercept_rate": intercept_rate, "n_sifted": len(records),
             "profilo_eve": getattr(profilo_eve, "nome", "?"),
             "profilo_depol": getattr(profilo_depol, "nome", "?"),
             "qber": qber,
             "loss_rate": 1 - len(results) / num_bits,
             "n_non_intercettati": len(puliti),
             "discordanze": n_disc}

    if verbose:
        print(f"\n--- r = {intercept_rate:.2f} ---")
        print(f"setacciati : {len(records)}   perdita {stats['loss_rate']:.1%}")
        print(f"QBER       : {qber:.2%}")
        print(f"coerenza   : {n_disc} discordanze su {len(puliti)} "
              f"posizioni non intercettate")
        for k in ("cf_canale", "cf_eve", "cf_rivelatore"):
            v = sum(r[k] for r in records) / len(records)
            print(f"{k:14s}: {v:.2%}")

    if con_ricevute:
        return stats, records, set(results)
    return stats, records


SUBBLOCKS = 4  # sotto-blocchi per il calcolo di qber_stability


def windows(records, size=200, subblocks=SUBBLOCKS, *, ricevute):
    """Aggrega i record per posizione in finestre temporali.

    Ogni finestra e' una riga del dataset: feature osservabili piu'
    frazioni causali (il target di O3).

    `ricevute` e' l'insieme degli indici di posizione arrivati a Bob, cioe'
    le chiavi di `results` in run_bb84. Serve a calcolare sift_rate e
    loss_rate PER FINESTRA.

    PERCHE' E' CAMBIATA LA FIRMA. Prima questa funzione non emetteva
    sift_rate ne' loss_rate: venivano aggiunte a valle da generate_dataset.py
    pescandole da `stats`, dove loss_rate e' `1 - len(results)/num_bits`,
    uno SCALARE PER CORSA replicato identico su tutte le finestre di quella
    corsa. Nella griglia da 94.200 finestre entrambe hanno esattamente un
    valore distinto per cella, su tutte e 600 le celle.
    Conseguenze: due qubit su cinque modellavano costanti gia' interamente
    determinate da `c`; la disparita' di dispersione fra osservabili
    risultava 47x quando fra quelle che variano davvero e' 1,22x, e su quel
    47x poggiava la scelta dei raggi per feature; e in modalita'
    autoregressiva due delle cinque componenti di w_{t-1} erano costanti
    lungo la traccia, quindi ripetevano la condizione invece di portare
    informazione temporale.

    DEFINIZIONE. La finestra k copre le posizioni grezze
    (limite[k-1], limite[k]], dove limite[k] e' l'indice dell'ultimo record
    setacciato della finestra e limite[-1] = -1. Le finestre tassellano
    quindi [0, ultima_setacciata] senza buchi e senza sovrapposizioni.

        sift_rate = size / n_grezze
        loss_rate = 1 - n_ricevute / n_grezze

    Corrisponde alle definizioni per corsa: min e max di sift_rate nella
    griglia, 0,078 e 0,400, sono 0,5*(1-perdita) a 40 km e a 5 km.

    CORRELAZIONE FRA LE DUE. sift_rate e loss_rate sono fortemente
    anticorrelate, ~-0,5 misurato: e' fisica, non un artefatto della
    definizione di finestra. Le posizioni setacciate sono un sottoinsieme
    delle ricevute, quindi sift_rate ~ 0,5*(1-loss_rate) piu' il rumore
    binomiale della coincidenza di base. Verificato confrontando questa
    definizione con l'alternativa a span grezzo fisso su 626 finestre: -0,480
    contro -0,508, cioe' la stessa. La definizione a conteggio fisso di
    setacciate non aggiunge correlazione spuria e resta quella adottata.
    `n_grezze` viene comunque emesso, come diagnostica.

    qber_stability: la finestra e' divisa in `subblocks` blocchi contigui
    di uguale lunghezza; si calcola il QBER di ciascuno e se ne prende la
    deviazione standard. Cattura l'instabilita' temporale entro la
    finestra, dimensione assente dalle altre osservabili.

    AVVERTENZA. Con QBER basso il numero di errori per sotto-blocco e'
    prossimo a zero e la statistica diventa rumorosa: a QBER 1,5 % su una
    finestra da 200 posizioni ci si attendono circa 3 errori distribuiti
    su 4 blocchi. Il comportamento va verificato e riportato.
    """
    ric = np.asarray(sorted(ricevute), dtype=np.int64)
    out = []
    limite_prec = -1
    for s in range(0, len(records) - size + 1, size):
        w = records[s:s + size]
        n = len(w)

        # --- copertura grezza della finestra ---
        limite = int(w[-1]["index"])
        n_grezze = limite - limite_prec
        n_ricevute = int(np.searchsorted(ric, limite, side="right")
                         - np.searchsorted(ric, limite_prec, side="right"))
        limite_prec = limite
        sift_rate = n / n_grezze
        loss_rate = 1.0 - n_ricevute / n_grezze

        errors = [r["error"] for r in w]
        qber = sum(errors) / n

        # --- QBER per base ---
        rect = [r for r in w if r["basis"] == 0]
        diag = [r for r in w if r["basis"] == 1]
        qber_rect = (sum(r["error"] for r in rect) / len(rect)) if rect else 0.0
        qber_diag = (sum(r["error"] for r in diag) / len(diag)) if diag else 0.0

        # --- instabilita' temporale ---
        bs = max(1, n // subblocks)
        sub = [errors[i:i + bs] for i in range(0, n, bs)]
        sub = [b for b in sub if len(b) >= bs // 2]  # scarta code corte
        sub_qber = [sum(b) / len(b) for b in sub]
        qber_stability = float(np.std(sub_qber)) if len(sub_qber) > 1 else 0.0

        # --- frazioni causali (target, non osservabili) ---
        cf = {k: sum(r[k] for r in w) / n
              for k in ("cf_canale", "cf_eve", "cf_rivelatore")}
        tot = sum(cf.values())

        out.append({
            "t_start": w[0]["t"],
            "n": n,
            # ---- osservabili ----
            "qber": qber,
            "qber_rect": qber_rect,
            "qber_diag": qber_diag,
            "basis_asymmetry": abs(qber_rect - qber_diag),  # candidata A
            "qber_stability": qber_stability,  # candidata B
            "n_subblocks": len(sub_qber),
            "sift_rate": sift_rate,
            "loss_rate": loss_rate,
            "n_grezze": n_grezze,  # per misurare la correlazione indotta
            # ---- condizioni effettive nella finestra (target, non osservabili) ----
            "rate_eff": sum(r.get("rate_eff", 0.0) for r in w) / n,
            "depol_eff": sum(r.get("depol_eff", 0.0) for r in w) / n,
            # ---- controfattuali e target ----
            **{f"qber_{k}": v for k, v in cf.items()},
            **{f"f_{k.split('_')[1]}": (v / tot if tot > 0 else 0.0)
               for k, v in cf.items()},
        })
    return out


def sanity_check():
    base = DEPOLAR_TOTAL / 2 + P_DETECTOR_ERR
    print(f"QBER di base atteso ~{base:.2%}\n")
    print(f"{'r':>6} {'QBER':>8} {'atteso':>8} {'canale':>8} "
          f"{'eve':>8} {'rivel.':>8} {'discord.':>12}")
    for r in [0.0, 0.25, 0.5, 1.0]:
        s, rec = run_bb84(intercept_rate=r, verbose=False)
        # composizione XOR delle cause, non somma
        att = base + r / 4 - 2 * base * (r / 4)
        cf = {k: sum(x[k] for x in rec) / len(rec)
              for k in ("cf_canale", "cf_eve", "cf_rivelatore")}
        print(f"{r:>6.2f} {s['qber']:>8.2%} {att:>8.2%} "
              f"{cf['cf_canale']:>8.2%} {cf['cf_eve']:>8.2%} "
              f"{cf['cf_rivelatore']:>8.2%} "
              f"{s['discordanze']:>5d}/{s['n_non_intercettati']:<6d}")
    print("\nGuarda per prima la riga r = 0.00: li' Eve non intercetta mai, "
          "quindi il\ncontrollo copre il 100% delle posizioni setacciate ed "
          "e' il caso con piu'\npotere. Discordanze diverse da zero gia' a "
          "r = 0.00 indicano un problema a\nmonte (misura di Bob o modello "
          "del canale); discordanze che crescono con r\nindicano le misure "
          "di Eve.")


if __name__ == "__main__":
    sanity_check()
    _, rec, ricevute = run_bb84(intercept_rate=0.5, verbose=True, con_ricevute=True)
    w = windows(rec, ricevute=ricevute)
    print(f"\nfinestre generate: {len(w)}")
    print("prima finestra:", {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in w[0].items()})