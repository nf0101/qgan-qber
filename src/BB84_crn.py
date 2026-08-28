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

import random
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


def stream(seed, tag, idx):
    """RNG indipendente per (esperimento, sorgente, posizione)."""
    return np.random.RandomState(abs(hash((seed, tag, idx))) % (2 ** 32))


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

    def __init__(self, prob, seed=SEED, tag="depol", log=None):
        super().__init__()
        self.prob, self._seed, self._tag = prob, seed, tag
        self.log = log if log is not None else {}

    def error_operation(self, qubits, delta_time=0, **kwargs):
        p = 0.25 * self.prob
        for q in qubits:
            if q is None:
                continue
            i = getattr(q, "idx", None)
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
    def __init__(self, node, intercept_rate, seed, log):
        super().__init__(node)
        self.rate, self.seed, self.log = intercept_rate, seed, log

    def run(self):
        while True:
            yield self.await_port_input(self.node.ports["qin"])
            msg = self.node.ports["qin"].rx_input()
            if msg is None:
                continue
            q = msg.items[0]
            i = msg.meta.get("header")

            rng = stream(self.seed, "eve", i)
            intercepted = rng.random_sample() < self.rate
            basis = int(rng.random_sample() < 0.5)   # estratto sempre
            outcome = None

            if intercepted:
                if basis == 1:
                    operate(q, H)
                m, _ = measure(q, observable=Z)
                outcome = int(m)
                q = create_qubits(1)[0]
                q.idx = i
                if outcome == 1:
                    operate(q, X)
                if basis == 1:
                    operate(q, H)

            self.log[i] = {"intercepted": intercepted,
                           "basis": basis, "outcome": outcome}
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
            m, _ = measure(q, observable=Z)
            raw = int(m)

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
             verbose=True):
    ns.sim_reset()
    rng = random.Random(seed)

    alice_n = Node("Alice", port_names=["qout"])
    eve_n = Node("Eve", port_names=["qin", "qout"])
    bob_n = Node("Bob", port_names=["qin"])

    depol1, depol2 = {}, {}

    def channel(name, length, dlog, tag):
        return QuantumChannel(name, length=length, models={
            "delay_model": FibreDelayModel(c=2e5),
            "quantum_loss_model": CRNFibreLoss(
                p_loss_init=0.0, p_loss_length=P_LOSS_LENGTH,
                seed=seed, tag=f"loss_{tag}"),
            "quantum_noise_model": CRNDepolar(
                DEPOLAR_SEG, seed=seed, tag=f"depol_{tag}", log=dlog),
        })

    ch1 = channel("ch_AE", L1, depol1, "s1")
    ch2 = channel("ch_EB", L2, depol2, "s2")

    alice_n.ports["qout"].connect(ch1.ports["send"])
    eve_n.ports["qin"].connect(ch1.ports["recv"])
    eve_n.ports["qout"].connect(ch2.ports["send"])
    bob_n.ports["qin"].connect(ch2.ports["recv"])

    bits = [rng.randint(0, 1) for _ in range(num_bits)]
    a_bases = [rng.randint(0, 1) for _ in range(num_bits)]
    b_bases = [rng.randint(0, 1) for _ in range(num_bits)]

    eve_log, results = {}, {}
    Bob(bob_n, b_bases, results, seed, P_DETECTOR_ERR).start()
    Eve(eve_n, intercept_rate, seed, eve_log).start()
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
            "index": i, "t": results[i]["t"], "basis": a_bases[i],
            "alice_bit": bits[i], "bob_bit": results[i]["bit"],
            "eve_intercepted": e["intercepted"],
            "p1": depol1.get(i), "p2": depol2.get(i),
            "det_flip": results[i]["det_flip"], **att,
        })

    qber = sum(r["error"] for r in records) / len(records) if records else float("nan")
    stats = {"intercept_rate": intercept_rate, "n_sifted": len(records),
             "qber": qber,
             "loss_rate": 1 - len(results) / num_bits}

    if verbose:
        print(f"\n--- r = {intercept_rate:.2f} ---")
        print(f"setacciati : {len(records)}   perdita {stats['loss_rate']:.1%}")
        print(f"QBER       : {qber:.2%}")
        for k in ("cf_canale", "cf_eve", "cf_rivelatore"):
            v = sum(r[k] for r in records) / len(records)
            print(f"{k:14s}: {v:.2%}")

    return stats, records


def windows(records, size=WINDOW):
    """Aggrega i record in finestre: e' la riga del dataset per la QGAN."""
    out = []
    for s in range(0, len(records) - size + 1, size):
        w = records[s:s + size]
        n = len(w)
        cf = {k: sum(r[k] for r in w) / n
              for k in ("cf_canale", "cf_eve", "cf_rivelatore")}
        tot = sum(cf.values())
        out.append({
            "t_start": w[0]["t"], "n": n,
            "qber": sum(r["error"] for r in w) / n,
            "qber_rect": (sum(r["error"] for r in w if r["basis"] == 0) /
                          max(1, sum(1 for r in w if r["basis"] == 0))),
            "qber_diag": (sum(r["error"] for r in w if r["basis"] == 1) /
                          max(1, sum(1 for r in w if r["basis"] == 1))),
            **{f"qber_{k}": v for k, v in cf.items()},
            # vettore di frazioni causali: il target di O3
            **{f"f_{k.split('_')[1]}": (v / tot if tot > 0 else 0.0)
               for k, v in cf.items()},
        })
    return out


def sanity_check():
    base = DEPOLAR_TOTAL / 2 + P_DETECTOR_ERR
    print(f"QBER di base atteso ~{base:.2%}\n")
    print(f"{'r':>6} {'QBER':>8} {'atteso':>8} {'canale':>8} "
          f"{'eve':>8} {'rivel.':>8}")
    for r in [0.0, 0.25, 0.5, 1.0]:
        s, rec = run_bb84(intercept_rate=r, verbose=False)
        # composizione XOR delle cause, non somma
        att = base + r / 4 - 2 * base * (r / 4)
        cf = {k: sum(x[k] for x in rec) / len(rec)
              for k in ("cf_canale", "cf_eve", "cf_rivelatore")}
        print(f"{r:>6.2f} {s['qber']:>8.2%} {att:>8.2%} "
              f"{cf['cf_canale']:>8.2%} {cf['cf_eve']:>8.2%} "
              f"{cf['cf_rivelatore']:>8.2%}")


if __name__ == "__main__":
    sanity_check()
    _, rec = run_bb84(intercept_rate=0.5, verbose=True)
    w = windows(rec)
    print(f"\nfinestre generate: {len(w)}")
    print("prima finestra:", {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in w[0].items()})