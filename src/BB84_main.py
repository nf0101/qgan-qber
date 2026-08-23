# Modified BB84_main.py with Synchronization Time, Classical Overhead, and Protocol Handling

import netsquid as ns
from netsquid.components.component import Message
from netsquid.components import QuantumChannel, QuantumProcessor, PhysicalInstruction
from netsquid.components.instructions import INSTR_MEASURE
from netsquid.components.models import FibreLossModel, DepolarNoiseModel, FibreDelayModel
from netsquid.nodes import Node
from netsquid.qubits import create_qubits, operate, measure, Z
from netsquid.qubits.operators import H, X
from netsquid.protocols import NodeProtocol, Protocol
from performance import PerformanceTracker
import random
import time

NUM_BITS = 1000

class AliceProtocol(NodeProtocol):
    def __init__(self, node, processor, bits, bases, perf):
        super().__init__(node)
        self.processor = processor
        self.bits = bits
        self.bases = bases
        self.perf = perf
        self.index = 0

    def run(self):
        while self.index < len(self.bits):
            bit = self.bits[self.index]
            basis = self.bases[self.index]

            qubit = create_qubits(1)[0]
            #qubit.name = f"q{self.index}"
            if bit == 1:
                operate(qubit, X)
            if basis == 1:
                operate(qubit, H)

            self.processor.put(qubit)
            tx_qubit = self.processor.pop(0)
            self.node.ports['qout'].tx_output(Message(tx_qubit, header=self.index))
            self.perf.record_epr_sent()
            self.index += 1
            yield self.await_timer(1e-6)  # simulate time delay

class BobProtocol(NodeProtocol):
    def __init__(self, node, processor, bases, results, perf):
        super().__init__(node)
        self.processor, self.bases, self.results, self.perf = processor, bases, results, perf

    def run(self):
        while True:
            yield self.await_port_input(self.node.ports['qin'])
            msg = self.node.ports['qin'].rx_input()

            if msg is None:
                continue
            i = msg.meta.get('header')  # ← al posto di int(qubit.name[1:])
            if len(self.results) < 3:
                print(f"[Bob] header={i}, meta={msg.meta}")
            qubit = msg.items[0]

            self.perf.record_qubit_received("bob")
            if self.bases[i] == 1:
                operate(qubit, H)
            m, _ = measure(qubit, observable=Z)

            self.results[i] = int(m)          # dizionario



def random_bits(n):
    return [random.randint(0, 1) for _ in range(n)]

def run_bb84():
    ns.sim_reset()
    perf = PerformanceTracker(num_pairs=NUM_BITS)
    perf.start_simulation()

    alice = Node("Alice", port_names=["qout"])
    bob = Node("Bob", port_names=["qin"])

    procA = QuantumProcessor("procA", num_positions=NUM_BITS,
        phys_instructions=[PhysicalInstruction(INSTR_MEASURE, duration=3700)])
    procB = QuantumProcessor("procB", num_positions=NUM_BITS,
        phys_instructions=[PhysicalInstruction(INSTR_MEASURE, duration=3700)])

    qchannel = QuantumChannel("qchannel", length=10,
        models={
            "delay_model": FibreDelayModel(c=2e5),
            "quantum_loss_model": FibreLossModel(p_loss_init=0.0, p_loss_length=0.2),
            "quantum_noise_model": DepolarNoiseModel(depolar_rate=0.04, time_independent=True)
        })
    alice.ports["qout"].connect(qchannel.ports["send"])
    bob.ports["qin"].connect(qchannel.ports["recv"])

    alice_bits = random_bits(NUM_BITS)
    alice_bases = random_bits(NUM_BITS)
    bob_bases = random_bits(NUM_BITS)
    bob_results = {}

    # Start time measurements for sync
    start_bob = time.time()
    bob_protocol = BobProtocol(bob, procB, bob_bases, bob_results, perf)
    bob_protocol.start()
    start_alice = time.time()
    alice_protocol = AliceProtocol(alice, procA, alice_bits, alice_bases, perf)
    alice_protocol.start()
    perf.set_sync_time(abs(start_bob - start_alice))

    # Simulate quantum transmission
    ns.sim_run()

    # Compare and sift key
    match_indices = [i for i in range(NUM_BITS) if alice_bases[i] == bob_bases[i]]
    valid_indices = [i for i in match_indices if i in bob_results]
    alice_key = [alice_bits[i] for i in valid_indices]
    bob_key = [bob_results[i] for i in valid_indices]

    # Classical communication: basis exchange
    perf.record_classical_message()  # Alice to Bob
    perf.record_classical_message()  # Bob to Alice

    mismatches = sum(a != b for a, b in zip(alice_key, bob_key))
    perf.record_basis_match(len(valid_indices))
    perf.record_mismatches(mismatches)
    perf.end_simulation()

    print(f"\n[BB84] Alice key: {''.join(map(str, alice_key))}")
    print(f"\n[BB84] Bob key:   {''.join(map(str, bob_key))}")
    perf.report()

if __name__ == "__main__":
    run_bb84()
