# performance.py

import time

Num_bits=1000     # Number of bits generate for keys (Should Change)

class PerformanceTracker:
    def __init__(self, num_pairs):
        self.num_pairs = num_pairs
        self.sent_epr_pairs = 0
        self.received_qubits_alice = 0
        self.received_qubits_bob = 0
        self.matched_bases = 0
        self.mismatched_bits = 0
        self.classical_messages = 0
        self.computation_time_alice = 0
        self.computation_time_bob = 0
        self.sync_time = 0
        self.sim_start = None
        self.sim_end = None

    def start_simulation(self):
        self.sim_start = time.time()

    def end_simulation(self):
        self.sim_end = time.time()

    def record_epr_sent(self):
        self.sent_epr_pairs += 1

    def record_qubit_received(self, party):
        if party == "alice":
            self.received_qubits_alice += 1
        elif party == "bob":
            self.received_qubits_bob += 1

    def record_basis_match(self, matched):
        self.matched_bases += matched

    def record_mismatches(self, mismatches):
        self.mismatched_bits += mismatches

    def record_classical_message(self):
        self.classical_messages += 1

    def record_computation_time(self, party, duration):
        if party == "alice":
            self.computation_time_alice += duration
        elif party == "bob":
            self.computation_time_bob += duration

    def set_sync_time(self, delay):
        self.sync_time = delay

    def report(self):
        duration = self.sim_end - self.sim_start if self.sim_end else 0
        raw_key_bits = self.matched_bases
        qber = (self.mismatched_bits / raw_key_bits) if raw_key_bits > 0 else 0
        loss_rate_alice = 1 - (self.received_qubits_alice / self.sent_epr_pairs)
        loss_rate_bob = 1 - (self.received_qubits_bob / self.sent_epr_pairs)
        avg_loss = (loss_rate_alice + loss_rate_bob) / 2
        throughput = raw_key_bits / duration if duration > 0 else 0
        avg_computation = (self.computation_time_alice + self.computation_time_bob) / 2 / self.num_pairs
        channel_loss= (Num_bits-raw_key_bits)/Num_bits

        print("\n=== BB84 Protocol Performance Report ===")
        print(f"Raw Key Rate:           {raw_key_bits} bits")
        print(f"QBER:                   {qber:.2%}")
        print(f"Latency:                {duration:.3f} seconds")
        print(f"Channel Loss Rate:      {channel_loss:.2%}")
        print(f"Throughput:             {throughput:.2f} bits/sec")
        print(f"Communication Overhead: {self.classical_messages} messages")
        print(f"Synchronization Time:   {self.sync_time:.6f} seconds")
        print(f"Computation Time/Round: {avg_computation:.6f} seconds")
        print("======================================\n")
