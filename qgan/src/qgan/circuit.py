"""Generatore quantistico parametrizzato.

Implementa SPEC-FASE2 §3 e §7.1-7.2.

    |0>^{⊗n} -> U_enc(z) -> U_cond(c) -> ansatz(θ)[1a metà]
              -> U_cond(c) -> ansatz(θ)[2a metà] -> misura -> marginali

Vincoli rispettati:
  - 5 qubit, uno per feature;
  - encoding latente RY(z_i) su ciascun qubit (niente ZZFeatureMap);
  - condizionamento RZ dipendente da c, con data re-uploading su 2 strati
    intervallati all'ansatz;
  - ansatz EfficientSU2, reps=4, entanglement lineare, 50 parametri;
  - lettura come probabilita' marginale per qubit;
  - gradienti con parameter-shift esatta;
  - TUTTE le valutazioni di circuito raggruppate in una sola chiamata a
    ``sim.run`` tramite ``parameter_binds`` (mai un ciclo di run).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

# Ampiezza dello shift della parameter-shift rule per generatori di Pauli
# con autovalori ±1/2 (RY, RZ): df/dθ = [f(θ+π/2) - f(θ-π/2)] / 2.
_SHIFT = np.pi / 2


class QuantumGenerator:
    """Circuito generatore e relativo motore di valutazione.

    Parameters
    ----------
    n_qubits : int
        Numero di qubit, pari al numero di feature (5 da specifica).
    reps : int
        Ripetizioni dell'ansatz EfficientSU2 (4 da specifica).
    d_c : int
        Dimensione del vettore di condizionamento. 0 = non condizionato.
    cond_layers : int
        Numero di strati di condizionamento (2 da specifica). Con 2 gli
        strati sono posti prima dell'ansatz e a meta' dell'ansatz.
    max_parallel_experiments : int
        Passato ad AerSimulator; 0 = usa tutti i core disponibili.
    max_circuits_per_call : int
        Limite di sicurezza sulla memoria. Le chiamate piu' grandi vengono
        spezzate in blocchi di questa dimensione. Il default e' molto
        superiore ai 3.200 circuiti di un gradiente su batch 32, quindi in
        regime nominale la chiamata resta unica.
    """

    def __init__(
        self,
        n_qubits: int = 5,
        reps: int = 4,
        d_c: int = 0,
        cond_layers: int = 2,
        max_parallel_experiments: int = 0,
        max_circuits_per_call: int = 32_768,
    ) -> None:
        if cond_layers not in (0, 1, 2) and d_c > 0:
            raise ValueError("cond_layers deve essere 0, 1 o 2")
        self.n_qubits = int(n_qubits)
        self.reps = int(reps)
        self.d_c = int(d_c)
        self.cond_layers = int(cond_layers) if self.d_c > 0 else 0
        self.latent_dim = self.n_qubits
        self.max_circuits_per_call = int(max_circuits_per_call)

        self.n_params = 2 * self.n_qubits * (self.reps + 1)

        self._z = ParameterVector("z", self.n_qubits)
        self._th = ParameterVector("th", self.n_params)
        self._c = ParameterVector("c", self.d_c) if self.d_c > 0 else []

        self.circuit = self._build()
        self.sim = AerSimulator(
            method="statevector",
            max_parallel_experiments=max_parallel_experiments,
        )
        self._tqc = transpile(self.circuit, self.sim, optimization_level=1)

        # mask[q, s] = 1 se il bit q dello stato s vale 1 (ordinamento Qiskit
        # little-endian: il bit meno significativo e' il qubit 0).
        dim = 2**self.n_qubits
        s = np.arange(dim)
        self._mask = np.array(
            [((s >> q) & 1) for q in range(self.n_qubits)], dtype=float
        )

        # Contatore diagnostico: circuiti eseguiti e chiamate a sim.run.
        self.n_circuit_evals = 0
        self.n_sim_calls = 0

    # ------------------------------------------------------------------ #
    # Costruzione del circuito
    # ------------------------------------------------------------------ #
    def _cond_layer(self, qc: QuantumCircuit) -> None:
        """Strato di condizionamento: RZ(pi * c_j) su ciascun qubit.

        Con d_c < n_qubits il vettore c viene ripetuto ciclicamente sui
        qubit. La ripetizione dello stesso strato in due punti distinti del
        circuito e' il data re-uploading richiesto dalla specifica: non
        introduce parametri addestrabili, quindi il conteggio resta 50.
        """
        if self.d_c == 0:
            return
        for q in range(self.n_qubits):
            qc.rz(np.pi * self._c[q % self.d_c], q)

    def _entangle(self, qc: QuantumCircuit) -> None:
        for q in range(self.n_qubits - 1):
            qc.cx(q, q + 1)

    def _build(self) -> QuantumCircuit:
        n, reps = self.n_qubits, self.reps
        qc = QuantumCircuit(n, name="generator")

        # --- encoding latente ------------------------------------------
        for i in range(n):
            qc.ry(self._z[i], i)
        qc.barrier()

        # --- condizionamento, primo strato -----------------------------
        if self.cond_layers >= 1:
            self._cond_layer(qc)
            qc.barrier()

        # --- ansatz EfficientSU2 ---------------------------------------
        # Struttura: rot_0 [ent rot_1] ... [ent rot_reps].
        # Lo strato di condizionamento intermedio va inserito DOPO un blocco
        # di entanglement e PRIMA di uno strato di rotazione, che comincia
        # con RY: cosi' non si fonde con l'RZ che lo precederebbe.
        mid = (reps + 1) // 2  # con reps=4 -> 2: cond fra rot_2 e rot_3
        k = 0
        for r in range(reps + 1):
            if r > 0:
                self._entangle(qc)
            if r == mid and self.cond_layers >= 2:
                qc.barrier()
                self._cond_layer(qc)
                qc.barrier()
            for q in range(n):
                qc.ry(self._th[k], q)
                qc.rz(self._th[k + 1], q)
                k += 2
        assert k == self.n_params

        qc.save_probabilities()
        return qc

    # ------------------------------------------------------------------ #
    # Valutazione
    # ------------------------------------------------------------------ #
    def _run(self, TH: np.ndarray, Z: np.ndarray, C: np.ndarray | None) -> np.ndarray:
        """Esegue N circuiti e restituisce le marginali, shape (N, n_qubits).

        TH: (N, n_params); Z: (N, n_qubits); C: (N, d_c) oppure None.
        Tutti i circuiti sono passati in un'unica chiamata a ``sim.run``
        tramite ``parameter_binds`` (spezzata solo oltre il limite di
        memoria ``max_circuits_per_call``).
        """
        N = TH.shape[0]
        if Z.shape != (N, self.n_qubits):
            raise ValueError(f"Z ha shape {Z.shape}, attesa {(N, self.n_qubits)}")
        if self.d_c > 0:
            if C is None or C.shape != (N, self.d_c):
                raise ValueError("C mancante o di shape errata")

        out = np.empty((N, self.n_qubits))
        step = self.max_circuits_per_call
        for lo in range(0, N, step):
            hi = min(lo + step, N)
            binds = {}
            for j, p in enumerate(self._th):
                binds[p] = TH[lo:hi, j]
            for j, p in enumerate(self._z):
                binds[p] = Z[lo:hi, j]
            for j, p in enumerate(self._c):
                binds[p] = C[lo:hi, j]
            res = self.sim.run(self._tqc, parameter_binds=[binds]).result()
            self.n_sim_calls += 1
            self.n_circuit_evals += hi - lo
            probs = np.asarray(
                [res.data(i)["probabilities"] for i in range(hi - lo)]
            )
            out[lo:hi] = probs @ self._mask.T
        return out

    def forward(
        self, theta: np.ndarray, Z: np.ndarray, C: np.ndarray | None = None
    ) -> np.ndarray:
        """Campioni generati: shape (B, n_qubits), valori in [0, 1]."""
        B = Z.shape[0]
        TH = np.repeat(theta[None, :], B, axis=0)
        return self._run(TH, Z, C)

    def forward_and_jacobian(
        self, theta: np.ndarray, Z: np.ndarray, C: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forward e jacobiana esatta in un'unica chiamata al simulatore.

        Returns
        -------
        X : (B, n_qubits)
        J : (B, n_qubits, n_params)  con J[b, q, p] = d x_{b,q} / d theta_p
        """
        B, P = Z.shape[0], self.n_params

        # Costruzione del blocco unico: 1 forward + 2P shift per campione.
        n_var = 1 + 2 * P
        TH = np.repeat(theta[None, None, :], B, axis=0)  # (B,1,P)
        TH = np.repeat(TH, n_var, axis=1)  # (B, n_var, P)
        # Indicizzazione diagonale: al blocco (1+p) va applicato lo shift al
        # solo parametro p. Con uno slice sull'asse 1 e un array sull'asse 2
        # NumPy produrrebbe il prodotto cartesiano, non la diagonale.
        idx = np.arange(P)
        TH[:, 1 + idx, idx] += _SHIFT
        TH[:, 1 + P + idx, idx] -= _SHIFT

        TH = TH.reshape(B * n_var, P)
        Zr = np.repeat(Z, n_var, axis=0)
        Cr = np.repeat(C, n_var, axis=0) if self.d_c > 0 else None

        vals = self._run(TH, Zr, Cr).reshape(B, n_var, self.n_qubits)
        X = vals[:, 0, :]
        plus = vals[:, 1 : 1 + P, :]  # (B, P, n_qubits)
        minus = vals[:, 1 + P :, :]
        J = np.transpose(0.5 * (plus - minus), (0, 2, 1))  # (B, n_qubits, P)
        return X, J

    # ------------------------------------------------------------------ #
    def circuits_per_gradient(self, batch_size: int) -> int:
        """Numero di circuiti di un passo di gradiente (diagnostico)."""
        return batch_size * (1 + 2 * self.n_params)

    def describe(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "reps": self.reps,
            "d_c": self.d_c,
            "cond_layers": self.cond_layers,
            "n_params": self.n_params,
            "depth": self._tqc.depth(),
            "size": self._tqc.size(),
        }
