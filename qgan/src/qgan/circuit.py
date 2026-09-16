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


def cond_index_map(d_c: int, n_qubits: int, n_layers: int) -> list[list[int]]:
    """Indici di ``c`` letti da ciascuno strato di condizionamento.

    Per ``d_c <= n_qubits`` riproduce ESATTAMENTE la mappatura congelata da
    SPEC-FASE2 §12 (ogni strato legge ``c[q % d_c]``): tutto cio' che e' gia'
    stato eseguito con d_c in {0, 1, 3, 5} resta bit-identico e la misura di
    O1 non va rifatta.

    Per ``d_c > n_qubits`` gli strati leggono blocchi consecutivi sfalsati di
    ``n_qubits``, cosi' che ogni componente compaia almeno una volta nel
    circuito. Non aggiunge parametri addestrabili ne' profondita': usa le
    posizioni di condizionamento che gia' esistono.

    La versione precedente calcolava ``c[q % d_c]`` in entrambi gli strati.
    Con d_c=8 e 5 qubit questo lasciava fuori dal circuito c[5], c[6] e c[7],
    cioe' le ultime tre osservabili della finestra precedente nella modalita'
    autoregressiva di §6. Il discriminatore riceveva invece tutte e otto le
    componenti.

    Con 2 strati e 5 qubit ci sono 10 posizioni per 8 componenti: due
    componenti compaiono due volte. La mappatura le fa cadere su c[0] e c[1],
    cioe' sulle prime due componenti del condizionamento statico, che sono
    quelle su cui RQ2 misura la risposta.

    Esempi
    ------
    d_c=3, 5 qubit, 2 strati -> [[0,1,2,0,1], [0,1,2,0,1]]   (invariato)
    d_c=8, 5 qubit, 2 strati -> [[0,1,2,3,4], [5,6,7,0,1]]   (copertura piena)
    """
    if d_c <= 0:
        return [[] for _ in range(n_layers)]
    capienza = n_qubits * n_layers
    if d_c > capienza:
        raise ValueError(
            f"d_c={d_c} eccede la capienza del condizionamento "
            f"({n_layers} strati x {n_qubits} qubit = {capienza} posizioni). "
            "Ridurre d_c, aumentare cond_layers, oppure dichiarare "
            "esplicitamente uno schema di encoding alternativo."
        )
    if d_c <= n_qubits:
        # ramo congelato: identico al comportamento precedente alla correzione
        return [[q % d_c for q in range(n_qubits)] for _ in range(n_layers)]
    return [[(layer * n_qubits + q) % d_c for q in range(n_qubits)]
            for layer in range(n_layers)]


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
        cond_scale: float = np.pi,
        max_parallel_experiments: int = 0,
        max_circuits_per_call: int = 32_768,
    ) -> None:
        if cond_layers not in (0, 1, 2) and d_c > 0:
            raise ValueError("cond_layers deve essere 0, 1 o 2")
        self.n_qubits = int(n_qubits)
        self.reps = int(reps)
        self.d_c = int(d_c)
        self.cond_layers = int(cond_layers) if self.d_c > 0 else 0
        self.cond_scale = float(cond_scale)
        self.latent_dim = self.n_qubits
        self.max_circuits_per_call = int(max_circuits_per_call)

        self.n_params = 2 * self.n_qubits * (self.reps + 1)

        self._z = ParameterVector("z", self.n_qubits)
        self._th = ParameterVector("th", self.n_params)
        self._c = ParameterVector("c", self.d_c) if self.d_c > 0 else []
        self._cond_map = cond_index_map(
            self.d_c, self.n_qubits, self.cond_layers
        )

        self.circuit = self._build()
        # Il difetto non deve poter rientrare in silenzio: se una componente
        # di c non compare fra i parametri del circuito, la costruzione
        # fallisce qui invece di produrre un modello che ignora parte del
        # condizionamento senza segnalarlo.
        if self.d_c > 0:
            presenti = {p.name for p in self.circuit.parameters}
            assenti = [p.name for p in self._c if p.name not in presenti]
            if assenti:
                raise RuntimeError(
                    f"componenti di c assenti dal circuito: {assenti}. "
                    "La mappatura di condizionamento non copre tutto c."
                )
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
    def _cond_layer(self, qc: QuantumCircuit, layer: int) -> None:
        """Strato di condizionamento: RZ(pi * c_j) su ciascun qubit.

        Gli indici letti da questo strato sono decisi da ``cond_index_map``.
        Con d_c <= n_qubits il vettore c viene ripetuto ciclicamente sui
        qubit, identico a entrambi gli strati; con d_c > n_qubits gli strati
        leggono blocchi sfalsati, cosi' che nessuna componente resti fuori.

        La ripetizione del condizionamento in due punti distinti del circuito
        e' il data re-uploading richiesto dalla specifica: non introduce
        parametri addestrabili, quindi il conteggio resta 50.
        """
        if self.d_c == 0:
            return
        for q, j in enumerate(self._cond_map[layer]):
            qc.rz(self.cond_scale * self._c[j], q)

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
            self._cond_layer(qc, 0)
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
                self._cond_layer(qc, 1)
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
    def conditioning_sensitivity(
        self, theta: np.ndarray, n: int = 300, seed: int = 0,
        latent_hi: float | None = None,
    ) -> dict:
        """Quanto l'uscita risponde a `c`, rispetto a quanto risponde a `z`.

        Diagnostico indispensabile: la presenza dei gate di condizionamento nel
        circuito non implica che l'uscita ne dipenda in misura utilizzabile. Se
        la sensibilita' e' molto minore della dispersione indotta dal latente,
        il segnale di `c` resta sommerso e il discriminatore, che riceve `c` in
        chiaro, vince il gioco senza che il generatore possa rispondere.
        """
        if self.d_c == 0:
            raise ValueError("il generatore non e' condizionato")
        rng = np.random.default_rng(seed)
        Z = rng.uniform(0.0, np.pi if latent_hi is None else latent_hi,
                        (n, self.n_qubits))
        lo = self.forward(theta, Z, np.zeros((n, self.d_c)))
        hi = self.forward(theta, Z, np.ones((n, self.d_c)))
        per_feature = np.abs(hi.mean(axis=0) - lo.mean(axis=0))
        spread = lo.std(axis=0)
        return {
            "per_feature": per_feature.tolist(),
            "media": float(per_feature.mean()),
            "minimo": float(per_feature.min()),
            "dispersione_latente": float(spread.mean()),
            "rapporto": float(per_feature.mean() / max(spread.mean(), 1e-12)),
        }

    def input_sensitivity(
        self, theta: np.ndarray, n: int = 200, seed: int = 0,
        latent_hi: float | list | None = None, delta: float = 1.0,
    ) -> np.ndarray:
        """Risposta dell'uscita a CIASCUNA componente di c, presa da sola.

        ``conditioning_sensitivity`` varia tutte le componenti insieme e
        riporta la risposta per feature di USCITA. E' quindi cieca a una
        componente di INGRESSO che non entra nel circuito: e' la ragione per
        cui la suite ha dichiarato «c influenza ogni feature» mentre tre
        delle otto componenti non comparivano fra i parametri del circuito.

        Returns
        -------
        (d_c,) : per ogni componente j, il massimo sulle feature di uscita di
        |media(x | c_j = delta) - media(x | c = 0)|. Una componente che non
        entra nel circuito da' esattamente 0.
        """
        if self.d_c == 0:
            raise ValueError("il generatore non e' condizionato")
        rng = np.random.default_rng(seed)
        hi = np.pi if latent_hi is None else latent_hi
        Z = rng.uniform(0.0, hi, (n, self.n_qubits))
        base = self.forward(theta, Z, np.zeros((n, self.d_c))).mean(axis=0)
        out = np.zeros(self.d_c)
        for j in range(self.d_c):
            C = np.zeros((n, self.d_c))
            C[:, j] = delta
            out[j] = float(
                np.abs(self.forward(theta, Z, C).mean(axis=0) - base).max()
            )
        return out

    def circuits_per_gradient(self, batch_size: int) -> int:
        """Numero di circuiti di un passo di gradiente (diagnostico)."""
        return batch_size * (1 + 2 * self.n_params)

    def describe(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "reps": self.reps,
            "d_c": self.d_c,
            "cond_layers": self.cond_layers,
            "cond_map": [list(m) for m in self._cond_map],
            "cond_scale": self.cond_scale,
            "n_params": self.n_params,
            "depth": self._tqc.depth(),
            "size": self._tqc.size(),
        }