"""QGAN ibrida: generatore quantistico, discriminatore classico.

Implementa l'interfaccia congelata in SPEC-FASE2 §2:

    class QGAN:
        def fit(self, X, c=None, **kw) -> "QGAN"
        def sample(self, n, c=None) -> np.ndarray

Nessun ramo condizionale sul dominio all'interno della classe. Il dominio
entra esclusivamente attraverso ``X`` e ``c``.
"""

from __future__ import annotations

import json
import time
import warnings

import numpy as np

from .circuit import QuantumGenerator
from .discriminator import Adam, Discriminator
from .metrics import median_bandwidth, mmd_rbf, mode_coverage


class QGAN:
    """Generative adversarial network con generatore variazionale.

    I valori di default riproducono la configurazione congelata
    (SPEC-FASE2 §3, §4, §7). I parametri dichiarati liberi in §12 —
    learning rate, batch, epoche, architettura del discriminatore —
    restano esposti, ma devono restare identici nei due domini.
    """

    def __init__(
        self,
        n_features: int = 5,
        d_c: int = 0,
        reps: int = 4,
        # --- liberi (§12), da tenere identici nei due domini ---
        lr_g: float = 0.01,
        lr_d: float = 0.001,
        batch_size: int = 32,
        epochs: int = 200,
        hidden: tuple[int, ...] = (32, 16),
        leak: float = 0.01,
        label_smoothing: float = 0.0,
        d_weight_decay: float = 0.0,
        # --- congelati ---
        theta_init: float = 0.1,
        patience: int = 20,
        # --- budget in passi (vedi nota nel docstring di fit) ---
        max_steps: int | None = None,
        eval_every: int | None = None,
        # --- pratici ---
        val_fraction: float = 0.2,
        n_monitor: int = 256,
        monitor_repeats: int = 3,
        mmd_smooth: int = 1,
        seed: int = 0,
        verbose: int = 1,
        max_parallel_experiments: int = 0,
    ) -> None:
        self.n_features = int(n_features)
        self.d_c = int(d_c)
        self.reps = int(reps)
        self.lr_g, self.lr_d = float(lr_g), float(lr_d)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.hidden, self.leak = tuple(hidden), float(leak)
        self.label_smoothing = float(label_smoothing)
        self.d_weight_decay = float(d_weight_decay)
        self.theta_init = float(theta_init)
        self.patience = int(patience)
        self.max_steps = None if max_steps is None else int(max_steps)
        self.eval_every = None if eval_every is None else int(eval_every)
        self.val_fraction = float(val_fraction)
        self.n_monitor = int(n_monitor)
        self.monitor_repeats = max(1, int(monitor_repeats))
        self.mmd_smooth = max(1, int(mmd_smooth))
        self.seed = int(seed)
        self.verbose = int(verbose)

        self.rng = np.random.default_rng(seed)
        self.gen = QuantumGenerator(
            n_qubits=self.n_features,
            reps=self.reps,
            d_c=self.d_c,
            max_parallel_experiments=max_parallel_experiments,
        )
        self.latent_dim = self.gen.latent_dim
        self.n_params = self.gen.n_params

        # theta vicino all'identita' (§7): uniforme in [-0.1, 0.1]
        self.theta = self.rng.uniform(
            -self.theta_init, self.theta_init, size=self.n_params
        )
        self.disc = Discriminator(
            d_in=self.n_features + self.d_c,
            hidden=self.hidden,
            leak=self.leak,
            lr=self.lr_d,
            weight_decay=self.d_weight_decay,
            rng=self.rng,
        )
        self.opt_g = Adam([self.theta], lr=self.lr_g)
        self.history: dict[str, list] = {}
        self.fitted_ = False

    # ------------------------------------------------------------------ #
    # Utilita'
    # ------------------------------------------------------------------ #
    def _check(self, X, c):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(
                f"X deve avere shape (n, {self.n_features}), ricevuta {X.shape}"
            )
        if X.min() < -1e-9 or X.max() > 1 + 1e-9:
            raise ValueError("X deve avere valori in [0, 1]")
        if self.d_c == 0:
            if c is not None:
                warnings.warn("d_c=0: il condizionamento fornito viene ignorato")
            return X, None
        if c is None:
            raise ValueError(f"d_c={self.d_c}: c e' obbligatorio")
        c = np.asarray(c, dtype=float)
        if c.ndim == 1:
            c = np.repeat(c[None, :], len(X), axis=0)
        if c.shape != (len(X), self.d_c):
            raise ValueError(
                f"c deve avere shape ({len(X)}, {self.d_c}), ricevuta {c.shape}"
            )
        if c.min() < -1e-9 or c.max() > 1 + 1e-9:
            raise ValueError("c deve avere valori in [0, 1]")
        return X, c

    def _latent(self, n: int) -> np.ndarray:
        return self.rng.uniform(0.0, np.pi, size=(n, self.latent_dim))

    def _disc_input(self, X: np.ndarray, c: np.ndarray | None) -> np.ndarray:
        return X if c is None else np.hstack([X, c])

    # ------------------------------------------------------------------ #
    # Addestramento
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X,
        c=None,
        X_val=None,
        c_val=None,
        modes=None,
        **kw,
    ) -> "QGAN":
        """Addestra il modello.

        Parameters
        ----------
        X : (n, n_features), valori in [0, 1]
        c : (n, d_c) o None
        X_val, c_val : insieme per il monitoraggio MMD e l'arresto anticipato.
            Se assenti si preleva ``val_fraction`` da X; se X e' troppo
            piccolo (< 50) si usa X stesso e viene emesso un avviso.
        modes : etichette di modalita' dei campioni reali, per la copertura.

        Notes
        -----
        Budget in passi. La specifica fissa 200 epoche e pazienza 20, ma
        l'epoca non e' un'unita' confrontabile fra i due domini: con la
        classe r2l a 1:1000 valgono 53 campioni e quindi 2 passi del
        generatore, mentre la griglia QBER ne conta 94.200. A parita' di
        epoche il generatore riceve tre ordini di grandezza di
        aggiornamenti in piu' in un dominio che nell'altro, il che rende
        O1 non confrontabile. ``max_steps`` e ``eval_every`` esprimono il
        budget e la cadenza del monitoraggio in passi del generatore. Con
        entrambi a ``None`` il comportamento coincide esattamente con
        quello descritto dalla specifica.

        Selezione del checkpoint. Con ``mmd_smooth`` maggiore di 1 il
        checkpoint migliore e l'arresto anticipato usano la media delle
        ultime valutazioni della MMD anziche' il singolo valore. Sotto le
        cinquanta unita' di validazione la MMD di monitoraggio e' rumorosa
        al punto che il minimo grezzo puo' cadere su un'oscillazione: e'
        quanto accaduto nel toy A, dove il checkpoint ripristinato aveva
        una MMD reale peggiore di quella all'inizializzazione. La media
        mobile e' riferita all'ultima valutazione della finestra, non al
        suo centro, quindi introduce un ritardo di ``mmd_smooth - 1``
        valutazioni: e' accettabile perche' il costo di un checkpoint
        lievemente in ritardo e' minore di quello di un checkpoint scelto
        dal rumore.
        """
        for k in kw:
            warnings.warn(f"argomento ignorato: {k}")
        X, c = self._check(X, c)
        n = len(X)
        modes = None if modes is None else np.asarray(modes)
        if modes is not None and len(modes) != n:
            raise ValueError("modes deve avere la stessa lunghezza di X")

        # --- insieme di monitoraggio ----------------------------------
        if X_val is None:
            if n >= 50 and self.val_fraction > 0:
                idx = self.rng.permutation(n)
                n_val = max(2, int(round(self.val_fraction * n)))
                vi, ti = idx[:n_val], idx[n_val:]
                X_val, c_val = X[vi], (c[vi] if c is not None else None)
                X, c = X[ti], (c[ti] if c is not None else None)
                modes = None if modes is None else modes[ti]
                n = len(X)
            else:
                warnings.warn(
                    "dataset troppo piccolo per uno split: l'MMD di arresto "
                    "e' calcolato sugli stessi dati di addestramento"
                )
                X_val, c_val = X, c
        else:
            X_val, c_val = self._check(X_val, c_val)

        B = min(self.batch_size, n)
        steps_per_epoch = max(1, int(np.ceil(n / B)))
        max_steps = self.max_steps or self.epochs * steps_per_epoch
        eval_every = self.eval_every or steps_per_epoch
        n_mon = min(self.n_monitor, len(X_val))
        # Larghezza di banda fissata una volta sui soli dati reali. Se fosse
        # ricavata a ogni valutazione dall'unione di reali e generati, il
        # kernel cambierebbe a ogni passo e le MMD successive non sarebbero
        # confrontabili fra loro: il minimo su cui si sceglie il checkpoint
        # sarebbe in parte un artefatto della banda.
        self.mmd_sigma_ = float(median_bandwidth(X_val, rng=self.rng))
        best = {"mmd": np.inf, "step": -1, "theta": self.theta.copy()}
        bad = 0
        self.history = {
            k: []
            for k in (
                "step",
                "epoch",
                "loss_d",
                "loss_g",
                "grad_var",
                "grad_norm",
                "mmd_val",
                "mmd_smooth",
                "mode_coverage",
                "mode_entropy",
                "seconds",
            )
        }

        if self.verbose:
            print(
                f"[QGAN] n={n} val={len(X_val)} batch={B} params={self.n_params} "
                f"d_c={self.d_c} passi={max_steps} valutazione ogni {eval_every} "
                f"circuiti/passo={self.gen.circuits_per_gradient(B)}"
            )

        recent: list[float] = []
        perm = self.rng.permutation(n)
        cur = 0
        acc = {"ld": 0.0, "lg": 0.0, "gv": 0.0, "gn": 0.0, "nb": 0}
        t0 = time.time()
        stop = False

        for step in range(1, max_steps + 1):
            if cur + B > n:
                perm = self.rng.permutation(n)
                cur = 0
            bidx = perm[cur : cur + B]
            cur += B
            Xb = X[bidx]
            cb = c[bidx] if c is not None else None
            Z = self._latent(len(bidx))

            # Unica chiamata al simulatore: forward + jacobiana. Il batch
            # generato serve sia al passo del discriminatore sia a quello del
            # generatore; theta non cambia fra i due, quindi X_fake resta
            # valido dopo l'aggiornamento di D.
            X_fake, J = self.gen.forward_and_jacobian(self.theta, Z, cb)

            acc["ld"] += self.disc.train_step(
                self._disc_input(Xb, cb),
                self._disc_input(X_fake, cb),
                label_smoothing=self.label_smoothing,
            )
            loss_g, dX = self.disc.generator_signal(self._disc_input(X_fake, cb))
            dX = dX[:, : self.n_features]  # la parte su c non e' generata
            grad = np.einsum("bq,bqp->p", dX, J)
            self.opt_g.step([grad])

            acc["lg"] += loss_g
            acc["gv"] += float(np.var(grad))
            acc["gn"] += float(np.linalg.norm(grad))
            acc["nb"] += 1

            if step % eval_every and step != max_steps:
                continue

            # --- monitoraggio ---------------------------------------
            mmds = []
            cov = ent = float("nan")
            for _ in range(self.monitor_repeats):
                sel = self.rng.choice(len(X_val), n_mon, replace=False)
                Xg = self.sample(n_mon, c_val[sel] if c_val is not None else None)
                mmds.append(mmd_rbf(X_val[sel], Xg, sigma=self.mmd_sigma_))
            mmd = float(np.mean(mmds))
            if modes is not None:
                mc = mode_coverage(X, modes, Xg)
                cov, ent = mc["coverage"], mc["entropy"]

            nb = max(acc["nb"], 1)
            self.history["step"].append(step)
            self.history["epoch"].append(step / steps_per_epoch)
            self.history["loss_d"].append(acc["ld"] / nb)
            self.history["loss_g"].append(acc["lg"] / nb)
            self.history["grad_var"].append(acc["gv"] / nb)
            self.history["grad_norm"].append(acc["gn"] / nb)
            self.history["mmd_val"].append(mmd)
            self.history["mode_coverage"].append(cov)
            self.history["mode_entropy"].append(ent)
            self.history["seconds"].append(time.time() - t0)

            if self.verbose and (len(self.history["step"]) % max(1, self.verbose) == 0):
                print(
                    f"  passo {step:5d} (ep {step/steps_per_epoch:6.1f})  "
                    f"L_D={acc['ld']/nb:.4f}  L_G={acc['lg']/nb:.4f}  "
                    f"MMD={mmd:.5f}  Var(grad)={acc['gv']/nb:.2e}  "
                    f"{time.time()-t0:.0f}s"
                )

            recent.append(mmd)
            if len(recent) > self.mmd_smooth:
                recent.pop(0)
            score = float(np.mean(recent))
            self.history["mmd_smooth"].append(score)

            if score < best["mmd"] - 1e-6:
                best = {"mmd": score, "step": step, "theta": self.theta.copy()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    if self.verbose:
                        print(f"  arresto anticipato al passo {step}")
                    stop = True

            acc = {"ld": 0.0, "lg": 0.0, "gv": 0.0, "gn": 0.0, "nb": 0}
            t0 = time.time()
            if stop:
                break

        self.theta = best["theta"]
        self.best_step_ = best["step"]
        self.best_epoch_ = best["step"] / steps_per_epoch
        self.best_mmd_ = best["mmd"]
        self.steps_run_ = step
        self.fitted_ = True
        if self.verbose:
            print(
                f"[QGAN] ripristinato il passo {best['step']} "
                f"(MMD {best['mmd']:.5f}); {self.gen.n_circuit_evals} circuiti "
                f"in {self.gen.n_sim_calls} chiamate"
            )
        return self

    # ------------------------------------------------------------------ #
    def sample(self, n: int, c=None) -> np.ndarray:
        """Genera ``n`` campioni, shape (n, n_features), valori in [0, 1]."""
        n = int(n)
        if self.d_c > 0:
            if c is None:
                raise ValueError(f"d_c={self.d_c}: c e' obbligatorio")
            c = np.asarray(c, dtype=float)
            if c.ndim == 1:
                c = np.repeat(c[None, :], n, axis=0)
            if len(c) != n:
                raise ValueError(f"c deve avere {n} righe, ne ha {len(c)}")
        else:
            c = None
        Z = self._latent(n)
        return self.gen.forward(self.theta, Z, c)

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        np.savez(
            path,
            theta=self.theta,
            **{f"W{i}": w for i, w in enumerate(self.disc.W)},
            **{f"b{i}": b for i, b in enumerate(self.disc.b)},
        )

    def load(self, path: str) -> "QGAN":
        d = np.load(path)
        self.theta = d["theta"]
        for i in range(len(self.disc.W)):
            self.disc.W[i][...] = d[f"W{i}"]
            self.disc.b[i][...] = d[f"b{i}"]
        self.fitted_ = True
        return self

    def save_history(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                {"config": self.config(), "history": self.history}, f, indent=2
            )

    def config(self) -> dict:
        d = self.gen.describe()
        d.update(
            lr_g=self.lr_g,
            lr_d=self.lr_d,
            batch_size=self.batch_size,
            epochs=self.epochs,
            hidden=list(self.hidden),
            leak=self.leak,
            label_smoothing=self.label_smoothing,
            d_weight_decay=self.d_weight_decay,
            theta_init=self.theta_init,
            patience=self.patience,
            max_steps=self.max_steps,
            eval_every=self.eval_every,
            monitor_repeats=self.monitor_repeats,
            mmd_smooth=self.mmd_smooth,
            latent_dim=self.latent_dim,
            seed=self.seed,
            disc_params=self.disc.n_params,
        )
        return d
