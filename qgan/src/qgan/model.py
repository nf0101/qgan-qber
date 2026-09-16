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

# Righe minime perche' uno strato di c entri nella MMD stratificata. Sotto
# questa soglia la MMD sul singolo strato e' dominata dal rumore di stima.
MIN_RIGHE_STRATO = 8


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
        monitor: str = "auto",
        latent_hi: float = np.pi,
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
        if monitor not in ("auto", "mmd", "mmd_stratificata"):
            raise ValueError("monitor: 'auto', 'mmd' o 'mmd_stratificata'")
        self.monitor = monitor
        # SPEC-FASE2 §3: z di dimensione 5, uniforme in [0, pi]. Il valore e'
        # congelato; e' esposto per poterlo MISURARE, non per cambiarlo
        # unilateralmente. Vedere FASE2.md §15 per la motivazione.
        self._latent_hi_raw = latent_hi
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
        # Scalare o vettore di 5 componenti, un raggio per qubit. La
        # dispersione condizionale richiesta puo' variare di un fattore 47 fra
        # osservabili (dominio QKD: qber_diag 0,071 contro loss_rate 0,0015),
        # e un raggio unico non puo' soddisfarle tutte.
        lh = np.atleast_1d(np.asarray(self._latent_hi_raw, dtype=float))
        if lh.size == 1:
            lh = np.repeat(lh, self.latent_dim)
        if lh.size != self.latent_dim:
            raise ValueError(
                f"latent_hi deve essere scalare o di {self.latent_dim} "
                f"componenti, ne ha {lh.size}")
        self.latent_hi = lh if lh.size > 1 and len(set(lh.tolist())) > 1 else float(lh[0])
        self._latent_hi_vec = lh

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

    def _set_theta(self, valori) -> None:
        """Aggiorna theta PRESERVANDO l'identita' dell'array.

        Adam tiene un riferimento all'array, non al nome: riassegnare
        ``self.theta`` lo scollega in silenzio dal modello. Il sintomo e' che
        un secondo ``fit`` dopo un ripristino di checkpoint o un ``load``
        lascia theta immobile, perche' l'ottimizzatore continua a modificare
        l'array precedente. La forma non cambia mai legittimamente
        (``n_params`` e' fissato alla costruzione), quindi non esiste un ramo
        di riallocazione: una forma diversa e' un errore.
        """
        valori = np.asarray(valori, dtype=float).ravel()
        if valori.shape != (self.n_params,):
            raise ValueError(
                f"theta deve avere {self.n_params} componenti, "
                f"ne ha {valori.size}"
            )
        self.theta[...] = valori

    def _latent(self, n: int) -> np.ndarray:
        return self.rng.uniform(
            0.0, self._latent_hi_vec[None, :], size=(n, self.latent_dim))

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
        *,
        riprendi: bool = False,
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
        riprendi : bool
            ``False`` (default) azzera lo stato di Adam del generatore prima
            di partire: un ``fit`` successivo a un altro ``fit`` o a un
            ``load`` riparte pulito. ``True`` lo conserva.

            Una ripresa corretta richiederebbe anche i pesi del
            discriminatore e lo stato di ENTRAMBI gli ottimizzatori.
            ``save``/``load`` non salvano gli stati degli ottimizzatori,
            quindi ``riprendi=True`` dopo un ``load`` e' rifiutato invece di
            fingere una ripresa che sarebbe parziale e non riproducibile.

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

        if riprendi:
            if getattr(self, "_da_load", False):
                raise ValueError(
                    "riprendi=True dopo load(): gli stati degli ottimizzatori "
                    "non sono salvati da save(), quindi la ripresa sarebbe "
                    "parziale. Usare riprendi=False, oppure estendere "
                    "save()/load() agli stati di Adam."
                )
        else:
            # Ricreare l'ottimizzatore sullo STESSO array theta: _set_theta
            # ne preserva l'identita', quindi il riferimento resta valido.
            self.opt_g = Adam([self.theta], lr=self.lr_g)
            if hasattr(self.disc, "reset_optimizer"):
                self.disc.reset_optimizer()
            elif self.fitted_:
                warnings.warn(
                    "riprendi=False azzera lo stato di Adam del generatore ma "
                    "non quello del discriminatore, che non espone "
                    "reset_optimizer(). Il fit non riparte da uno stato "
                    "completamente pulito."
                )

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

        # Strati di c per il monitoraggio condizionale
        modo = self.monitor
        if modo == "auto":
            modo = "mmd_stratificata" if self.d_c > 0 else "mmd"
        strati = None
        if modo == "mmd_stratificata":
            if c_val is None:
                raise ValueError("monitoraggio stratificato senza c")
            # Binning su ciascun asse di c: 2 livelli per asse, al piu' 8
            # strati. Gli strati sotto MIN_RIGHE_STRATO vengono saltati dal
            # monitoraggio; se lo sono TUTTI la MMD e' NaN a ogni valutazione,
            # nessun checkpoint batte inf, e alla fine verrebbe ripristinata
            # l'inizializzazione con fitted_=True: un fit vacuo che si dichiara
            # riuscito.
            #
            # La risposta non e' abbandonare la stratificazione — e' proprio il
            # criterio che RQ2 richiede, e la MMD aggregata seleziona contro il
            # condizionamento — ne' fermarsi. Si riduce la GRANULARITA' finche'
            # gli strati non sono popolati: con 40 righe 8 strati sono vuoti,
            # 2 strati da 20 no. Il numero di assi effettivamente usati finisce
            # in ``strati_assi_`` e in ``config()``, perche' MMD calcolate su
            # granularita' diverse non sono confrontabili fra loro.
            b = (c_val > 0.5).astype(int)
            assi_max = min(self.d_c, 3)

            def _strati_su(n_assi: int) -> np.ndarray:
                s = np.zeros(len(c_val), dtype=int)
                for j in range(n_assi):
                    s = s * 2 + b[:, j]
                return s

            for n_assi in range(assi_max, 0, -1):
                strati = _strati_su(n_assi)
                conteggi = np.bincount(strati, minlength=2**n_assi)
                utili = int((conteggi >= MIN_RIGHE_STRATO).sum())
                if utili >= 2:
                    break

            self.strati_assi_ = int(n_assi)
            self.strati_utili_ = utili
            self.strati_totali_ = int((conteggi > 0).sum())

            if utili == 0:
                raise ValueError(
                    f"nemmeno con un solo asse di condizionamento gli strati "
                    f"raggiungono {MIN_RIGHE_STRATO} righe di validazione "
                    f"({len(c_val)} righe in {self.strati_totali_} strati): "
                    "la MMD stratificata non puo' produrre valori finiti. "
                    "Aumentare la validazione, oppure dichiarare "
                    "esplicitamente monitor='mmd' accettando che la MMD "
                    "aggregata seleziona contro il condizionamento."
                )
            if utili == 1:
                warnings.warn(
                    f"un solo strato utile su {self.strati_totali_}: la MMD "
                    "cosiddetta stratificata e' di fatto aggregata su un "
                    "sottoinsieme e non misura la fedelta' condizionale."
                )
            elif n_assi < assi_max:
                warnings.warn(
                    f"monitoraggio stratificato ridotto da {assi_max} a "
                    f"{n_assi} assi di c ({utili} strati utili su "
                    f"{self.strati_totali_}, {len(c_val)} righe di "
                    "validazione): la MMD non e' confrontabile con quella di "
                    "corse a granularita' diversa."
                )
            elif utili < self.strati_totali_:
                warnings.warn(
                    f"monitoraggio su {utili} strati di {self.strati_totali_}: "
                    f"{self.strati_totali_ - utili} sotto soglia vengono "
                    "ignorati."
                )
        self.monitor_ = modo
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
                "d_su_reali",
                "d_su_falsi",
                "theta_ampiezza",
                "sensibilita_c",
                "mmd_val",
                "mmd_smooth",
                "mode_coverage",
                "mode_entropy",
                "seconds",
            )
        }

        if self.verbose:
            circ = self.gen.circuits_per_gradient(B)
            print("=" * 92)
            print(f"QGAN  {self.n_features} qubit  {self.n_params} parametri  "
                  f"d_c={self.d_c}  discriminatore {list(self.hidden)} "
                  f"({self.disc.n_params} parametri)")
            print(f"  dati        n={n}  validazione={len(X_val)}  "
                  f"monitoraggio={n_mon}  batch={B}")
            print(f"  ottimizzaz. lr_g={self.lr_g}  lr_d={self.lr_d}  "
                  f"theta init +-{self.theta_init}  "
                  f"z in [0,{np.array2string(self._latent_hi_vec, precision=3, separator=',')}]  "
                  if self._latent_hi_vec.std() > 1e-9 else
                  f"z in [0,{self._latent_hi_vec[0]:.3f}]  "
                  f"smoothing={self.label_smoothing}  wd={self.d_weight_decay}")
            print(f"  budget      {max_steps} passi "
                  f"({max_steps/steps_per_epoch:.0f} epoche)  "
                  f"valutazione ogni {eval_every}  pazienza {self.patience}  "
                  f"mmd_smooth={self.mmd_smooth}  monitor={self.monitor_}")
            print(f"  costo       {circ} circuiti/passo  "
                  f"{circ*max_steps/1e6:.1f}M totali")
            if self.d_c > 0:
                d0 = self.gen.conditioning_sensitivity(
                    self.theta, n=200, seed=self.seed,
                    latent_hi=float(self._latent_hi_vec.mean()))
                print(f"  condiz.     sensibilita' a c {d0['media']:.4f}  "
                      f"dispersione da z {d0['dispersione_latente']:.4f}  "
                      f"rapporto {d0['rapporto']:.3f}"
                      + ("   <-- sotto 0,05: c e' sommerso dal latente"
                         if d0["rapporto"] < 0.05 else ""))
            print("-" * 92)
            cols = (f"{'passo':>6s} {'ep':>6s} {'L_D':>7s} {'L_G':>7s} "
                    f"{'D(re)':>6s} {'D(fa)':>6s} {'MMD':>8s} {'MMDs':>8s} "
                    f"{'Var(g)':>9s} {'|th|':>6s}")
            if self.d_c > 0:
                cols += f" {'c/z':>6s}"
            cols += f" {'s':>6s} {'resta':>7s}"
            print(cols)

        recent: list[float] = []
        perm = self.rng.permutation(n)
        cur = 0
        acc = {k: 0.0 for k in ("ld", "lg", "gv", "gn", "dr", "df", "nb")}
        t0 = time.time()
        t_start = time.time()
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
            acc["dr"] += float(
                self.disc.predict_proba(self._disc_input(Xb, cb)).mean()
            )
            acc["df"] += float(
                self.disc.predict_proba(self._disc_input(X_fake, cb)).mean()
            )
            acc["nb"] += 1

            if step % eval_every and step != max_steps:
                continue

            # --- monitoraggio ---------------------------------------
            mmds = []
            cov = ent = float("nan")
            for _ in range(self.monitor_repeats):
                sel = self.rng.choice(len(X_val), n_mon, replace=False)
                Xg = self.sample(n_mon, c_val[sel] if c_val is not None else None)
                if strati is None:
                    mmds.append(mmd_rbf(X_val[sel], Xg, sigma=self.mmd_sigma_))
                else:
                    # MMD stratificata su c. La MMD aggregata NON vede il
                    # condizionamento: un generatore che ignora c puo'
                    # eguagliare la distribuzione complessiva, che e' la
                    # stessa cosa. Usarla per l'arresto anticipato seleziona
                    # il checkpoint precedente allo sviluppo del
                    # condizionamento. Stratificando su c si misura la
                    # fedelta' condizionale, che e' cio' che RQ2 richiede.
                    vals = []
                    for k in range(strati.max() + 1):
                        m = strati == k
                        if m.sum() < MIN_RIGHE_STRATO:
                            continue
                        Xk = X_val[m]
                        gk = self.sample(len(Xk), c_val[m])
                        vals.append(mmd_rbf(Xk, gk, sigma=self.mmd_sigma_))
                    mmds.append(float(np.mean(vals)) if vals else np.nan)
            mmd = float(np.mean(mmds))
            # Diagnostico chiave nel caso condizionato: quanto l'uscita
            # risponde a c rispetto a quanto risponde al latente. Se resta
            # sotto 0,05 il discriminatore, che riceve c in chiaro, vince
            # senza che il generatore possa rispondere. Costa 2x100 circuiti
            # contro le migliaia del tratto di addestramento.
            sens = float("nan")
            if self.d_c > 0:
                sens = self.gen.conditioning_sensitivity(
                    self.theta, n=100, seed=self.seed,
                    latent_hi=self._latent_hi_vec.tolist(),
                )["rapporto"]
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
            self.history["d_su_reali"].append(acc["dr"] / nb)
            self.history["d_su_falsi"].append(acc["df"] / nb)
            self.history["theta_ampiezza"].append(float(np.abs(self.theta).mean()))
            self.history["sensibilita_c"].append(sens)
            self.history["mmd_val"].append(mmd)
            self.history["mode_coverage"].append(cov)
            self.history["mode_entropy"].append(ent)
            self.history["seconds"].append(time.time() - t0)

            recent.append(mmd)
            if len(recent) > self.mmd_smooth:
                recent.pop(0)
            score = float(np.mean(recent))
            self.history["mmd_smooth"].append(score)


            if self.verbose and (len(self.history["step"]) % max(1, self.verbose) == 0):
                dt = time.time() - t0
                fatti = step
                resta = (time.time() - t_start) / max(fatti, 1) * (max_steps - fatti)
                riga = (
                    f"{step:6d} {step/steps_per_epoch:6.1f} "
                    f"{acc['ld']/nb:7.4f} {acc['lg']/nb:7.4f} "
                    f"{acc['dr']/nb:6.3f} {acc['df']/nb:6.3f} "
                    f"{mmd:8.5f} {score:8.5f} "
                    f"{acc['gv']/nb:9.2e} "
                    f"{np.abs(self.theta).mean():6.3f}"
                )
                if self.d_c > 0:
                    riga += f" {sens:6.3f}"
                riga += f" {dt:6.1f} {resta/60:6.1f}m"
                if score <= best["mmd"] + 1e-9:
                    riga += "  *"
                if acc["ld"] / nb < 0.40:
                    riga += "  D vince"
                print(riga, flush=True)

            if score < best["mmd"] - 1e-6:
                best = {"mmd": score, "step": step, "theta": self.theta.copy()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    if self.verbose:
                        print(f"  arresto anticipato al passo {step}")
                    stop = True

            acc = {k: 0.0 for k in ("ld", "lg", "gv", "gn", "dr", "df", "nb")}
            t0 = time.time()
            if stop:
                break

        if best["step"] < 0 or not np.isfinite(best["mmd"]):
            raise RuntimeError(
                f"fit terminato senza alcuna valutazione finita della MMD "
                f"(best_step_={best['step']}, best_mmd_={best['mmd']}). "
                "theta e' rimasto all'inizializzazione: nessun addestramento "
                "utilizzabile e' avvenuto. Il modello NON viene dichiarato "
                "addestrato."
            )
        self._set_theta(best["theta"])
        self._da_load = False
        self.best_step_ = best["step"]
        self.best_epoch_ = best["step"] / steps_per_epoch
        self.best_mmd_ = best["mmd"]
        self.steps_run_ = step
        self.fitted_ = True
        if self.verbose:
            h = self.history
            print("-" * 92)
            print(f"ripristinato il passo {best['step']} di {step} "
                  f"(MMD {best['mmd']:.5f})")
            print(f"  MMD        {h['mmd_smooth'][0]:.5f} -> "
                  f"{best['mmd']:.5f} al migliore -> "
                  f"{h['mmd_smooth'][-1]:.5f} alla fine")
            print(f"  L_D        {h['loss_d'][0]:.4f} -> {h['loss_d'][-1]:.4f}"
                  f"   minimo {min(h['loss_d']):.4f}"
                  + ("   (sotto 0,40: il discriminatore ha vinto)"
                     if min(h["loss_d"]) < 0.40 else ""))
            print(f"  Var(grad)  {h['grad_var'][0]:.2e} -> "
                  f"{h['grad_var'][-1]:.2e}")
            print(f"  |theta|    {h['theta_ampiezza'][0]:.3f} -> "
                  f"{h['theta_ampiezza'][-1]:.3f}")
            if self.d_c > 0:
                print(f"  c/z        {h['sensibilita_c'][0]:.3f} -> "
                      f"{h['sensibilita_c'][-1]:.3f}")
            print(f"  costo      {self.gen.n_circuit_evals} circuiti in "
                  f"{self.gen.n_sim_calls} chiamate, "
                  f"{(time.time()-t_start)/60:.1f} min")
            print("=" * 92)
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
        self._set_theta(d["theta"])
        # save() non salva gli stati degli ottimizzatori: un fit successivo
        # con riprendi=True sarebbe una ripresa parziale, e viene rifiutato.
        self._da_load = True
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
            monitor=self.monitor,
            latent_hi=self._latent_hi_vec.tolist(),
            mmd_smooth=self.mmd_smooth,
            latent_dim=self.latent_dim,
            seed=self.seed,
            disc_params=self.disc.n_params,
            monitor_=getattr(self, "monitor_", None),
            strati_assi=getattr(self, "strati_assi_", None),
            strati_utili=getattr(self, "strati_utili_", None),
            strati_totali=getattr(self, "strati_totali_", None),
            best_step=getattr(self, "best_step_", None),
            steps_run=getattr(self, "steps_run_", None),
        )
        return d