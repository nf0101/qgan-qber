"""Discriminatore classico e ottimizzatore Adam, in NumPy puro.

SPEC-FASE2 §4: rete feed-forward 5 -> 32 -> 16 -> 1, LeakyReLU con pendenza
0,01, uscita sigmoidale. Nel caso condizionale l'ingresso e' la
concatenazione di feature e condizionamento, 5 + d_c.

L'implementazione e' in NumPy e non in un framework di deep learning per due
ragioni: la rete e' minuscola e il collo di bottiglia e' il simulatore
quantistico, e l'appendice della specifica documenta la fragilita'
dell'ambiente rispetto alle dipendenze. Serve inoltre il gradiente della
perdita rispetto all'INGRESSO del discriminatore, che va poi concatenato con
la jacobiana del circuito: averlo esplicito rende la catena verificabile.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def bce_with_logits(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Entropia incrociata binaria, forma numericamente stabile."""
    return np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))


class Adam:
    """Adam su una lista di array, aggiornati in-place."""

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.p = params
        self.wd = float(wd)
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads) -> None:
        self.t += 1
        bc1 = 1 - self.b1**self.t
        bc2 = 1 - self.b2**self.t
        for i, (p, g) in enumerate(zip(self.p, grads)):
            if self.wd:
                g = g + self.wd * p
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            p -= self.lr * (self.m[i] / bc1) / (np.sqrt(self.v[i] / bc2) + self.eps)


class Discriminator:
    """MLP con uscita logit. La sigmoide e' inglobata nella perdita."""

    def __init__(
        self,
        d_in: int,
        hidden=(32, 16),
        leak: float = 0.01,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng or np.random.default_rng(0)
        self.leak = float(leak)
        dims = [d_in, *hidden, 1]
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            # He, adeguata alle LeakyReLU
            scale = np.sqrt(2.0 / dims[i])
            self.W.append(rng.normal(0.0, scale, size=(dims[i], dims[i + 1])))
            self.b.append(np.zeros(dims[i + 1]))
        self.params = self.W + self.b
        self.opt = Adam(self.params, lr=lr, wd=weight_decay)
        self.n_params = sum(p.size for p in self.params)

    # -------------------------------------------------------------- #
    def _act(self, z):
        return np.where(z > 0, z, self.leak * z)

    def _dact(self, z):
        return np.where(z > 0, 1.0, self.leak)

    def forward(self, X: np.ndarray):
        """Restituisce (logits, cache)."""
        a = X
        cache = [X]
        L = len(self.W)
        for i in range(L):
            z = a @ self.W[i] + self.b[i]
            cache.append(z)
            a = self._act(z) if i < L - 1 else z
            cache.append(a)
        return a.ravel(), cache

    def _backward(self, cache, dlogits: np.ndarray):
        """Backprop. dlogits: (B,) gia' mediato sul batch.

        Returns (grads_params, dL/dX).
        """
        L = len(self.W)
        gW = [None] * L
        gb = [None] * L
        delta = dlogits[:, None]  # (B,1) gradiente sull'uscita lineare
        for i in reversed(range(L)):
            a_prev = cache[2 * i]  # attivazione in ingresso al layer i
            gW[i] = a_prev.T @ delta
            gb[i] = delta.sum(axis=0)
            d_in = delta @ self.W[i].T
            if i > 0:
                z_prev = cache[2 * i - 1]
                delta = d_in * self._dact(z_prev)
            else:
                delta = d_in
        return gW + gb, delta

    # -------------------------------------------------------------- #
    def train_step(
        self, X_real: np.ndarray, X_fake: np.ndarray, label_smoothing: float = 0.0
    ) -> float:
        """Un passo di discesa. Ritorna la perdita del discriminatore.

        ``label_smoothing`` abbassa l'etichetta dei campioni reali da 1 a
        1 - eps. Con minoritarie di poche decine di campioni il
        discriminatore tende a memorizzare l'insieme reale, saturando il
        segnale che arriva al generatore; l'attenuazione dell'etichetta e'
        la contromisura piu' economica e non tocca nulla di congelato.
        """
        X = np.vstack([X_real, X_fake])
        y = np.concatenate([
            np.full(len(X_real), 1.0 - float(label_smoothing)),
            np.zeros(len(X_fake)),
        ])
        logits, cache = self.forward(X)
        loss = bce_with_logits(logits, y).mean()
        dlogits = (_sigmoid(logits) - y) / len(y)
        grads, _ = self._backward(cache, dlogits)
        self.opt.step(grads)
        return float(loss)

    def generator_signal(self, X_fake: np.ndarray) -> tuple[float, np.ndarray]:
        """Perdita non saturante del generatore e suo gradiente su X_fake.

        L_G = mean_b BCE(D(x_b), 1). Il gradiente restituito include gia' il
        fattore 1/B, quindi puo' essere contratto direttamente con la
        jacobiana del circuito.
        """
        B = len(X_fake)
        logits, cache = self.forward(X_fake)
        loss = bce_with_logits(logits, np.ones(B)).mean()
        dlogits = (_sigmoid(logits) - 1.0) / B
        _, dX = self._backward(cache, dlogits)
        return float(loss), dX

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(X)
        return _sigmoid(logits)

    def state(self) -> list[np.ndarray]:
        return [p.copy() for p in self.params]

    def load(self, st: list[np.ndarray]) -> None:
        for p, s in zip(self.params, st):
            p[...] = s
