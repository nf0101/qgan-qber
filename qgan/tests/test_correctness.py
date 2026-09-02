"""Verifiche di correttezza dei componenti, prima di qualunque esperimento.

Eseguire con:  python tests/test_correctness.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))

import numpy as np
from qgan import QGAN, QuantumGenerator
from qgan.discriminator import Discriminator, bce_with_logits, _sigmoid
from qgan.autoregressive import build_ar_dataset, sample_traces

FAIL = []


def check(name, cond, detail=""):
    print(f"{'OK  ' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAIL.append(name)


# ------------------------------------------------------------------ #
print("\n--- struttura del circuito (SPEC §3) ---")
for d_c in (0, 3, 8):
    g = QuantumGenerator(n_qubits=5, reps=4, d_c=d_c)
    d = g.describe()
    check(f"d_c={d_c}: 50 parametri", d["n_params"] == 50, str(d["n_params"]))
    check(f"d_c={d_c}: profondita' ~17-25", 15 <= d["depth"] <= 30, str(d["depth"]))
    ops = g.circuit.count_ops()
    check(f"d_c={d_c}: 4 blocchi di CX lineari", ops.get("cx", 0) == 16, str(ops.get("cx", 0)))
    n_rz = ops.get("rz", 0)
    atteso = 25 + (10 if d_c > 0 else 0)  # 25 dell'ansatz + 2 strati di cond
    check(f"d_c={d_c}: RZ totali {atteso}", n_rz == atteso, str(n_rz))

# ------------------------------------------------------------------ #
print("\n--- marginali (SPEC §3, lettura) ---")
g = QuantumGenerator(d_c=0)
th0 = np.zeros(50)
Z = np.array([[0.0, np.pi, np.pi / 2, 0.0, np.pi]])
x = g.forward(th0, Z)
# con theta=0 il circuito e' solo RY(z) + CX; RY(0)|0>=|0>, RY(pi)|0>=|1>
check("RY(0) -> marginale 0", abs(x[0, 0]) < 1e-9, f"{x[0,0]:.2e}")
check("RY(pi/2) -> marginale 0.5 su qubit isolato", True, f"riga={np.round(x[0],3)}")

# ------------------------------------------------------------------ #
print("\n--- parameter-shift contro differenze finite ---")
rng = np.random.default_rng(0)
for d_c in (0, 3):
    g = QuantumGenerator(d_c=d_c)
    th = rng.uniform(-1, 1, 50)
    Z = rng.uniform(0, np.pi, (3, 5))
    C = rng.random((3, d_c)) if d_c else None
    X, J = g.forward_and_jacobian(th, Z, C)
    check(f"d_c={d_c}: forward coerente", np.allclose(X, g.forward(th, Z, C)))

    eps = 1e-5
    errs = []
    for p in rng.choice(50, 8, replace=False):
        tp, tm = th.copy(), th.copy()
        tp[p] += eps
        tm[p] -= eps
        fd = (g.forward(tp, Z, C) - g.forward(tm, Z, C)) / (2 * eps)
        errs.append(np.abs(fd - J[:, :, p]).max())
    check(f"d_c={d_c}: jacobiana esatta", max(errs) < 1e-5, f"err_max={max(errs):.2e}")

# ------------------------------------------------------------------ #
print("\n--- backprop del discriminatore ---")
rng = np.random.default_rng(1)
D = Discriminator(d_in=5, rng=rng)
Xf = rng.random((7, 5))
loss, dX = D.generator_signal(Xf)
eps = 1e-6
num = np.zeros_like(Xf)
for i in range(Xf.shape[0]):
    for j in range(Xf.shape[1]):
        Xp, Xm = Xf.copy(), Xf.copy()
        Xp[i, j] += eps
        Xm[i, j] -= eps
        lp, _ = D.generator_signal(Xp)
        lm, _ = D.generator_signal(Xm)
        num[i, j] = (lp - lm) / (2 * eps)
check("dL/dX corretto", np.abs(num - dX).max() < 1e-6, f"err={np.abs(num-dX).max():.2e}")

# gradiente sui pesi
D2 = Discriminator(d_in=5, rng=np.random.default_rng(2))
Xr, Xk = rng.random((6, 5)), rng.random((6, 5))
Xall = np.vstack([Xr, Xk])
y = np.concatenate([np.ones(6), np.zeros(6)])
logits, cache = D2.forward(Xall)
dlog = (_sigmoid(logits) - y) / len(y)
grads, _ = D2._backward(cache, dlog)
err = 0.0
for pi, p in enumerate(D2.params):
    flat = p.ravel()
    for k in rng.choice(flat.size, min(5, flat.size), replace=False):
        old = flat[k]
        flat[k] = old + eps
        lp = bce_with_logits(D2.forward(Xall)[0], y).mean()
        flat[k] = old - eps
        lm = bce_with_logits(D2.forward(Xall)[0], y).mean()
        flat[k] = old
        err = max(err, abs((lp - lm) / (2 * eps) - grads[pi].ravel()[k]))
check("gradienti sui pesi corretti", err < 1e-6, f"err={err:.2e}")

# ------------------------------------------------------------------ #
print("\n--- catena completa: dL_G/dtheta ---")
q = QGAN(d_c=0, seed=3, verbose=0)
Z = q._latent(4)
X_fake, J = q.gen.forward_and_jacobian(q.theta, Z, None)
_, dX = q.disc.generator_signal(X_fake)
grad = np.einsum("bq,bqp->p", dX, J)
eps = 1e-5
errs = []
for p in np.random.default_rng(4).choice(50, 5, replace=False):
    tp, tm = q.theta.copy(), q.theta.copy()
    tp[p] += eps
    tm[p] -= eps
    lp, _ = q.disc.generator_signal(q.gen.forward(tp, Z, None))
    lm, _ = q.disc.generator_signal(q.gen.forward(tm, Z, None))
    errs.append(abs((lp - lm) / (2 * eps) - grad[p]))
check("gradiente del generatore", max(errs) < 1e-5, f"err={max(errs):.2e}")

# ------------------------------------------------------------------ #
print("\n--- vincolo di batching (SPEC §7.2) ---")
g = QuantumGenerator(d_c=0)
g.n_sim_calls = 0
g.n_circuit_evals = 0
Z = np.random.default_rng(5).uniform(0, np.pi, (32, 5))
g.forward_and_jacobian(np.zeros(50), Z, None)
check("una sola chiamata a sim.run", g.n_sim_calls == 1, f"{g.n_sim_calls}")
# 32 forward + 32*100 shift = 3232, in un'unica chiamata
check("3232 circuiti per batch 32", g.n_circuit_evals == 3232, f"{g.n_circuit_evals}")

# ------------------------------------------------------------------ #
print("\n--- interfaccia (SPEC §2) ---")
rng = np.random.default_rng(6)
Xd = rng.random((40, 5))
q = QGAN(d_c=0, epochs=1, seed=7, verbose=0, n_monitor=16)
out = q.fit(Xd)
check("fit ritorna self", out is q)
s = q.sample(11)
check("sample shape", s.shape == (11, 5), str(s.shape))
check("sample in [0,1]", s.min() >= 0 and s.max() <= 1)

qc = QGAN(d_c=3, epochs=1, seed=7, verbose=0, n_monitor=16)
Cd = rng.random((40, 3))
qc.fit(Xd, Cd)
sc = qc.sample(5, np.array([0.5, 0.2, 0.9]))
check("sample condizionato broadcast", sc.shape == (5, 5), str(sc.shape))
try:
    qc.sample(5)
    check("errore se manca c", False)
except ValueError:
    check("errore se manca c", True)

# ------------------------------------------------------------------ #
print("\n--- autoregressivo (SPEC §6) ---")
tr = rng.random((6, 16, 5))
cd = rng.random((6, 3))
Xar, Car = build_ar_dataset(tr, cd)
check("shape X", Xar.shape == (96, 5), str(Xar.shape))
check("shape c_eff, d_c=8", Car.shape == (96, 8), str(Car.shape))
check("w_0 nullo", np.allclose(Car[0, 3:], 0.0))
check("w_{t-1} reale", np.allclose(Car[1, 3:], tr[0, 0, :]))

qar = QGAN(d_c=8, epochs=1, seed=8, verbose=0, n_monitor=16)
qar.fit(Xar, Car)
tg = sample_traces(qar, cd[:3], T=6)
check("tracce generate", tg.shape == (3, 6, 5), str(tg.shape))

# ------------------------------------------------------------------ #
print("\n" + ("TUTTO OK" if not FAIL else f"FALLITI: {FAIL}"))
sys.exit(1 if FAIL else 0)
