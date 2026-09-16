"""Verifiche di correttezza dei componenti, prima di qualunque esperimento.

Eseguire con:  python tests/test_correctness.py
"""

import sys, os, itertools, tempfile, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from qgan import QGAN, QuantumGenerator
from qgan.circuit import cond_index_map
from qgan.discriminator import Discriminator, bce_with_logits, _sigmoid
from qgan.autoregressive import (
    autocorrelation, autocorrelation_nulla, build_ar_dataset, sample_traces,
    valore_w0,
)

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


def _marginali_theta_zero(z, n_catene=4):
    """Marginali esatte con theta=0, derivate senza simulatore.

    Con theta=0 le rotazioni dell'ansatz sono l'identita': restano gli RY(z)
    e le catene di CX, che sui basis state agiscono come XOR prefisso. Un
    z_i in {0, pi} da' |0> o |1>; z_i = pi/2 da' una sovrapposizione, quindi
    si enumera e si media.

    Sostituisce un controllo che aveva condizione letterale ``True`` e
    quindi passava qualunque cosa il simulatore restituisse.
    """
    fissi = [None if abs(zi - np.pi / 2) < 1e-12
             else (0 if abs(zi) < 1e-12 else 1) for zi in z]
    liberi = [i for i, v in enumerate(fissi) if v is None]
    acc = np.zeros(len(z))
    for combo in itertools.product([0, 1], repeat=len(liberi)):
        b = [0 if v is None else v for v in fissi]
        for i, val in zip(liberi, combo):
            b[i] = val
        for _ in range(n_catene):
            for q in range(len(b) - 1):
                b[q + 1] ^= b[q]
        acc += np.array(b, dtype=float)
    return acc / 2 ** len(liberi)


atteso = _marginali_theta_zero(Z[0])
check("marginali esatte con theta=0", np.abs(x[0] - atteso).max() < 1e-9,
      f"simulato={np.round(x[0],3)} atteso={np.round(atteso,3)}")

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
print("\n--- sensibilita' al condizionamento (SPEC §5) ---")
# Verificare che i gate di condizionamento esistano NON basta: va verificato
# che variare c cambi l'uscita, e di quanto rispetto alla variabilita' che il
# latente introduce da solo. L'assenza di questo controllo ha permesso di
# arrivare alla validazione di RQ2 senza accorgersi che il condizionamento
# era sommerso dal latente di un fattore dodici.
rng = np.random.default_rng(11)
for d_c in (3, 8):
    g = QuantumGenerator(d_c=d_c)
    th = rng.uniform(-0.1, 0.1, 50)
    d = g.conditioning_sensitivity(th, n=300, seed=11)
    sens, disp = d["media"], d["dispersione_latente"]
    check(f"d_c={d_c}: c influenza l'uscita", sens > 1e-3, f"sens={sens:.4f}")
    check(f"d_c={d_c}: c influenza OGNI feature", d["minimo"] > 1e-6,
          f"min={d['minimo']:.2e}")
    check(
        f"d_c={d_c}: rapporto sensibilita'/dispersione noto",
        0.02 < sens / disp,
        f"{sens/disp:.3f}  (documentato ~0,09 con latente in [0,pi]; "
        f"se scende sotto 0,02 il condizionamento e' inutilizzabile)",
    )

# ------------------------------------------------------------------ #
print("\n--- copertura del condizionamento (difetto 1) ---")
# La sensibilita' qui sopra varia TUTTE le componenti di c insieme e riporta
# la risposta per feature di USCITA: e' cieca a una componente di INGRESSO
# che non entra nel circuito. Con d_c=8 e 5 qubit la mappatura c[q % d_c]
# lasciava fuori c[5], c[6] e c[7] — le ultime tre osservabili della finestra
# precedente — e questa suite passava lo stesso.

for d_c in (1, 2, 3, 4, 5):
    m = cond_index_map(d_c, 5, 2)
    atteso_m = [[q % d_c for q in range(5)]] * 2
    check(f"d_c={d_c}: mappatura invariante (congelata da §12)",
          m == atteso_m, str(m))

m8 = cond_index_map(8, 5, 2)
check("d_c=8: tutte le componenti mappate",
      set(m8[0]) | set(m8[1]) == set(range(8)),
      f"{m8}  mancanti={sorted(set(range(8)) - (set(m8[0]) | set(m8[1])))}")

for d_c in (3, 8):
    g = QuantumGenerator(d_c=d_c)
    nomi = {p.name for p in g.circuit.parameters}
    mancanti = {f"c[{j}]" for j in range(d_c)} - nomi
    check(f"d_c={d_c}: ogni c compare fra i parametri del circuito",
          not mancanti, f"mancanti={sorted(mancanti)}")

rng = np.random.default_rng(11)
for d_c in (3, 8):
    g = QuantumGenerator(d_c=d_c)
    th = rng.uniform(-0.1, 0.1, 50)
    s = g.input_sensitivity(th, n=200, seed=11)
    check(f"d_c={d_c}: ogni ingresso, da solo, cambia l'uscita",
          float(s.min()) > 1e-6,
          f"minimo {s.min():.2e} su c[{int(np.argmin(s))}]  "
          f"tutti={np.round(s, 4).tolist()}")

for d_c, cl in ((8, 1), (11, 2)):
    try:
        QuantumGenerator(d_c=d_c, cond_layers=cl)
        check(f"d_c={d_c} con {cl} strati: rifiutato", False, "nessun errore")
    except ValueError:
        check(f"d_c={d_c} con {cl} strati: rifiutato", True, "ValueError")

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
print("\n--- allineamento di w_0 (difetto 8) ---")
# build_ar_dataset calcolava la media, sample_traces ripiegava in silenzio
# su zero: addestramento e generazione libera partivano da stati diversi e
# il divario finiva attribuito al modello.
w0v = valore_w0(tr, "media")
Xw, Cw = build_ar_dataset(tr, cd, w0="media", w0_value=w0v)
check("l'addestramento parte da w_0", np.allclose(Cw[0, 3:], w0v),
      f"{np.round(w0v, 4).tolist()}")
try:
    sample_traces(qar, cd[:2], T=3, w0="media")
    check("sample_traces rifiuta 'media' senza valore", False, "nessun errore")
except ValueError:
    check("sample_traces rifiuta 'media' senza valore", True, "ValueError")
check("valore_w0 e' None per 'zero'", valore_w0(tr, "zero") is None)

# ------------------------------------------------------------------ #
print("\n--- autocorrelazione: memoria o livello? (difetto 2) ---")
# Tracce SENZA alcuna memoria temporale: livello casuale costante per
# traccia, rumore indipendente nel tempo. La metrica aggregata deve
# comunque restituire un valore altissimo — e' il difetto.
n_t, T_t = 400, 16
liv = np.random.default_rng(31).uniform(0, 1, (n_t, 1, 1))
tr_nm = liv + np.random.default_rng(32).normal(0, 0.01, (n_t, T_t, 1))
agg = float(autocorrelation(tr_nm, 4, mode="aggregata")[0, 0])
intra = float(autocorrelation(tr_nm, 4, mode="intra")[0, 0])
res = float(autocorrelation(tr_nm, 4, mode="residua", livello=liv)[0, 0])
bias = -1.0 / (T_t - 1)
check("aggregata scambia il livello per memoria", agg > 0.9, f"{agg:.5f}")
check("intra riconosce l'assenza di memoria", abs(intra - bias) < 0.03,
      f"{intra:.5f}  (atteso ~{bias:.5f}, bias di centratura)")
check("residua riconosce l'assenza di memoria", abs(res) < 0.03, f"{res:.5f}")
nul = float(autocorrelation_nulla(tr_nm, 4, mode="aggregata", seed=1,
                                  repliche=5)[0, 0])
check("la baseline nulla smaschera la metrica aggregata",
      abs(nul - agg) < 0.02,
      f"con memoria distrutta {nul:.5f} contro {agg:.5f} misurata")
try:
    autocorrelation(tr_nm, 4)
    check("mode e' obbligatorio", False, "nessun errore")
except TypeError:
    check("mode e' obbligatorio", True, "TypeError")

# ------------------------------------------------------------------ #
print("\n--- ciclo di vita di theta (difetto 6) ---")
# Adam tiene un riferimento all'array, non al nome. Riassegnare self.theta al
# ripristino del checkpoint lo scollegava in silenzio: un secondo fit non
# muoveva piu' il generatore, e la variazione massima di theta era 0,0.
rng = np.random.default_rng(21)
Xd2 = rng.random((60, 5))
q2 = QGAN(d_c=0, epochs=2, seed=12, verbose=0, n_monitor=16)
q2.fit(Xd2)
check("Adam resta agganciato a theta dopo il ripristino",
      q2.opt_g.p[0] is q2.theta)
prima = q2.theta.copy()
q2.fit(Xd2)
check("un secondo fit muove theta",
      np.abs(q2.theta - prima).max() > 0,
      f"variazione massima {np.abs(q2.theta - prima).max():.2e}")

perc = os.path.join(tempfile.gettempdir(), "qgan_regressione.npz")
q2.save(perc)
q3 = QGAN(d_c=0, epochs=1, seed=13, verbose=0, n_monitor=16)
q3.load(perc)
check("load preserva l'identita' dell'array", q3.opt_g.p[0] is q3.theta)
check("load carica davvero theta", np.allclose(q3.theta, q2.theta))
try:
    q3.fit(Xd2, riprendi=True)
    check("riprendi=True dopo load rifiutato", False, "nessun errore")
except ValueError:
    check("riprendi=True dopo load rifiutato", True, "ValueError")
os.remove(perc)

# ------------------------------------------------------------------ #
print("\n--- monitoraggio senza strati utili (difetto 7) ---")
# Otto righe di validazione, una per angolo del cubo di condizionamento:
# ogni strato ha una riga sola, tutti vengono saltati, la MMD e' NaN a ogni
# valutazione e nessun checkpoint batte inf. Prima veniva ripristinata
# l'inizializzazione con fitted_=True: un fit vacuo che si dichiarava
# riuscito, con mmd_val=[nan] e best_step_=-1.
# Caso recuperabile: 40 righe e 8 strati non funzionano, ma 2 strati da 20
# si'. La granularita' si riduce, la stratificazione resta.
qred = QGAN(d_c=3, epochs=1, seed=9, verbose=0, n_monitor=16)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    qred.fit(np.random.default_rng(23).random((40, 5)),
             np.random.default_rng(24).random((40, 3)))
check("granularita' ridotta invece di fallire", qred.strati_assi_ < 3,
      f"assi={qred.strati_assi_} utili={qred.strati_utili_}")
check("il fit condizionato ha davvero un checkpoint", qred.best_step_ >= 0,
      f"best_step_={qred.best_step_}")

# Caso non recuperabile: 8 righe, una per angolo del cubo.
qbad = QGAN(d_c=3, epochs=1, seed=9, verbose=0, monitor="mmd_stratificata")
Xb = np.random.default_rng(22).random((8, 5))
Cb = np.array(list(itertools.product([0.0, 1.0], repeat=3)))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        qbad.fit(Xb, Cb)
        check("fit rifiuta una validazione senza strati utili", False,
              "nessun errore sollevato")
    except ValueError as e:
        check("fit rifiuta una validazione senza strati utili",
              "strat" in str(e).lower(), str(e)[:60])
check("fitted_ resta False dopo il rifiuto", not qbad.fitted_)

# ------------------------------------------------------------------ #
print("\n" + ("TUTTO OK" if not FAIL else f"FALLITI: {FAIL}"))
sys.exit(1 if FAIL else 0)