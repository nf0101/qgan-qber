"""Da dove viene la dipendenza fra posizioni nella perdita?

Sulla griglia rigenerata, sift_rate e loss_rate hanno autocorrelazione lag 1
entro cella di +0,059 e +0,026, contro un nullo esatto — stesso stimatore,
stessa definizione di finestra, posizioni indipendenti — di -0,008 e -0,006:
21 e 10 errori standard. Nel codice la perdita di ogni qubit pesca da
stream(seed, "loss_s*", i), che per costruzione non ha memoria.

Due prove ESATTE, posizione per posizione, senza incertezza statistica.

  A. INDIPENDENZA DALL'ACCOPPIAMENTO. L'accoppiamento tocca solo le misure.
     Se la perdita dipende soltanto dai flussi per posizione, l'insieme delle
     posizioni ricevute deve essere IDENTICO con e senza accoppiamento. Se
     cambia, la perdita consuma lo stato casuale globale di NetSquid, che
     misura_accoppiata reinizializza prima di ogni misura.

  B. PREVEDIBILITA' DAL FLUSSO. Se lose_qubit usa l'rng che gli passiamo,
     l'insieme delle ricevute e' calcolabile a mano dai flussi loss_s1 e
     loss_s2. Se non coincide, la perdita non pesca da dove il codice dice.

    python src/diagnosi_perdita.py
"""

import sys

import numpy as np

import BB84_crn as B


def ricevute_previste(seed, num_bits, L1, L2, p_ll):
    """Posizioni che sopravvivono a entrambe le tratte, dai soli flussi."""
    p1 = 1 - 10 ** (-L1 * p_ll / 10)
    p2 = 1 - 10 ** (-L2 * p_ll / 10)
    ok = set()
    for i in range(num_bits):
        u1 = B.stream(seed, "loss_s1", i).random_sample()
        u2 = B.stream(seed, "loss_s2", i).random_sample()
        if u1 >= p1 and u2 >= p2:
            ok.add(i)
    return ok


def acf1_indicatore(ric, num_bits):
    """Autocorrelazione lag 1 dell'indicatore di ricezione, per posizione."""
    x = np.zeros(num_bits)
    x[list(ric)] = 1.0
    x -= x.mean()
    return float((x[:-1] * x[1:]).sum() / (x ** 2).sum())


def main(num_bits=20000, seed=1):
    print(f"{num_bits} posizioni, seme {seed}, r=0 (nessuna misura di Eve)\n")
    esiti = {}
    for acc in (True, False):
        B.ACCOPPIA_MISURE = acc
        _, _, ric = B.run_bb84(intercept_rate=0.0, num_bits=num_bits,
                               seed=seed, verbose=False, con_ricevute=True)
        esiti[acc] = ric
        print(f"  accoppiamento {'attivo    ' if acc else 'disattivato'}  "
              f"ricevute {len(ric):6d}   acf lag 1 indicatore "
              f"{acf1_indicatore(ric, num_bits):+.4f}")

    prev = ricevute_previste(seed, num_bits, B.L1, B.L2, B.P_LOSS_LENGTH)
    se = 1 / np.sqrt(num_bits)
    print(f"  previste dai flussi         ricevute {len(prev):6d}   "
          f"acf lag 1 indicatore {acf1_indicatore(prev, num_bits):+.4f}")
    print(f"  (errore standard dell'acf per posizione: {se:.4f})\n")

    diff_a = len(esiti[True] ^ esiti[False])
    diff_on = len(esiti[True] ^ prev)
    diff_off = len(esiti[False] ^ prev)
    print(f"A. ricevute con vs senza accoppiamento : {diff_a:6d} posizioni diverse")
    print(f"B. ricevute con accoppiamento vs flussi: {diff_on:6d} posizioni diverse")
    print(f"   ricevute senza accopp.     vs flussi: {diff_off:6d} posizioni diverse\n")

    if diff_a == 0 and diff_on == 0:
        print("Esito: la perdita pesca ESATTAMENTE dai flussi per posizione e")
        print("l'accoppiamento non la tocca. La dipendenza osservata sulla")
        print("griglia viene da altro: prossimo sospetto, le basi di Alice e Bob")
        print("o la definizione delle finestre nel CSV.")
    elif diff_a > 0:
        print("Esito: l'insieme delle ricevute CAMBIA con l'accoppiamento. La")
        print("perdita consuma lo stato casuale globale di NetSquid, che")
        print("misura_accoppiata reinizializza prima di ogni misura: la")
        print("correzione CRN ha agganciato le perdite alle misure. Va corretto")
        print("prima di usare la griglia per sift_rate e loss_rate.")
    else:
        print("Esito: l'accoppiamento non cambia la perdita, ma la perdita non")
        print("coincide con quella prevista dai flussi. lose_qubit non usa l'rng")
        print("che gli passiamo nel modo atteso: va letto il sorgente NetSquid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())