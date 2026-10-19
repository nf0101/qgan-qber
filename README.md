# ⚛️ Progetto Quantum: Hybrid QGAN for Security Data Imbalance &amp; QBER Causal Attribution

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Qiskit 2.5.2](https://img.shields.io/badge/Qiskit-2.5.2-purple.svg)](https://qiskit.org/)
[![NetSquid 1.1.8](https://img.shields.io/badge/NetSquid-1.1.8-teal.svg)](https://netsquid.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Un'architettura generativa avversaria quantistica ibrida (**QGAN**) sviluppata per affrontare lo **sbilanciamento strutturale dei dati** nella sicurezza informatica classica (**Intrusion Detection Systems - IDS**) e nelle reti a Distribuzione di Chiavi Quantistiche (**Quantum Key Distribution - QKD**), abilitando la **scomposizione causale** e la **ricostruzione di scenari fisici non visti**.

---

## 📌 Panoramica del Progetto

I dataset di sicurezza soffrono di uno sbilanciamento marcato: il traffico benigno e il normale rumore operativo del canale dominano numericamente le osservazioni, mentre gli attacchi avversari o i tentativi di intercettazione (_eavesdropping_) costituiscono una frazione estremamente ridotta ed episodica.

Questo progetto implementa una singola architettura **QGAN ibrida quantistico-classica** condivisa su due domini con la medesima rappresentazione vettoriale ($d = 5$ osservabili):

1. **Dominio Classico (IDS):** _Data augmentation_ sintetica della classe minoritaria sul dataset **NSL-KDD** (attacchi R2L/Probe in regime sbilanciato 1:1000).
2. **Dominio Quantistico (QKD):** Generazione condizionata di tracce di canale su una simulazione **BB84** in **NetSquid** e scomposizione causale continua del Quantum Bit Error Rate ($\text{QBER}_{\text{tot}}$) nelle sue componenti fisiche.

---

## 🎯 Obiettivi Scientifici e Domande di Ricerca

- **O1 (Trasferibilità Architetturale):** Verificare la portabilità di un'architettura quantistica condivisa ($N=5$ qubit) tra domini di sicurezza eterogenei mantenendo congelata la dimensionalità della rappresentazione (_Feature Freezing_).
- **RQ2 (Generazione Condizionata e Generalizzazione):** Verificare se il generatore condizionato riesca a ricostruire le metriche di canale su configurazioni di celle fisiche escluse dall'addestramento (_out-of-grid / held-out_).
- **O3 (Attribuzione Causale del QBER):** Scomporre analiticamente l'errore totale osservato nel canale quantistico nel vettore delle frazioni causali costitutive:
  $\text{QBER}_{\text{oss}} = \text{QBER}_{\text{attacco}} + \text{QBER}_{\text{rumore}} + \text{QBER}_{\text{rivelatore}} \implies y = (f_{\text{eve}}, f_{\text{canale}}, f_{\text{rivelatore}})$

---

## 🏗️ Architettura del Sistema

### 1. Generatore Quantistico ($G_\theta$) — _Qiskit 2.5.2_

- **Registro:** 5 qubit ($N=5$), ciascuno mappato su un'osservabile fisica ($\text{QBER}_{\text{tot}}$), $(\text{QBER}_X$), $(\text{QBER}_Z$), $(\text{sift\_rate}$), $(\text{loss\_rate}$).
- **Encoding Latente:** Rotazioni $RY(z_i)$ con raggio calibrato ($r=0.82$) per QKD, $(r=0.93$) per IDS.
- **Ansatz Quantistico:** Topology `EfficientSU2` con 4 ripetizioni (`reps=4`), entanglement lineare adiacente e 50 parametri addestrabili $\theta$.
- **Condizionamento:** Iniezione delle variabili fisiche ($c$) tramite rotazioni $RZ(\pi \cdot c)$ condotte in _data re-uploading_ a due strati.
- **Lettura:** Misurazione della probabilità marginale dello stato ($|1\rangle$) per ciascun qubit.

### 2. Discriminatore Classico ($D_\phi$) — _NumPy Puro_

- **Topologia:** Perceptron Multilivello (MLP) feed-forward $(5 + d_c \to 32 \to 16 \to 1$).
- **Caratteristiche:** Implementato senza framework pesanti per eliminare l'overhead e calcolare analiticamente la Jacobiana ($J_D = \frac{\partial D_\phi}{\partial \hat{x}}$), concatenata con la _Parameter-Shift Rule_ per l'aggiornamento dei parametri del circuito quantistico.

### 3. Simulazione Fisica QKD — _NetSquid 1.1.8_

- Protocollo BB84 con fotoni singoli, canale in fibra soggetti ad attenuazione ($\text{dB/km}$), rumore depolarizzante ed eavesdropper con attacco _Intercept-Resend_ ($I_e \in [0.0, 1.0]$).
- Griglia di **120/360 celle fisiche** simulate con 5 semi stocastici per un totale di **94.170 finestre temporali**.
- Estrazione dei controfattuali causali esatti tramite il modulo **Common Random Numbers (CRN)**.

---

## 📊 Risultati Chiave della Baseline (Fase 1 - QKD)

I risultati del regressore baseline _Multi-Output Random Forest_ per la stima del vettore target $y = (f_{\text{eve}}, f_{\text{canale}}, f_{\text{rivelatore}})$ fissano il termine di paragone per il progetto:

| Protocollo di Valutazione | MAE Complessivo | MAE ($f_{\text{eve}}$) | MAE ($f_{\text{canale}}$) | MAE ($f_{\text{rivelatore}}$) | Note e Significato                      |
| :------------------------ | :-------------: | :--------------------: | :-----------------------: | :---------------------------: | :-------------------------------------- |
| **Random Split (80/20)**  |   **0.1218**    |         0.1350         |          0.1426           |            0.0876             | Capacità di interpolazione classica     |
| **Seed Split**            |   **0.1218**    |         0.1349         |          0.1413           |            0.0892             | Stabilità al rumore stocastico NetSquid |
| **Cell Split (Held-Out)** |   **0.1738**    |       **0.1907**       |        **0.1918**         |          **0.1389**           | **BENCHMARK RQ2** (Degrado +42.7%)      |

---

## 🛠️ Installazione e Requisiti

### Requisiti

- **Python:** `&gt;= 3.11` (Consigliato `3.12`)
- **Framework Quantistici:** `qiskit == 2.5.2`, `qiskit-aer == 0.17.2`, `netsquid == 1.1.8`
- **Librerie Scientifiche:** `numpy == 1.26.4`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`

### Configurazione Ambiente

```bash
# Clona la repository
git clone https://github.com/tuo-username/progetto-quantum.git
cd progetto-quantum

# Crea ed attiva l'ambiente virtuale
python3 -m venv .venv
source .venv/bin/activate

# Installa le dipendenze
pip install -r requirements.txt

```

---

## 🚀 Guida all'Uso Rapido

Tutti i comandi vanno lanciati **dalla radice del repository**, con il
virtualenv attivo: gli script risolvono i percorsi dei dati rispetto alla
directory corrente.

```bash
python -m venv .venv
source .venv/bin/activate          # fish: source .venv/bin/activate.fish
pip install -r requirements.txt
```

NetSquid richiede registrazione e un indice pacchetti dedicato; senza di
esso si possono comunque eseguire tutti i passi a valle della generazione,
purché il file di griglia sia già presente in `data/`.

L'ordine sotto è quello di dipendenza: ogni fase consuma ciò che produce
la precedente. Le fasi 4 e 5 sono indipendenti fra loro e si possono
invertire.

### 1. Generazione della griglia QKD

Simula BB84 su NetSquid sulle 120 configurazioni fisiche × 5 semi, con
numeri casuali comuni fra condizioni, e scrive una riga per finestra da
200 bit setacciati.

```bash
# stima dei tempi prima di impegnare la macchina
python src/generate_dataset.py --benchmark

# griglia ridotta, per verificare che la catena giri
python src/generate_dataset.py --smoke

# griglia completa, in parallelo  ->  data/qber_grid.csv
python src/generate_dataset.py

# opzionale: griglia non stazionaria  ->  data/qber_grid_ns.csv
python src/generate_dataset.py --ns
```

È la fase lunga: `--benchmark` dice quanto, su quella macchina, prima di
lanciarla. `--workers N` limita i processi, `--serial` li elimina.

Risultato atteso: **94.170 finestre** su 120 celle, di cui 24 interne.

### 2. Baseline classica dell'attributore

Addestra la Random Forest multi-output sui quattro protocolli di
valutazione (`random`, `seed`, `interior`, `boundary`) e produce lo skill
di ciascuno. Il divario fra `random` e `interior` è il riferimento
numerico di RQ2.

```bash
python src/qkd/baseline_attributor.py --dati data/qber_grid.csv --final
```

Atteso: skill 0,564 / 0,403 / 0,467 / 0,516, divario RQ2 = **0,097**.

### 3. Verifiche di O3

Due proprietà distinte, verificate separatamente: l'identità XOR
dell'attribuzione e l'accoppiamento dei flussi casuali fra condizioni.

```bash
# identita' XOR per enumerazione esaustiva + residuo di additivita'
python strumenti/verifica_xor.py --dati data/qber_grid.csv

# accoppiamento CRN fra condizioni diverse
python src/qkd/prova_crn.py
```

Atteso: zero violazioni su tutte le combinazioni di ingresso; 259 finestre
a denominatore nullo su 94.170 (0,275%); residuo di additivita' positivo,
media +0,006. Sull'accoppiamento: **0 divergenze su 299 posizioni comuni**
con CRN attivo, 87 su 299 senza.

### 4. Ramo IDS - augmentation e valutazione fuori distribuzione

Dieci repliche indipendenti dello split (sottocampione, PCA e validazione
rigenerati per replica), due semi di addestramento ciascuna.

```bash
# cosa farebbe, senza eseguire
python lancia_repliche.py --dry-run

# pipeline completa: pre-processing, verifica, augmentation, analisi
python lancia_repliche.py --parallele 5
```

Le quattro fasi si possono lanciare separatamente con `--solo`:

```bash
python lancia_repliche.py --solo pre        # dieci split indipendenti
python lancia_repliche.py --solo verifica   # controlla che siano distinti
python lancia_repliche.py --solo aug        # le venti corse di augmentation
python lancia_repliche.py --solo analisi    # intervalli e test dei segni
```

Una singola corsa, fuori dalla pipeline:

```bash
python src/ids/ids_augmentation.py --classe r2l --ratio 1_1000 --seeds 5
```

L'analisi finale, su una metrica alla volta:

```bash
python src/ids/analisi_repliche.py --dir results/ids_repliche --metrica pr_auc
python src/ids/analisi_repliche.py --dir results/ids_repliche --metrica f1
```

Atteso su `KDDTest+`: QGAN con correzione affine **+0,1237** di PR-AUC su
SMOTE (10/10 repliche), QGAN senza correzione +0,0034 con intervallo che
contiene lo zero.

Il controllo decisivo — QGAN contro gaussiana a momenti corrispondenti —
si ottiene ripetendo con `--reps 8` e confrontando le due profondità.

Verifica dichiarata nel paper, su quanto il troncamento in `[0,1]` incida
sul test:

```bash
python strumenti/verifica_clipping.py --classe r2l --ratio 1_1000
```

Atteso: 2,93% delle righe di `KDDTest+` con almeno una componente
troncata, **tutte benigne, zero su 2.754 attacchi**.

### 5. Ramo QKD - generazione condizionata (RQ2)

Prima di addestrare, conviene misurare quanto segnale condizionale la
griglia contenga oltre la media: se l'eccesso sul pavimento è nullo,
nessun modello distribuzionale potrà battere un regressore della media.

```bash
python strumenti/diagnosi_griglia.py --dati data/qber_grid.csv
```

La corsa vera, cinque semi, con il ricentraggio diagnostico e il pavimento
calcolato dentro la corsa sulle stesse celle:

```bash
python qgan/scripts/rq2_reale.py \
    --dati data/qber_grid.csv \
    --seeds 5 --ricentra --backend denso \
    --out qgan/results/rq2_omogeneo.jsonl
```

Senza `--ricentra` la corsa produce i soli contrasti grezzi; con il flag
aggiunge i livelli ricentrati, il pavimento interno e i due eccessi.

Contrasti appaiati per (corsa, cella) con intervalli bootstrap incrociati:

```bash
# i confronti grezzi: condizionato contro i quattro riferimenti
python strumenti/contrasto_campi.py \
    --dir qgan/results/rq2_omogeneo.jsonl --famiglie mae,mmd

# i due eccessi sul pavimento omogeneo
python strumenti/contrasto_campi.py \
    --dir qgan/results/rq2_omogeneo.jsonl \
    --famiglie mmd --riferimento interno_ricentrato
```

Atteso: contro l'interpolazione lineare −0,046 di MAE e −0,063 di MMD,
0 corse su 5; dopo il ricentraggio la MMD del condizionato passa da
0,0657 a 0,00060 (il **99,1%** del divario era errore di media); eccesso
del segnale disponibile +0,00014 e del generatore +0,00085 sul pavimento
−0,00025, cinque corse su cinque entrambi, intervalli non sovrapposti.

### 6. Compilazione del paper

```bash
cd docs
pdflatex short_paper.tex
bibtex   short_paper
pdflatex short_paper.tex
pdflatex short_paper.tex
```

Le tre passate servono a risolvere i riferimenti incrociati e la
bibliografia. La figura della pipeline è in TikZ nel sorgente: non
richiede file esterni.

### Strumenti di ispezione

Non fanno parte della pipeline, servono a leggere i risultati già
prodotti.

```bash
# elenca le corse presenti e la loro configurazione
python ispeziona_risultati.py --filtro 'ids_*'

# confronta le configurazioni di due o piu' corse, campo per campo
python confronta_config.py --risultati qgan/results

# contrasti fra gruppi di corse (p.es. ablazioni, capacita' del discriminatore)
python strumenti/contrasto_gruppi.py \
    --dir qgan/results/o1_qkd_h32 qgan/results/o1_qkd_h16 \
    --contro o1_qkd_h32 --quantita skill
```

### Note sulla riproducibilità

I semi derivano da `SeedSequence` con identificatori fissi: a parità di
comando e di versione delle dipendenze, le corse si riproducono
esattamente. Gli script di verifica che estraggono campioni usano
generatori casuali **propri**, seminati in modo deterministico, così da
non alterare il flusso delle misure principali.

Le campagne a molte repliche girano su un simulatore a vettore di stato
denso (`--backend denso`), vettorizzato sul minibatch. Con `--backend aer`
si ottengono gli stessi risultati più lentamente.

---

## 👥 Autori e Riconoscimenti

### Autori del Progetto

- **Alfonso Zappia** (Matricola: `NF22500012`) — _Università degli Studi di Salerno_
- **Nicola Frugieri** (Matricola: `0522501966`) — _Università degli Studi di Salerno_

### Tutor e Docenti

- **Tutor di Ricerca:** Dott. Franco Cirillo, Dott.ssa Maria Caterina d'Aloia
- **Docente Referente:** Prof. Christiancarmine Esposito
- **Dipartimento:** Dipartimento di Informatica, Università degli Studi di Salerno (UNISA)

---
