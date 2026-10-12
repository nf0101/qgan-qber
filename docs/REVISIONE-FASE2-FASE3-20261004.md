# Revisione di Appendice B, FASE2 e Fase 3 — 4 ottobre 2026

Sostituisce `APPENDICE-B-REVISIONE-20260916.md`, che copriva solo la risposta
all'audit. Qui ci sono anche la Fase 3 e le voci nuove emerse dopo.

Unità di inferenza dichiarata ovunque. Dove non diversamente indicato: venti
corse indipendenti, intervalli bootstrap al 95% ricampionati sull'unità
dichiarata, test dei segni binomiali bilaterali.

---

# Parte I — Appendice B

## B.2 — Raggio del latente

**Decisione: 0,8200 per il ramo condizionato QKD** (era 0,7342). Il ramo non
condizionato resta a π su QKD (B.5), ma **non su IDS** — vedi B.14.

**I raggi per feature sono ritirati.** La motivazione era una disparità di
dispersione di 47× fra osservabili, che derivava da `sift_rate` e `loss_rate`
costanti entro cella. Sulla griglia rigenerata la disparità è **3×**.

Dispersioni per osservabile, entro configurazione, in unità normalizzate
min-max — il confronto con dispersioni grezze porta fuori strada:

| | vecchia griglia | rigenerata |
|---|---|---|
| qber | 0,05754 | 0,05336 |
| qber_rect | 0,07024 | 0,07109 |
| qber_diag | 0,07138 | 0,07136 |
| sift_rate | 0,00318 | **0,03584** |
| loss_rate | 0,00152 | **0,02084** |
| disparità | 47× | **3×** |
| quota di varianza dovuta al seme | 4,8% | **0,2%** |

Il crollo della quota del seme è la firma del difetto corretto: quella
varianza apparteneva interamente alle due colonne costanti per corsa.

**Confronto end-to-end, venti semi appaiati, 5.400 passi:**

| condizione | skill |
|---|---|
| raggio unico 0,8200 | +0,346 ± 0,038 |
| per feature | +0,353 ± 0,038 |
| unico sulle QBER, stretto sulle due basse | +0,346 ± 0,035 |

Confronti appaiati: `per feature − unico` +0,008 ± 0,044 (9/20, p = 0,82);
`per feature − misto` +0,008 ± 0,036 (11/20, p = 0,82). **Nessuna differenza.**
O1 perde uno dei tre elementi di non trasferibilità, e B.2 si semplifica in un
raggio unico senza motivazione da riscrivere: la motivazione sparisce.

## B.3 — Criterio di arresto

**Decisione invariata: MMD stratificata su `c` nel caso condizionato.**

Aggiunta: quando gli strati non sono popolati il monitoraggio **riduce la
granularità** — 3, poi 2, poi 1 asse di `c` — invece di restituire NaN. Prima,
con tutti gli strati sotto soglia, nessun checkpoint batteva `inf` e veniva
ripristinata l'inizializzazione con `fitted_ = True`: un fit vacuo che si
dichiarava riuscito. Il numero di assi usato è registrato in `strati_assi_`:
MMD a granularità diverse non sono confrontabili.

Invarianza verificata: ogni corsa che alla granularità piena aveva almeno due
strati popolati si comporta esattamente come prima.

## B.6 — Ordine delle osservabili sui qubit

**Decisione invariata per QKD: ordine naturale.** Allineati i default che la
contraddicevano, inclusa la codifica interna del toy autoregressivo, che usava
ancora l'ordine b6.

**Ma l'ordine non è trasferibile.** L'accoppiamento qubit-latente misurato è
`q0 +1,00  q1 +0,99  q2 +0,99  q3 +1,00  q4 +0,21`: il qubit 4 è il meno
pilotabile. Su QKD, con disparità 3×, l'ordine conta poco. Su IDS, con 9,8×,
l'ordine naturale assegna a `q4` la componente che richiede la dispersione
minima — esattamente il contrario del criterio. Vedi B.14 per l'esito.

## B.7 — Stato iniziale `w_0` autoregressivo

**Decisione invariata: media dell'addestramento.** MMD alla prima finestra
0,0307 con `media` contro 0,1332 con `zero`, un fattore 4,3 su venti semi.
Sull'autocorrelazione non ha effetto (+0,014 ± 0,017, 13/20), ed è corretto:
tocca solo la prima finestra.

Corretto un rischio dell'API che non aveva contaminato risultati:
`sample_traces(w0='media')` senza valore ripiegava in silenzio su zero. Ora il
valore si calcola una volta con `valore_w0()` e l'omissione è un errore.

## B.8 — Sottocampione della griglia

**Decisione riscritta: nessuna dipendenza rilevabile da `n` fra 480 e 28.800.**
Il precedente «massimo a n ≈ 7.680» era un artefatto del difetto 3.

Quel massimo veniva da una curva in cui lo stesso flusso RNG generava la
griglia e poi sceglieva le celle escluse: per il seme 0 si passava da
`{22, 26, 42, 38}` a `{38, 26, 25, 41}` solo aumentando `--per-cella`. In più
le finestre di valutazione erano un limite superiore, non un valore fisso.

Corretto con flussi separati per ruolo e un test chiavizzato sull'indice della
cella: **stesse celle escluse a tutte le numerosità su tutti e venti i semi,
32 finestre di test ovunque.**

| n_train | 1.350 passi | 5.400 passi |
|---|---|---|
| 480 | +0,309 ± 0,036 | +0,355 ± 0,049 |
| 1.920 | +0,245 ± 0,057 | — |
| 7.680 | +0,234 ± 0,055 | — |
| 28.800 | +0,235 ± 0,043 | +0,331 ± 0,033 |

Confronto appaiato fra gli estremi: a 1.350 passi −0,074 ± 0,042 (8/20,
p = 0,50); a 5.400 **−0,024 ± 0,050 (9/20, p = 0,82)**. Liberato il budget la
differenza si avvicina ancora di più allo zero.

Unito al costo, già misurato indipendente dalla numerosità (2,18 milioni di
circuiti per corsa a qualunque `n`), è un'affermazione utile sull'economia del
metodo: su questo bersaglio la numerosità non è una leva.

## B.9 (nuova) — Accoppiamento CRN delle misure quantistiche

Il modulo dichiarava tre proprietà. Due valevano, una no.

**Stabilità dei flussi fra processi: non valeva.** Il seed derivava da
`abs(hash((seed, tag, idx)))` con `tag` stringa, e l'hash delle stringhe in
Python è randomizzato per processo: `(1, 'eve', 10)` dava 126.733.112 in un
processo e 425.770.420 in un altro. Sostituito con `SeedSequence` e
identificatori numerici.

**Coerenza fra catena modellata ed eseguita: valeva già.** Zero discordanze fra
errore analitico e osservato, in entrambe le configurazioni e su tutte e 600 le
celle. `attribute()` era corretto.

**Accoppiamento fra condizioni: non valeva.** `measure()` pescava dallo stato
casuale globale di NetSquid, mai inizializzato e non toccato da `sim_reset()`.
Cambiando `intercept_rate` cambia il numero di misure, quindi il flusso globale
veniva consumato diversamente e l'esito di Bob alla stessa posizione era
estratto da un punto diverso — ed è proprio quella casualità a generare il QBER
da intercettazione.

Corretto fissando lo stato globale dal flusso della posizione prima di ogni
misura. Verifica sulle posizioni intercettate in entrambe le corse a r = 0,25 e
r = 0,50, che sono nidificate perché la soglia cambia e il flusso no:

| | esito di Bob divergente |
|---|---|
| con accoppiamento | **0 / 299** |
| senza | **87 / 299** (29,1%) |

Non è un test che possa passare per caso: circa metà delle posizioni comuni ha
esito deterministico, e sull'altra metà due estrazioni indipendenti
divergerebbero al 50%.

**Conseguenza.** Le griglie precedenti al 16 settembre 2026 non sono riusabili
per conclusioni causali, né rigenerabili bit per bit: vanno conservate come
artefatti versionati con la loro impronta.

## B.10 (nuova) — Osservabili calcolate per finestra

`sift_rate` e `loss_rate` erano scalari **per corsa** replicati su tutte le
finestre della cella: un solo valore distinto per cella, su tutte e 600.

Tre conseguenze: due qubit su cinque modellavano costanti già determinate da
`c`; la disparità di dispersione risultava 47× invece di 3×, e su quel 47%
poggiavano i raggi per feature; in modalità autoregressiva due delle cinque
componenti di `w_{t-1}` erano costanti lungo la traccia.

Ora calcolate sullo span grezzo della finestra `k`, che copre le posizioni
`(limite[k-1], limite[k]]`. Le finestre tassellano senza buchi né
sovrapposizioni.

    sift_rate = size / n_grezze
    loss_rate = 1 - n_ricevute / n_grezze

Coerente con le definizioni per corsa: minimo e massimo di `sift_rate` nella
griglia, 0,078 e 0,400, sono 0,5·(1−perdita) a 40 km e a 5 km. Verifica: valori
distinti per cella da **1** a una mediana di **104** e **154** su ~157 finestre.

L'anticorrelazione fra le due (~−0,5) è fisica: le posizioni setacciate sono un
sottoinsieme delle ricevute. Verificato contro una definizione a span grezzo
fisso: −0,480 contro −0,508.

**Osservazione aperta.** L'autocorrelazione lag 1 di `sift_rate` fra finestre
consecutive vale +0,059 contro un nullo simulato di −0,008, circa 3,5 errori
standard su venti sequenze. Ogni componente testata da sola è pulita: perdita
esatta per posizione (zero discordanze su 20.000), uniformi senza struttura a
blocchi, basi nulle su 200 semi. Registrata come aperta; non tocca nessun
obiettivo.

## B.11 (nuova) — Allocazione di `c` sugli strati di condizionamento

`_cond_layer` leggeva `c[q % d_c]` in entrambi gli strati: con `d_c = 8` su
cinque qubit, `c[5]`, `c[6]`, `c[7]` **non comparivano nel circuito**. In
`concat(c_statico, w_precedente)` significa che il generatore vedeva due sole
delle cinque osservabili precedenti. Il discriminatore le riceveva tutte.

**Decisione: schema `avanti`**, `[[0,1,2,3,4], [5,6,7,0,1]]` per `d_c = 8`.
Progettato come **no-op per `d_c ≤ n_qubit`**: il ramo congelato da §12 resta
bit-identico su circuito, forward e jacobiana. Nessun risultato con `d_c` in
{0, 1, 3, 5} va rifatto, **e O1 non va rimisurato per questo**.

**Alternativa provata e scartata: `coda`**, che mette l'intera finestra
precedente nel primo strato, dandole +46% di sensibilità a scapito del
condizionamento statico. Venti semi appaiati, `coda` meno `avanti`:

| | differenza | segni | p |
|---|---|---|---|
| autocorrelazione intra | −0,059 ± 0,031 | 7/20 | 0,26 |
| MMD teacher forcing | +0,011 ± 0,005 | 13/20 | 0,26 |
| MMD generazione libera | +0,011 ± 0,005 | 13/20 | 0,26 |

Nessun confronto significativo, segno sfavorevole su tre indicatori: si tiene
`avanti`. L'opzione resta nel codice. Nota: l'asimmetria fra strati, mediata su
cinque inizializzazioni, vale 2,9× — non i 18× che una singola
inizializzazione suggeriva.

## B.12 (nuova) — Profili temporali e griglia non stazionaria

Il simulatore produce condizioni **stazionarie**: ogni sorgente stocastica è
indipendente per posizione e nessun parametro varia nel tempo. Le finestre di
una cella sono quindi scambiabili, e l'autocorrelazione delle osservabili di
errore è compatibile con zero: `qber` −0,0026, `qber_diag` −0,0004, contro un
nullo di −0,0064. §6 modella una dipendenza che i dati non hanno.

Per metterlo alla prova sul terreno giusto, i profili temporali diventano un
**parametro**, con `costante` come default: il caso stazionario è un caso
particolare, non una modalità alternativa. Regressione verificata — con profili
costanti il codice riproduce la griglia precedente con scarto **esattamente
0,0** su 18.816 finestre.

| profilo | cosa produce | acf del QBER entro cella |
|---|---|---|
| (stazionario) | nulla | +0,036 |
| `deriva_eve` | AR(1) sul tasso di Eve: memoria vera | **+0,691** |
| `intermittente` | onda quadra: punto di cambiamento | **+0,705** |
| `entrambi` | composizione dei due | **+0,742** |

Due scelte di progetto meritano di essere riportate. La deriva sulla
**depolarizzazione** è satura: quel parametro contribuisce al QBER con
`dep/2 ≤ 0,0225` contro un rumore binomiale per finestra di ~0,025, e un
modello analitico la ferma a +0,11 anche a sigma 1,0. Spostata sul tasso di
Eve, che contribuisce `r/4`. E il periodo dell'intermittente è più corto della
traccia, così **ogni traccia da 16 finestre contiene almeno una transizione**;
un gradino unico ne avrebbe lasciate otto su nove senza.

Griglia: 3 profili × 2 tassi × 2 lunghezze × 2 depolarizzazioni × **15 semi** =
360 celle, 56.496 finestre. Quindici semi e non cinque perché
`profilo_deriva(seed)` dipende solo dal seme: **il numero di semi è il numero di
forme di deriva distinte**, e con cinque l'addestramento ne vedrebbe tre.

## B.13 (nuova) — Simulatore denso

Backend alternativo ad Aer, vettorizzato sull'asse del batch: le N valutazioni
di una chiamata diventano una sola sequenza di operazioni NumPy su un tensore
`(N, 2, 2, 2, 2, 2)`.

| | scarto su forward e jacobiana | un passo di gradiente |
|---|---|---|
| `d_c` = 0, 1, 3, 5, 8 | **≤ 8,9·10⁻¹⁶** | Aer 418,6 ms → denso **42,3 ms** |

La libreria resta ad Aer per default, che è il riferimento di SPEC §7.2; gli
script usano `denso` e il backend entra in `firma_config`. La suite ha una
regressione che confronta i due: se il circuito cambia senza che `_probs_denso`
sia aggiornato, il test fallisce.

Il guadagno pratico è maggiore di 9,9×: Aer satura già tutti i core, il denso
ne usa uno, quindi il fattore si moltiplica per il numero di processi
paralleli. Ottanta corse in cinque minuti invece di mezza giornata — è ciò che
ha reso possibili le venti repliche indipendenti.

## B.14 (nuova) — Raggio e ordine sono parametri per dominio

Su IDS il ramo non condizionato usava π per B.5, valore corretto per QKD e mai
misurato qui. Le cinque componenti PCA della minoritaria `r2l` hanno dispersione
`[0,128, 0,113, 0,042, 0,028, 0,013]`, media 0,065; a raggio π il generatore
all'inizializzazione ne produce 0,331, **cinque volte troppo**. Il raggio
calibrato è **0,9327**.

La disparità fra feature è **9,8×**, contro i 3× di QKD. All'inizializzazione,
ordine e raggi per feature **insieme** riproducono il bersaglio esattamente
(errore 1,00×); separatamente no (3,11× e 5,34×).

**Dopo l'addestramento nessuna delle quattro combinazioni funziona**: MMD fra
0,37 e 0,43 contro il ~0 delle baseline classiche, e `sd generata / reale` con
la feature più stretta a 2,1–4,3×. È il sesto caso del risultato metodologico di
§12: le proprietà all'inizializzazione non predicono il generatore addestrato.

La decisione operativa è quindi: **raggio e ordine sono parametri per dominio e
vanno calibrati separatamente**, ma su IDS la calibrazione non è sufficiente.

## B.15 (nuova) — Riduzione da sei a cinque osservabili: confermata

Il confronto originale era fra due insiemi da **sei**: le cinque attuali più
`basis_asymmetry` oppure `qber_stability`. L'esito documentato è «nessuna delle
due», ma era misurato su `qber_grid_small.csv`, con un solo seme e con due
delle cinque osservabili degeneri.

Rivalutato sulla griglia non stazionaria, dove la dinamica temporale esiste
davvero:

| profilo | corr con \|Δ rate_eff\| | corr con `qber` | sd misurata | nullo |
|---|---|---|---|---|
| deriva_eve | **0,008** | 0,409 | 0,0163 | 0,0444 |
| entrambi | 0,294 | 0,605 | 0,0172 | 0,0357 |
| intermittente | 0,147 | 0,584 | 0,0161 | 0,0359 |

`qber_stability` **non cattura la transizione di regime nemmeno su tracce che
ne hanno**, ed è in larga parte ridondante con `qber` — l'effetto binomiale per
cui a QBER più alto la varianza fra sotto-blocchi cresce meccanicamente.

La decisione regge, e per una ragione migliore di quella documentata. Nota che
il bias del confronto originale era **favorevole** alla decisione presa: con due
osservabili degeneri dentro le cinque c'era più varianza residua da spiegare,
quindi il contributo della sesta era semmai sopravvalutato.

---

# Parte II — Appendice A

## `max_steps` era sottodimensionato

Non è una preferenza, è una misura. Con `max_steps = 1350`, `eval_every = 25` e
pazienza 20 valutazioni (cioè 500 passi):

| budget | corse al tetto | `steps_run` medio | `best_step` medio |
|---|---|---|---|
| 1.350 | **17–18 / 20** | 1.350 | 1.319–1.329 |
| 5.400 | **1 / 20** | 3.344–3.432 | 2.882–2.954 |

A 1.350 la pazienza non scattava mai. La convergenza sul dominio QKD cade fra
2.900 e 3.400 passi. **Proposta: 5.400**, dichiarando che i numeri a 1.350 erano
troncati.

---

# Parte III — Sezioni FASE2 da riscrivere

## §11.3 e §11.4 — O1

**La larghezza del discriminatore regge.** Sweep a 5.400 passi sul bersaglio
QKD corretto, venti semi:

| hidden | skill | best_step |
|---|---|---|
| 8 | +0,057 ± 0,070 | 2290 |
| 16 | +0,165 ± 0,070 | 2372 |
| **32,16** | **+0,346 ± 0,038** | 2944 |
| 64,32 | +0,408 ± 0,045 | 2742 |

`h8 − h32x16` = −0,289 (4/20, p = 0,012): il discriminatore piccolo non
funziona su QKD, confermato al budget corretto. `h64x32 − h32x16` = +0,063 ±
0,048 (9/20, p = 0,82): non significativo, quindi `h32x16` resta difendibile
come configurazione comune.

**Il «fattore cinque» va corretto in eccesso.** Gli 875 passi su `qkd` erano il
minimo di traiettoria entro un budget di 1.350, quindi troncati. La convergenza
misurata a 5.400 è **~2.900**. Se i 175 passi su `ids` reggono — e lì il
troncamento è improbabile — il rapporto passa da cinque a **circa diciassette**.
La rivendicazione di §11.4 non cade: si rafforza. Resta da rimisurare il lato
`ids` allo stesso budget, l'unico numero della tabella 11.3 ancora preso sotto
un tetto stringente.

**I raggi per feature escono dall'elenco** (B.2). Entrano ordine e raggio come
parametri per dominio (B.14), e la capacità richiesta (Parte IV).

## §16.3.1 — RQ2 sul bersaglio sintetico

Venti corse indipendenti, configurazione P32, budget 1.350.

| | media | IC95 per corsa | segni |
|---|---|---|---|
| skill condizionato | +0,2445 | [+0,1289, +0,3470] | 18/20 |
| skill incondizionato | −0,2468 | [−0,3310, −0,1648] | 2/20 |
| condizionato − incondizionato | +0,4913 | [+0,3737, +0,5990] | 19/20 |
| MAE banale − MAE condizionato | +0,0186 | [+0,0107, +0,0262] | 18/20 |
| guadagno MMD | +0,1458 | [+0,1031, +0,1844] | 18/20 |
| risposta al condizionamento | +0,9047 | [+0,8647, +0,9401] | 20/20 |

**Due conclusioni cambiano.** `banale − condizionato` passa da «non si può
affermare» (p = 0,16) a 18 segni su 20. E `condizionato − incondizionato`
sopravvive con un intervallo che non contiene lo zero — prima il bootstrap dava
[−0,069, +0,375] mentre il `p` accanto valeva 0,0016: le due statistiche si
contraddicevano perché erano calcolate su unità diverse.

**Ma l'inferenza va dichiarata su due livelli.** Con corse e celle come
raggruppamenti incrociati:

| | fra corse | fra celle | residuo |
|---|---|---|---|
| varianza dello skill | 19,9% | **59,6%** | 20,5% |
| varianza del divario | 16,8% | **65,6%** | 17,5% |

| quantità | per corsa | incrociata |
|---|---|---|
| skill | [−0,026, +0,236] | [−0,228, +0,425] |
| divario banale − cond | [+0,011, +0,026] | [−0,003, +0,044] |

Sul divario l'intervallo per corsa esclude lo zero, quello incrociato no.
**L'affermazione difendibile è: sulle celle di questa griglia il condizionamento
migliora la ricostruzione in modo riproducibile fra esecuzioni; l'estensione a
celle non viste non è stabilita.** La griglia sintetica ha 8 celle interne, e la
varianza fra celle domina: il rimedio non è statistico ma di disegno.

Cautela da riportare: il `p` minimo ottenibile con venti segni concordi è
2·2⁻²⁰ ≈ 0,00002. È il pavimento del test, non l'ampiezza dell'effetto.

## §17 — Generazione autoregressiva

«Funziona» non regge.

**La metrica è stata validata prima del modello.** Sul toy AR(1) con φ = 0,75 la
lettura residua sulle tracce reali dà **+0,7501 ± 0,0017** contro 0,7500 vero.

**La metrica precedente misurava il livello, non la memoria:**

| lettura, tracce reali | valore | con la memoria distrutta a permutazioni |
|---|---|---|
| residua (livello noto) | +0,750 | — |
| intra (media empirica) | +0,606 | −0,064 |
| **aggregata (la precedente)** | **+0,936** | **+0,822** |

**L'88% era differenza di livello fra tracce.** Il confronto 0,818 contro 0,969
di §17.3 misurava quasi solo quello.

**Sul sintetico il modello cattura circa il 29% della memoria** (+0,129 ± 0,023
sulla scala intra, dove il nullo è −0,064 e il vero +0,606). Attenzione a un
artefatto: la lettura residua sulle sintetiche dà +0,672 ma centra sul livello
vero, quindi un errore di livello entra nel residuo come costante per traccia e
una costante ha autocorrelazione 1 a ogni lag.

**Sui dati reali cattura zero.** Con la pipeline su griglia non stazionaria, su
tre bersagli con memoria genuina:

| bersaglio | acf reale | autoregressivo | per finestra | nullo |
|---|---|---|---|---|
| stazionario (controllo) | −0,027 | +0,038 | −0,064 | −0,066 |
| deriva_eve | +0,102 | **−0,046** | −0,068 | −0,067 |
| intermittente | +0,363 | **−0,057** | −0,069 | −0,067 |
| entrambi | +0,326 | **−0,053** | −0,058 | −0,068 |

Il confronto appaiato con un modello **per finestra** (`d_c = 3`, che per
costruzione non può avere memoria) non mostra differenze: 12, 11, 11 segni su
20, p fra 0,50 e 0,82.

Il rapporto di deriva 2,22 ± 0,45 non è confrontabile con l'1,17 riportato: con
il vecchio encoding il generatore vedeva due sole osservabili precedenti su
cinque. Una deriva maggiore è la conseguenza attesa di un condizionamento
autoregressivo che ora funziona davvero.

Riallocare `c` fra gli strati (B.11) non migliora.

---

# Parte IV — Fase 3: dati reali

Esito negativo su entrambi i domini, per ragioni **diverse**. È questo che lo
rende riportabile: non «non ha funzionato», ma «sappiamo dove sta il limite in
ciascun caso e abbiamo escluso le spiegazioni alternative».

## QKD — su griglia densa l'interpolazione vince

Disegno: 24 configurazioni interne su 120, 12 escluse con tutti i loro semi,
~9.410 finestre di valutazione. Nessun sottocampionamento: le configurazioni
escluse non entrano in addestramento in alcun modo.

| riferimento | MAE medie | MMD |
|---|---|---|
| QGAN condizionata | 0,0423 | 0,0671 |
| banale (media globale) | 0,1395 | 0,3487 |
| vicino più prossimo | 0,0595 | 0,1460 |
| **interpolazione lineare** | **0,0031** | **0,0017** |

Skill contro il riferimento banale **+0,70 sul MAE e +0,81 sulla MMD**; contro
l'interpolazione **−12,9 e −39,5**. Su una griglia regolare a tre assi con 108
configurazioni in addestramento, la distribuzione condizionale varia poco fra
celle adiacenti e l'interpolazione empirica è quasi al livello del rumore di
campionamento.

**Esperimento sulla densità**, configurazioni escluse fisse, cinque semi:

| conf. in addestramento | MMD QGAN | MMD lineare | skill vs lineare | escluse fuori inviluppo |
|---|---|---|---|---|
| 108 | 0,066 | 0,003 | −26,5 | 0,0 |
| 54 | 0,051 | 0,007 | −35,9 | 0,0 |
| 24 | 0,129 | 0,018 | −8,7 | 0,4 |
| 12 | 0,129 | 0,099 | −1,9 | 2,0 |
| 6 | 0,251 | 0,411 | **+0,34 ± 0,22** | **9,4 / 12** |

Il crossover esiste ma cade dove il 78% delle celle escluse è fuori
dall'inviluppo convesso: lì il riferimento non interpola più, estrapola. E la
QGAN è degradata di quattro volte. **È una vittoria fra due metodi che falliscono
entrambi**, a 1,5 errori standard.

Lo skill contro il riferimento banale resta positivo a ogni densità (+0,57 a
+0,86): il condizionamento si impara. È il confronto con i metodi classici che
non regge.

## IDS — con 53 campioni la distribuzione non è apprendibile

Minoritaria `r2l` a 1:1000, PCA locale, da 53 a 269 campioni (il livello 1:200,
già misurato con campioni **veri** in `baseline.csv`). Valutazione su
validation; il test resta intatto. Cinque semi.

| variante | F1 `rf` | MMD contro i 53 reali |
|---|---|---|
| nessuna | 0,5506 ± 0,0043 | — |
| **bilanciato** (`class_weight`) | **0,7732 ± 0,0011** | — |
| duplicazione | 0,7126 ± 0,0089 | −0,008 |
| rumore | 0,7349 ± 0,0037 | −0,008 |
| SMOTE | 0,7740 ± 0,0056 | −0,007 |
| **QGAN** | **0,5773 ± 0,0101** | **+0,365** |

Confronti appaiati contro `bilanciato`: duplicazione −0,061 (0/5), rumore
−0,038 (0/5), SMOTE +0,001 (3/5), **QGAN −0,196 (0/5)**. Nessuna augmentation
sintetica batte `class_weight`, che costa zero; la QGAN perde di molto e non
supera nemmeno il non far nulla.

**Il meccanismo è identificato.** Il generatore **omogeneizza le dispersioni**:

| | z1 | z2 | z3 | z4 | z5 |
|---|---|---|---|---|---|
| sd reale | 0,128 | 0,112 | 0,042 | 0,028 | **0,013** |
| sd generata | 0,056 | 0,071 | 0,043 | 0,015 | **0,048** |
| rapporto | 0,44 | 0,63 | 1,01 | 0,53 | **3,65** |

Le medie sono corrette (scarto massimo 0,075): è la forma a non essere
rappresentabile. La feature più stretta esce quasi quattro volte troppo larga, e
con 53 campioni il classificatore ha bisogno proprio di quella.

**Quattro spiegazioni alternative escluse per misura.** Raggio ricalibrato
(0,93 contro π): MMD 0,47 → 0,37. Ordine delle feature sui qubit: nessun
effetto. Raggi per feature: nessun effetto. Le quattro combinazioni di ordine e
raggio danno tutte MMD fra 0,37 e 0,43.

**Ablazione su `reps`** — una misura, non una modifica di SPEC §12 — con il
raggio ricalibrato per ciascuna profondità, cinque semi:

| reps | parametri | MMD | F1 `rf` | QGAN − bilanciato |
|---|---|---|---|---|
| 4 | 50 | **0,327** | 0,564 | −0,209 (0/5) |
| 6 | 70 | 0,431 | 0,590 | −0,183 (0/5) |
| 8 | 90 | 0,475 | 0,575 | −0,198 (0/5) |
| 10 | 110 | 0,553 | 0,560 | −0,213 (0/5) |
| 12 | 130 | 0,502 | 0,570 | −0,203 (0/5) |

**La MMD peggiora al crescere della profondità.** Più capacità su 53 campioni
danneggia, come ci si attende da un sovradattamento in uno spazio più grande.
Il collo di bottiglia è la numerosità, non l'espressività dell'ansatz: **SPEC
§12 non va aperto su `reps`**, e la riduzione a cinque qubit non è messa in
discussione.

Controprova sulla numerosità: ripetendo a 1:200 (269 campioni reali) la MMD
resta 0,419. Non è la scarsità di questo particolare campione, è il regime.

**Limite da dichiarare accanto al risultato**: le cinque feature sono componenti
PCA che spiegano il **34%** della varianza. L'augmentation opera in uno spazio
che perde due terzi dell'informazione — è il vincolo dei cinque qubit di §12.

---

# Parte V — Il filo metodologico

È il contributo più trasferibile del lavoro, e si è rafforzato.

**Le proprietà all'inizializzazione non predicono il generatore addestrato.
Sei casi su sei.** Ai quattro già documentati si aggiungono: la calibrazione di
ordine e raggi per feature su IDS, che all'inizializzazione riproduceva il
bersaglio esattamente e dopo l'addestramento no; e l'ablazione su `reps`, dove
la misura statica prometteva che `reps = 8` bastasse.

**Gli errori stanno nel codice di misura.** Otto su nove in Fase 2, e dei dodici
difetti chiusi in questa revisione solo uno (l'encoding) è nel modello.

**I protocolli di verifica costruiti qui**, ciascuno un controllo che poteva
fallire e non ha:

- baseline a memoria distrutta per l'autocorrelazione, che smaschera una metrica
  la quale misura il livello invece della dinamica;
- prova di accoppiamento CRN su condizioni nidificate (0/299 contro 87/299);
- inferenza a raggruppamenti incrociati, dove ricampionare un solo asse
  sottostima la varianza;
- controllo negativo su bersaglio stazionario, che distingue «il modello cattura
  memoria» da «il modello inventa autocorrelazione»;
- regressione di invarianza bit-per-bit quando si corregge un ramo mai eseguito,
  che permette di correggere senza invalidare;
- riferimenti classici forti (interpolazione, SMOTE, `class_weight`) invece di
  una costante.

Per un ambito in cui gran parte della letteratura riporta positivi su bersagli
non controllati, è un contributo difendibile a prescindere dal segno dei
risultati.

---

# Parte VI — Cosa resta

**Da misurare.** La convergenza sul lato `ids` a 5.400 passi: è l'unico numero
della tabella 11.3 ancora preso sotto un tetto stringente, e determina se il
fattore cinque diventa diciassette.

**Da scrivere.** La riscrittura piana dell'Appendice B esistente, cresciuta per
accumulo su otto voci con due proposte ritirate al suo interno e una tabella
B.7/B.8 che non corrisponde ai titoli delle sezioni. Questo documento è
materiale per quella riscrittura, non la riscrittura.

**Da decidere.** L'impostazione della tesi. Il risultato principale di Fase 3 è
negativo su entrambi i domini, e una tesi così va strutturata come tale
dall'introduzione, non corretta in un capitolo finale. Gli elementi che reggono
sono O1 (misurato, con un elemento in più), O3 (il framework CRN verificato,
il contributo più pulito), RQ2 sul sintetico con il limite dichiarato, e il filo
metodologico.

**Aperte e registrate, non bloccanti.** L'autocorrelazione di `sift_rate` a
+0,059 (3,5 errori standard) con tutte le componenti individualmente pulite; e
la questione se `qber_stability` meriti una rivalutazione in un disegno dove la
dinamica temporale sia il bersaglio.
