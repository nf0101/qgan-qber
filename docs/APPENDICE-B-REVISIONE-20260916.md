# Appendice B — revisione del 16 settembre 2026

Riscrittura delle voci toccate dall'audit del 15 settembre e dalle verifiche
del giorno successivo. Ogni voce riporta la decisione corrente, il numero che
la sostiene e cosa la rende diversa dalla versione precedente.

Unità di inferenza dichiarata: **la corsa**. Venti repliche indipendenti dove
non diversamente indicato. Gli intervalli sono bootstrap al 95% ricampionati
sulle corse; i test dei segni sono binomiali bilaterali.

---

## B.2 — Raggio del latente

**Decisione: 0,8200 per il ramo condizionato** (era 0,7342). Il ramo non
condizionato resta a π (B.5).

Il valore precedente era calcolato su una griglia in cui `sift_rate` e
`loss_rate` erano scalari per corsa replicati su tutte le finestre. Sulla
griglia rigenerata la dispersione media entro configurazione passa da 0,041 a
0,050 e il raggio corrispondente da 0,7342 a 0,8200.

Dispersioni per osservabile, entro configurazione, in **unità normalizzate
min-max** — il confronto con dispersioni grezze porta fuori strada:

| | vecchia griglia | griglia rigenerata |
|---|---|---|
| qber | 0,05754 | 0,05336 |
| qber_rect | 0,07024 | 0,07109 |
| qber_diag | 0,07138 | 0,07136 |
| sift_rate | 0,00318 | **0,03584** |
| loss_rate | 0,00152 | **0,02084** |
| disparità dichiarata | 47× | **3×** |
| quota di varianza dovuta al seme | 4,8% | **0,2%** |

La quota del seme che crolla da 4,8% a 0,2% è la firma del difetto corretto:
quasi tutta quella varianza apparteneva alle due colonne costanti per corsa.

**Raggi per feature: motivazione da riscrivere, decisione sospesa.** Non
«disparità di 47× fra osservabili», che non era fisica, ma una disparità reale
di 3×. I raggi calibrati sulla griglia nuova sono
`0,6758 / 1,1005 / 0,9424 / 0,5118 / 1,0330`, intervallo 2,2× contro i 9×
precedenti. Non riusare `0,1719 / 1,0907 / 0,9721 / 0,1226 / 1,0547`.

Un raggio unico non copre i 3×: sul seme 0 a n=1.920 le tre osservabili QBER
escono compresse (rapporti generato/vero 0,62 / 0,46 / 0,60) e le due basse
dilatate (1,20 / 1,64). Se i raggi per feature servano ancora va deciso con un
confronto end-to-end a tre condizioni — raggio unico, per feature, e unico
sulle tre che variano con raggio stretto sulle due basse — non con una misura
statica. **Questa è una regola già stabilita dal progetto: le proprietà
all'inizializzazione non predicono quelle del generatore addestrato.**

---

## B.3 — Criterio di arresto

**Decisione invariata: MMD stratificata su `c` nel caso condizionato.**

Aggiunta: quando gli strati non sono popolati, il monitoraggio **riduce la
granularità** — 3, poi 2, poi 1 asse di `c` — invece di restituire NaN. Prima,
con tutti gli strati sotto soglia, nessun checkpoint batteva `inf` e veniva
ripristinata l'inizializzazione con `fitted_ = True`: un fit vacuo che si
dichiarava riuscito.

Il numero di assi effettivamente usato è registrato in `strati_assi_` e nei
risultati: MMD calcolate su granularità diverse non sono confrontabili fra
loro. Se nemmeno con un asse si raggiungono due strati utili, il fit viene
rifiutato con un errore.

Invarianza verificata: ogni corsa che alla granularità piena aveva almeno due
strati popolati si comporta esattamente come prima.

---

## B.6 — Ordine delle osservabili sui qubit

**Decisione invariata: ordine naturale, nessuna permutazione.**

Allineati i default che ancora contraddicevano la decisione: `--ordine` in
`cond_validation.py` e la codifica interna del toy in
`autoregressive_validation.py`, che usava ancora l'ordine b6. Le varianti
storiche restano come opzioni esplicite.

---

## B.7 — Stato iniziale w_0 nella generazione autoregressiva

**Decisione invariata: media dell'addestramento.**

Sostenuta da un numero, su venti semi: la MMD alla prima finestra vale 0,0307
con `media` contro 0,1332 con `zero`, un fattore 4,3.

Sull'autocorrelazione `w_0` non ha effetto — differenza +0,014 ± 0,017, segni
13/20 — ed è corretto, perché tocca solo la prima finestra.

Corretto un rischio dell'API che non aveva contaminato risultati:
`sample_traces(w0='media')` senza valore ripiegava in silenzio su zero. Ora il
valore si calcola una volta con `valore_w0()` e si passa a entrambe le fasi;
l'omissione è un errore.

---

## B.8 — Sottocampione della griglia

**Decisione riscritta: nessuna dipendenza rilevabile da `n` fra 480 e 28.800.**
Il precedente «massimo a n ≈ 7.680» era un artefatto.

Quel massimo veniva da una curva in cui lo stesso flusso RNG generava la
griglia e poi sceglieva le celle escluse: cambiare la numerosità cambiava le
celle di test. Per il seme 0 si passava da `{22, 26, 42, 38}` a
`{38, 26, 25, 41}` solo aumentando `--per-cella`. In più il numero di finestre
di valutazione era un limite superiore, non un valore fisso.

Corretto con flussi separati per ruolo e un test generato a parte, chiavizzato
sull'indice della cella: **stesse celle escluse a tutte le numerosità su tutti
e venti i semi, 32 finestre di test ovunque.**

Skill condizionato, venti corse indipendenti:

| n_train | 1.350 passi | 5.400 passi |
|---|---|---|
| 480 | +0,309 ± 0,036 | +0,355 ± 0,049 |
| 1.920 | +0,245 ± 0,057 | — |
| 7.680 | +0,234 ± 0,055 | — |
| 28.800 | +0,235 ± 0,043 | +0,331 ± 0,033 |

Confronto appaiato per seme fra gli estremi:

| budget | P480 − P8 | segni | p |
|---|---|---|---|
| 1.350 | −0,074 ± 0,042 | 8/20 | 0,50 |
| 5.400 | −0,024 ± 0,050 | 9/20 | **0,82** |

Liberato il budget la differenza si avvicina ancora di più allo zero.

Unito al costo, che il progetto ha già misurato indipendente dalla numerosità
— 2,18 milioni di circuiti per corsa a qualunque `n` — è un'affermazione utile
sull'economia del metodo: su questo bersaglio la numerosità non è una leva.

---

## B.9 (nuova) — Accoppiamento CRN delle misure quantistiche

Il modulo dichiarava tre proprietà. Due valevano, una no.

**Stabilità dei flussi fra processi: non valeva.** Il seed derivava da
`abs(hash((seed, tag, idx)))` con `tag` stringa, e l'hash delle stringhe in
Python è randomizzato per processo: `(1, 'eve', 10)` dava 126.733.112 in un
processo e 425.770.420 in un altro. Sostituito con `SeedSequence` e
identificatori numerici delle sorgenti.

**Coerenza fra catena modellata e catena eseguita: valeva già.** Zero
discordanze fra errore analitico e osservato, in entrambe le configurazioni e
su tutte e 600 le celle della rigenerazione. `attribute()` era corretto.

**Accoppiamento fra condizioni: non valeva.** `measure()` pescava dallo stato
casuale globale di NetSquid, mai inizializzato e non toccato da `sim_reset()`.
Cambiando `intercept_rate` cambia il numero di misure eseguite, quindi il
flusso globale veniva consumato diversamente e l'esito di Bob alla stessa
posizione era estratto da un punto diverso — ed è proprio quella casualità a
generare il QBER da intercettazione.

Corretto fissando lo stato globale dal flusso della posizione prima di ogni
misura. Verifica sulle posizioni intercettate in entrambe le corse a r = 0,25
e r = 0,50, che sono nidificate perché la soglia cambia e il flusso no:

| | esito di Bob divergente |
|---|---|
| con accoppiamento | **0 / 299** |
| senza | **87 / 299** (29,1%) |

Non è un test che possa passare per caso: circa metà delle posizioni comuni ha
esito deterministico, e sull'altra metà due estrazioni indipendenti
divergerebbero al 50%.

**Conseguenza.** Le griglie generate prima del 16 settembre 2026 non sono
riusabili per conclusioni causali. Non sono nemmeno rigenerabili bit per bit,
perché il seme effettivo dipendeva da un hash randomizzato: vanno conservate
come artefatti versionati con la loro impronta, non riprodotte.

---

## B.10 (nuova) — Osservabili calcolate per finestra

`sift_rate` e `loss_rate` erano `len(rec)/num_bits` e `stats["loss_rate"]`,
cioè **scalari per corsa** replicati identici su tutte le finestre della cella:
un solo valore distinto per cella, su tutte e 600.

Tre conseguenze, tutte reali: due qubit su cinque modellavano costanti già
interamente determinate da `c`; la disparità di dispersione fra osservabili
risultava 47× invece di 3×, e su quel 47× poggiava la scelta dei raggi per
feature; in modalità autoregressiva due delle cinque componenti di `w_{t-1}`
erano costanti lungo la traccia, quindi ripetevano la condizione invece di
portare informazione temporale.

Ora calcolate sullo span grezzo della finestra k, che copre le posizioni
`(limite[k-1], limite[k]]` dove `limite[k]` è l'indice dell'ultimo record
setacciato. Le finestre tassellano senza buchi né sovrapposizioni.

    sift_rate = size / n_grezze
    loss_rate = 1 - n_ricevute / n_grezze

Coerente con le definizioni per corsa: minimo e massimo di `sift_rate` nella
griglia, 0,078 e 0,400, sono 0,5·(1−perdita) a 40 km e a 5 km.

Verifica sulla griglia rigenerata: valori distinti per cella passano da **1** a
una mediana di **104** (`sift_rate`) e **154** (`loss_rate`), su circa 157
finestre.

**Anticorrelazione fra le due, ~−0,5: è fisica, non un artefatto.** Le
posizioni setacciate sono un sottoinsieme delle ricevute, quindi
`sift_rate ≈ 0,5·(1 − loss_rate)`. Verificato confrontando questa definizione
con l'alternativa a span grezzo fisso: −0,480 contro −0,508.

**Osservazione aperta.** L'autocorrelazione lag 1 di `sift_rate` fra finestre
consecutive vale +0,059 contro un nullo simulato di −0,008, circa 3,5 errori
standard su venti sequenze. Ogni componente testata da sola è pulita: perdita
esatta per posizione (zero discordanze su 20.000), uniformi di perdita senza
struttura a blocchi, basi nulle su 200 semi. Registrata come aperta; non tocca
nessun obiettivo corrente. Il test definitivo, se servisse affermare l'assenza
di memoria anche su `sift_rate`, è rigenerare con più semi.

---

## B.11 (nuova) — Allocazione di `c` sugli strati di condizionamento

La mappatura per `d_c > n_qubit` non era definita: `_cond_layer` leggeva
`c[q % d_c]` in entrambi gli strati, e con `d_c = 8` su cinque qubit le
componenti `c[5]`, `c[6]`, `c[7]` **non comparivano nel circuito**. In
`concat(c_statico, w_precedente)` significa che il generatore vedeva due sole
delle cinque osservabili della finestra precedente. Il discriminatore le
riceveva tutte e otto.

**Decisione: schema `avanti`**, `[[0,1,2,3,4], [5,6,7,0,1]]` per `d_c = 8`.

Progettato come **no-op per `d_c ≤ n_qubit`**: il ramo congelato da §12 resta
bit-identico, verificato su circuito, forward e jacobiana. Nessun risultato già
prodotto con `d_c` in {0, 1, 3, 5} va rifatto, e **O1 non va rimisurato**.

**Alternativa provata e scartata: schema `coda`**, `[[3,4,5,6,7],
[0,1,2,3,4]]`, che mette l'intera finestra precedente nel primo strato, il più
influente. Dà alla finestra precedente +46% di sensibilità, togliendolo al
condizionamento statico: il totale è conservato, è un baratto.

Venti semi appaiati, `coda` meno `avanti`:

| | differenza | segni | p |
|---|---|---|---|
| autocorrelazione intra (meglio alto) | −0,059 ± 0,031 | 7/20 | 0,26 |
| MMD teacher forcing (meglio basso) | +0,011 ± 0,005 | 13/20 | 0,26 |
| MMD generazione libera (meglio basso) | +0,011 ± 0,005 | 13/20 | 0,26 |

Nessun confronto è significativo, ma il segno è sfavorevole su tre indicatori
su quattro: si tiene `avanti`. L'opzione resta nel codice con `avanti` come
default.

Nota metodologica: l'asimmetria fra strati esiste ma è modesta. Mediata su
cinque inizializzazioni vale 2,9×, non i 18× che una singola inizializzazione
suggeriva.

---

## Appendice A — `max_steps` è sottodimensionato

Non è una preferenza, è una misura. Con `max_steps = 1350`, `eval_every = 25` e
pazienza 20 valutazioni:

| budget | corse al tetto | `steps_run` medio | `best_step` medio |
|---|---|---|---|
| 1.350 | **17–18 / 20** | 1.350 | 1.319–1.329 |
| 5.400 | **1 / 20** | 3.344–3.432 | 2.882–2.954 |

A 1.350 la pazienza non scattava mai: al termine del budget il modello stava
ancora migliorando. La convergenza cade fra 2.900 e 3.400 passi.

**Proposta: 5.400 per il condizionamento QKD**, dichiarando che i numeri a
1.350 erano troncati. Col simulatore denso il costo è un decimo di prima.

**Ricade su O1.** Il «fattore cinque nei passi alla convergenza, 175 contro
875» è stato misurato sotto lo stesso tetto: va verificato che 875 non fosse
anch'esso troncato prima di tenerlo fra i risultati di trasferibilità.

---

## FASE2 §16.3.1 — da riscrivere

Venti corse indipendenti, configurazione P32, budget 1.350.

| | media | IC95 | segni |
|---|---|---|---|
| skill condizionato | +0,2445 | [+0,1289, +0,3470] | 18/20 |
| skill incondizionato | −0,2468 | [−0,3310, −0,1648] | 2/20 |
| condizionato − incondizionato | +0,4913 | [+0,3737, +0,5990] | 19/20 |
| MAE banale − MAE condizionato | +0,0186 | [+0,0107, +0,0262] | 18/20 |
| guadagno MMD incond − cond | +0,1458 | [+0,1031, +0,1844] | 18/20 |
| risposta al condizionamento | +0,9047 | [+0,8647, +0,9401] | 20/20 |

**Due conclusioni cambiano.** `banale − condizionato` passa da «non si può
affermare» (p = 0,16) a 18 segni su 20: il generatore condizionato ricostruisce
le celle escluse meglio della media della griglia. E `condizionato −
incondizionato` sopravvive con un intervallo che **non contiene lo zero** —
prima l'intervallo bootstrap dava [−0,069, +0,375], contenendo lo zero, mentre
il `p` accanto valeva 0,0016: le due statistiche si contraddicevano perché
erano calcolate su unità diverse.

L'incondizionato è negativo ovunque: peggio della media della griglia. Il
condizionamento non migliora di poco, fa la differenza fra funzionare e non
funzionare.

**Tre cautele da riportare nel testo.**

Il `p` minimo ottenibile con venti segni concordi è 2·2⁻²⁰ ≈ 0,00002: è il
pavimento del test, non una misura dell'ampiezza dell'effetto. Riportare segni
e intervallo, non il `p` da solo.

Il bersaglio è sintetico, con relazione fra `c` e osservabili monotona e liscia.
Condizione necessaria, non sufficiente.

L'analisi **generalizza sulle corse, non sulle celle**: le celle fisiche
ricompaiono fra semi. Per generalizzare anche su quelle serve un'analisi che
consideri entrambi i raggruppamenti — è l'ultimo pezzo aperto del difetto 4
dell'audit, e richiede la riscrittura di `analisi_appaiata.py`.

---

## FASE2 §17 — da riscrivere

«Generazione autoregressiva. Funziona.» non regge.

**La metrica è stata validata prima del modello.** Sul toy AR(1) con φ = 0,75,
la lettura residua sulle tracce reali dà **+0,7501 ± 0,0017** contro un valore
vero di 0,7500.

**La metrica precedente misurava il livello, non la memoria.** Su venti semi:

| lettura, tracce reali | valore | con la memoria distrutta a permutazioni |
|---|---|---|
| residua (livello noto) | +0,750 | — |
| intra (media empirica) | +0,606 | −0,064 |
| **aggregata (la precedente)** | **+0,936** | **+0,822** |

Delle 0,936 della metrica aggregata, 0,822 restano anche azzerando ogni
dipendenza temporale: **l'88% era differenza di livello fra tracce.** Il
confronto 0,818 contro 0,969 di §17.3 misurava quasi solo quello.

**Il modello cattura circa il 29% della memoria temporale.** Sulla scala intra,
dove il nullo è −0,064 e il vero +0,606, le tracce sintetiche stanno a
**+0,129 ± 0,023**. La lettura aggregata lo conferma per un'altra via: 0,832
contro un proprio nullo di 0,822.

Attenzione a un artefatto: la lettura residua sulle sintetiche dà +0,672, ma
centra sul livello **vero**, quindi un errore di livello del modello entra nel
residuo come costante per traccia e una costante ha autocorrelazione 1 a ogni
lag. È anche la spiegazione del MAE di 0,299 su otto lag contro un divario di
0,078 a lag 1. Sulle sintetiche va usata la lettura intra.

**Il rapporto di deriva 2,22 ± 0,45 non è confrontabile con l'1,17 riportato.**
Con il vecchio encoding il generatore vedeva due sole osservabili precedenti su
cinque, quindi dipendeva meno da `w_{t-1}` e accumulava meno errore. Una deriva
maggiore è la conseguenza attesa di un condizionamento autoregressivo che ora
funziona davvero.

**Non c'è un aggiustamento economico.** Riallocare `c` fra gli strati (B.11) non
migliora: peggiora, senza significatività.

**E i dati veri non pongono il problema.** Sulla griglia rigenerata
l'autocorrelazione lag 1 entro cella delle osservabili di errore è compatibile
con l'assenza di memoria — `qber` −0,0026, `qber_diag` −0,0004, contro un nullo
di −0,0064. Ogni sorgente stocastica del simulatore è indipendente per
posizione e nessun parametro varia nel tempo: le finestre di una cella sono
scambiabili.

---

## Decisioni da prendere insieme

**1. Il destino di §6.** Due vie. Ritirare la generazione autoregressiva dalle
rivendicazioni, tenendola come macchina validata: funziona meccanicamente, non
deriva, non collassa, ma riproduce il livello e non la dinamica — e i dati veri
non hanno dinamica da riprodurre. Oppure estendere il simulatore perché produca
condizioni variabili nel tempo, per esempio Eve che si attiva a metà traccia,
dando all'autoregressione un senso. La prima è a costo zero, la seconda apre
una Fase 4.

**2. `max_steps`.** Portarlo a 5.400, con la verifica a ricaduta su O1.

**3. Raggi per feature.** Il confronto end-to-end a tre condizioni, dopo il
quale B.2 si chiude.

**4. Difetto 4 nella forma piena.** Riscrivere `analisi_appaiata.py` con
un'analisi che consideri corse e celle come due raggruppamenti. Le venti corse
indipendenti che servono ora esistono.

**5. Fase 3.** Non è mai stata avviata. Nessun modello è stato addestrato su
NSL-KDD o sulla griglia QBER: tutta la calibrazione è su bersagli sintetici,
deliberatamente.
