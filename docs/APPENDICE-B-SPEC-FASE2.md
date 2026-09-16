# Appendice B a SPEC-FASE2 — Revisione dopo la validazione di RQ2

Documento di decisione. Modifica tre punti della specifica e uno
dell'Appendice A, sulla base delle misure riportate in `../../../../Scaricati/FASE2.md` §12–§15.

A differenza dell'Appendice A, che fissava elementi già dichiarati liberi da
§12, **questo documento tocca elementi congelati**. Va approvato da entrambe
le parti prima di essere applicato, e le modifiche vanno riportate nel paper
come scelte motivate, non come dettagli implementativi.

---

## B.1 Riepilogo delle decisioni

| | Elemento | Stato in SPEC | Decisione |
|---|---|---|---|
| B.2 | Raggio del latente `z` | congelato, §3 | si congela **la regola**, non il valore |
| B.3 | Criterio di arresto nel caso condizionato | congelato, §7.3 e §9 | MMD **stratificata su `c`** |
| B.4 | Definizione dello skill score | convenzione di Fase 1-B | su MAE **aggregate**, non media di rapporti |
| B.5 | Baseline non condizionata | congelato, §5 | usa il **proprio** raggio del latente |
| B.6 | Ordine delle osservabili sui qubit | non fissato | **nessuna permutazione**, vedi B.6 |
| B.7 | `w_0` nella generazione autoregressiva | congelato, §6 | media dell'addestramento, non zero |
| B.8 | Sottocampione della griglia | non fissato | n ≈ 7.680, vedi `../../../../Scaricati/FASE2.md` §18 |

---

## B.2 Raggio del latente

### Cosa dice la specifica

§3: «Il vettore latente `z` ha dimensione 5, campionato uniformemente in
`[0, π]`.»

### Perché va cambiato

Con `[0, π]` la validazione di RQ2 su bersaglio sintetico dà skill **−1,02**,
cioè peggio del riferimento banale. Con `[0, π/8]` dà **+0,390 ± 0,240** su
cinque semi, con il generatore condizionato che batte l'incondizionato sulle
celle escluse in tutte e cinque le corse.

Non è un miglioramento marginale: senza questa modifica RQ2 non ha risultato, e
RQ2 è il contributo della Parte 2.

### La regola

Il raggio del latente controlla direttamente la dispersione della distribuzione
condizionale generata. La regola è:

> Il raggio del latente si sceglie perché la dispersione condizionale generata
> eguagli la dispersione entro cella dei dati.

Si calcola prima di eseguire qualunque addestramento, e non richiede ricerca.
La dispersione all'inizializzazione in funzione del raggio:

| `latent_hi` | dispersione da `z` |
|---|---|
| π | 0,33 |
| π/2 | 0,15 |
| π/4 | 0,047 |
| π/8 | 0,0125 |

approssimativamente `latent_hi ≈ (π/8)·√(σ / 0,0125)`, dove `σ` è la
deviazione standard entro cella, **calcolata nello spazio riscalato in `[0,1]`
e non in unità grezze**.

Sul bersaglio sintetico, `σ` vale 0,019 e π/8 dà dispersione generata 0,017:
rapporto 0,89. A π/4 era 1,74.

### Misura sui dati reali

Eseguita con `scripts/latent_radius_qkd.py` sulla griglia a 75.330 righe, seme
5 escluso, 120 configurazioni.

| | |
|---|---|
| dispersione entro configurazione | 0,04077 |
| dispersione entro cella | 0,03979 |
| quota di varianza dovuta al seme | **4,8 %** |
| raggio unico risultante | 0,7342 = π/4,3 |

La previsione di π/8 formulata prima della misura era **sbagliata di circa un
fattore due**: il riscalamento in `[0,1]` cambia il numero, come annotato.

La quota del 4,8 % è il dato più importante: quasi tutta la dispersione a `c`
fissato è predicibile da `c`, quindi non esiste un tetto intrinseco basso per
RQ2. Se fosse stata sopra il 50 %, l'esperimento sarebbe stato limitato
superiormente a prescindere dal generatore.

### I raggi per feature: proposta ritirata

**Questa sezione riporta una proposta formulata e poi ritirata dai dati.** È
conservata perché il motivo per cui era plausibile e sbagliata è informativo.

Le cinque osservabili hanno dispersioni condizionali che differiscono di un
**fattore 47**:

| osservabile | σ |
|---|---|
| qber_diag | 0,07138 |
| qber_rect | 0,07024 |
| qber | 0,05754 |
| sift_rate | 0,00318 |
| loss_rate | 0,00152 |

Ha senso fisico: il sifting rate dipende dalla corrispondenza fra basi e resta
intorno a 0,5; la perdita è quasi deterministica data la lunghezza. Con un
raggio unico a 0,7342 tutte le osservabili escono intorno a 0,040, cioè
`loss_rate` risulterebbe **26 volte troppo dispersa**.

Il latente ha dimensione 5, una componente per qubit, e §3 applica `RY(z_i)`
sul qubit `i`. I raggi possono quindi essere **per feature**, e
all'inizializzazione funzionano: con `0,1719 · 1,0907 · 0,9721 · 0,1226 ·
1,0547` il rapporto fra dispersione ottenuta e richiesta resta fra 0,97 e 1,31
su tutte e cinque, contro un fattore 26 di errore su `loss_rate` con raggio
unico.

**Ma la validazione end-to-end non lo conferma.** Le dispersioni per
osservabile a fine addestramento, media su tre semi:

| osservabile | bersaglio | ottenuta | rapporto |
|---|---|---|---|
| sift_rate | 0,0032 | 0,0326 | 10,2 |
| qber_rect | 0,0702 | 0,0398 | 0,6 |
| qber | 0,0575 | 0,0324 | 0,6 |
| loss_rate | 0,0015 | 0,0412 | 27,5 |
| qber_diag | 0,0714 | 0,0279 | 0,4 |

**La disparità passa da 48× nei dati a 1,5× nel generato.** I raggi iniziali
vengono cancellati: man mano che θ cresce, l'entanglement riporta tutte le
marginali alla stessa scala. La regola controlla il punto di partenza, non
quello di arrivo, e quello che conta è il secondo.

Sul confronto appaiato con il raggio unico, su bersaglio con le dispersioni
reali:

| seme | per feature | raggio unico |
|---|---|---|
| 0 | +0,234 | +0,374 |
| 1 | +0,167 | +0,404 |
| 2 | +0,245 | −0,272 |
| media | +0,215 ± 0,042 | +0,169 ± 0,382 |

Portato a cinque semi, il quadro cambia ancora:

| | 3 semi | 5 semi |
|---|---|---|
| raggio unico | +0,169 ± 0,382 | **+0,185 ± 0,273** |
| per feature | +0,215 ± 0,042 | +0,115 ± 0,197 |
| favorevoli ai per feature | 1/3 | **1/5** |

Il raggio unico vince in media e su quattro semi su cinque; la deviazione dei
raggi per feature sale da 0,042 a 0,197, quindi la stabilità apparente a tre
semi era il caso. Differenze appaiate media −0,070, `t` = −0,45.

### Decisione: raggio unico

Si adotta il **raggio unico 0,7342**, calcolato dalla dispersione condizionale
media. È più semplice, resta dentro la forma della specifica, ed è
migliore su quattro semi su cinque — anche se la differenza non è
statisticamente distinguibile.

L'errore di metodo è stato raccomandare i raggi per feature sulla base di una
misura all'inizializzazione: è precisamente il tipo di conclusione che la
validazione end-to-end serve a smentire.

### Limite da dichiarare nel paper

**Il generatore riproduce la posizione delle celle, non la loro forma.** Le
dispersioni per osservabile collassano da un fattore 48 a un fattore 1,5,
indipendentemente dai raggi iniziali. Per l'attribuzione causale della Fase 4
potrebbe non essere determinante, perché quel che serve è che i campioni
sintetici cadano nella regione giusta; ma va riportato, non scoperto in
revisione.

### Un raggio unico non basta per la forma

### Il valore per l'altro dominio

- **QKD**: raggio unico `0,7342`, dalla dispersione condizionale media 0,04077.
- **IDS**: `π` uniforme. La configurazione base è non condizionata, quindi non
  esiste una dispersione condizionale da eguagliare e il latente è l'unica
  fonte di diversità del generatore.
- **IDS**: `π`. La configurazione base è non condizionata, quindi non esiste
  una dispersione condizionale da eguagliare, e il latente è l'unica fonte di
  diversità del generatore.

### Conseguenza su O1

È la **seconda voce non trasferibile**, dopo la larghezza del discriminatore
(§11.3). Con una differenza sostanziale: qui la regola di scelta è nota a
priori, quindi la non trasferibilità è *prevedibile* dal rumore condizionale
del dominio, non scoperta per tentativi.

Va riportata nella tabella di trasferibilità con la stessa struttura usata per
il discriminatore: ottimo per dominio, valore comune, costo del compromesso.
Il costo di forzare il valore comune `π` è di 1,41 punti di skill sul dominio
QKD, da −1,02 a +0,39.

L'affermazione di O1 cambia forma: da «la stessa configurazione si trasferisce»
a «la stessa architettura con una regola di adattamento dichiarata si
trasferisce». Restano condivisi qubit, ansatz, encoding, schema di
condizionamento, lettura, gradienti, discriminatore, learning rate, batch,
budget e protocollo di valutazione. **Va tenuto il conto esplicito delle voci
adattate**: al momento due su una dozzina, ancora difendibile, ma è una
grandezza che va monitorata e dichiarata.

---

## B.3 Criterio di arresto nel caso condizionato

### Cosa dice la specifica

§7.3: «Su Maximum Mean Discrepancy calcolata su un insieme di validazione.»
§9 elenca la MMD fra le metriche di qualità distribuzionale.

### Perché va cambiato

La MMD aggregata **non può vedere il condizionamento**. Confronta l'insieme dei
campioni generati con l'insieme dei reali; un generatore che ignora `c` eguaglia
la distribuzione aggregata senza sforzo, perché l'aggregato è la marginale
della condizionale.

Il criterio quindi seleziona *contro* il condizionamento. Misurato: il
checkpoint ripristinato cadeva al passo 225, con `c/z` pari a 0,207, mentre al
passo 725 `c/z` valeva 0,820 e quel modello veniva scartato.

Correggendolo, la correlazione della risposta è passata da +0,305 ± 0,381, con
segno che cambiava fra semi, a +0,796 ± 0,249.

### La modifica

Nel caso condizionato la MMD di monitoraggio è calcolata **separatamente su
strati di `c`** e mediata. Gli strati sono due livelli per asse, fino a otto.
Un generatore che ignora `c` fallisce in ogni strato, perché confrontato con la
condizionale e non con la marginale.

Nel caso non condizionato nulla cambia.

### Questione aperta

La risoluzione degli strati è grossolana rispetto alle celle su cui RQ2 viene
valutato. È sufficiente a non essere cieca, che era il difetto, ma non è
garantito che sia sensibile alla granularità richiesta. **Se in Fase 4 lo skill
sui dati reali risultasse basso, alzare la risoluzione degli strati è la prima
cosa da provare**, ed è una riga di codice.

---

## B.4 Definizione dello skill score

Fase 1-B §2 definisce lo skill come `1 − MAE_modello / MAE_banale`. La prima
implementazione di `conditional_validation.py` lo calcolava **cella per cella e
poi ne faceva la media**, che è una media di rapporti dove serve un rapporto di
medie.

Con quattro celle, una in cui il riferimento banale è casualmente ottimo
produce un rapporto enorme che domina la media. Sul bersaglio a π/4 la
differenza fra le due formule era **+0,020 contro +0,292**, con deviazione
0,169 contro 0,063.

Si adotta la formula su MAE aggregate, coerente con Fase 1-B. Va verificato che
`baseline_attributor.py` usi la stessa convenzione, così che i numeri delle due
fasi siano confrontabili.

---

## B.5 Baseline non condizionata

§5 stabilisce che la configurazione non condizionata è una baseline
obbligatoria, perché isola il contributo del condizionamento.

Perché isoli davvero, deve girare **al proprio raggio del latente**, non a
quello del ramo condizionato. Senza `c`, il latente è la sua unica fonte di
diversità: a raggio stretto non copre più la griglia e il confronto è truccato
a favore del condizionato.

Misurato: il guadagno sulla MMD passa da **+0,587 con baseline penalizzata a
+0,220 con baseline al proprio ottimo**. Più della metà del guadagno era
demerito del confronto.

È la stessa obiezione già sollevata in Fase 1-A §5 su ADASYN e in Fase 1-B §5
su HistGradientBoosting.

---

## B.6 Ordine delle osservabili sui qubit

L'entanglement lineare non è simmetrico: il qubit terminale della catena è
sempre bersaglio e mai controllo. Misurato, l'accoppiamento fra il latente di
un qubit e la dispersione della propria marginale vale 1,00 · 0,99 · 0,99 ·
1,00 sui primi quattro e **0,21 sul quinto**.

Ne segue che restringere il raggio del quinto qubit non ne riduce la
dispersione, perché il 79 % di questa arriva dagli altri attraverso
l'entanglement. **L'osservabile che richiede la dispersione più stretta non va
collocata lì.**

§1.2 elenca le cinque osservabili ma non le assegna ai qubit, quindi l'ordine è
libero. Il criterio è: sul qubit meno pilotabile l'osservabile che richiede più
dispersione.

### La regola era sbagliata, ed è stata ribaltata

L'ordine `sift_rate, qber_rect, qber, loss_rate, qber_diag` metteva `qber_diag`,
la dispersione maggiore, sul qubit terminale. Il test end-to-end lo smentisce in
**entrambi i domini**.

Dominio IDS, non condizionato, `scripts/ordine_qubit_ids.py`, riduzione di MMD
su cinque semi:

| ordine | riduzione MMD | sigma generata / vera |
|---|---|---|
| naturale (varianza decrescente) | **+87,3 % ± 4,0** | 1,28 … 2,66 |
| casuale | +86,7 % ± 6,6 | 1,46 … 2,26 |
| invertito (varianza crescente) | +71,1 % ± 12,0 | 3,22 … 0,95 |

`naturale − invertito` = +0,161, `t` = 2,56, **5/5 semi**.
`naturale − casuale` = +0,005, `t` = 0,36.

Dominio QKD, condizionato, `--ordine` di `conditional_validation.py`, skill su
cinque semi:

| ordine | skill |
|---|---|
| decrescente | **+0,321 ± 0,222** |
| naturale (§1.2) | +0,294 ± 0,217 |
| b6 | +0,191 ± 0,271 |

`decrescente − b6` = +0,131, `t` = 1,76, **5/5 semi** (`p` = 0,031 al test dei
segni, che con cinque osservazioni è il criterio appropriato).
`decrescente − naturale` = +0,028, `t` = 1,12.

**In entrambi i domini non è un ordine a essere buono: è quello che mette la
dispersione maggiore sul qubit terminale a essere cattivo.** La regola corretta
è l'opposto di quella formulata: su quel qubit il generatore fatica a
*produrre* dispersione, non solo a comprimerla.

La conclusione regge su due domini con condizionamento diverso (assente contro
`d_c = 3`), raggi diversi (π contro 0,7342) e disparità fra osservabili diverse
(3,6× contro 47×).

### Decisione: nessuna permutazione

Poiché `naturale` è statisticamente equivalente all'ottimo in entrambi i domini,
**si usa l'ordine naturale**: §1.2 per il QKD, varianza spiegata decrescente per
le componenti PCA dell'IDS. Nessuna permutazione, nessuna scelta da
giustificare, preparazione dei dati più semplice.

L'unico vincolo è **non** collocare sul qubit terminale l'osservabile o la
componente a dispersione maggiore.

### Nota di metodo

È la quarta regola derivata da una misura all'inizializzazione che la verifica
end-to-end smentisce, dopo l'inizializzazione di θ (§13), i raggi per feature
(§16.2) e il sottocampione (§18.1). Il pattern documentato in `../../../../Scaricati/FASE2.md` §17.5
regge: **su questa architettura le proprietà del generatore all'inizializzazione
non predicono quelle del generatore addestrato.**

**Per il dominio IDS l'ordine resta da decidere.** Le componenti principali
hanno un ordine naturale per varianza spiegata, ma il criterio di B.6 riguarda
la varianza *condizionale*, che nel caso non condizionato non è definita. Va
deciso se applicare l'ordine PCA naturale, dichiarandolo, o se il qubit
terminale meriti la componente a varianza maggiore anche lì. **Sarebbe la terza
voce adattata per dominio**, e conviene contarla.

## B.7 Cosa deve fare il lato QKD

1. ~~Aggiornare `src/qgan/`~~ — fatto.
2. ~~Calcolare la dispersione condizionale~~ — fatto, B.2.
3. ~~Ricavare i raggi~~ — fatto, B.2.
4. ~~Decidere sull'ordinamento di B.6~~ — fatto: nessuna permutazione,
   ordine naturale in entrambi i domini.
5. ~~Applicare il confronto appaiato per cella di SPEC §11~~ — fatto. Non
   riduce l'incertezza: la variabilità è fra celle, non fra semi. Vedi
   `../../../../Scaricati/FASE2.md` §16.3.
6. ~~Decidere il trattamento della prima finestra~~ — fatto: `w_0` è la media
   delle finestre di addestramento, non zero. Rapporto fra primo passo e
   successivi da 16,3 a 3,7. Modifica a §6 della specifica, vedi
   `../../../../Scaricati/FASE2.md` §17.2.
7. ~~Dimensionare il sottocampione~~ — fatto: n ≈ 7.680, vedi `../../../../Scaricati/FASE2.md` §18.
8. Costruire la pipeline di RQ2 sui dati reali: addestrare la QGAN sulle celle
   di training, generare finestre sintetiche per le celle escluse dal
   protocollo `interior`, riaddestrare l'attributore con augmentation e
   confrontare lo skill **appaiato per cella** contro lo 0,282 di baseline.
   Non ancora scritta.

Le baseline di SPEC §10 sono state riverificate a cinque osservabili e
riprodotte: `interior` 0,282 ± 0,116, `random` 0,442 contro 0,444 dichiarato.
Restano valide e non vanno rigenerate.

---

## B.8 Limiti di questa appendice

Tutte le misure che la motivano sono su **bersagli sintetici**. La scelta è
deliberata e coerente con §9.1 di `../../../../Scaricati/FASE2.md`: calibrare sui dati reali
reintrodurrebbe l'asimmetria che Fase 1 ha evitato. Ma ne segue che la regola
di B.2 è **verificata su un solo dominio simulato**, e la sua validità sui dati
QKD reali è una previsione da controllare, non un fatto acquisito.

Il bersaglio sintetico ha inoltre una relazione fra `c` e le osservabili
monotona e liscia, più facile di quella reale, dove Fase 1-B §4 documenta che
`f_canale` ha skill 0,088 sull'interpolazione. Un esito positivo qui è
condizione necessaria e non sufficiente.

Lo skill a π/8 ha deviazione 0,240 su cinque semi, dominata dalla selezione
delle celle escluse: il MAE del riferimento banale varia di un fattore 1,66 fra
semi. Il confronto appaiato previsto da §11 della specifica non è ancora stato
applicato e ridurrebbe questa componente.

`loss_rate` è il candidato più probabile a fallire: finisce sul qubit 3 con
rapporto sensibilità-su-dispersione 0,18, il più basso dei cinque. Il raggio
piccolo che le serve per la dispersione le comprime anche la risposta a `c`,
che la sposta di circa un terzo della sua stessa larghezza. Va guardata
separatamente e non nella media sulle cinque osservabili.

Il regime avversariale resta quello con `L_D` fra 0,14 e 0,38 e varianza del
gradiente fino a 0,58. Funziona ma non è sano nel senso usuale, e il tentativo
di renderlo tale ha portato lo skill a −0,79 ± 1,09.
