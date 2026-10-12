# Audit della conclusione IDS di Fase 3 — 5 ottobre 2026

## Ambito

Questo audit verifica la conclusione IDS esposta in
`REVISIONE-FASE2-FASE3-20261004.md` confrontandola con:

- `src/ids/ids_augmentation.py`;
- il preprocessing in `src/ids/preprocess_nslkdd.py`;
- l'implementazione della QGAN e della MMD;
- i risultati grezzi in `results/ids_1_1000`, `results/ids_1_200` e
  `results/ids_reps*`.

L'audit riguarda soprattutto l'affermazione «IDS — con 53 campioni la
distribuzione non è apprendibile» e le interpretazioni causali che seguono.

## Verdetto

Il risultato operativo regge: **nella configurazione provata, sullo split IDS
fisso e sulla validation usata, l'augmentation QGAN non è competitiva con
`class_weight`, rumore o SMOTE**.

Non reggono invece, nella forma attuale, le conclusioni più generali secondo
cui:

- la distribuzione IDS non sarebbe apprendibile;
- il collo di bottiglia sarebbe stato identificato nella numerosità;
- il meccanismo della perdita di prestazione sarebbe stato identificato
  nell'omogeneizzazione delle dispersioni;
- la QGAN non supererebbe nemmeno l'assenza di augmentation.

I dati permettono di concludere che **la QGAN e il protocollo di addestramento
provati falliscono il confronto operativo**. Non distinguono fra un limite
dell'architettura, un problema di ottimizzazione e un limite dei dati.

## 1. Contraddizione nel confronto con l'assenza di augmentation

La tabella del documento riporta:

| variante | F1 RF medio |
|---|---:|
| nessuna | 0,5506 |
| QGAN | 0,5773 |

La QGAN ha quindi un F1 medio superiore di **+0,0267**, con differenza positiva
in quattro semi su cinque. La frase «non supera nemmeno il non far nulla» è
numericamente errata.

Con il test dei segni binomiale bilaterale dichiarato dal documento:

- QGAN contro nessuna augmentation: +0,0267, 4/5, `p = 0,375`;
- QGAN contro `class_weight`: −0,1959, 0/5, `p = 0,0625`.

Il confronto con `class_weight` mostra una differenza grande e coerente, ma
cinque segni concordi non raggiungono la soglia convenzionale 0,05 con un test
bilaterale.

La formulazione corretta è che la QGAN migliora lievemente il punto stimato
rispetto all'assenza di augmentation, senza evidenza statistica con cinque
repliche, e perde nettamente rispetto a `class_weight`.

## 2. La conclusione dipende dall'F1 alla soglia 0,5

`ids_augmentation.py` calcola l'F1 sulle etichette restituite direttamente da
`predict()`. Per i classificatori usati questo equivale alla soglia predefinita
0,5. I risultati RF completi raccontano una storia più articolata:

| variante | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|
| nessuna | 0,5506 | 0,8420 | 0,9488 |
| bilanciato | 0,7732 | 0,8526 | 0,9482 |
| duplicazione | 0,7126 | 0,8560 | 0,9546 |
| rumore | 0,7349 | **0,8964** | **0,9779** |
| SMOTE | **0,7740** | 0,8870 | 0,9758 |
| QGAN | 0,5773 | 0,8559 | 0,9688 |

La QGAN resta peggiore dei riferimenti classici migliori, ma in PR-AUC supera
leggermente sia l'assenza di augmentation sia `class_weight`.

La precisione RF è inoltre circa 1,0 in quasi tutte le condizioni. Le
differenze di F1 riflettono soprattutto il recall ottenuto alla soglia 0,5.
`class_weight` modifica anche la calibrazione e la frontiera decisionale, perciò
il confronto sul solo F1 predefinito non misura esclusivamente la qualità dei
campioni generati.

Per una conclusione generale sul classificatore occorre dichiarare in anticipo
la metrica primaria e il protocollo per la soglia. PR-AUC è una scelta naturale
in questo regime fortemente sbilanciato. Se l'F1 resta la metrica primaria, la
soglia dovrebbe essere scelta in un insieme interno distinto dalla validation
usata per il confronto finale.

## 3. Le cinque corse non sono cinque campioni IDS indipendenti

Il preprocessing usa un solo split train/validation con seed 42 e seleziona una
sola minoritaria di 53 righe. Tutte le corse riutilizzano:

- gli stessi 53 esempi minoritari;
- gli stessi esempi maggioritari;
- la stessa validation.

I semi da 0 a 4 cambiano l'inizializzazione della QGAN, la generazione dei
campioni sintetici e la casualità dei classificatori. Misurano quindi la
stabilità algoritmica **condizionata su uno specifico dataset**, non la
variabilità rispetto alla scelta dei 53 attacchi.

Questo è sufficiente per affermare che l'esito è stabile rispetto alle
inizializzazioni provate. Non permette di estendere direttamente il risultato
ad altri campioni di 53 casi o alla popolazione IDS.

Per l'inferenza richiesta dalla conclusione servono più sottocampioni
indipendenti o split ripetuti, con l'intera pipeline — PCA locale inclusa —
riadattata in ogni replica.

## 4. «Il meccanismo è identificato» è eccessivo

I risultati mostrano che i campioni QGAN hanno dispersioni distorte. Non
dimostrano che tale distorsione causi la perdita di F1.

Mancano almeno due controlli causali:

1. un'ablazione che corregga le dispersioni dei sintetici lasciandone invariato
   il resto e misuri il recupero di F1;
2. una misura dell'importanza di `z5` o un'ablazione delle feature che dimostri
   che il classificatore «ha bisogno proprio» della componente più stretta.

Una dispersione errata può derivare da una famiglia generativa incapace di
rappresentare il bersaglio, ma anche da ottimizzazione instabile, selezione del
checkpoint o collasso dell'addestramento avversario.

La tabella documentale delle dispersioni non rappresenta inoltre le cinque
repliche dell'ablazione `reps=4`. Su queste corse il rapporto medio fra
dispersione generata e reale di `z5` è 3,15, con valori osservati da 2,09 a
3,90. Il fenomeno esiste, ma il singolo valore 3,65 non ne descrive la
variabilità.

Una formulazione sostenibile è: «la perdita di prestazione è associata a una
forte distorsione delle dispersioni, in particolare sulla componente più
stretta». L'identificazione causale del meccanismo resta aperta.

## 5. L'ablazione su `reps` non identifica il collo di bottiglia

La MMD peggiora mediamente aumentando `reps`, ma questo non dimostra da solo un
sovradattamento né esclude un limite di ottimizzazione. Cambiando la profondità
cambiano contemporaneamente:

- il numero di parametri;
- il paesaggio di ottimizzazione;
- la scala e la varianza dei gradienti;
- il raggio del latente calibrato;
- il numero di passi utile prima dell'arresto anticipato.

Learning rate, budget e criterio di arresto restano invece fissi. Un modello
più profondo può quindi ottenere risultati peggiori perché richiede un diverso
protocollo di addestramento, non necessariamente perché sovradatta i 53 punti.

La controprova con 269 campioni reali produce ancora una MMD media di circa
0,423. Questo mostra che aumentare i dati da 53 a 269 **non risolve il problema
con la configurazione corrente**. Non sostiene però la frase «il collo di
bottiglia è la numerosità»; semmai rende insufficiente quella spiegazione.

La scelta pragmatica di non riaprire SPEC §12 su `reps` è difendibile: nessuna
profondità provata migliora l'esito operativo. La causa del fallimento resta
indeterminata.

## 6. La MMD finale della QGAN non è out-of-sample

La configurazione IDS usa `val_fraction=0.0`. La QGAN addestra sui 53 punti e
usa gli stessi punti per:

- monitorare la MMD;
- scegliere il checkpoint;
- calcolare la MMD finale contro i sintetici.

La MMD finale non misura quindi la generalizzazione distribuzionale a esempi
minoritari non visti. La selezione del checkpoint tende semmai a renderla
ottimistica. Il fatto che rimanga molto alta rafforza il risultato negativo
operativo, ma non autorizza una conclusione generale sull'apprendibilità della
distribuzione.

## 7. Correzioni statistiche e documentali

### Barre `±`

I valori `±` della tabella IDS sono errori standard della media:

| variante | media | errore standard |
|---|---:|---:|
| nessuna | 0,5506 | 0,0043 |
| bilanciato | 0,7732 | 0,0011 |
| duplicazione | 0,7126 | 0,0089 |
| rumore | 0,7349 | 0,0037 |
| SMOTE | 0,7740 | 0,0056 |
| QGAN | 0,5773 | 0,0101 |

Non sono intervalli bootstrap al 95%, come potrebbe far pensare
l'intestazione generale del documento. La tabella deve etichettarli
esplicitamente come `media ± SE`, oppure sostituirli con gli intervalli
dichiarati dal protocollo.

### Numerosità finale

Le varianti non hanno tutte la stessa numerosità minoritaria finale:

- `nessuna` e `bilanciato`: 53 campioni reali;
- duplicazione, rumore, SMOTE e QGAN: 269 campioni fra reali e sintetici.

`class_weight="balanced"` è un riferimento forte e legittimo, ma non è una
condizione alla stessa numerosità. Assegna pesi che bilanciano le classi e non
equivale ad aggiungere campioni fino al rapporto 1:200.

### SMOTE contro `class_weight`

SMOTE ha F1 medio 0,7740 contro 0,7732 di `class_weight`, differenza +0,00085
con tre segni favorevoli su cinque. Non c'è evidenza convincente di superiorità,
ma la frase corretta è «nessuna augmentation mostra un vantaggio dimostrato su
`class_weight`», non «nessuna augmentation lo batte».

### PCA e informazione

Le cinque componenti spiegano circa il 34% della varianza totale dopo la
trasformazione. Questo non significa che conservino il 34% dell'informazione o
che perdano due terzi dell'informazione discriminante. Varianza spiegata e
informazione utile alla classificazione non sono equivalenti.

La limitazione della PCA va dichiarata, ma usando «varianza» invece di
«informazione».

## 8. Provenienza e riproducibilità

Al momento dell'audit:

- `ids_augmentation.py` contiene modifiche non consolidate;
- il documento di revisione non è versionato;
- i risultati IDS usati nelle tabelle non sono versionati;
- le corse riportano `d93064f+modificato` come versione del codice.

`corsa.versione_codice()` registra il commit corrente e la presenza di modifiche
locali, ma non l'impronta del diff. Due versioni diverse del working tree
ricevono quindi la stessa etichetta `d93064f+modificato`.

Prima di usare le tabelle come risultato definitivo conviene conservare
insieme:

- commit esatto del codice;
- configurazione completa;
- risultati grezzi;
- eventuale hash del diff quando il working tree non è pulito;
- script che genera le statistiche e le tabelle del documento.

## 9. Difetto secondario dell'interfaccia

`n_agg` è calcolato come `max(0, n_finale - len(Xmin))`. Se `n_finale` è
minore o uguale alla numerosità esistente, le varianti sintetiche ricevono
`n=0`; la MMD non è definita per un campione vuoto e il percorso non è gestito.

Il caso è raggiungibile, per esempio, usando il default `--n-finale 269` con
`--ratio 1_200`, che contiene già 269 campioni minoritari. Lo script dovrebbe
rifiutare esplicitamente `n_finale <= n_minoritaria`, oppure saltare la
generazione e la MMD in modo definito.

## Elementi che reggono

L'audit non ha rilevato problemi nelle seguenti parti del disegno:

- scaler e PCA sono adattati sui soli dati di training;
- validation e test sono trasformati con la proiezione corrispondente;
- il test rimane intatto;
- la MMD usa una banda fissata sui dati reali, coerente fra varianti;
- duplicazione, rumore e SMOTE aggiungono soltanto dati al training;
- i campioni sintetici non entrano nella validation;
- le medie F1 della tabella principale corrispondono ai risultati grezzi;
- la QGAN perde contro `class_weight` in tutte e cinque le corse;
- raggio, ordine e profondità provati non recuperano la qualità distribuzionale
  o la prestazione a valle.

Questi punti rendono solida la conclusione negativa ristretta al protocollo
eseguito.

## Formulazione consigliata per la revisione

> Sullo split fisso `r2l` 1:1000 con PCA locale, l'augmentation QGAN testata
> non è competitiva con i riferimenti classici. Su cinque inizializzazioni
> algoritmiche ottiene F1 RF 0,577 contro 0,773 con `class_weight`, mentre
> rumore e SMOTE raggiungono rispettivamente 0,735 e 0,774. Rispetto
> all'assenza di augmentation il punto stimato migliora di 0,027, senza
> evidenza conclusiva con cinque repliche. La QGAN presenta inoltre MMD 0,365
> contro valori prossimi a zero per le augmentation classiche. Calibrazione
> del raggio, permutazione delle feature e aumento di `reps` non risolvono il
> problema nelle configurazioni provate. Gli esperimenti non distinguono fra
> limite architetturale, difficoltà di ottimizzazione e insufficienza dei dati.

## Verifiche ancora necessarie

Prima di promuovere il risultato a conclusione generale di Fase 3:

1. ripetere l'esperimento su più sottocampioni indipendenti della minoritaria,
   riadattando ogni volta la PCA locale;
2. dichiarare una metrica primaria e un protocollo per la soglia;
3. riportare almeno F1, PR-AUC e recall per tutti i riferimenti;
4. separare i dati usati per l'arresto della QGAN dalla MMD finale, quando la
   numerosità lo consente;
5. eseguire una sola valutazione conclusiva sul test rimasto intatto;
6. versionare codice, risultati grezzi e script di aggregazione.

## Verifica tecnica eseguita

La compilazione sintattica di `ids_augmentation.py`, `model.py`, `metrics.py` e
`preprocess_nslkdd.py` è riuscita. La suite `pytest` non è stata eseguita perché
`pytest` non è installato nell'ambiente corrente.
