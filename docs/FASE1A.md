# Fase 1-A — Dominio IDS

Report tecnico. Dataset NSL-KDD, ambiente Python 3.11 con scikit-learn e imbalanced-learn.
Il documento registra le decisioni prese, l'evidenza su cui poggiano e i risultati ottenuti.

---

## 1. Dati

I file `KDDTrain+.txt` e `KDDTest+.txt` sono stati scaricati tramite `download_nslkdd.py` da mirror pubblici su GitHub, poiché la distribuzione ufficiale del Canadian Institute for Cybersecurity richiede la compilazione di un modulo web.

L'integrità è stata verificata confrontando tre mirror indipendenti, che restituiscono file identici al byte. Lo SHA-256 del training set è `1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95`; i conteggi sono 125.973 record di training e 22.544 di test, coerenti con i valori canonici di NSL-KDD.

I file `.txt` non contengono l'intestazione. I nomi delle colonne sono stati confrontati posizione per posizione con quelli dichiarati in `KDDTrain+.arff`, senza differenze sulle 41 feature. L'ARFF dichiara 42 attributi e denomina `class` l'ultimo, mentre i `.txt` ne hanno 43 perché includono anche il livello di difficoltà, assente dall'ARFF: la colonna 42 corrisponde quindi a `class`.

La convergenza di tre fonti indipendenti è un'evidenza forte ma non equivale a un confronto con l'originale ufficiale, che resta non verificato.

---

## 2. Preprocessing

### 2.1 Riduzione a sei componenti

Il numero di componenti è un vincolo di progetto e non il risultato di un'ottimizzazione. Le feature osservabili nel dominio QKD sono sei, e l'allineamento fra i due domini è la condizione perché l'obiettivo O1, la trasferibilità architetturale, sia misurabile.

Il costo di questo vincolo è una varianza cumulata del 40,2 % per il sottoinsieme probe e del 38,1 % per r2l. La verifica che ciò non comprometta il compito è riportata in §3.1.

La strategia di riduzione è stata scelta confrontando quattro alternative su validation, con la selezione supervisionata calcolata sul regime di sbilanciamento più severo.

| Strategia | F1 | PR-AUC |
|---|---|---|
| PCA-6 | 0,9929 | 0,9995 |
| PCA-20 seguita da mutua informazione, top 6 | 0,9906 | 0,9995 |
| Mutua informazione su colonne grezze, top 6 | 0,9811 | 0,9949 |
| Importanza Random Forest su colonne grezze, top 6 | 0,9931 | 0,9997 |
| Riferimento: tutte le 67 colonne codificate | 0,9983 | 1,0000 |

Le differenze rientrano nel rumore. È stata quindi adottata la PCA semplice, che non utilizza le etichette e non introduce alcuna questione di fuga di informazione.

Un primo confronto fra le stesse strategie era stato eseguito sul test set, con esiti diversi: la seconda strategia risultava superiore di 4,6 punti di F1. Quel confronto costituiva selezione del modello sui dati di valutazione ed è stato scartato e rifatto su validation.

### 2.2 Variabili categoriche

La colonna `service` assume 70 valori distinti. Un one-hot completo porterebbe da 41 a oltre 110 colonne, in larga parte binarie e sparse, con il rischio che le componenti principali finiscano per descrivere il servizio anziché il comportamento del traffico. Si conservano perciò i 15 servizi più frequenti e si raggruppa la coda nella categoria `other`. Le colonne `protocol_type` e `flag`, con 3 e 11 valori, sono codificate integralmente. La lista dei servizi frequenti è determinata sul solo training.

### 2.3 Classi minoritarie

La categoria DoS è stata scartata perché dominata da un singolo attacco, `neptune`, con 41.214 campioni su circa 45.000: la classe risulterebbe artificialmente omogenea e la generazione di campioni sintetici troppo facile.

Sono state selezionate due categorie. La prima, probe, comprende satan, ipsweep, portsweep, nmap, mscan e saint: quattro attacchi di numerosità comparabile nel training, quindi una distribuzione multimodale. La seconda, r2l, comprende quindici etichette ed è notoriamente la categoria più difficile di NSL-KDD.

L'uso di entrambe permette di confrontare, a parità di numerosità della minoritaria, l'effetto della separabilità intrinseca da quello della scarsità di dati.

### 2.4 Rapporti di sbilanciamento

I rapporti 1:200, 1:500 e 1:1000 derivano dalla curva di difficoltà prodotta da `difficulty_curve.py`. A rapporti inferiori la baseline supera F1 0,97 e l'augmentation non avrebbe margine per dimostrare alcun guadagno.

| Rapporto | Campioni minoritari | Recall probe | Recall r2l |
|---|---|---|---|
| 1:10 | 5387 | 0,992 | — |
| 1:50 | 1077 | 0,964 | — |
| 1:100 | 538 | 0,940 | 0,911 |
| 1:200 | 269 | 0,907 | 0,851 |
| 1:500 | 107 | 0,817 | 0,722 |
| 1:1000 | 53 | 0,687 | 0,610 |

Le celle vuote corrispondono a rapporti non realizzabili per r2l, che dispone di soli 796 campioni minoritari nel training.

### 2.5 Suddivisione dei dati

Il training è stato suddiviso in 80 % di addestramento e 20 % di validation, con stratificazione sulla classe. Scaler, PCA e lista dei servizi sono adattati esclusivamente su dati di training. Il test non è stato usato per alcuna decisione e sarà impiegato una sola volta, per i risultati finali.

---

## 3. Baseline senza augmentation

### 3.1 Effetto della rappresentazione

Per verificare se il degrado osservato nei regimi di scarsità dipenda dalla proiezione o dal classificatore, ogni configurazione è stata generata in due varianti: `local`, con PCA adattata sul sottocampione sbilanciato, e `global`, con PCA adattata sul training completo.

La varianza spiegata risulta quasi identica, fra il 37,5 % e il 38,1 % nella variante locale contro il 40,2 % in quella globale. Il motivo è che il sottocampionamento riguarda solo la classe minoritaria, mentre i circa 53.900 campioni normali restano invariati: la PCA vede quindi sostanzialmente gli stessi dati in entrambe le varianti.

La differenza di F1 fra le due varianti, con Random Forest, è la seguente.

| Classe | 1:200 | 1:500 | 1:1000 |
|---|---|---|---|
| probe | −0,003 | −0,013 | −0,042 |
| r2l | +0,010 | +0,003 | +0,009 |

Con l'unica eccezione di probe a 1:1000, dove la deviazione standard passa da 0,003 a 0,014, la rappresentazione non costituisce il collo di bottiglia. Il degrado è quindi attribuibile al classificatore, che è il livello su cui l'augmentation può intervenire.

### 3.2 Dipendenza dal modello

Sono stati valutati tre classificatori con cinque semi ciascuno e riponderazione delle classi. Random Forest e HistGradientBoosting degradano in modo analogo al crescere dello sbilanciamento, il che indica che il degrado è una proprietà del problema e non un artefatto del modello scelto.

F1 a 1:1000, variante globale:

| Modello | probe | r2l |
|---|---|---|
| Random Forest | 0,836 ± 0,003 | 0,795 ± 0,002 |
| HistGradientBoosting | 0,856 ± 0,026 | 0,754 ± 0,045 |
| Logistic Regression | 0,849 ± 0,000 | 0,179 ± 0,000 |

### 3.3 Separabilità di r2l

La regressione logistica su r2l produce F1 di circa 0,18 con recall di 0,98, comportamento tipico di un classificatore che assegna quasi ogni campione alla classe positiva. La metrica diagnostica è la PR-AUC, che vale 0,127 con riponderazione e 0,021 senza, a fronte di una prevalenza della classe di circa 0,015.

Senza riponderazione il modello è dunque appena sopra il caso. Non si tratta di un problema di soglia, poiché la PR-AUC misura la qualità dell'ordinamento indipendentemente da essa: un modello lineare non separa r2l dal traffico normale nello spazio a sei componenti. Il risultato è coerente con la natura degli attacchi remote-to-local, che consistono in sessioni apparentemente legittime con contenuto malevolo e non presentano la firma grossolana di una scansione di porte.

Le prestazioni dei modelli ad albero su r2l dipendono quindi da frontiere di decisione non lineari. Ne segue che la generazione di campioni sintetici realistici è per questa classe più impegnativa che per probe: il generatore dovrà catturare una struttura non lineare a partire da 53 esempi.

### 3.4 Variabilità nei regimi di scarsità

HistGradientBoosting su r2l a 1:1000 presenta una deviazione standard di 0,045 su F1 e 0,066 su PR-AUC, un ordine di grandezza superiore a quella di Random Forest nella stessa configurazione.

Nei regimi di forte scarsità il problema non riguarda quindi soltanto il valore atteso della metrica ma anche la sua riproducibilità fra esecuzioni. La riduzione di questa variabilità costituisce un possibile beneficio dell'augmentation, distinto dal miglioramento della media e raramente riportato in letteratura.

---

## 4. Metodi di ricampionamento

Sette metodi sono stati confrontati con Random Forest, cinque semi e variante globale della PCA. Il parametro `k_neighbors` è stato adattato alla numerosità della minoritaria, poiché il valore predefinito di 5 non è applicabile con 53 campioni.

Guadagno di F1 rispetto all'assenza di intervento:

| Classe | Rapporto | ADASYN | SMOTE | SMOTE-Tomek | Borderline | class_weight | Random over |
|---|---|---|---|---|---|---|---|
| probe | 1:200 | +0,041 | +0,043 | +0,043 | +0,035 | +0,021 | +0,016 |
| probe | 1:500 | +0,046 | +0,031 | +0,033 | +0,038 | −0,008 | −0,013 |
| probe | 1:1000 | +0,058 | +0,045 | +0,045 | −0,005 | −0,008 | −0,017 |
| r2l | 1:200 | +0,028 | +0,050 | +0,049 | +0,034 | +0,035 | +0,022 |
| r2l | 1:500 | +0,068 | +0,064 | +0,064 | +0,064 | +0,066 | +0,046 |
| r2l | 1:1000 | +0,116 | +0,103 | +0,100 | +0,092 | +0,066 | +0,050 |

ADASYN risulta il metodo migliore in quattro configurazioni su sei, e in particolare nei regimi di maggiore scarsità. Il comportamento è coerente con il suo criterio di funzionamento, che concentra la generazione nelle regioni in cui la classe minoritaria è più difficile da apprendere anziché distribuirla uniformemente.

Contrariamente all'aspettativa iniziale, SMOTE ottiene buoni risultati anche su r2l, con un guadagno di 0,103 a 1:1000. L'interpolazione opera fra vicini prossimi, quindi localmente, e l'unione di molti segmenti locali approssima adeguatamente anche una varietà curva: la mancata separabilità lineare globale documentata in §3.3 non impedisce l'interpolazione locale.

L'oversampling per duplicazione casuale peggiora le prestazioni in tre configurazioni su sei. Il dato funge da controllo negativo, indicando che il guadagno di SMOTE deriva dall'interpolazione e non dal semplice riequilibrio numerico delle classi.

Borderline-SMOTE su probe a 1:1000 risulta lievemente peggiore dell'assenza di intervento. Con 53 campioni l'identificazione della frontiera fra le classi è instabile, il che è confermato dalla deviazione standard, la più elevata fra le varianti di SMOTE in quella configurazione.

---

## 5. Riferimento per il confronto con la QGAN

Il termine di paragone per l'augmentation generativa è ADASYN, non l'assenza di intervento. I valori da superare, in F1 su validation, sono i seguenti.

| Classe | Rapporto | ADASYN | Senza augmentation |
|---|---|---|---|
| probe | 1:1000 | 0,9015 | 0,8439 |
| r2l | 1:1000 | 0,8460 | 0,7299 |

Ne discende una revisione dell'analisi dei rischi riportata nel documento di progetto. Il rischio più immediato non è che un generatore classico eguagli quello quantistico, ma che un metodo di ricampionamento elementare eguagli entrambi: ADASYN ha costo computazionale trascurabile, non richiede addestramento e non presenta iperparametri da ottimizzare.

Nell'eventualità, il contributo del lavoro si sposta sul dataset, sul benchmark e sull'attribuzione causale, che restano validi indipendentemente dall'esito del confronto sull'augmentation. È un'ulteriore ragione per mantenere O3 come contributo principale.

---

## 6. Artefatti

| File | Contenuto |
|---|---|
| `data/ids/{cls}_train_{ratio}_{mode}.csv` | dodici dataset di training: due classi, tre rapporti, due varianti di PCA |
| `data/ids/{cls}_val.csv`, `{cls}_test.csv` | validation e test, variante globale |
| `data/ids/{cls}_val_{ratio}_local.csv` | validation e test per la variante locale |
| `data/ids/meta.json` | parametri, varianza spiegata, numerosità |
| `data/ids/feature_selection.json` | confronto fra le quattro strategie di riduzione |
| `results/baseline.csv` | baseline su tutte le configurazioni |
| `results/difficulty_curve.csv` | degrado al variare del rapporto |
| `results/resampling.csv` | confronto fra i sette metodi di ricampionamento |

Tutti gli artefatti sono rigenerabili con gli script contenuti in `src/ids/`.

---

## 7. Esclusioni deliberate

Il test set non è stato utilizzato in alcuna fase. Gli iperparametri dei classificatori non sono stati ottimizzati: la baseline deve essere ragionevole e non ottimizzata, poiché il confronto con un modello ottimizzato ne falserebbe l'esito. L'augmentation generativa è oggetto della Fase 3 e non è stata affrontata.

---

## 8. Questioni aperte

Le due classi minoritarie risultano meno distanti del previsto: a 1:1000 l'F1 vale 0,84 per probe e 0,80 per r2l. L'uso di entrambe resta utile, ma non consente di sostenere una differenza marcata di difficoltà.

Non è stato quantificato il costo del vincolo delle sei componenti. Un confronto con dieci o venti componenti sarebbe opportuno qualora la questione emergesse in fase di revisione.

La soglia di decisione non è stata calibrata. Le metriche a soglia fissa, F1 e recall, potrebbero migliorare con una calibrazione, che tuttavia introdurrebbe un ulteriore grado di libertà da giustificare.