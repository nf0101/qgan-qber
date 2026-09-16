# Fase 2 — QGAN condivisa

Report tecnico. Implementazione dell'architettura congelata in `../../../../Scaricati/SPEC-FASE2.md`,
verifiche di correttezza, misura del costo e validazione su dati sintetici.
Il documento registra le decisioni prese, l'evidenza su cui poggiano e le due
questioni che richiedono una revisione della specifica.

Ambiente: Python 3.12, Qiskit 2.5.2, Aer 0.17.2, NumPy 2.x. Il dominio QKD
resta su un ambiente separato con NetSquid, secondo l'appendice della
specifica; i due comunicano solo tramite CSV.

---

## 1. Implementazione

Il pacchetto `src/qgan/` è organizzato in modo che il dominio entri
esclusivamente attraverso `X` e `c`, come richiesto da §2. Nessun ramo
condizionale sul dominio compare all'interno delle classi.

`circuit.py` contiene il generatore. Cinque qubit, encoding latente `RY(z_i)`,
ansatz EfficientSU2 con `reps=4` ed entanglement lineare per 50 parametri,
lettura come probabilità marginale per qubit. Il condizionamento è applicato
con rotazioni `RZ(π·c)` su due strati: il primo subito dopo l'encoding, il
secondo dopo il terzo blocco di entanglement, cioè a metà dell'ansatz.

La posizione del secondo strato non è arbitraria. Uno strato di rotazione di
EfficientSU2 termina con `RZ`, quindi collocarvi immediatamente dopo il
condizionamento produrrebbe due `RZ` consecutivi sullo stesso qubit, che si
fondono in una sola rotazione: il condizionamento si ridurrebbe a uno
scostamento del parametro addestrabile. Inserendolo dopo l'entanglement, prima
di uno strato che comincia con `RY`, la fusione non avviene. Il conteggio dei
gate lo conferma: 35 `RZ` nel caso condizionato contro 25 in quello non
condizionato, cioè i due strati da cinque effettivamente presenti.

Il vettore `c` viene ripetuto ciclicamente sui cinque qubit quando `d_c < 5`.
La ripetizione dello stesso strato in due punti distinti è il data re-uploading
di §3 e non introduce parametri addestrabili, per cui il conteggio resta 50 in
tutte e tre le configurazioni previste (`d_c` pari a 0, 3 e 8).

`discriminator.py` implementa la rete `5 → 32 → 16 → 1` in NumPy, con Adam.
La scelta di non usare un framework di deep learning ha due ragioni: il collo
di bottiglia è il simulatore quantistico e non la rete, e serve
esplicitamente il gradiente della perdita rispetto all'*ingresso* del
discriminatore, da concatenare con la jacobiana del circuito. Averlo scritto
a mano rende la catena verificabile in modo diretto, cosa di cui si è poi
avuta necessità.

`model.py` espone la classe `QGAN` con l'interfaccia congelata. `metrics.py`
raccoglie il protocollo di valutazione di §9. `autoregressive.py` costruisce
il dataset con teacher forcing, campiona le tracce ricorsivamente e confronta
le autocorrelazioni, che è la misura della deriva prevista da §6.

### 1.1 Vincolo di batching

Il vincolo di §7.2 è rispettato in modo strutturale e non per disciplina di
scrittura. Il metodo `forward_and_jacobian` costruisce un'unica matrice di
binding contenente il forward e tutti gli shift, e la passa in una sola
chiamata a `sim.run` tramite `parameter_binds`. Su batch 32 sono **3.232
circuiti in una chiamata**: i 3.200 shift previsti dalla specifica più i 32
forward, che il conteggio di §7.2 ometteva. Il test di correttezza verifica
sia il numero di circuiti sia il numero di chiamate.

Anche il campionamento delle tracce autoregressive è raggruppato: tutte le
tracce avanzano di un passo insieme, per cui servono `T` chiamate e non
`n_tracce × T`.

---

## 2. Verifiche di correttezza

`tests/test_correctness.py` copre struttura del circuito, gradienti e
interfaccia. Tutte le verifiche passano.

La jacobiana ottenuta con la parameter-shift rule è stata confrontata con
differenze finite centrali su parametri estratti a caso, in configurazione
condizionata e non: l'errore massimo resta sotto 1e-5, cioè nell'ordine
dell'errore di troncamento delle differenze finite stesse. La verifica è stata
ripetuta sulla catena completa, dal gradiente della perdita del generatore
fino a θ, e sui gradienti del discriminatore rispetto ai pesi e all'ingresso.

Una prima versione della jacobiana era errata di 0,3 in valore assoluto.
La causa era l'indicizzazione della matrice degli shift: con uno slice sul
primo asse e un array sul secondo, NumPy produce il prodotto cartesiano e non
la diagonale, per cui lo shift veniva applicato a tutti i parametri
contemporaneamente. La forma corretta usa due array indicizzati insieme. Il
caso è istruttivo perché il forward restava esatto e le perdite avevano un
andamento plausibile: senza il confronto con le differenze finite l'errore
sarebbe passato inosservato fino alla Fase 3.

---

## 3. Costo, verifica di §7.1

La specifica riporta 9,22 ms per una valutazione isolata e 1,37 ms per una
raggruppata, con un fattore 6,5. La rimisura su una macchina diversa dà 2,43 ms
e 0,25–0,40 ms, con un fattore fra 6 e 9. Le due misure assolute differiscono
perché differisce l'hardware, ma **il fattore di guadagno del raggruppamento è
confermato**, ed è quella la quantità su cui poggia la scelta della
parameter-shift esatta.

Un gradiente su batch 32 costa 1,22 s. Da qui il costo di un'epoca:

| n | batch | s/epoca | 200 epoche |
|---|---|---|---|
| 53 | 2 | 2,4 | 8 min |
| 200 | 7 | 8,5 | 28 min |
| 630 | 20 | 24,3 | 81 min |
| 2.000 | 63 | 76,7 | 4 h 15 |
| 5.000 | 157 | 191,0 | 10 h 37 |

La stima di §7.1, «un addestramento da 200 epoche resta nell'ordine della
mezz'ora», vale quindi per dataset di poche centinaia di campioni. Il dominio
IDS vi rientra: la minoritaria r2l a 1:1000 conta 53 campioni. Il dominio QKD
no: la griglia ha 94.200 finestre, e un addestramento su tutte richiederebbe
giorni. **Il sottocampionamento della griglia in Fase 3 non è un'opzione ma un
vincolo**, e la dimensione del sottocampione va scelta e dichiarata, non
lasciata implicita.

---

## 4. Validazione su dati sintetici

Il bersaglio è una miscela di quattro modalità in `[0,1]^5` con deviazione
standard 0,05, scelta per riprodurre la struttura della classe probe, che
contiene quattro attacchi distinti.

### 4.1 Espressività

Per separare la questione dell'espressività da quella della stabilità
avversariale, il discriminatore è stato rimosso e θ ottimizzato direttamente
sulla MMD, il cui gradiente rispetto ai campioni generati è analitico.

La MMD scende da 0,099 a 0,010 in 200 passi. Le medie generate coincidono con
quelle reali entro 0,03 su tutte e cinque le feature, e le deviazioni standard
entro 0,06. **L'architettura congelata ha quindi l'espressività necessaria e la
catena circuito-jacobiana è corretta.**

La detectability resta però a 0,98. Il generatore produce, per θ fissato, una
mappa liscia e deterministica dal latente `z` alle cinque marginali: cattura
posizione e dispersione, non la struttura fine di una miscela a modalità
separate. È un limite dell'architettura, non dell'implementazione, e va
riportato come tale nel confronto con ADASYN.

Anche questa prova ha richiesto una correzione. Il termine repulsivo del
gradiente della MMD aveva il segno invertito, e il generatore collassava sulla
media con deviazione standard 0,01 contro 0,22. L'errore era nello script
diagnostico e non nel modello, ma produceva l'apparenza di un fallimento
dell'architettura. Il gradiente è ora verificato numericamente.

### 4.2 Instabilità del ciclo avversariale

Con i valori di §7 il ciclo avversariale non converge. Su cento passi la MMD
peggiora, da 0,073 a 0,157, cioè il modello addestrato è peggiore di quello
inizializzato.

Confronto fra configurazioni, stesso bersaglio e stesso insieme di
validazione indipendente da 256 campioni:

| Configurazione | MMD iniziale | MMD finale | migliore |
|---|---|---|---|
| specifica, `lr_g` 0,01 `lr_d` 0,001 | 0,073 | 0,157 | 0,157 |
| pari, 0,005 / 0,005 | 0,073 | 0,084 | 0,088 |
| pari lenta, 0,001 / 0,001 | 0,073 | 0,073 | 0,071 |
| D più rapido, 0,002 / 0,005 | 0,073 | 0,075 | **0,065** |

L'asimmetria della specifica, che dà al generatore un passo dieci volte più
rapido del discriminatore, è il fattore destabilizzante. È un'asimmetria
insolita: la prassi nelle GAN è che il discriminatore sia almeno altrettanto
rapido, altrimenti il generatore insegue un segnale che non è ancora
informativo.

Correggere i learning rate non basta. Su quattrocento passi con la
configurazione migliore la perdita del discriminatore scende da 0,60 a 0,18
mentre quella del generatore sale da 0,88 a 2,78, e la varianza del gradiente
cresce di due ordini di grandezza, da 1,8e-3 a 2,2e-1. La MMD tocca 0,056 al
passo 100 e poi risale a 0,30. Il discriminatore, con oltre settecento
parametri contro 96 campioni reali, memorizza l'insieme di addestramento; il
generatore riceve da quel momento un segnale saturo.

Riducendo il discriminatore a un solo strato nascosto da otto unità, con
attenuazione dell'etichetta reale a 0,85 e decadimento dei pesi a 1e-3, la
divergenza scompare: la perdita del discriminatore resta intorno a 0,68, che è
il valore di indifferenza, e la varianza del gradiente resta stabile intorno a
1e-4. Ma in questo regime la MMD non migliora: passa da 0,077 a 0,16 con una
deriva lenta.

Il quadro è quindi netto. **Il generatore sa rappresentare il bersaglio, ma il
segnale avversariale non lo porta a farlo.** O il discriminatore vince e il
generatore diverge, oppure il gioco è in equilibrio e il generatore non
apprende. Sono i due regimi tipici delle GAN su dataset piccoli, e qui si
manifestano già su un bersaglio giocattolo.

Le contromisure — attenuazione dell'etichetta, decadimento dei pesi,
discriminatore ridotto — sono state implementate e sono disponibili come
parametri. §12 dichiara libera l'architettura interna del discriminatore
proprio «se una configurazione diversa risultasse necessaria alla stabilità»,
e questo è il caso previsto. Restano però da esplorare sistematicamente, ed è
lavoro che va fatto sulla macchina su cui girerà la Fase 3.

---

## 5. Due questioni che toccano la specifica

### 5.1 L'epoca non è un'unità confrontabile fra i due domini

Con la classe r2l a 1:1000 il dominio IDS ha 53 campioni, cioè due passi del
generatore per epoca. La griglia QBER ne ha 94.200, cioè quasi tremila. A
parità di 200 epoche il generatore riceve tre ordini di grandezza di
aggiornamenti in più in un dominio che nell'altro.

Poiché O1 misura la trasferibilità architetturale confrontando i due domini,
un budget espresso in epoche rende il confronto privo di significato: una
differenza di prestazione fra i due domini sarebbe indistinguibile da una
differenza di quantità di addestramento. Lo stesso vale per la pazienza
dell'arresto anticipato, che a 20 epoche corrisponde a quaranta passi in un
dominio e a sessantamila nell'altro.

L'implementazione espone `max_steps` e `eval_every` in passi del generatore.
Con entrambi a `None` il comportamento coincide esattamente con quello della
specifica, per cui nulla è cambiato di fatto. **Va però deciso e congelato un
budget in passi**, ed è una decisione che riguarda entrambe le parti.

### 5.2 Il rapporto fra i learning rate

`lr_g` dieci volte `lr_d` è la configurazione meno stabile fra quelle provate.
§12 dichiara liberi entrambi i valori, purché identici nei due domini, quindi
la modifica non richiede di riaprire nulla di congelato. Va però scelta sulla
base di una ricerca sistematica e non del primo valore che converge, perché è
un parametro che entra nel confronto di O1.

---

## 6. Artefatti

| File | Contenuto |
|---|---|
| `src/qgan/circuit.py` | generatore, valutazione batch, parameter-shift |
| `src/qgan/discriminator.py` | rete classica, Adam, attenuazione dell'etichetta |
| `src/qgan/model.py` | classe `QGAN`, interfaccia di §2, ciclo di addestramento |
| `src/qgan/metrics.py` | MMD, Wasserstein, KS, detectability, copertura |
| `src/qgan/autoregressive.py` | teacher forcing, campionamento tracce, autocorrelazione |
| `tests/test_correctness.py` | verifiche di struttura, gradienti, interfaccia |
| `scripts/benchmark_epoch.py` | misura del costo, verifica di §7.1 |
| `scripts/diagnose_expressivity.py` | ottimizzazione diretta su MMD |
| `scripts/lr_sweep.py` | confronto fra configurazioni di learning rate |
| `scripts/toy_validation.py` | tre prove su dati sintetici |
| `results/*.json` | esiti delle prove sopra |

---

## 7. Esclusioni deliberate

Nessun esperimento è stato eseguito sui domini reali: è oggetto della Fase 3.
Gli iperparametri non sono stati ottimizzati oltre quanto necessario a
documentare l'instabilità. Il condizionamento è implementato e verificato
strutturalmente, ma la prova che sia *effettivo*, cioè che i campioni generati
seguano `c`, richiede un addestramento stabile e resta da completare: è il
prerequisito di RQ2 e va chiuso prima della Fase 4.

---

## 8. Passi immediati

Il punto 2 di §13 è completato per la parte di implementazione e verifica; la
validazione su dati sintetici ha prodotto un esito che richiede una decisione
anziché una conferma. Il punto 3 è completato. Restano:

1. Concordare il budget in passi e la configurazione dei learning rate,
   e congelarli come gli altri elementi comuni.
2. Ricerca sistematica sulla stabilità, sui soli dati sintetici, prima di
   toccare i domini reali. La grandezza da guardare è la perdita del
   discriminatore: se scende sotto 0,4 il gioco è già perso.
3. Se nessuna configurazione avversariale converge, esiste una via
   documentata: sostituire l'entropia incrociata con una perdita a kernel
   sulla MMD, che nella prova di §4.1 raggiunge 0,010. Comporterebbe però una
   modifica a §7 e cambierebbe la denominazione del metodo, per cui va
   discussa con il relatore e non decisa fra noi.
4. Revisione incrociata dell'implementazione, punto 4 di §13.

---

## 9. Protocollo di calibrazione

Le decisioni su budget e learning rate sono rimandate a una ricerca
sistematica, implementata in `scripts/coordinate_search.py`.

### 9.1 Dove si calibra

Esclusivamente su bersagli sintetici. In Fase 1 le baseline sono state
lasciate deliberatamente non ottimizzate, con la motivazione che confrontare
un modello ottimizzato con una baseline non ottimizzata ne falserebbe
l'esito. Calibrare la QGAN su NSL-KDD o sulla griglia QBER, anche solo su
validation, reintrodurrebbe la stessa asimmetria, ed è la prima obiezione
che un revisore solleverebbe.

I due bersagli riproducono i domini nelle statistiche che contano e non nel
contenuto. Il primo ha 53 campioni e quattro modalità, come r2l a 1:1000; il
secondo ne ha 2.000, condizionamento a tre dimensioni e una relazione monotona
ma non lineare fra `c` e le osservabili, come la griglia sottocampionata.

### 9.2 Il batch va deciso per primo

Con 53 campioni e batch 32 il dataset produce due minibatch, il secondo da 21,
largamente sovrapposti fra epoche consecutive: il discriminatore vede sempre
gli stessi dati. Con batch 16 se ne ottengono tre distinti e il costo per passo
si dimezza, da 1,22 s a 0,45 s. Poiché il costo reale è `passi × batch × 101`
valutazioni di circuito, ridurre il batch sposta budget dalla qualità della
stima del gradiente al numero di aggiornamenti, che è lo scambio giusto nel
regime osservato. **Batch 16 in entrambi i domini.**

### 9.3 Ricerca per coordinate, non griglia piena

Il fattore dominante misurato in §4.2 è la larghezza del discriminatore, non i
learning rate, per cui una griglia piena sprecherebbe corse. Primo stadio con
discriminatore fissato a `(16, 8)` e sei combinazioni di `lr_d` e del rapporto
`lr_g/lr_d`; secondo stadio, con i learning rate vincenti, su tre larghezze.

### 9.4 Criterio di selezione

Non la sola MMD finale. La divergenza è preannunciata dal crollo della perdita
del discriminatore: nella corsa di §4.2 `L_D` scende sotto 0,4 e solo dopo la
MMD risale. Si scartano quindi anzitutto le configurazioni in cui `L_D` scende
sotto 0,45 in un qualunque momento, e fra le superstiti si prende la MMD più
bassa su un insieme di test indipendente. Una configurazione che tocca un buon
valore e poi diverge non è utilizzabile, perché il checkpoint verrebbe scelto
dal rumore.

Il filtro va superato su **entrambi** i bersagli. Se una configurazione è
stabile solo in uno, quello è già un risultato di O1 e va riportato, non
aggirato scegliendo valori diversi per dominio. Lo script si rifiuta di
raccomandare in quel caso.

### 9.5 Selezione del checkpoint

La classe accetta ora `mmd_smooth`: il checkpoint migliore e l'arresto
anticipato usano la media delle ultime valutazioni anziché il singolo valore.
Nel toy A il checkpoint ripristinato era stato scelto su un minimo che si è
poi rivelato rumore di un insieme di validazione da diciannove campioni, e
aveva una MMD reale peggiore di quella all'inizializzazione. Sotto le
cinquanta unità di validazione l'arresto anticipato su MMD non è affidabile,
e per i regimi più scarsi vale la pena considerare un budget fisso senza
arresto anticipato: è più difendibile di un checkpoint scelto dal rumore.

### 9.6 Lettura del budget

Il budget non si postula, si legge dalle curve. Per ciascuna corsa si individua
il passo in cui il minimo corrente della MMD arriva entro il 2 % dal minimo
finale, e `max_steps` è posto a una volta e mezza il massimo fra le mediane dei
due bersagli. Si prende il massimo e non la media perché il rischio è
asimmetrico: con l'arresto anticipato che ripristina il checkpoint migliore un
budget eccessivo costa solo calcolo, mentre uno insufficiente tronca
l'addestramento e falsa il confronto.

### 9.7 Il vincolo di comparabilità, per intero

A parità di passi i due domini ricevono la stessa quantità di ottimizzazione ma
un'esposizione ai dati molto diversa: con 53 campioni ciascuno viene rivisto
centinaia di volte, con 94.200 quasi nessuno. A parità di epoche succede
l'inverso. Non esiste una scelta neutra: si elimina un confondente e se ne
introduce un altro.

Poiché O1 misura se la stessa architettura regge in due domini, il confondente
da eliminare è la quantità di ottimizzazione, quindi i passi. È una scelta da
dichiarare nel paper con la sua motivazione. Ne segue anche che il
sottocampionamento della griglia QBER va dimensionato di conseguenza:
addestrare su duemila finestre anziché su 94.200 riavvicina il rapporto fra
esposizione e ottimizzazione a quello del dominio IDS, il che è un argomento in
più per sottocampionare in modo aggressivo.

### 9.8 Costo

Lo stadio 1 completo, con tre semi e 1.500 passi, sono 36 corse per 87 milioni
di circuiti, circa **6,8 ore**; lo stadio 2 ne aggiunge 18 per altre 3,4. Un
piano ridotto che conserva il criterio: stadio 1 sul solo bersaglio `ids`, che
è il più severo, con due semi e 1.000 passi — 12 corse, circa 1,5 ore — poi
verifica delle due configurazioni migliori sul bersaglio `qkd`, e infine
stadio 2 completo. Circa cinque ore in tutto.

La ricerca accoda i risultati a un file JSONL, per cui è interrompibile e
riprendibile: le combinazioni già presenti vengono saltate.

### 9.9 Cosa congelare al termine

Batch, `lr_g`, `lr_d`, larghezza del discriminatore, `max_steps`,
`eval_every`, `mmd_smooth` e pazienza, da aggiungere come appendice a
`../../../../Scaricati/SPEC-FASE2.md` prima di toccare i dati reali. Sono tutti elementi che §12
dichiara liberi, per cui non serve riaprire nulla di congelato, ma vanno
fissati insieme perché entrano nel confronto di O1.

---

## 10. Esito del passaggio di rifiuto

Sei configurazioni sul solo bersaglio `ids`, un seme, seicento passi, come da
§9.8. Circa mezz'ora di calcolo.

### 10.1 Un errore di misura trovato dai dati stessi

La prima lettura dei risultati conteneva un'incoerenza: per una configurazione
la MMD minima di monitoraggio valeva 0,0725 mentre quella finale sul test
valeva 0,0477, cioè inferiore al minimo osservato. È impossibile se le due
misurano con lo stesso metro, e infatti non lo facevano.

`mmd_rbf` ricavava la larghezza di banda del kernel dall'unione dei due
campioni. Poiché uno dei due è quello generato, che cambia a ogni valutazione,
**il kernel cambiava a ogni passo e le MMD successive non erano confrontabili
fra loro**. Il minimo su cui si sceglie il checkpoint era in parte un
artefatto della banda, non una proprietà del modello.

La banda è ora fissata una volta sui soli dati reali di validazione e riusata
per tutta la corsa. Lo stesso vale per il confronto fra configurazioni nella
ricerca. I risultati ottenuti con la banda variabile sono conservati in
`results/quick_banda_variabile.jsonl` ma non vanno usati.

L'errore è dello stesso tipo di quello sul confronto fra strategie di
riduzione svolto sul test set in Fase 1-A: non produce numeri assurdi, produce
numeri plausibili e sbagliati.

### 10.2 Il criterio di rifiuto era sbagliato

Avevo derivato dalla corsa lunga di §4.2 la regola che un crollo della perdita
del discriminatore sotto 0,45 preannunci la divergenza della MMD, e l'avevo
usata come filtro secco. Sulle sei corse la regola non regge: le due
configurazioni con discriminatore `(32, 16)` hanno `L_D` a 0,286 e 0,167,
quindi ben sotto la soglia, e riducono comunque la MMD di oltre il 45 %.
Applicato alla lettera il filtro avrebbe scartato configurazioni funzionanti.

La perdita del discriminatore resta un indicatore diagnostico, ma il criterio
di selezione è la profondità del minimo della MMD, misurata con il kernel
fisso. Lo script è stato corretto di conseguenza.

### 10.3 Risultati

Con il kernel fisso, le due configurazioni migliori rieseguite:

| Configurazione | `lr_g` | `lr_d` | D | `L_D` min | MMD iniziale | MMD al migliore | riduzione | passo |
|---|---|---|---|---|---|---|---|---|
| `h8_g0.001_d0.005` | 0,001 | 0,005 | (8,) | 0,594 | 0,0904 | **0,0393** | 56,5 % | 300 |
| `h32x16_g0.002_d0.002` | 0,002 | 0,002 | (32,16) | 0,286 | 0,0890 | 0,0486 | 45,4 % | 150 |

**Esiste una configurazione avversariale che funziona.** Non serve proporre al
relatore la sostituzione della funzione di costo: era la domanda a cui il
passaggio di rifiuto doveva rispondere, ed è risolta.

Tre osservazioni.

Il rapporto vincente è **invertito rispetto alla specifica**: il
discriminatore è cinque volte più rapido del generatore, non un decimo. È
coerente con la diagnosi di §4.2, dove il generatore inseguiva un segnale non
ancora informativo.

Il discriminatore vincente è **il più piccolo provato**, un solo strato da otto
unità. Con cinquantatré campioni reali la capacità è un difetto, non un
pregio. §12 prevede esplicitamente questo caso fra gli elementi liberi.

Entrambe le configurazioni **peggiorano dopo il minimo**: da 0,0393 a 0,1347 e
da 0,0486 a 0,1142 entro il passo 600. L'arresto anticipato con ripristino del
checkpoint non è quindi un accessorio ma una condizione di funzionamento, e va
dichiarato come tale nel paper.

### 10.4 Valori proposti

Letti dalle curve, non postulati:

| Grandezza | Valore | Origine |
|---|---|---|
| `batch_size` | 16 | §9.2 |
| `lr_g` | 0,001 | §10.3 |
| `lr_d` | 0,005 | §10.3 |
| discriminatore | `(8,)` | §10.3 |
| `max_steps` | 450 | 1,5 × altopiano a 300 |
| `eval_every` | 25 | §9.6 |
| `mmd_smooth` | 3 | §9.5 |
| pazienza | 20 valutazioni | 500 passi |

Sono proposte, non ancora decisioni: poggiano su un solo seme e su un solo
bersaglio. Prima di congelarle servono la conferma sul bersaglio `qkd` e la
ripetizione su almeno due semi, che è il piano ridotto di §9.8 e costa circa
un'ora e tre quarti.

Il valore di `max_steps` va inoltre riletto dopo la conferma su `qkd`: se
l'altopiano cade più tardi, si prende il massimo fra i due, per l'asimmetria
del rischio descritta in §9.6.

---

## 11. Calibrazione a cinque semi e tabella di trasferibilità

Due configurazioni, due bersagli sintetici, cinque semi. Su `ids` entrambe a
1.350 passi; su `qkd` `h32x16` a 1.200 passi con cinque semi, `h8` a 600,
1.200 e 2.000 passi con due semi.

### 11.1 Dominio a bassa numerosità (`ids`, 53 campioni)

Riduzione della MMD al checkpoint migliore, per seme:

| Configurazione | s0 | s1 | s2 | s3 | s4 | media | altopiani |
|---|---|---|---|---|---|---|---|
| `h8_g0.001_d0.005` | 64,7 | 33,4 | 47,9 | 60,0 | 45,0 | **50,2 ± 11,1** | 300–350 |
| `h32x16_g0.002_d0.002` | 41,8 | 30,5 | 46,6 | 44,4 | 29,0 | 38,5 ± 7,3 | 125–175 |

Il confronto va fatto **appaiato**: `make_target` costruisce il bersaglio da un
seme indipendente dalla configurazione, quindi a parità di seme le due corse
vedono gli stessi dati. Differenze a favore di `h8`: 22,9 · 2,9 · 1,3 · 15,6 ·
16,0; media 11,7 punti, `t = 2,82` su 4 gradi di libertà, `p = 0,048`, cinque
segni su cinque concordi.

A medie e deviazioni separate le due distribuzioni si sovrappongono e la
differenza non risulterebbe stabilita. L'appaiamento è la stessa correzione
prevista in §11 della specifica per il confronto su RQ2, e per la stessa
ragione: elimina la componente di variabilità dominante, che qui è la
realizzazione del bersaglio.

### 11.2 Dominio a numerosità intermedia (`qkd`, 2.000 campioni)

`h32x16_g0.002_d0.002`, cinque semi a 1.200 passi:

| s0 | s1 | s2 | s3 | s4 | media | altopiani |
|---|---|---|---|---|---|---|
| 67,4 | 68,8 | 59,3 | 73,8 | 65,1 | **66,9 ± 5,4** | 800–1.200 |

`h8_g0.001_d0.005` non converge su questo dominio. Su budget di 600, 1.200 e
2.000 passi l'altopiano cade a 550, 1.150 e 1.950, cioè sempre nell'ultimo
tratto: la MMD continua a scendere lentamente senza stabilizzarsi, e le
riduzioni misurate (fra il 34 e il 60 %) non sono valori di convergenza ma
fotografie a istanti arbitrari.

### 11.3 Tabella di trasferibilità — risultato di O1

| | `ids` | `qkd` |
|---|---|---|
| Ottimo per dominio | `h8` — 50,2 % | `h32x16` — 66,9 % |
| Configurazione comune (`h32x16`) | 38,5 % | 66,9 % |
| Costo del compromesso | **−11,7 punti** (appaiato, `p` = 0,048) | nessuno |
| Passi alla convergenza | ~175 | ~875 |

L'ottimo architetturale **non è trasferibile**: la larghezza del
discriminatore che funziona meglio su cinquantatré campioni non è quella che
funziona meglio su duemila. La configurazione comune è `h32x16` perché è
l'unica a convergere in entrambi i domini, e il costo di questa scelta è
misurato, non stimato.

### 11.4 Il numero di passi è esso stesso una misura di trasferibilità

Gli altopiani differiscono di un fattore cinque: 175 passi su `ids`, 875 su
`qkd`. Con l'arresto anticipato che ripristina il checkpoint migliore,
l'ottimizzazione effettiva è quella, indipendentemente dal valore di
`max_steps`, che determina solo quanto calcolo si spreca dopo il minimo.

Questo qualifica il ragionamento di §5.1. Fissare il budget in passi elimina
il confondente della quantità di ottimizzazione *nominale*, ma quella
*effettiva* resta diversa perché diverso è il punto di convergenza. Non è un
difetto da correggere: il numero di passi alla convergenza va riportato come
riga della tabella di O1 accanto alla MMD, non nascosto in un iperparametro.

### 11.5 Un limite che va riportato accanto ai guadagni

La detectability è compresa fra 0,990 e 0,999 in tutte e dieci le corse su
`ids`, e analoga su `qkd`. Un classificatore separa quasi perfettamente i
campioni reali da quelli sintetici anche nella configurazione migliore.

La MMD scende, le marginali si avvicinano, ma i campioni restano
riconoscibili. È coerente con il limite architetturale osservato in §4.1: per
θ fissato la mappa dal latente alle cinque marginali è liscia e
deterministica, e cattura posizione e dispersione ma non la struttura fine.

Ne segue una previsione da verificare in Fase 3. Se l'augmentation dovesse
migliorare l'F1 su r2l, il meccanismo non sarà la verosimiglianza dei campioni
generati — che è bassa — ma qualcos'altro, verosimilmente un effetto di
regolarizzazione o di copertura dello spazio. Conviene dichiararlo prima di
essere costretti a spiegarlo in sede di revisione.

### 11.6 Prossimo passo: efficacia del condizionamento

`scripts/conditional_validation.py` chiude il punto lasciato aperto in §7,
usando la configurazione congelata. Misura tre cose distinte, che e' facile
confondere: se i campioni generati rispondono a `c`; se il generatore
ricostruisce celle della griglia escluse dall'addestramento; e se lo fa meglio
di un generatore incondizionato addestrato sugli stessi dati.

La terza e' quella che decide. Un generatore condizionato puo' superare le
prime due e restare inutile: se la distribuzione complessiva e' gia' vicina a
quella di ogni singola cella, l'incondizionato la eguaglia senza sforzo, e il
condizionamento non aggiunge nulla. E' la stessa struttura del punto di
decisione posto a fine Fase 1 nella roadmap.

Il disegno e' appaiato per cella, come prescritto da §11 della specifica: per
ciascuna cella esclusa si confrontano condizionato e incondizionato sugli
stessi dati, il che elimina la componente di variabilita' dominante.

Va eseguito a budget pieno e su almeno tre semi prima della Fase 4:

    python scripts/conditional_validation.py --seeds 3

### 11.7 Asimmetrie da colmare

Il confronto non è del tutto simmetrico: `h8` su `qkd` poggia su due semi,
contro i cinque delle altre celle. La conclusione che non converge è però
basata sulla posizione dell'altopiano su tre budget diversi, non sulla media,
ed è per questo robusta all'esiguità dei semi. Se in fase di scrittura la cella
dovesse essere riportata numericamente, va portata a cinque semi.

---

## 12. Due errori nella misura del condizionamento

La prima esecuzione di `conditional_validation.py` a budget pieno ha dato MMD
0,389 su entrambi i rami e per tutti e tre i semi, con guadagno 0,0003 ± 0,0006
e correlazione della risposta 0,08 ± 0,53. Letto alla lettera sarebbe stato
l'esito che chiude RQ2. Era invece una misura rotta, per due difetti distinti
nello script.

**Banda del kernel ricavata dalla singola cella.** Una cella ha dispersione
circa 0,02 per feature, quindi banda 0,043, contro 0,330 della griglia intera.
Con un kernel così stretto qualunque campione generato a più di pochi
centesimi dalla cella dà similarità nulla: la MMD satura vicino al suo massimo
per qualsiasi generatore e smette di distinguere. Non stava misurando la
qualità della ricostruzione, ma il fatto che entrambi i rami fossero
«lontani», senza quantificare quanto. La banda è ora unica, ricavata
dall'intera griglia.

È la stessa classe di errore già trovata in §10.1 e non l'ho riconosciuta
riscrivendo lo script. Vale la pena enunciare la regola in forma generale:
**una metrica a kernel è confrontabile solo a banda fissata, e la banda va
scelta sulla scala della variabilità che si vuole risolvere**, non su quella
del sottoinsieme che si sta valutando.

**Validazione su un angolo della griglia.** L'insieme di monitoraggio erano le
prime 256 righe, che per come è costruita la griglia corrispondono a otto celle
tutte con il primo asse a zero. L'arresto anticipato guardava una fetta non
rappresentativa. Ora è un campione casuale su tutte le celle di addestramento.

**Controllo diagnostico aggiunto.** Lo script riporta la MMD sul training per
il condizionato, per l'incondizionato e per un generatore non addestrato. Senza
questo confronto un guadagno nullo resta ambiguo fra «il condizionamento non
serve» e «nessuno dei due generatori ha appreso», che è l'ambiguità in cui la
prima esecuzione era caduta. Se la MMD sul training non scende rispetto
all'inizializzazione, lo script si rifiuta di interpretare il confronto.

Lo script va rieseguito a budget pieno. I risultati precedenti sono da
scartare.

### 12.1 L'asimmetria della MMD contro una cella stretta

La prima esecuzione corretta ha dato guadagno −0,40 ± 0,07 su tre semi, con la
MMD del condizionato fra 0,68 e 0,84 contro 0,26–0,39 dell'incondizionato.
Letto alla lettera, il condizionamento peggiorerebbe la ricostruzione rispetto
a un generatore che ignora `c`, il che non è un'affermazione sensata.

La causa è una proprietà della metrica. Contro una cella di dispersione 0,02
valutata a banda 0,33:

| Generatore | MMD |
|---|---|
| condizionato, errore di posizione 0,00 | 0,0003 |
| condizionato, errore 0,05 | 0,1164 |
| condizionato, errore 0,10 | 0,4082 |
| condizionato, errore 0,20 | 1,1843 |
| condizionato, errore 0,30 | 1,7136 |
| **incondizionato, nessuna posizione tentata** | **0,6183** |

La MMD penalizza una risposta sbagliata molto più di una non-risposta. Il
generatore ampio incassa una rendita da copertura: intercettando massa vicino
a qualunque cella, non supera mai 0,62 qualunque cosa accada. Il condizionato
lo batte solo se colloca la cella entro circa 0,12.

È l'analogo del `DummyRegressor` sulla media di Fase 1-B, dove però lo skill
score rendeva il riferimento interpretabile. Qui mancava.

### 12.2 Skill score sulla posizione

Aggiunto allo script, con la definizione di Fase 1-B §2:
`1 − MAE_modello / MAE_banale`, dove il riferimento banale è la media della
griglia di addestramento, cioè la migliore predizione ignorando `c`. Vale 1 se
la posizione è ricostruita esattamente, 0 se equivale a ignorare `c`, negativo
se peggio. Non concede rendita al generatore ampio.

Lo script riporta anche la dispersione generata accanto a quella vera, perché
la MMD confonde errore di posizione ed errore di ampiezza, e le due hanno
rimedi diversi.

### 12.3 Cosa ci si aspetta di leggere

Tre esiti distinti, che le misure precedenti non separavano:

- **skill vicino a zero o negativo** — il condizionamento non trasporta
  informazione sulla posizione. RQ2 non ha spazio in questa forma, e va rivista
  l'implementazione del condizionamento prima dello scope.
- **skill positivo ma guadagno MMD negativo** — la posizione è ricostruita
  meglio del banale, ma non abbastanza da superare la rendita del generatore
  ampio. Le due misure divergono e vanno riportate entrambe.
- **entrambi positivi** — il condizionamento funziona e RQ2 procede.

L'evidenza indipendente disponibile non è incoraggiante: la correlazione della
risposta vale +0,357, −0,099 e +0,658 sui tre semi, cambia segno ed è
compatibile con il rumore. Un condizionamento efficace darebbe correlazioni
alte e concordi.

---

## 13. Perché il condizionamento non funziona

Lo skill negativo di §12 ha una causa misurata. `scripts/conditioning_sweep.py`
misura la sensibilità dell'uscita a `c`, rapportata alla dispersione che il
latente `z` introduce da solo.

**Ampiezza della rotazione di condizionamento** (θ ±0,1, latente in `[0,π]`):

| `cond_scale` | rapporto c/z |
|---|---|
| π/2 | 0,059 |
| π | 0,089 |
| 2π | **0,000** |
| 4π, 8π | 0,000 |

A `2π` la sensibilità si annulla esattamente, perché `RZ(2π)` è l'identità a
meno di fase globale. `π` è già l'ottimo: l'ampiezza non è la leva.

**Ampiezza dell'inizializzazione di θ** (`cond_scale` = π, latente in `[0,π]`):

| `theta_init` | rapporto c/z |
|---|---|
| ±0,1 (specifica) | 0,089 |
| ±0,3 | 0,266 |
| ±0,5 | 0,432 |
| ±1,0 | **0,817** |
| ±1,5 | 0,532 |

**È l'inizializzazione prescritta da §7 a tenere chiuso il canale di
condizionamento.** Un fattore nove separa `±0,1` da `±1,0`.

Il meccanismo è circolare: il generatore parte con `c` sommerso dal latente, il
discriminatore riceve `c` in chiaro e verifica la coerenza con `x` vincendo
subito, il segnale che torna al generatore satura, e θ non cresce mai
abbastanza da aprire il canale. È coerente con l'osservazione che la MMD sul
training del condizionato resta a 0,15 contro 0,084 dell'incondizionato:
aggiungere `c` peggiora anche l'apprendimento della distribuzione complessiva.

La scala del latente conferma dall'altro lato: restringendola a `[0, π/4]` il
rapporto sale a 0,437, non perché `c` conti di più ma perché `z` conta di meno.

### 13.1 Ipotesi formulate e smentite

Due meccanismi erano stati proposti prima della misura, ed entrambi sono
risultati falsi. Sono riportati perché il fatto che siano plausibili e
sbagliati è esso stesso informativo.

- **Condizionamento con `RY` anziché `RZ`**, sul ragionamento che `RY` agisce
  direttamente sulle popolazioni mentre `RZ` agisce sulle fasi: 0,038 contro
  0,052, nessun miglioramento.
- **Cancellazione fra i due strati di data re-uploading**, che a θ ≈ 0 sono
  separati da un blocco quasi identico all'identità: uno strato dà 0,0303, due
  danno 0,0283, differenza trascurabile.

### 13.2 Il conflitto da portare al relatore

L'inizializzazione di θ è congelata in §7 e la motivazione dichiarata è la
mitigazione del barren plateau. Allargarla apre il condizionamento ma va contro
quella motivazione, e i due effetti vanno misurati insieme: l'ampiezza che apre
il canale è anche quella che avvicina al plateau. Il monitoraggio della varianza
del gradiente è già implementato, quindi il confronto è diretto.

Va notato che il rapporto peggiora oltre `±1,0`, scendendo a 0,532 a `±1,5`:
esiste un ottimo interno, e non è la coda della curva.

### 13.3 La lacuna nei test che ha permesso tutto questo

`tests/test_correctness.py` verificava che i gate di condizionamento fossero
presenti e che la jacobiana rispetto a θ fosse esatta, ma non che variare `c`
cambiasse l'uscita. La presenza dei gate non implica che l'uscita ne dipenda in
misura utilizzabile.

È ora presente un test che asserisce una sensibilità minima per feature e un
rapporto c/z non degenere, e il metodo `QuantumGenerator.conditioning_sensitivity`
lo rende disponibile anche in fase di addestramento.

### 13.4 Il guadagno di c/z non è tutto guadagno

La terza tabella di §13 va letta con attenzione. Fra θ ±0,1 e ±1,0 il rapporto
passa da 0,089 a 0,817, ma per due ragioni distinte:

| | θ ±0,1 | θ ±1,0 | fattore |
|---|---|---|---|
| sensibilità a `c` | 0,0296 | 0,1256 | ×4,2 |
| dispersione da `z` | 0,3306 | 0,1534 | ÷2,2 |
| rapporto | 0,089 | 0,817 | ×9,2 |

Meno della metà del guadagno viene da un condizionamento più efficace; il
resto viene dal fatto che il latente conta di meno. Ma il latente è ciò che
fornisce diversità ai campioni: un generatore che risponde bene a `c` e
produce una nuvola stretta è a rischio di mode collapse, che §8 chiede
esplicitamente di monitorare.

A θ ±1,5 si vede il rovescio della curva: dispersione 0,109 e sensibilità che
ricrolla a 0,059. L'ottimo è interno, e va cercato guardando tre grandezze
insieme e non una.

### 13.5 L'esperimento da fare

`conditional_validation.py` accetta ora `--theta-init` e riporta a ogni corsa
le tre grandezze in conflitto: rapporto c/z, varianza minima del gradiente
(indicatore di barren plateau, motivo per cui §7 prescrive l'inizializzazione
vicina all'identità) e dispersione generata contro reale.

    for t in 0.1 0.3 0.5 1.0; do
      python scripts/conditional_validation.py --seeds 3 --theta-init $t \
          --out results/theta_$t.jsonl
    done

Dodici corse, circa cinquanta minuti. Il criterio di lettura non è il solo
skill: una configurazione che raggiunge skill positivo con dispersione generata
molto inferiore a quella reale ha barattato il condizionamento con la
diversità, e va scartata anche se il numero è buono.

Se nessun valore di `theta_init` porta lo skill sopra zero mantenendo
dispersione e varianza del gradiente accettabili, il limite non è
l'inizializzazione ma la lettura a probabilità marginale, e la conversazione
con il relatore riguarda §3 e non §7.

---

## 14. Il criterio di arresto era cieco al condizionamento

La corsa a budget pieno con diagnostica ha corretto la diagnosi di §13, che era
sbagliata.

`c/z` **non si richiude**: sale da 0,072 a 0,820 nel corso dell'addestramento,
un fattore undici. Il canale di condizionamento si apre da solo man mano che θ
cresce, e l'inizializzazione a `±0,1` non lo impedisce — la ritarda soltanto.
La lettura data in §13, che attribuiva il fallimento all'inizializzazione, era
un'inferenza da uno sweep su θ *casuali*, e non si trasferisce a θ *appresi*.

Il problema è la selezione del checkpoint. Traiettoria della corsa condizionata:

| passo | MMD (media mobile) | `c/z` | `L_D` |
|---|---|---|---|
| 25 | 0,288 | 0,072 | 0,657 |
| **225** | **0,165 ← migliore** | **0,207** | 0,226 |
| 475 | 0,179 | 0,487 | 0,080 |
| 725 | 0,196 | 0,820 | 0,062 |

**Viene ripristinato il passo 225, cioè il modello di prima che il
condizionamento si sviluppi.** Al passo 725, con `c/z` quadruplicato, la MMD è
peggiorata e quel modello viene scartato dall'arresto anticipato.

### 14.1 Perché la MMD aggregata non può vedere il condizionamento

La MMD di monitoraggio confronta l'insieme dei campioni generati con l'insieme
dei reali. Un generatore che ignora completamente `c` può eguagliare la
distribuzione aggregata senza alcuno sforzo, perché l'aggregato è per
costruzione la stessa cosa: la marginale su `c` della distribuzione
condizionale.

Il criterio di selezione è quindi cieco rispetto all'unica proprietà che RQ2
deve misurare, e in pratica seleziona contro di essa: fra un modello che
riproduce bene l'aggregato ignorando `c` e uno che condiziona ma paga
qualcosa sull'aggregato, sceglie il primo.

Questo spiega interamente lo skill negativo di §12, senza bisogno delle ipotesi
di §13, ed è coerente con il fatto che `theta_init 0.5` non migliorasse nulla.

### 14.2 Correzione

`QGAN` accetta ora `monitor`, con default `auto`: `mmd` nel caso non
condizionato, **`mmd_stratificata`** in quello condizionato. La MMD viene
calcolata separatamente su strati di `c` — due livelli per asse, fino a otto
strati — e mediata. Un generatore che ignora `c` fallisce in ogni strato,
perché confrontato con la distribuzione condizionale e non con la marginale.

Sul bersaglio di prova la MMD stratificata parte da 0,621, molto più alta della
aggregata, e scende a 0,406 mentre `c/z` sale da 0,146 a 0,773: le due
grandezze ora si muovono nella stessa direzione, dove prima erano in
opposizione.

### 14.3 Cosa va rifatto

Tutte le corse condizionate precedenti usavano il criterio cieco, quindi
`conditional_validation.py` va rieseguito. L'esperimento su `theta_init`
risponde a una domanda che si è rivelata mal posta e non è prioritario: il
canale si apre comunque, ed è la selezione a scartarlo.

Da rifare invece è la validazione di RQ2 con il monitoraggio stratificato, che
è la misura che dirà se il condizionamento funziona quando non viene
attivamente selezionato contro.

---

## 15. RQ2 funziona: il raggio del latente

Le sezioni 12–14 hanno inseguito tre spiegazioni del fallimento del
condizionamento. Due erano sbagliate, la terza era corretta ma parziale. La
causa dominante è un'altra, e la sequenza delle correzioni è riportata per
intero perché il modo in cui i numeri hanno smentito le ipotesi è esso stesso
il risultato.

### 15.1 Le ipotesi, in ordine

| Ipotesi | Esito |
|---|---|
| §13 — inizializzazione di θ troppo vicina all'identità | **smentita**: `c/z` sale da 0,07 a 0,82 durante l'addestramento; il canale si apre da solo |
| §13.1 — condizionamento `RY` anziché `RZ` | smentita: 0,038 contro 0,052 |
| §13.1 — cancellazione fra i due strati di re-uploading | smentita: 0,0303 contro 0,0283 |
| §14 — criterio di arresto cieco al condizionamento | **confermata, parziale**: risposta da +0,31 ± 0,38 a +0,80 ± 0,25, skill da −2,10 a −1,55 |
| discriminatore troppo forte | **smentita**: indebolirlo porta lo skill a −0,79 ± 1,09, con varianza otto volte maggiore |
| §15 — raggio del latente | **confermata**: skill da −1,02 a +0,48 |

### 15.2 Il risultato

Discriminatore `(32,16)`, nessuno smoothing, monitoraggio stratificato.

| latente | skill | guadagno MMD | risposta `r` | dispersione per cella |
|---|---|---|---|---|
| π (congelato in §3) | −1,021 ± 0,738 | −0,327 | +0,429 ± 0,516 | 5–8× |
| π/2 | +0,084 ± 0,182 | +0,149 (3/3) | +0,752 ± 0,049 | 3,7× |
| π/4, 5 semi | +0,290 ± 0,062 | +0,278 (5/5) | +0,940 ± 0,032 | 1,74× |
| **π/8, 5 semi** | **+0,390 ± 0,240** | **+0,220 ± 0,092 (5/5)** | **+0,960 ± 0,035** | **0,89×** |

La riga π/8 è l'unica misurata con la baseline incondizionata al proprio
raggio ottimale (§15.6); le altre tre la penalizzano e il loro guadagno sulla
MMD è di conseguenza gonfiato. Lo skill non ne è affetto, perché è misurato
contro la media del training e non contro quel ramo.

Il generatore condizionato batte l'incondizionato sulle celle escluse in tutte
e cinque le corse a π/8, e in tutte le corse a π/4.

Skill per seme a π/8: +0,624 · +0,485 · +0,341 · −0,000 · +0,502. Calcolato
mettendo in comune le venti celle dei cinque semi vale **+0,411**, contro
+0,390 come media delle medie: i due valori sono vicini, ma il secondo porta
una deviazione di 0,240 che descrive la selezione delle celle e non la qualità
del modello.

### 15.2.1 Il seme 3

Skill esattamente nullo, con MAE del condizionato e del riferimento banale
entrambe a 0,0467. Non è una corsa fallita: la risposta vale +0,987, la MMD
sulle celle escluse 0,178 contro 0,236 dell'incondizionato, e la dispersione
per cella 0,0149 contro 0,0190 vera. Il condizionamento ha funzionato quanto
negli altri semi.

A cambiare è il denominatore. Il MAE del riferimento banale varia di un fattore
1,66 fra semi, da 0,0467 a 0,0776: sul seme 3 le quattro celle escluse sono
capitate insolitamente vicine alla media della griglia, quindi predire la media
è insolitamente buono.

È il fenomeno documentato in Fase 1-B §4 sul protocollo `interior`: la
deviazione standard non è rumore di stima ma variabilità dovuta a quali celle
vengono escluse. Qui è più acuto che sui dati reali, perché la griglia
sintetica ha otto celle interne contro le ventiquattro di quella vera.

La correzione prevista è già in §11 della specifica: confronto appaiato sulla
medesima cella, mediando le differenze anziché due skill calcolati
separatamente. I dati per applicarla sono nel JSONL, nel campo `per_cella`.

### 15.3 Perché π/8 è l'ottimo, e non serve continuare lo sweep

Il raggio del latente controlla direttamente la dispersione della distribuzione
condizionale. A π/8 quella generata vale 0,017 contro 0,019 dei dati, rapporto
0,89; a π/4 valeva 1,74. È appena arrivata al valore corretto: restringere
ancora la porterebbe sotto, generando celle più strette del vero.

Ne segue una regola che non richiede ricerca: **il raggio del latente va scelto
perché la dispersione condizionale generata eguagli quella dei dati.** Si
calcola dal rumore condizionale del dominio, non si cerca.

### 15.4 Una seconda riga per la tabella di O1

Le finestre QKD reali hanno rumore condizionale intorno a 0,03, dello stesso
ordine dei 0,019 del bersaglio sintetico: π/8 dovrebbe essere circa corretto
anche lì. Sul dominio IDS il caso base è non condizionato, quindi il latente è
l'unica fonte di diversità e deve restare ampio.

**Il raggio ottimale del latente non è trasferibile fra i due domini**, accanto
alla larghezza del discriminatore già documentata in §11.3. Con la differenza
che qui la regola di scelta è nota, quindi la non trasferibilità è prevedibile
e non va misurata per tentativi.

### 15.5 Limiti da dichiarare

Il guadagno sulla MMD era gonfiato nelle prime misure, ed è stato corretto
(§15.6). Resta il numero meno solido dei due: lo skill è misurato contro la
media del training, che non dipende da alcun generatore.

Il regime avversariale resta quello con `L_D` fra 0,14 e 0,33 e varianza del
gradiente fino a 0,58. Funziona, ma non è sano nel senso usuale. Il tentativo
di renderlo tale — discriminatore a otto unità, attenuazione dell'etichetta,
decadimento dei pesi — ha portato lo skill a −0,79 ± 1,09: su questa
architettura il discriminatore che vince è anche l'unico che trasmette
informazione. La soglia `L_D < 0,40` resta un indicatore diagnostico, ma è la
seconda volta che va ritirata come criterio di scarto, dopo §10.2.

La dispersione globale a π/8 è 0,137 contro 0,162 reale, rapporto 0,84: il
generatore copre la griglia un po' meno del vero. È il prezzo del latente
stretto e va riportato.

### 15.6 La baseline penalizzata, e la sua correzione

Le prime misure passavano lo stesso raggio del latente a entrambi i rami. È un
errore: senza `c` il latente è l'unica fonte di diversità del generatore
incondizionato, e a raggio stretto non copre più la griglia. A π/8 il suo skill
scendeva fra −0,38 e −1,06, contro −0,14 e −0,38 al proprio raggio ottimale.

Il guadagno sulla MMD ne risultava gonfiato: **+0,587 con baseline penalizzata,
+0,220 con baseline al proprio ottimo**. Più della metà del guadagno non era
merito del condizionamento ma demerito del confronto.

È la stessa obiezione sollevata in Fase 1-B §5 sulla scelta di
HistGradientBoosting anziché Random Forest come termine di paragone, e in
Fase 1-A §5 su ADASYN: confrontare con il modello più debole gonfia
artificialmente il guadagno. Lo script accetta ora `--latent-hi-uncond`, con
default π, e non eredita più il raggio del ramo condizionato.

### 15.7 Cosa resta

Riaprire l'Appendice A: il raggio del latente è congelato in §3 della
specifica, e ora esiste una misura a cinque semi che chiede di cambiarlo, con
una regola di scelta al posto di un valore.

Applicare il confronto appaiato per cella previsto da §11, che eliminerebbe la
componente di variabilità dominante nello skill.

Rifare, se si vuole la tabella di §15.2 omogenea, le righe π/2 e π/4 con la
baseline al proprio raggio. Non è urgente: la riga che porta il risultato è
quella a π/8, e lo skill delle altre resta valido.


---

## 16. Bersaglio eterogeneo: i raggi per feature non reggono

La validazione di §15 usava un bersaglio con dispersioni condizionali uniformi.
La griglia QKD reale ha un fattore 48 fra la piu' alta e la piu' bassa, quindi
i risultati non si trasferivano automaticamente. `conditional_validation.py`
accetta ora `--target qkd`, che riproduce le dispersioni misurate.

### 16.1 RQ2 regge, ma debolmente

Cinque semi, entrambe le configurazioni del raggio:

| | media | per seme |
|---|---|---|
| raggio unico 0,7342 | **+0,185 ± 0,273** (4/5) | +0,374 · +0,404 · −0,272 · +0,167 · +0,252 |
| raggi per feature | +0,115 ± 0,197 (4/5) | +0,234 · +0,167 · +0,245 · −0,230 · +0,157 |

**Il numero da riportare e' +0,185 ± 0,273**, non il +0,390 di §15: quello era
ottenuto su un bersaglio con dispersioni uniformi che i dati veri non hanno.
Meno della meta', con una deviazione che supera la meta' della media e un seme
che fallisce.

### 16.1.1 I tre semi ingannavano

Con tre semi i raggi per feature davano +0,215 ± 0,042 contro +0,169 ± 0,382
del raggio unico, e la conclusione registrata era che l'unico vantaggio fosse
la varianza, nove volte minore. Con cinque semi:

- il raggio unico vince in media, +0,185 contro +0,115;
- vince su **quattro semi su cinque** (con tre era 2 su 3);
- la deviazione dei raggi per feature sale da 0,042 a 0,197: la stabilita'
  apparente era il caso.

Differenze appaiate −0,140 · −0,237 · +0,517 · −0,397 · −0,095, media −0,070,
`t` = −0,45. Nessuna configurazione e' dimostrabilmente migliore, ma la
direzione si e' invertita. La decisione per il raggio unico resta e non poggia
piu' solo sulla semplicita'.

E' il secondo caso in questa fase in cui tre semi producono una conclusione che
cinque ribaltano, dopo la lettura sulla dispersione fra semi di §11.1. Le
configurazioni non vanno confrontate sotto i cinque semi.

### 16.2 Le dispersioni si omogeneizzano

Fattore 48 nei dati, **1,5 nel generato**, indipendentemente dai raggi
iniziali. I raggi fissano la dispersione all'inizializzazione; l'entanglement
la riporta a scala comune man mano che theta cresce.

Ne segue che la regola dell'Appendice B.2 e' formulata sulla grandezza
sbagliata, e che i raggi per feature non sono dimostrabilmente migliori del
raggio unico: su due semi su tre il raggio unico e' anzi superiore, con
differenza appaiata `t` = 0,2.

### 16.3 Il confronto appaiato: applicato, e non riduce l'incertezza

Applicato con `scripts/analisi_appaiata.py`, che aggrega le 4 celle di ciascuno
dei 5 semi in 20 osservazioni.

| | raggio unico | per feature |
|---|---|---|
| skill per seme | +0,185 ± 0,273 | +0,115 ± 0,196 |
| skill messo in comune | **+0,188** | +0,137 |
| intervallo bootstrap 95 % | **[−0,069, +0,375]** | [−0,087, +0,291] |

**La previsione che l'appaiamento avrebbe ridotto l'incertezza e' sbagliata.**
Avevo attribuito la deviazione alla variabilita' del denominatore fra semi;
l'intervallo su 20 celle e' largo quanto la stima per seme e contiene lo zero.
La variabilita' e' genuinamente fra celle, non fra semi.

Test appaiati, raggio unico:

| confronto | media | t | p | segni |
|---|---|---|---|---|
| banale − condizionato | +0,0121 | 1,45 | 0,164 | 12/20 |
| **incondizionato − condizionato** | **+0,0370** | **3,67** | **0,0016** | **16/20** |
| banale − incondizionato | −0,0249 | −3,21 | 0,0046 | 5/20 |

Confronto fra le due configurazioni del raggio, appaiato sulle stesse celle:
`p` = 0,66, 8 segni su 20. **Indistinguibili.** La decisione per il raggio
unico resta, ora per parsimonia e non per prestazione.

### 16.3.1 Cosa si puo' affermare, e cosa no

**Non si puo' affermare** che il generatore condizionato ricostruisca le celle
escluse meglio della media della griglia: `p` = 0,16, dodici segni su venti,
cioe' quasi una moneta. Lo skill +0,188 e' positivo ma non distinguibile da
zero.

**Si puo' affermare** che il condizionamento migliora significativamente la
ricostruzione rispetto a un generatore altrimenti identico che ignora `c`:
`p` = 0,0016, sedici segni su venti. E' la domanda di RQ2, e la risposta e'
positiva.

Il controllo regge: il generatore incondizionato e' peggiore del riferimento
banale, come dev'essere.

La formulazione corretta per il paper e' quindi «il condizionamento trasporta
informazione utilizzabile sulle celle non viste», non «il generatore
condizionato ricostruisce le celle non viste».

### 16.3.2 Conseguenza per la Fase 4

Il termine di paragone per l'augmentation e' lo skill 0,282 di
HistGradientBoosting sul protocollo `interior`. Se il generatore non batte in
modo dimostrabile nemmeno la media della griglia sulla ricostruzione delle
celle, **l'aspettativa che l'augmentation migliori un attributore gia' a 0,282
va ricalibrata verso il basso.**

Non e' necessariamente fatale: l'augmentation potrebbe aiutare per un
meccanismo diverso dalla fedelta' dei campioni — regolarizzazione, copertura
dello spazio — che e' la stessa ipotesi gia' annotata in §11.5 a proposito
della detectability. Ma va dichiarata come previsione prima della Fase 4, non
come spiegazione dopo.

### 16.4 La deviazione fra celle

Con media +0,185 e deviazione 0,273, l'errore copre il 150 % del risultato. I
due semi che falliscono — il 2 con raggio unico, il 3 con raggi per feature —
hanno entrambi il riferimento banale insolitamente basso, 0,0685 e 0,0459
contro 0,0586–0,0784 degli altri: e' il meccanismo di §15.2.1, dove le celle
escluse cadono vicine alla media della griglia e il banale diventa difficile da
battere.

Applicato l'appaiamento (§16.3), la componente resta. Con quattro celle
interne su otto disponibili nella griglia sintetica, e ventiquattro in quella
reale, il numero di celle e' il fattore limitante — la stessa conclusione a cui
Fase 1-B §8 era arrivata per il protocollo `interior`.

Ampliare la griglia sintetica per avere piu' celle interne e' l'unica via per
stringere l'intervallo, ed e' lavoro di Fase 4 non di Fase 2.

### 16.5 Cosa va nel paper

**Il generatore riproduce la posizione delle celle, non la loro forma.** E' un
limite dell'architettura, non della calibrazione, e va riportato accanto ai
risultati di RQ2.

Va riportato anche il percorso: la raccomandazione dei raggi per feature era
stata formulata su una misura all'inizializzazione e smentita dalla validazione
end-to-end. E' il quinto caso in questa fase in cui una diagnosi plausibile e'
risultata sbagliata, dopo l'inizializzazione di theta, il condizionamento con
RY, la cancellazione fra strati di re-uploading e il discriminatore troppo
forte.

---

## 17. Generazione autoregressiva: funziona

§6 prescrive la generazione autoregressiva per le tracce QBER — una finestra
alla volta, condizionando su `c` e sulla finestra precedente, con
`d_c = 3 + 5 = 8`. Il codice era implementato e passava i test strutturali, ma
non era mai stato addestrato: il toy C previsto da §13 punto 2 non era stato
eseguito. È la modalità operativa dell'intero dominio QKD.

Cinque semi, 1350 passi, `T = 16`, 120 tracce di addestramento, raggio del
latente 0,7342 (B.2), bersaglio con le dispersioni condizionali reali.

### 17.1 I tre rischi di §6, tutti negativi

**Deriva assente.** Il rapporto fra MMD in generazione libera e in teacher
forcing vale **1,17 ± 0,20**, e su un seme è sotto 1. L'accumulo di errore
dovuto al condizionamento sulla finestra generata anziché su quella reale — il
rischio principale della modalità autoregressiva — non si manifesta.

**Nessun accumulo lungo la traccia.** La MMD ai passi da 2 a 15 ha pendenza
`+0,00012` per passo, cioè piatta. Su due semi i valori sono lievemente
negativi, normale per lo stimatore non distorto quando la distribuzione è ben
riprodotta.

**Nessun collasso.** La variabilità nel tempo è **1,45 ± 0,17** volte quella
reale: le tracce sono semmai troppo mosse, non congelate.

### 17.2 Il primo passo è il punto debole

`w_0` è posto a zero da §6, quindi la prima finestra è generata senza
informazione sul passato:

| | MMD |
|---|---|
| passo 0 | **0,1752** |
| passi 1–15 | 0,0107 |
| ultimo passo | 0,0091 |

Sedici volte peggio, con rientro immediato al passo 2.

Tre modalità confrontate, cinque semi ciascuna:

| `w0` | MMD passo 0 | MMD 1–fine | rapporto | deriva | scarto acf |
|---|---|---|---|---|---|
| `zero` (§6) | 0,1752 | 0,0107 | 16,3 | 1,17 | 0,151 |
| **`media`** | **0,0460** | 0,0124 | **3,7** | 1,21 | 0,140 |
| `scarta` | 0,0704 | 0,0108 | 6,5 | 1,24 | 0,136 |

**Si adotta `media`**: `w_0` è la media delle finestre di addestramento, uno
stato plausibile anche se non specifico della traccia. Il primo passo migliora
di quasi quattro volte e il costo sui successivi è del 16 %. Su una traccia da
sedici finestre si guadagna 0,129 sulla prima e si perde 0,026 in totale sulle
altre quindici.

`scarta` — buttare la prima finestra generata — non risolve: **rapporto 6,5**.
Scartarla sposta il problema invece di eliminarlo, perché la seconda finestra
resta condizionata su una prima prodotta da uno stato iniziale arbitrario.
Correggere lo stato iniziale è più efficace che nasconderne l'effetto.

Deriva e autocorrelazione sono invarianti fra le tre modalità, 1,17–1,24 e
0,136–0,151: `w_0` tocca solo l'avvio, come atteso.

**Resta un fattore 3,7.** `media` usa la media globale dell'addestramento, non
specifica della configurazione: condizionarla su `c` dovrebbe ridurre ancora il
residuo. Non implementato. Per la Fase 4 conta poco — con `T = 16` la prima
finestra è un sedicesimo dei dati e le osservabili di §1.2 sono statistiche di
singola finestra — ma diventerebbe rilevante su tracce corte.

`w0_value` è calcolato sulle tracce di **addestramento** e riusato in
generazione: usare quelle di validazione sarebbe una fuga di informazione,
piccola ma reale.

### 17.3 L'autocorrelazione resta attenuata

Sintetica **0,818 ± 0,060** contro 0,969 reale, scarto 0,151. Coerente con la
dispersione temporale in eccesso: tracce più mosse hanno meno memoria.

La variabilità fra semi è notevole, da 0,720 a 0,870, e corrisponde alla
dispersione del rapporto `c/z` osservata durante l'addestramento, che andava da
2,1 a 10,0. Con `d_c = 8` il condizionamento include la finestra precedente, e
quanto il generatore riesca a usarla dipende fortemente dall'inizializzazione.

**La dipendenza temporale è il pezzo meno affidabile del meccanismo**:
riprodotta, ma attenuata e in misura che varia fra realizzazioni.

### 17.4 Conseguenza per la Fase 4

Il dominio QKD ha una via di esecuzione: il generatore produce tracce con la
statistica marginale corretta e la dinamica temporale approssimata.

Se l'attribuzione causale dipendesse dalla struttura temporale, l'attenuazione
sarebbe un limite. Le cinque osservabili di §1.2 sono però tutte statistiche di
singola finestra; `qber_stability`, che era l'unica feature temporale
considerata, è stata valutata e scartata in §1.4. Il limite conta quindi meno
di quanto potrebbe sembrare, ma va dichiarato — e va riconsiderato se in Fase 4
si volesse reintrodurre una feature temporale.

### 17.5 Il raggio del latente: uno scambio, non un ottimo

Il raggio 0,7342 di B.2 e' calcolato sulla dispersione condizionale a `c`
fissato, con `d_c = 3`. Nel caso autoregressivo `d_c = 8` e la distribuzione
condizionale e' piu' stretta, perche' include la finestra precedente: il raggio
corretto dovrebbe essere minore.

La dispersione condizionale a `c_eff` fissato e' la deviazione dei residui
rispetto alla predizione dalla finestra precedente, misurata **direttamente**
sui dati e non stimata da un modello: 0,02615 contro 0,03222 a `c` fissato,
rapporto 0,81. Interpolando sulla curva di calibrazione ne segue un raggio di
**0,585**.

Tre valori, cinque semi ciascuno:

| raggio | scarto acf | dispersione temporale / reale | MMD passi 1–15 |
|---|---|---|---|
| 0,450 | **0,117** | **1,14** | 0,0222 |
| 0,585 (derivato) | 0,145 | 1,21 | 0,0217 |
| 0,7342 (attuale) | 0,151 | 1,45 | **0,0107** |

**Non esiste un ottimo interno.** Le due grandezze si muovono in modo monotono
e opposto: restringere il latente migliora la struttura temporale — scarto di
autocorrelazione giu' del 23 %, dispersione temporale verso 1 — e peggiora
del doppio la qualita' delle finestre. Il valore derivato non e' il migliore su
nessuna colonna.

**Il raggio derivato analiticamente non identifica un ottimo.** E' la terza
regola ricavata dalla struttura dei dati che non regge alla verifica, dopo
l'inizializzazione di theta (§13) e i raggi per feature (§16.2). Il pattern e'
consistente: le grandezze calcolate all'inizializzazione o derivate dai dati
vengono riassorbite dall'addestramento avversariale, che ha una sua dinamica.

### 17.6 Decisione e limite da dichiarare

Si mantiene **0,7342**. La MMD sulle finestre e' la metrica primaria di qualita'
e vale il doppio; l'autocorrelazione entra solo attraverso una struttura
temporale che le cinque osservabili di §1.2 non usano — sono tutte statistiche
di singola finestra, e `qber_stability`, l'unica temporale considerata, e'
stata scartata in §1.4.

Lo scambio va riportato nel paper: **il generatore non puo' ottimizzare
simultaneamente la fedelta' marginale e la dinamica temporale**, e il raggio
del latente e' la manopola che sceglie fra le due. Se in Fase 4 si volesse
reintrodurre una feature temporale, la scelta andrebbe rifatta al contrario.

---

## 18. Sottocampione della griglia: il costo non dipende da n

### 18.1 Una premessa sbagliata, ripetuta in tre punti

§3 di questo documento, l'Appendice A.2 e la lista dei bloccanti affermavano
che il sottocampionamento della griglia QBER e' «un vincolo e non un'opzione»
perche' 94.200 finestre costerebbero ore per corsa.

**Era vero nel regime a epoche.** Da quando il budget e' espresso in passi del
generatore — decisione presa in A.2 — il costo vale
`max_steps x batch x 101` valutazioni di circuito e **non dipende dalla
numerosita' del dataset**. A 1350 passi e batch 16 sono 2,18 milioni di
circuiti, che `n` valga 480 o 94.200.

Quello che `n` cambia e' l'esposizione ai dati:

| n | estrazioni | viste per finestra | mai estratte |
|---|---|---|---|
| 53 (IDS r2l 1:1000) | 21.600 | 407,5 | 0 % |
| 1.920 | 21.600 | 11,2 | 0 % |
| 7.680 | 21.600 | 2,8 | 6 % |
| 94.200 | 21.600 | 0,2 | 79,5 % |

Il vincolo reale e' la memoria, non il calcolo: `mmd_rbf` costruisce una
matrice kernel `n x n`, e il controllo diagnostico di
`conditional_validation.py` la calcolava sull'intero insieme di addestramento —
7,5 GB a 30.720 righe. Ora e' sottocampionato a 1.024.

### 18.2 La curva ha un massimo

Quattro numerosita', cinque semi ciascuna, bersaglio con le dispersioni reali,
valutazione tenuta fissa a 32 finestre per cella esclusa cosi' che cambiare `n`
non migliori anche la stima del MAE.

| n | viste | skill | guadagno |
|---|---|---|---|
| 480 | 45,0 | −0,051 ± 0,500 | +0,048 |
| 1.920 | 11,2 | +0,185 ± 0,257 | +0,140 |
| **7.680** | **2,8** | **+0,259 ± 0,196** | **+0,151** |
| 28.800 | 0,8 | −0,023 ± 0,270 | +0,081 |

Skill e guadagno hanno il massimo nello stesso punto, il che rende il risultato
piu' credibile di quanto sarebbe una sola metrica. L'interpretazione e'
plausibile: sotto le tre viste per finestra il generatore non rivede mai i
campioni e non estrae struttura; sopra le dieci ha troppa poca varieta'.

**Il massimo non e' pero' stabilito.** La differenza fra 7.680 e 28.800 vale
0,282 con errore combinato 0,149, cioe' `t` = 1,89. Le tre numerosita' centrali
non sono distinguibili fra loro con cinque semi.

### 18.3 Decisione

**n ≈ 7.680**, circa 130 finestre per configurazione. E' il massimo osservato su
due metriche indipendenti e i valori estremi sono chiaramente peggiori. Va
dichiarato che le tre numerosita' centrali non sono statisticamente separate,
quindi la scelta e' ragionevole ma non ottimale in senso stretto.

Stringere richiederebbe dieci semi su 1.920, 7.680 e 28.800.

### 18.4 Conseguenza per O1

L'argomento di A.2 — sottocampionare per riavvicinare il rapporto fra
esposizione e ottimizzazione a quello del dominio IDS — non regge. Con 53
campioni IDS e 7.680 finestre QKD l'esposizione resta diversa di due ordini di
grandezza, e nessun sottocampione ragionevole la equipara: portarla a 53
finestre significherebbe 45 viste ciascuna, che e' il punto peggiore della
curva.

L'esposizione ai dati va quindi **dichiarata come differente fra i domini e
riportata nella tabella di trasferibilita'**, accanto al numero di passi alla
convergenza gia' registrato in §11.4, invece di essere forzata a costo della
qualita'.

### 18.5 Tre letture sbagliate su questa domanda

Vale la pena registrarle, perche' la sequenza e' istruttiva.

1. **Il sottocampionamento e' un vincolo di costo.** Falso: il costo non
   dipende da `n` nel regime a passi.
2. **La curva e' monotona crescente, quindi conviene la griglia intera.**
   Falso: a 28.800 lo skill torna negativo.
3. **Il massimo e' a 7.680.** Suggerito da due metriche, non dimostrato: la
   differenza con le numerosita' adiacenti e' al limite della significativita'.


---

## 19. Ordine delle osservabili sui qubit: la regola era invertita

Appendice B.6 prescriveva di collocare sul qubit terminale l'osservabile che
richiede piu' dispersione, sul ragionamento che li' il raggio del latente non
riesce comunque a comprimerla. Era derivata dalla matrice di accoppiamento
misurata all'inizializzazione: 1,00 sui primi quattro qubit e 0,21 sul quinto,
che nella catena di entanglement lineare e' sempre bersaglio e mai controllo.

Il test end-to-end la smentisce in entrambi i domini. I numeri sono in
Appendice B.6; la struttura del risultato e' identica nei due casi: l'ordine
che mette la dispersione maggiore sul qubit terminale perde su cinque semi su
cinque, mentre tutti gli altri ordini sono fra loro indistinguibili.

**Non e' un ordine a essere buono, e' uno a essere cattivo.** Sul qubit
terminale il generatore fatica a PRODURRE dispersione, non solo a comprimerla —
il contrario di quanto la misura statica suggeriva.

Si adotta l'ordine naturale in entrambi i domini: §1.2 per il QKD, varianza
spiegata decrescente per le componenti PCA dell'IDS. L'unico vincolo e' non
collocare la dispersione maggiore sul qubit terminale.

### 19.1 Il conto delle regole smentite

| Regola, derivata da misura statica | Esito end-to-end | Dove |
|---|---|---|
| l'inizializzazione di theta chiude il condizionamento | smentita | §13, §14 |
| i raggi per feature riproducono le dispersioni | smentita | §16.2 |
| il sottocampione e' un vincolo di costo | smentita | §18.1 |
| sul qubit terminale la dispersione maggiore | **smentita** | §19 |

Quattro su quattro. E' un risultato metodologico da riportare nel paper, non un
elenco di errori: **su questa architettura le proprieta' del generatore
all'inizializzazione non predicono quelle del generatore addestrato.** Chiunque
calibri una QGAN variazionale partendo da misure statiche incontrera' lo stesso
problema, ed e' un'abitudine diffusa in letteratura.
