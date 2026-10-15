# Risultati IDS di protocolli precedenti

Spostati qui perche' i loro numeri non sono confrontabili con quelli
del capitolo. L'unica directory viva e' `results/ids_finale`.

## confronti su singolo split

- `ids_1_1000`
- `ids_1_200`

PROTOCOLLO SUPERATO. Cinque semi su UN solo campione di 53 attacchi
(ids_1_1000) o 269 (ids_1_200), quello fissato dal seme 42. Da qui
veniva il divario di -0.196 della QGAN, che su dieci split
indipendenti vale -0.112: il seme 42 era la replica piu' ostile
delle dieci, con -0.206 contro -0.049 della migliore. Si tengono
perche' sono l'origine del numero corretto.

## prima ablazione causale

- `ids_ablazione.jsonl`

PROTOCOLLO SUPERATO. La prima corsa con qgan_sd e smote_distorta,
un seme solo: e' da qui che veniva l'ipotesi sulle dispersioni per
feature (F1 0.783 -> 0.566 distorcendo SMOTE). Su dieci repliche e
sul test quell'ipotesi non regge in nessuno dei due versi. Si tiene
perche' documenta come l'ipotesi e' nata, non come conferma.

## Protocollo finale, per confronto

Dieci repliche indipendenti dello split (semi 42, 101, 202, 303,
404, 505, 606, 707, 808, 909) x due semi di addestramento; tredici
varianti; soglia scelta fuori campione sul training e, per il test,
sul validation; conferma su KDDTest+ mai usato per nessuna scelta;
inferenza con bootstrap incrociato su repliche x semi e unita' di
riferimento la replica. Metrica dichiarata prima della misura:
F1@0.5 sul validation; sul test si riporta la PR-AUC, perche' con
prior 1:1000 in addestramento e 1:5 nel test una soglia fissa
misura lo scarto fra i prior e non la qualita' del metodo.
