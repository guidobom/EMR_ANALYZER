# Evidenze atomiche dal Lessico condiviso

La fase **Storia clinica → Estrai evidenze** usa `LexiconExtractor`, versione
`shared_lexicon_atomic_v1`. Il catalogo WHO non è richiesto per i testi narrativi.
Prima dell’avvio si scelgono modello e parametri nella finestra della pipeline.
Il registro deve contenere almeno un termine confermato nel Lessico condiviso.

## Riconoscimento

L’estrattore acquisisce una copia del catalogo: identificativi stabili, termini,
definizioni, campi ed esempi/controesempi. Tutti i termini vengono sottoposti al
modello: la somiglianza lessicale sceglie soltanto gli esempi illustrativi, non
esclude entità. Non è richiesta una coincidenza letterale con gli esempi.
Il modello può emettere soltanto gli identificativi del catalogo. Le proposte
aperte di nuovi termini restano nella funzione di annotazione del Lessico e
richiedono conferma umana prima di diventare termini estraibili nel registro.

Si preferisce l’intero documento, per collegare anche riferimenti temporali
lontani. Se prompt, catalogo, schema e budget della risposta non entrano nel
contesto, vengono suddivisi catalogo e testo, mantenendo frasi intere e contesto
adiacente. Nessun termine o segmento viene eliminato per far entrare il prompt.
Una frase/scheda indivisibile troppo grande produce un errore esplicito.
Una risposta troncata viene suddivisa quando possibile; gli errori di validazione
prevedono un solo recupero. I risultati parziali validi sono conservati e il
documento rimane da completare. Il controllo formale non garantisce che il
modello non ometta eventi: sensibilità e precisione vanno misurate su referti
annotati indipendenti dagli esempi del catalogo.

Ogni occorrenza conserva citazione originale e offset, termine del catalogo,
asserzione, certezza, soggetto, stato, tipo di evento e attributi espliciti.
Le citazioni sono controllate sul testo corrente; sono ammesse differenze di
soli spazi/a capo, salvando comunque la versione originale. Gli esempi non
possono essere citati come evidenze del paziente in analisi.

## Date

La data dell’evento è separata dalla data del documento. L’LLM cita l’espressione
temporale e il codice la normalizza. Giorno, mese, anno, intervallo e data
approssimativa conservano la propria precisione. Le date sconosciute restano
sconosciute, senza sostituzione automatica con la data del referto.
La data del referto è usata solo con attribuzione contestuale esplicita a una
osservazione corrente, o come ancora di un’espressione relativa risolvibile.
La citazione temporale, gli offset e gli eventuali motivi di revisione sono
conservati in `date_provenance`. Date non risolvibili non eliminano l’evento.

## Deduplicazione

Il database conserva tutte le occorrenze documentali. La vista e la successiva
costruzione degli eventi usano una proiezione deduplicata per paziente e termine.
Con date precise al giorno, si raggruppano le occorrenze compatibili entro
**15 giorni inclusi dalla prima osservazione**; la data rappresentativa è la più
antica, indipendentemente dalla data del documento che la cita. Non si concatena
la finestra: 1, 14 e 27 marzo diventano due eventi, al 1 e al 27 marzo.

Negazioni, dubbi, soggetti, stati clinici e attributi incompatibili non vengono
fusi. Una ripetizione storica dello stesso evento può essere fusa con la prima
osservazione attuale. Date ignote, mensili, annuali, intervalli e approssimazioni
restano separate. Somministrazioni, procedure e misurazioni distinte non vengono
accorpate per semplice vicinanza: per deduplicarle occorrono la stessa data e
un identificatore esplicito dello stesso episodio. Nei casi ambigui si conserva
la separazione. Tutte le fonti e i dettagli originali restano ispezionabili in
`source_occurrences`; eliminare un documento ricalcola la proiezione dalle fonti
rimanenti. La proiezione non viene risalvata al posto delle occorrenze.

## Ripresa e compatibilità

Prompt, parametri del modello, testo e catalogo partecipano alle chiavi di cache.
Modificare termini, definizioni o esempi richiede la nuova estrazione dei documenti
interessati; i risultati completati identici sono riutilizzati. La nuova versione
invalida i manifesti del precedente estrattore ICD-11. All’elaborazione di ciascun
documento, i nuovi risultati sostituiscono i vecchi output automatici narrativi;
i documenti originali e le annotazioni del Lessico non vengono modificati.

Le evidenze di laboratorio strutturate continuano a essere generate in modo
deterministico soltanto fuori range; non vengono fuse con la regola temporale del
Lessico. La loro eventuale codifica ICD-11/LOINC resta distinta dall’estrazione
narrativa guidata dal catalogo locale.


## Laboratori come evidenze di supporto

L’estrazione conserva tutte le anomalie fuori range. Nella fase **Crea eventi
clinici**, la valutazione delle relazioni stabilisce quali sostengano un evento
clinico già documentato. Sono necessari un legame specifico affidabile valutato
dal modello o accettato dal revisore, stesso paziente/soggetto e compatibilità
temporale. Le sole date vicine, una relazione puramente temporale o le regole
provvisorie di categoria non bastano. Eventi negati o ipotetici non vengono
confermati dai risultati di laboratorio; la certezza della diagnosi non cambia.

I risultati collegati diventano `diagnostic_support` dell’evento e non generano
una voce autonoma duplicata. Un laboratorio può sostenere più eventi distinti,
ma non può essere il ponte che li fonde. La vista delle evidenze li mostra come
figli dell’evento, mantenendo valore, data e accesso alla fonte. Senza un legame
affidabile restano visibili come reperti autonomi non associati: mancanza di
associazione non significa prova dell’assenza di un legame clinico.

Prima di eseguire **Crea eventi clinici**, i laboratori non ancora valutati
restano autonomi. Tutte le occorrenze originali restano nel database; la modifica
riguarda relazioni e rappresentazione, non la cancellazione dei valori.
