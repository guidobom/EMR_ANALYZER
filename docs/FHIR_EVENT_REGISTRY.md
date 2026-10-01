# Registro eventi FHIR R4

Il pulsante **Storia clinica → 1 · Genera registro FHIR** produce
`<workspace>/<paziente>/clinical_events.fhir.json` (Bundle collection).
**Visualizza FHIR** mostra il risultato. Attribuzione dei PDF, anonimizzazione e
parser dei risultati di laboratorio non sono modificati.

## Cataloghi

Da **Lessico condiviso → Catalogo SNOMED CT… / Catalogo LOINC…** si importano le
terminologie condivise. LOINC accetta ZIP, `Loinc.csv` o cartella e preferisce
`LoincTable/Loinc.csv` alle copie negli accessori. Importa anche la variante
italiana quando presente. La distribuzione locale `Loinc_2.83/` è esclusa da Git.

LOINC propone corrispondenze solo con campione esplicito, unità compatibili con
quelle esemplificative, scala numerica, tempo puntuale e metodo non specificato.
Questi filtri sono volutamente conservativi e non coprono tutti gli esami.
Il modello sceglie fra candidati italiani/inglesi senza salvare automaticamente
associazioni approvate. Le associazioni confermate manualmente sono vincolate
ad analita, campione e unità. Senza codice il risultato viene comunque conservato.

## Contenuto

Tutti i risultati strutturati, normali e anomali, diventano Observation; il testo
clinico viene analizzato da un unico estrattore aperto con citazioni e date
verificate, poi codificato con candidati SNOMED. Il file include Patient pseudonimo,
DocumentReference, Device, Provenance e Basic per la copertura di elaborazione.
Diagnosis → Condition; reperti/sintomi → Observation; procedure → Procedure;
terapie → MedicationStatement; familiarità positiva → FamilyMemberHistory.
Stato, certezza, soggetto, attributi originali e citazioni rimangono disponibili.
Le date generiche delle condizioni sono estensioni `event-date`, senza attribuirle
arbitrariamente all'esordio. Prescrizioni esplicite diventano MedicationRequest; somministrazioni esplicite e
datate diventano MedicationAdministration. Nei casi insufficientemente specificati
il contesto originale viene conservato in MedicationStatement senza inventare date.

Le estensioni applicative usano `urn:emr-analyzer:fhir:StructureDefinition:`.
Non è dichiarata conformità a un profilo pubblicato: va ancora formalizzato un
Implementation Guide con le relative StructureDefinition.

## Validazione e continuità

Il writer verifica riferimenti interni, identità, campi minimi e valori JSON finiti,
poi sostituisce atomicamente il file. Se `fhirclient` è installato, verifica anche
le risorse tramite i modelli R4. Non è una certificazione di conformità HL7 né una
validazione terminologica contro un server ufficiale. Il download di questa
libreria opzionale non è stato autorizzato durante l'implementazione.

La risorsa Basic esplicita fallimenti, copertura e numero di eventi non codificati.
Non si elimina un evento per forzare completezza della codifica. Non si applica
la fusione a 15 giorni agli eventi di questa pipeline. Il database delle evidenze
rimane una proiezione per le schermate e il consolidamento esistenti, che dovranno
essere riprogettati separatamente; il Bundle include anche risultati normali.

Gli estrattori separati ICD-11, a catalogo chiuso e SNOMED precedente sono stati
rimossi; il nuovo ingresso è `clinical/event_extraction.py`, con infrastruttura
comune di citazione/checkpoint in `clinical/grounded_sources.py`.

## Estrazione e codifica congiunte (v2)

Prima della chiamata LLM, la ricerca locale recupera candidati per ogni frase
e per finestre delle frasi lunghe. Ogni candidato include riferimento breve,
codice, termine preferito inglese, FSN e frasi di pertinenza. Il modello sceglie
il riferimento nella stessa risposta che identifica evento, citazione e data.
I riferimenti non validi non causano la perdita di un evento altrimenti valido.

Solo gli eventi non codificati passano alla ricerca aggiuntiva con query inglese
e alla selezione aggregata di recupero. Senza indice vettoriale multilingue o
associazioni italiane validate, il recupero inglese può rimanere frequente. Nessun
modello viene scaricato automaticamente. Gli eventi senza codice sono conservati.

Il budget iniziale è sei candidati per query, massimo 64 concetti per blocco,
selezionati alternando le frasi. I testi vengono suddivisi se il contesto non
basta; in una frase indivisibile si riducono i candidati/esempi opzionali prima
di rinunciare all'estrazione. I candidati sono ricalcolati per ciascun sottoblocco.
La nuova versione invalida i checkpoint precedenti. I contatori distinguono
codifiche congiunte e codifiche di recupero. La correttezza clinica e i tempi con
un modello reale richiedono ancora valutazione su documenti annotati.
# Modalità Thinking

Nella configurazione LLM aperta prima dell'analisi, il campo «Modalità di
ragionamento», sotto «Slot simultanei», permette di scegliere «No thinking»
(predefinito) oppure «Thinking» per llama.cpp/GGUF. La preferenza è salvata
indipendentemente per ciascuna pipeline. Serve un modello e un motore che
supportino il ragionamento; la scelta non aggiunge questa capacità ai modelli
che ne sono privi. Per vLLM rimane disponibile solo No thinking.

La scelta è trasmessa per richiesta, anche durante il conteggio del prompt;
il server non forza più globalmente il ragionamento disattivato. Dopo
l'aggiornamento riavviare l'applicazione per ricreare i server locali.
Solo la risposta finale viene utilizzata come JSON clinico. Il thinking
consuma parte del limite dei token di risposta e può aumentare i tempi;
raggiungere il limite continua a produrre un errore di risposta troncata.
Cambiare modalità modifica l'impronta dell'estrattore, invalidando i checkpoint
precedenti. Le verifiche sintetiche coprono persistenza, richieste, selettore
e checkpoint; non sostituiscono una prova di inferenza sul modello scelto.

## Annotazione compatta v3

`fhir_events_compact_v3` sostituisce l'estrazione/codifica congiunta v2.
Il prompt `compact_events_system` richiede sei campi per evento: frase,
ancora testuale, tipo, assertion, certainty e subject. Etichetta, stato,
attributi, relazioni e riferimenti temporali sono facoltativi. Il codice
ricostruisce citazione e provenienza e normalizza le date. Le relazioni
source-grounded sono proposte conservate nell'estensione FHIR
`extraction-context`; non diventano automaticamente causalità confermate.

I gruppi iniziali sono dimensionati a 200–600 token stimati di testo obiettivo
in funzione del budget di risposta, con frasi di contesto. I riferimenti temporali
sono numerati sull'intero documento e possono essere usati anche da un gruppo
lontano. La mappa è euristica: intervalli o espressioni non riconosciuti restano
estraibili tramite citazione letterale. Un riferimento non equivale a una data
attribuita: è il modello a proporre il collegamento, da verificare clinicamente.

La ricerca SNOMED avviene dopo l'annotazione, sulle menzioni. Se manca un indice
multilingue, vengono richieste query inglesi brevi in batch; i codici restano
limitati ai candidati del catalogo importato. Un catalogo assente lascia gli
eventi senza codice, da rivedere. Nessun catalogo viene importato automaticamente.

Le correzioni conservano le annotazioni valide. Il checkpoint completo contiene
l'unione delle annotazioni validate; una correzione vuota non risolve errori
precedenti. Un troncamento conserva gli oggetti JSON completi e validati, ma
richiede comunque di suddividere/rileggere il gruppo per cercare omissioni.
La posizione sorgente rende stabili gli identificativi fra suddivisioni.
Diagnostica e risposte incomplete sono salvate nel database del workspace,
non nel lessico comune. La revisione ha al massimo due tentativi per gruppo;
la suddivisione termina alla singola frase, segnalando problemi irrisolti.

Il riuso delle risposte è limitato allo stesso paziente e a prompt identici
(testo, contesto, date, esempi e configurazione): le citazioni e gli identificativi
vengono ricostruiti per il documento corrente. Non è riuso semantico automatico
fra frasi diverse, né trasferimento di informazioni fra pazienti.
I vecchi checkpoint non sono compatibili. PDF, anonimizzazione e parser
laboratorio non vengono modificati. Il file FHIR finale continua a essere scritto
al termine del run; i checkpoint intermedi permettono ripresa e diagnostica.

Verifiche sintetiche: date a precisione mensile, date distanti, falsa negazione,
riuso con diversa provenienza, correzione parziale persistita, troncamento,
deduplicazione delle riletture, selezione SNOMED e costruzione FHIR. Nessun
benchmark clinico o incremento di sensibilità è stato misurato sul modello reale.

## Contesto temporale compatto v4

`fhir_events_compact_v4` sostituisce la mappa con frasi ripetute: ogni voce
contiene soltanto ID, espressione temporale e fonte (S locale o D distante).
Le frasi visibili restano una sola volta nel testo. Per un collegamento a una
fonte distante proposto dal modello, viene recuperata la frase originale e
verificato il legame in batch di massimo quattro eventi, con risposta massima
di 512 token. Un rifiuto o errore conserva l'evento senza data, da rivedere.
Non è un'assegnazione automatica basata sulla vicinanza, né garantisce di
recuperare tutte le date lontane: la richiesta dipende dalla proposta del modello.
Il digest del testo completo protegge i checkpoint anche da modifiche alle
frasi non incluse nel prompt iniziale.

Le correzioni riportano solo gli eventi indicati negli errori, senza ricopiare
quelli già validi. Mantengono il testo del gruppo per verificare la provenienza.
Sul documento DOC_010343 la mappa passa da 11.421 a 1.838 caratteri (-83,9%).
Questo misura il contenuto della mappa, non la velocità di inferenza. La
suddivisione iniziale resta di nove gruppi; la pianificazione finale dipende
anche da esempi didattici, tokenizzazione e contesto configurato.

## Relazioni separate v5

`fhir_events_compact_v5` elimina `relations` dallo schema della prima estrazione.
La pipeline salva gli eventi estratti nel database prima della codifica e delle
relazioni (oltre ai checkpoint dei gruppi). Dopo la codifica, un passaggio mirato
cerca collegamenti fra eventi già presenti, nei brani con indizi espliciti come
«sospeso per», «dovuto a», «regredito», «trattato con», «prima di» e «dopo».
I brani senza questi indizi non richiedono chiamate per le relazioni: questa
selezione non filtra gli eventi ma limita la copertura delle relazioni.

Il nuovo prompt `event_relations_system` richiede solo estremi, tipo e citazione
di supporto. Ogni finestra comprende la frase con l'indizio e le adiacenti;
è elaborata una volta per esecuzione, con massimo 1.536 token di risposta.
Finestre con più di 20 eventi o contesto insufficiente sono segnate da rivedere.
Relazioni valide vengono conservate anche se altre falliscono; gli errori non
riestraggono gli eventi e non innescano retry del passaggio di annotazione.
Stato e problemi sono conservati separatamente in `relation_review`, esportati
nel contesto FHIR e conteggiati nel riepilogo GUI. I collegamenti sono proposte,
non conferme cliniche. Il file FHIR finale viene ancora scritto a fine run.

Verificato con risposte sintetiche: salvataggio precedente al collegamento,
assenza di relazioni nello schema iniziale, conservazione dei collegamenti
validi, isolamento di citazioni invalide e di un errore del modello. Nessuna
misura di velocità sul modello reale è stata effettuata per questa versione.

## Estrazione intermedia v7: selezione della fonte

`fhir_events_referenced_v7` sostituisce v6 nell'ingresso `EventExtractor`.
Il prompt istituzionale `compact_events_system` è ora v4. Il modello restituisce
sette campi obbligatori: `s`, `span`, `label`, `type`, `assertion`, `certainty`,
`subject`. `span=[prima,ultima]` indica parole numerate (estremi inclusi) nella
frase: il programma ricostruisce la citazione dagli offset originali, preservando
anche ritorni a capo e refusi. Non è richiesta la rigenerazione della citazione.
La selezione può comunque essere clinicamente sbagliata: la verifica degli
indirizzi non dimostra la correttezza dell'interpretazione del modello.

Il nome sintetico è obbligatorio. Valore, unità e andamento sono attributi del
parametro nominato; nomi generici isolati come «in aumento» vengono respinti e
inviati alla correzione mirata. Sintomi diversi possono condividere la stessa
citazione negativa senza collidere nell'identificativo. Valori scalari verificati
sono esportati come `valueQuantity`, compresi i comparatori. Valori/unità non
riscontrati rimangono proposte da rivedere, senza una quantità FHIR validata.

Un riferimento temporale incoerente non elimina l'evento: la data rimane
sconosciuta, con proposta originale e motivo di revisione. Un riferimento di
frase errato può essere ricollocato solo quando la citazione temporale compare
in un'unica frase visibile. «Esegue in data odierna» può sostenere la data del
documento; «referto disponibile in data odierna» non dimostra la data dell'esame.
Le date dell'intestazione tecnica sono escluse dall'indice delle date cliniche.
Il riuso storico resta subordinato a una data storica verificata senza avvisi.

La versione invalida i checkpoint precedenti senza cancellarli. Riavviare
l'applicazione e usare il prompt istituzionale v4 (eventuali prompt personalizzati
con `anchor` devono essere aggiornati). L'importazione RF2 e la successiva
selezione dei candidati SNOMED International restano operative.

Verifiche del 25/09/2026: 11 casi offline (citazioni, negazioni multiple,
quantità/comparatori, unità inventate, date odierne e incoerenti, intervalli
invalidi e frasi lunghe), integrazione con LLM simulato/checkpoint/riuso fra
referti/FHIR/isolamento paziente e 4 passaggi reali P068 con proposte controllate.
Non è stato rilanciato il benchmark completo né misurata inferenza reale v7.
Gli indici aumentano il prompt e riducono il testo della risposta: il beneficio
netto e la sensibilità vanno misurati sul campione revisionato, non presunti.
