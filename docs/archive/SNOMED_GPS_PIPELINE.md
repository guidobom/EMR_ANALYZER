# Estrazione aperta e codifica SNOMED GPS

## Attivazione

1. Riavvia l'applicazione e apri **Lessico condiviso → Catalogo SNOMED CT…**.
2. Premi **Importa catalogo SNOMED CT…** e scegli il TXT originale della release (non Readme.txt).
   Importazione e indice testuale sono atomici, in un worker con annullamento.
3. Verifica release e conteggi. L'opzione «Usa estrazione aperta + ricerca SNOMED»
   è attiva dopo la prima importazione. Il catalogo è condiviso fra workspace,
   nel file `snomed_gps.db` accanto al registro del lessico.
4. In **Storia clinica → Estrai evidenze** scegli il modello come di consueto.
   Lo stesso modello esegue estrazione e selezione dei codici; non serve un
   lessico già popolato. I risultati precedenti vengono rielaborati quando cambia
   il catalogo/la pipeline, tramite l'invalidazione dei digest.

Il pulsante «Analizza documento con LLM» nel Lessico conserva la funzione di
proporre esempi da annotare: la nuova pipeline codifica il registro delle evidenze.

## Ricerca e indice vettoriale

La ricerca testuale FTS5 è disponibile subito: il LLM produce una breve query
inglese insieme al nome italiano dell'evento. Le associazioni italiane confermate
nel dialogo sono salvate separatamente dai termini inglesi ufficiali.

Per aggiungere ricerca semantica multilingue, usa **Crea indice vettoriale da
modello locale…** e seleziona una cartella SentenceTransformer già disponibile
localmente, adatta al confronto italiano/inglese. Non vengono scaricati modelli.
La costruzione opera in background, mostra avanzamento ed è annullabile.
L'indice è legato all'impronta del GPS; una nuova importazione lo disattiva fino
alla ricostruzione. I ranking testuale, vettoriale e delle associazioni curate
vengono combinati senza interpretare la similarità come probabilità clinica.

## Contratto della pipeline

- Analisi aperta del testo, con citazioni esatte, negazione, certezza, soggetto,
  date documentate, attributi e regole delle schede disponibili come esempi.
- Tutti i concetti GPS attivi possono essere candidati, senza gerarchie inferite.
- Ricerca per menzione (12 candidati), seguita da chiamate aggregate fino a 8
  menzioni; riduzione del batch se il contesto non basta.
- Il modello può selezionare solo un codice candidato o astenersi. Il codice è
  verificato nuovamente nel catalogo. La scelta è una proposta, non una
  certificazione dell'appropriatezza clinica.
- Risultati senza codice, errori del server e contesto insufficiente mantengono
  le menzioni con stato «da rivedere» e motivazione.
- Checkpoint separati per estrazione e codifica; cambi di catalogo, associazioni,
  prompt, modello e indice invalidano il riuso.
- Storia clinica ed export mostrano codice, release, FSN e motivo della proposta.
- La deduplicazione considera anche tipo di evento e contesto. La data e la
  citazione restano quelle del documento; SNOMED non fornisce date o relazioni
  fra episodi. Il consolidamento usa le regole cliniche dell'applicazione.
- I laboratori strutturati conservano la pipeline deterministica e il collegamento
  come supporto clinico. Questa implementazione non ricodifica automaticamente
  tutte le righe di laboratorio in SNOMED.

## Limiti e validazione

Il GPS è una raccolta piatta: nessuna gerarchia, ECL, ragionamento ontologico o
sostituzione automatica dei concetti inattivi. Il collegamento LLM dei testi ai
termini è distinto dai servizi ontologici SNOMED. Restano da verificare le
condizioni d'uso dello specifico impiego LLM/embedding e l'accuratezza su referti
italiani annotati. Consultare la guida della release e SNOMED International.

L'importazione è stata provata sull'intero GPS 20260101 in un database temporaneo:
429.674 record, 378.553 attivi; circa 1,5 secondi. Cinque query testuali esemplificative
hanno impiegato 0,1–1,8 ms. Non sono benchmark di accuratezza clinica né di latenza
LLM. I test del vettoriale usano un encoder sintetico; le prestazioni con un modello
multilingue reale devono essere misurate sul sistema dell'utente.
