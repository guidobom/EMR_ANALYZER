> Documento storico: questi metodi di estrazione sono stati rimossi dal branch di sviluppo. Vedere [pipeline ICD-11 attuale](ICD11_PIPELINE.md).

# Proposta di semplificazione dell’estrazione atomica

Analisi del 14 settembre 2026. Proposta progettuale: nessuna modifica al motore, ai prompt attivi o all’esecuzione in corso. Le misure provengono da letture del codice, dai contatori del server locale e da una ricostruzione deterministica dei blocchi; non sono risultati di un nuovo confronto clinico.

## Esito

Raccomandazione: una richiesta principale per unità di testo, validazione locale e al massimo una richiesta mirata che riunisca errori e possibili omissioni. Separare la ripresa dell’elaborazione dal riuso clinico tra referti; mantenere le singole occorrenze e il loro contesto. Introdurre questa semplificazione prima di cambiare contemporaneamente modello, schema e dimensione dei blocchi.

“Un solo prompt” non deve significare un unico messaggio per l’intero dossier: servono ancora limiti di risposta, contesto sufficiente per negazioni/date e gestione esplicita delle estrazioni incomplete.

## Misure disponibili

- 266 documenti con testo attivo; 183 narrativi destinati all’LLM e 83 referti di laboratorio elaborati deterministicamente.
- Documento narrativo mediano: 852 token; 95° percentile: 2.680; massimo: 4.169. Conteggio con tokenizer del Qwen3-14B locale.
- Il motore divide il testo in blocchi di massimo 2.800 caratteri. Prima del riuso: 369 blocchi.
- Il piano di riuso interessa 173 documenti e 2.344 occorrenze di 481 blocchi distinti. Il testo iniziale passa da 630.222 a 384.458 caratteri: riduzione del 39,0% circa. Sono caratteri esclusi dalla prima lettura, non un risparmio netto di tempo verificato.
- Dopo riuso e filtro locale: 238 blocchi iniziali. Senza riuso sarebbero 369, ossia il 55% circa in più. Non includono correzioni, recuperi, verifiche del riuso o divisioni adattive.
- Prompt di sistema e formato chat: circa 1.664 token. Input ordinario mediano: 2.398 token; massimo: 2.992. La parte fissa pesa quindi circa il 69% dell’input mediano, ma i contatori del server mostrano che gran parte del prefisso viene già riutilizzata dalla cache.
- Limite adattivo di risposta: fino a 4.407 token sui blocchi ricostruiti; richiesta ordinaria più risposta: massimo 7.321 token. Non sono lunghezze effettivamente generate.
- Generazione osservata: circa 15 token/s per slot con tre slot. Mille token di risposta richiedono circa 67 secondi su uno slot a quel ritmo; non è una previsione del tempo totale di un documento.
- Il riepilogo delle chiamate e dei token viene scritto nell’audit a fine esecuzione. Durante la run non abbiamo una ripartizione persistente completa del costo per correzione, recupero e verifica del riuso.

## Dove nasce la complessità

Il percorso ordinario combina istruzione di sistema e task nello stesso messaggio: non sono due chiamate. Per ciascun blocco avvengono estrazione, normalizzazione/validazione locale, eventuale correzione e possibile recupero delle categorie non rappresentate. Con i valori predefiniti si può arrivare a tre chiamate per blocco prima di considerare il limite di output. Il recupero è euristico: una categoria già citata non prova che tutti i fatti siano stati estratti; una categoria mancante non prova da sola che esista un’omissione.

Se l’output raggiunge il limite, il blocco viene diviso e il percorso ripetuto. Esiste poi un secondo livello: estrazione del testo non ripetuto, trasferimento delle evidenze dei passaggi identici, verifica delle sorgenti senza corrispondenza, eventuale verifica del segmento di destinazione e, in caso di errore, rilettura del documento completo.

I documenti con dipendenze di riuso possono rimanere nello stato “running” anche quando hanno evidenze persistite. La finalizzazione segue una barriera tra fasi, rendendo l’avanzamento difficile da interpretare. Le evidenze parziali possono essere durevoli senza rappresentare un checkpoint sufficiente a saltare tutto il lavoro al riavvio.

Infine il contratto contiene una contraddizione: il prompt breve vieta di deduplicare; il task richiede di unire ripetizioni identiche. Il modello svolge già numerose operazioni: individuazione dei fatti, classificazione, polarità, cronologia, stati terapeutici e compilazione di payload specifici. Parte della normalizzazione è ripetuta nel codice. Le citazioni sono già espresse tramite riferimenti numerici e diversi campi sono già facoltativi: queste ottimizzazioni non vanno presentate come nuove.

## Soluzione A: semplificazione del percorso attuale — raccomandata come primo intervento

1. Un contratto coerente: un solo prompt principale, mantenendo inizialmente lo schema e le regole cliniche correnti. La deduplicazione locale riguarda esclusivamente ripetizioni identiche; episodi, stati e date distinti rimangono separati. Ridurre formulazioni ripetute senza cancellare distinzioni cliniche.
2. Un solo passaggio locale per normalizzazione, citazioni, tipi, unità e contraddizioni verificabili. Correggere soltanto ciò che è determinabile dalla fonte; un’ambiguità clinica non diventa una correzione automatica.
3. Riunire gli oggetti non validi e i possibili fatti mancanti in una sola richiesta mirata. Inviare solo il contesto necessario, inclusi antecedenti per date, soggetti e negazioni. Non ripetere l’intera risposta valida. Chiedere sostituzioni e aggiunte identificabili e validarle prima del merge.
4. Al massimo un recupero semantico per blocco. Se fallisce, conservare gli elementi validi e registrare esplicitamente gli elementi irrisolti e il documento incompleto. Una risposta non valida non deve essere trasformata silenziosamente in successo.
5. Trattare l’output troncato come problema distinto: dividere il solo blocco coinvolto, con un budget complessivo di tentativi anche per i blocchi figli. Conservare i risultati dei blocchi già conclusi.
6. Registrare ogni chiamata: motivo, tempi di attesa e inferenza, token, troncamenti, errori e nuovi fatti validi. Mostrare blocchi estratti, documenti in attesa di riuso e documenti finalizzati separatamente.

Vantaggio strutturale: per un blocco che oggi richiede sia correzione sia recupero si passa da tre a due chiamate. Nessun risparmio di chiamate è garantito sui blocchi già conclusi alla prima risposta. Va verificato se la richiesta combinata mantiene lo stesso richiamo.

## Soluzione B: unità di estrazione persistenti e riuso senza cascata di verifiche

Evoluzione più ampia: costruire unità con frasi numerate e contesto esplicito; ogni unità ha stato, risultato e metriche persistenti. Una sola coda globale alimenta i tre slot. I documenti diventano insiemi di occorrenze delle unità e si finalizzano appena tutte le proprie dipendenze sono risolte.

Distinguere due cache:

- **Ripresa esatta della stessa richiesta:** chiave composta da contenuto, contesto, modello/quantizzazione, prompt, schema e impostazioni rilevanti. Il risultato può essere riutilizzato senza nuove chiamate quando l’identità è esatta.
- **Riuso tra documenti:** ammesso solo se sono equivalenti anche contesto clinico e ancoraggio temporale. Un testo identico può descrivere visite o stati differenti. Non basta un hash della frase e non basta una similarità embedding.

Per ogni unità salvare anche il risultato vuoto validamente restituito, distinguendolo da errore, troncamento o mancata elaborazione. Questo evita di usare l’assenza di una corrispondenza tra evidenze come motivo automatico di un’altra lettura. “Vuoto valido” certifica l’esecuzione, non l’assenza di omissioni; i controlli di qualità restano necessari.

Questa architettura può eliminare la cascata sorgente → destinazione → documento completo e rendere la ripresa meno costosa. Richiede però una migrazione accurata delle provenienze e un confronto con il riuso attuale, che già riduce sensibilmente il testo iniziale. Non assumere che rimuovere semplicemente il riuso acceleri il progetto.

## Soluzione C: blocchi più grandi o documento intero per i testi brevi

Simulazione della sola segmentazione, mantenendo il piano di riuso:

| Limite in caratteri | Blocchi iniziali | Variazione rispetto a oggi |
|---|---:|---:|
| 2.800 | 238 | — |
| 4.000 | 204 | −14,3% |
| 5.600 | 186 | −21,8% |

Sono riduzioni di blocchi, non di tempo misurato. Il testo completo può entrare nel contesto mentre la sua risposta strutturata supera il limite. Per 4.000/5.600 caratteri occorre ricalcolare budget di input, risposta e correzione: la raccomandazione precedente di 16.384 token riguarda i blocchi attuali, non qualsiasi variante.

Usare dimensioni adattive ai token e alla densità attesa dei fatti, rispettando sezioni, antecedenti e confini documentali. Non concatenare referti diversi. Provare prima 4.000 caratteri sullo stesso campione: il vantaggio teorico è modesto e può essere annullato da risposte troncate o omissioni. Una chiamata per frase aumenta invece il numero di richieste e non è la prima scelta per velocizzare questa configurazione.

## Prompt candidato per la soluzione A

Bozza di struttura, non sostituto clinicamente validato del task v12. Nella prima implementazione vanno conservate le distinzioni cliniche del contratto corrente, evitando di cambiare contemporaneamente i criteri di inclusione.

> Estrai dal testo numerato tutti i fatti clinici espliciti previsti dallo schema. Per ogni fatto conserva concetto, riferimenti alla fonte e, quando documentati, polarità, stato, data, valore, unità e attributi specifici. Separa fatti indipendenti e transizioni differenti. Unisci soltanto ripetizioni identiche all’interno dello stesso blocco, conservando tutti i riferimenti. Non dedurre diagnosi, causalità, gravità o date assenti. Distingui azioni programmate da eseguite e assenza di un fatto da un risultato negativo. Rispetta le regole di inclusione clinica e laboratoristica del contratto. Restituisci esclusivamente il JSON previsto; non produrre spiegazioni.

Il prompt mirato usa lo stesso contratto e aggiunge soltanto un elenco strutturato di problemi e fonti pertinenti. Non introdurre un LLM pianificatore, un revisore LLM sistematico o una seconda classificazione universale: ricreerebbero la cascata che si vuole eliminare.

## Come scegliere e verificare

Prima aggiungere misure persistenti senza cambiare le decisioni estrattive. Poi confrontare separatamente: baseline attuale; prompt coerente + recupero unificato; eventuale architettura a unità; blocchi più grandi. Mantenere invariati modello, quantizzazione e condizioni hardware all’interno di ogni confronto. Separare avvio a freddo e funzionamento con cache.

Usare referti con annotazioni cliniche indipendenti, includendo documenti densi, negazioni, date retrospettive, transizioni farmacologiche, anomalie di laboratorio narrative e ripetizioni tra referti. Per il riuso serve un test sull’intero dossier; per la qualità generale serve anche un campione di altri pazienti, non solo il workspace corrente.

Misurare tempo totale, mediana e 95° percentile per documento, token generati per fatto corretto, chiamate per blocco, guadagno di fatti validi di ogni recupero, precisione, richiamo, accuratezza degli attributi, documenti incompleti e duplicati. Nessuna percentuale di accelerazione è dimostrata dalle sole simulazioni. Definire prima del test il margine accettabile sul richiamo e verificare separatamente gli errori clinicamente rilevanti; l’assenza di regressioni in un campione piccolo non prova equivalenza.

## Riferimenti

- `emr_analyzer/clinical/atomic_evidence.py`: estrazione, budget, schema, validazione, recuperi e supplementi deterministici.
- `emr_analyzer/clinical/registry_builder.py`: pianificazione, riuso, verifiche, checkpoint e audit.
- `emr_analyzer/clinical/block_reuse.py`: regole di riuso esatto.
- `docs/ATOMIC_EVIDENCE_BENCHMARK_2026-08-26.md`: evidenza storica che modificare il recupero può diminuire il richiamo; non è un confronto della run attuale.
- [Documentazione ufficiale llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md): supporto a JSON vincolato da schema, elaborazione parallela e monitoraggio. Queste funzionalità non attestano la correttezza clinica delle risposte.
