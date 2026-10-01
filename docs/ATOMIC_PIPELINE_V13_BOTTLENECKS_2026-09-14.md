> Documento storico: questi metodi di estrazione sono stati rimossi dal branch di sviluppo. Vedere [pipeline ICD-11 attuale](ICD11_PIPELINE.md).

# Analisi dei colli di bottiglia v13 — 14 settembre 2026

Analisi in sola lettura dell'esecuzione iniziata alle 15:55:12 Europe/Rome, con Qwen3-14B e tre worker. Nessuna inferenza di prova, modifica alla configurazione o interruzione della sessione clinica. Le statistiche seguenti riguardano le prime dieci chiamate concluse, osservate alle 16:04:47; non costituiscono un benchmark dell'intero dossier.

## Misure

| Fase | Chiamate concluse | Token generati | Secondi cumulati delle chiamate |
|---|---:|---:|---:|
| Estrazione | 6 | 12.080 | 867,0 |
| Recupero | 4 | 8.582 | 596,5 |

I secondi sono sommati tra worker concorrenti, non sono il tempo trascorso dell'esecuzione. I recuperi rappresentano il 40,8% di questi secondi e il 41,5% dei token prodotti. Tre recuperi hanno terminato normalmente e uno è stato troncato dopo 3.485 token e 234,2 secondi. Velocità di decodifica ponderata: 14,54 token/s nelle estrazioni, 14,42 nei recuperi, per worker. Nelle prime quattro chiamate circa il 95% del tempo interno al modello era decodifica, non lettura del prompt.

Alle 16:04:47 tre risultati di gruppo erano salvati come `needs_review`; nessuno dei 183 documenti narrativi risultava concluso. I 183 manifest `running` includono documenti in coda: i worker effettivi sono tre. Le chiamate osservate riguardavano ancora i primi tre documenti. Una barra documentale quasi ferma non equivale a un server bloccato.

I tre slot risultano occupati e riutilizzano token di prompt in cache. Il contesto è 32.768 token per slot. Gli input delle prime dieci chiamate sono compresi tra 1.530 e 2.240 token per le estrazioni e tra 1.636 e 2.167 per i recuperi: la saturazione del contesto non spiega queste latenze. Lo swap è rimasto sostanzialmente stabile: nella finestra estesa osservata nessun nuovo swap-out e solo 36 pagine di swap-in (circa 0,56 MiB). Non emerge paging intenso; questo non dimostra che la memoria o la concorrenza siano ottimali.

## Cause verificate nel codice e nei risultati

### 1. Recupero troppo ampio e privo della memoria dei fatti validi

`GroupedAtomicExtractor._run_group` invia nuovamente l'intero prompt originale e lo stesso schema di tutte le categorie, aggiungendo soltanto problemi, riferimenti mancanti e categorie. L'istruzione di non ripetere fatti validi non è accompagnata dai fatti della precedente risposta. Le chiamate sono indipendenti: il modello non conosce quella risposta.

Nei risultati persistiti di un gruppo ci sono sette coppie di oggetti identici salvo maiuscole/minuscole. La deduplicazione JSON intermedia distingue quelle differenze. La deduplicazione clinica successiva può eliminarle, ma non recupera il tempo già speso a generarle. Un recupero troncato attiva `_divide`, che riparte con nuove estrazioni dei sottogruppi.

Correzione proposta: recupero limitato alle frasi effettivamente problematiche, con il contesto necessario e identificativi stabili; includere una rappresentazione compatta dei fatti già coperti e dei problemi specifici. Separare l'eventuale riparazione di oggetti invalidi dall'estrazione di nuovi fatti nel contratto della singola chiamata di recupero. Non ridurre il limite di output senza verificare che non aumentino i troncamenti.

### 2. Copertura basata su categorie lessicali, non sui fatti

`_coverage_recovery_plan` considera coperta una frase se trova un atomo del tipo previsto dalle espressioni regolari. Nel campione sono presenti falsi allarmi e omissioni reali.

Riproduzioni locali, senza LLM:

- Una frase con «lesioni epatiche sospette», già rappresentata da un reperto radiologico, viene segnalata come diagnosi mancante per la parola «sospette».
- «Lesione compresa nei limiti di exeresi» può essere segnalata come procedura mancante: la presenza della parola «exeresi» non basta a dimostrare che la frase descriva un nuovo intervento.
- Nei gruppi reali restano anche una procedura senza atomo, un intervento classificato come diagnosi e una frase di familiarità non rappresentata. Disabilitare indiscriminatamente il controllo nasconderebbe quindi problemi reali.

Correzione proposta: distinguere fatto assente, fatto presente in categoria plausibile, errore di classificazione e frase legittimamente esclusa. Accettare equivalenze solo quando sostenute dal significato della fonte; non equiparare genericamente tutte le categorie e non considerare una frase completa soltanto perché è citata una volta.

### 3. Incompatibilità tra il controllo ereditato e gli integratori v12

Il controllo di copertura esenta alcune procedure eseguite perché presume un successivo integratore deterministico. La classe base esegue effettivamente integratori per procedure esplicite, reperti negativi e risoluzioni; la nuova implementazione `extract_document` non li richiama.

Riproduzione: con «Eseguita colonscopia.» e nessun atomo, il controllo non richiede alcun recupero. È una lacuna di affidabilità distinta dal costo dei falsi positivi. Non è dimostrato che questa specifica omissione sintetica sia avvenuta nei documenti elaborati.

Correzione proposta: rendere il controllo v13 indipendente da integratori non eseguiti, oppure reintegrare soltanto integratori verificati per citazioni, date e soggetto. Non trasferire automaticamente integratori che potrebbero attribuire al paziente eventi dei familiari.

### 4. Risposte verbose e generazione autoregressiva

Le prime estrazioni producono in media circa 2.013 token ciascuna. A circa 14–16 token/s per worker, un output di 2.000 token richiede già circa 125–143 secondi di sola generazione. Alcuni concetti arrivano a 227 caratteri e ripetono porzioni ampie del reperto. La fonte è già conservata localmente: non occorre usarne una parafrasi estesa come etichetta.

Correzione proposta: etichette concise e attributi strutturati che conservino sede, misure, stato e data; evitare campi vuoti o ripetuti. Misurare i token per fatto corretto, oltre ai token/s. La riduzione dei token non deve eliminare dettagli clinici.

### 5. Primo passaggio v13 senza il riuso di blocchi v12

Il ramo grouped disabilita il vecchio riuso tra blocchi di documenti diversi. La cache v13 richiede l'identità del testo completo oltre a richiesta, contesto, modello e metadati. È conservativa ma limita il riuso della storia clinica copiata nei referti successivi.

La precedente analisi locale del medesimo workspace stimava 630.222 caratteri narrativi totali e 384.458 dopo il riuso v12: circa il 39% di testo evitabile con quel metodo. Non è una misura di accelerazione della v13 né una garanzia che tutto quel riuso sia semanticamente trasferibile.

Correzione proposta: riuso a livello di gruppo con impronta del contesto rilevante, ancore temporali e soggetto; rimaterializzazione delle occorrenze nel documento di destinazione. Non usare il solo testo identico quando date relative o antecedenti cambiano.

### 6. Checkpoint e costi minori

I gruppi `needs_review` sono durevoli ma non eleggibili per la cache dei risultati conclusi. Una nuova esecuzione li rigenera interamente, anche quando contengono molti fatti validi. Il checkpoint dovrebbe conservare separatamente il risultato valido e i problemi residui, permettendo una ripresa mirata senza dichiarare completo ciò che non lo è.

Il pianificatore chiama il tokenizer per ogni frase e prima della generazione conta nuovamente il prompt completo. Questo introduce richieste locali evitabili con memoizzazione o stima iniziale seguita dal conteggio esatto. I dati disponibili non lo indicano però come costo dominante rispetto a minuti di generazione per chiamata.

## Ordine degli interventi

1. Correggere il contratto del recupero e il controllo di copertura, includendo il problema degli integratori mancanti.
2. Rendere riutilizzabili i risultati parziali validi e ridurre la verbosità dell'output senza perdita di attributi.
3. Introdurre riuso sicuro dei gruppi tra documenti e misurare token/fatto e secondi/documento.
4. Solo successivamente confrontare uno, due e tre worker e contesti più piccoli sullo stesso campione. Il numero di worker ottimale non è determinabile dai soli token/s per worker; occorre confrontare il throughput complessivo con accuratezza comparabile.

La sessione osservata è rimasta in esecuzione. Non sono state applicate correzioni alla pipeline durante questa analisi.
