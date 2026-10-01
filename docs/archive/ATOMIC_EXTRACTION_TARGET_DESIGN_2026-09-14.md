> Documento storico: questi metodi di estrazione sono stati rimossi dal branch di sviluppo. Vedere [pipeline ICD-11 attuale](ICD11_PIPELINE.md).

# Estrazione atomica, storia clinica e interrogazione: architettura proposta

14 settembre 2026. Documento progettuale fondato sull’ispezione del codice, sulle misure del workspace e su fonti primarie. Non è una validazione clinica né una modifica della pipeline in esecuzione. Estende `ATOMIC_EXTRACTION_SIMPLIFICATION_2026-09-14.md` e precisa la proposta di lavorare per frase.

## Decisione proposta

Usare un’estrazione ibrida, ancorata alle fonti: laboratorio deterministico; testo narrativo analizzato una volta in gruppi adattivi di frasi con contesto; risultati compatti; controlli locali; un solo recupero semantico selettivo; memorizzazione incrementale. Gli eventi longitudinali e le risposte alle domande sono proiezioni ricostruibili delle evidenze, dei valori strutturati e delle fonti.

L’obiettivo operativo è minimizzare il tempo per dossier rispettando vincoli di richiamo, precisione, attribuzione temporale e completezza delle risposte, misurati su un riferimento indipendente. Non esiste evidenza che una configurazione massimizzi contemporaneamente tutti questi criteri su qualsiasi referto. L’architettura rende i compromessi osservabili e modificabili senza perdere i dati originali.

## Vincoli del progetto e misure

Il workspace esaminato contiene 266 documenti, dei quali 183 narrativi e 83 laboratoristici. Il testo narrativo mediano è 852 token, il massimo 4.169. Con il riuso corrente, i blocchi iniziali sono 238. La modalità per frase ricostruisce invece 4.719 unità potenzialmente analizzabili: non tutte sono frasi cliniche autonome, perché il segmentatore usa punteggiatura e un limite di lunghezza.

Pertanto la precedente indicazione di 4–8 frasi per chiamata non è una configurazione ottimale dimostrata: potrebbe aumentare le richieste. Il numero di frasi deve essere variabile e subordinato a token, densità dei fatti, sezioni e budget di risposta. Non confondere unità di attribuzione con unità di inferenza.

Il riuso attuale sottrae circa il 39% dei caratteri alla prima lettura, ma introduce verifiche e dipendenze di finalizzazione. Il prompt fisso occupa circa 1.664 token ed è in parte già in cache. La generazione osservata è circa 15 token/s per slot. La riduzione di token generati e riletture è quindi una priorità da misurare, senza assumere che ogni taglio del prompt produca un’accelerazione proporzionale.

## 1. Tre livelli di informazione, conservati separatamente

| Livello | Contenuto | Uso |
|---|---|---|
| Fonti e dati strutturati | Testo clinico versionato, paragrafi/frasi, metadati e tutti i risultati di laboratorio | Verifica, recupero di omissioni, domande quantitative |
| Evidenze atomiche | Singole affermazioni supportate, con qualificatori e provenienza | Indice clinico verificabile e base per eventi |
| Eventi e storia cronologica | Raggruppamenti di evidenze, transizioni e sintesi con riferimenti | Visualizzazione e domande longitudinali |

Una ripetizione documentale non è automaticamente un nuovo evento. Viceversa due osservazioni simili non sono necessariamente lo stesso evento. La normalizzazione terminologica non decide l’identità degli episodi.

L’atomicità corrisponde a un’affermazione clinica interpretabile con tutti i qualificatori necessari. La dimensione di una lesione appartiene al reperto; non deve diventare un evento separato privo di sede. Due sintomi distinti nella stessa frase possono invece richiedere due atomi. L’obiettivo non è massimizzare il numero di righe.

Ogni atomo deve poter mantenere: soggetto/esperiente, tipo, concetto originale, asserzione, stato, dati quantitativi con operatore e unità, attributi clinici espliciti, espressione temporale, fonti principali e fonti di contesto, versione dell’estrazione, stato di verifica. Familiarità e anamnesi altrui non devono trasformarsi in diagnosi del paziente.

## 2. Laboratorio deterministico, con criterio di inclusione esplicito

Mantenere tutti i valori nel repository del laboratorio. Nell’elenco atomico includere solo quelli per cui sia documentata l’anomalia rispetto al riferimento pertinente, come richiesto dall’utente. Distinguere un confronto numerico verificato da un flag di anomalia del laboratorio; un risultato qualitativo necessita di una regola esplicita, non di un intervallo inventato. Il comportamento corrente ammette anche anomalie qualitative e flag: questa distinzione deve essere visibile nel motivo d’inclusione.

Usare il range riportato per quella misura, senza applicare indiscriminatamente il range della prima osservazione a tutta la serie. Conservare materiale, unità, metodo quando disponibile, limiti e data di prelievo distinta da quella del referto. Trattare `<`, `>`, `≤`, `≥` come valori censurati: il valore della soglia non è la misura esatta e può non consentire di decidere se il risultato è fuori range.

Le anomalie citate nelle visite restano estraibili dal testo: un referto di laboratorio mancante non deve cancellare una leucocitosi esplicitamente documentata. Un numero senza range o indicazione di anomalia non deve essere dichiarato anomalo dall’LLM sulla base di conoscenze generali.

La domanda “quando il TSH è tornato nel range?” deve consultare anche i risultati normali nella tabella laboratorio, pur non creando atomi per essi. La separazione tra momento clinicamente rilevante, emissione e riferimenti è coerente con i concetti di `effective`, `issued` e `referenceRange` di [FHIR Observation R5](https://hl7.org/fhir/R5/observation-definitions.html). Non è necessario far generare al modello l’intera rappresentazione FHIR.

## 3. Frasi come ancoraggio; gruppi adattivi come richiesta

Segmentare localmente preservando offset, ordine, intestazioni, elenchi, tabelle e provenienza disponibile. Proteggere abbreviazioni cliniche, decimali, date e interruzioni grafiche. Nessun LLM preliminare per riassumere o riscrivere il documento.

Ogni frase ha un ID stabile legato alla versione della fonte. Selezionare tutte le frasi cliniche o ambigue; escludere automaticamente soltanto contenuto certamente amministrativo. Un filtro per parole chiave non deve decidere da solo ciò che il modello potrà vedere.

Costruire gruppi contigui della stessa sezione, con budget reale di token e risposta. Ogni frase è obiettivo una sola volta; le frasi di contesto possono comparire in più richieste senza generare nuovamente i propri fatti. Includere intestazione temporale pertinente e antecedenti necessari; usare il seguito quando risolve una dipendenza. Non estrarre da documenti diversi nella stessa richiesta.

Un punto iniziale sperimentale è confrontare budget di fonte nell’ordine di 800, 1.200 e 1.600 token, sempre calcolando input totale e risposta. Non sono preset convalidati: frasi dense possono richiedere gruppi più piccoli. Il contesto di 16.384 token è un riferimento per la configurazione esaminata; deve essere ricontrollato dopo modifiche a prompt, schema o recuperi.

L’evidenza scientifica sulle relazioni temporali tra frasi motiva la conservazione del contesto, non una specifica finestra universale. [D’Souza e Ng, LREC 2014](https://aclanthology.org/L14-1129/). Gli esperimenti di [Lost in the Middle](https://arxiv.org/abs/2307.03172) mostrano limiti nell’uso dei contesti lunghi sui modelli e task valutati: non dimostrano che finestre brevi siano sempre migliori sul nostro Qwen.

## 4. Una sola estrazione con risposta essenziale e informazione completa

Chiedere al modello solo decisioni semantiche che il codice non può ricostruire con certezza. Mantenere inizialmente la struttura a categorie già implementata, evitando una migrazione dello schema mentre si misura il nuovo flusso.

- Conservare concetto esplicito e qualificatori essenziali: negazione, ipotesi, soggetto, azione pianificata/eseguita, transizioni terapeutiche, sede, misure e date supportate.
- Usare riferimenti a frasi o porzioni di fonte; il codice ricostruisce il testo citato e risolve gli offset. Il modello non deve generare pagine, ID paziente, hash o offset numerici da indovinare.
- Distinguere frasi obiettivo e frasi di supporto. La fonte di contesto è citabile quando sostiene il farmaco, il soggetto o la data di un fatto della frase obiettivo.
- Non far ripetere una seconda volta nome del farmaco, stato, data o misura quando sono già presenti nello stesso oggetto. I campi realmente distinti restano distinti: nome commerciale e principio attivo non sono sempre intercambiabili.
- Omettere campi assenti, spiegazioni, catene di ragionamento, grading non esplicito e valutazioni generiche di “importanza”. La normalizzazione terminologica avviene dopo e conserva sempre il testo originale.
- Salvare le misure collegate a reperti anche quando sono multiple; comprimere la serializzazione non deve rimuoverle.

La letteratura dimostra che gli LLM possono affrontare estrazione di span e relazioni cliniche, ma non valida automaticamente questa pipeline italiana o i modelli installati. [Agrawal et al., EMNLP 2022](https://aclanthology.org/2022.emnlp-main.130/).

## 5. Controlli locali e recupero limitato

Dopo la risposta, verificare sintassi/schema, esistenza dei riferimenti, integrità della citazione e coerenze verificabili. La presenza delle parole nella fonte non certifica che l’interpretazione sia corretta; in particolare non prova da sola la corretta negazione o l’attribuzione della data.

Produrre un unico elenco di problemi: oggetti invalidi, misure non rappresentate, antecedenti irrisolti, segnali clinici plausibilmente omessi. Fare al massimo una richiesta semantica mirata per gruppo, con il minimo contesto sufficiente. Non chiedere universalmente “ricontrolla tutto”. Studi su compiti di ragionamento distinguono la capacità di correggere un errore noto dalla capacità di trovarlo: è una motivazione progettuale per il feedback specifico, non una prova clinica. [Tyen et al., ACL 2024](https://aclanthology.org/2024.findings-acl.826/).

Se il limite di output viene raggiunto, trattarlo come troncamento: dividere il solo gruppo coinvolto, senza accettare una risposta parziale come completa. Un budget globale limita i tentativi anche sui figli. Gli elementi irrisolti restano visibili e recuperabili.

Per i casi semanticamente difficili, la richiesta mirata può essere instradata a un modello più capace solo dopo un confronto che ne dimostri il beneficio. Accodare questi casi e caricare il secondo modello in una fase dedicata evita continui cambi di modello e doppia occupazione della memoria. Nessuna seconda lettura universale.

Un controllore non intercetta tutte le omissioni: valutare anche un campione casuale di gruppi non segnalati e quelli marcati “nessun fatto”. La copertura di elaborazione delle frasi è una metrica operativa, non la sensibilità clinica.

## 6. Persistenza e riuso con contesto

Ogni richiesta ha risultato, versione, metriche e stato persistenti: pending, running, completed, completed_empty, needs_review, failed. Un’interruzione non cancella i gruppi conclusi. La scrittura del risultato e dello stato deve essere atomica e il merge idempotente.

La cache della stessa richiesta include contenuto obiettivo, contesto, modello/digest, prompt, schema e impostazioni che influiscono sul risultato. Il riuso tra documenti è più restrittivo: stessa frase senza stesso ancoraggio non basta. Salvare occorrenze separate e collegarle a un risultato riutilizzabile solo quando le condizioni sono verificate. La similarità semantica propone candidati; non autorizza fusioni automatiche.

Valutare il risparmio netto del riuso, comprese le verifiche. Finalizzare ogni documento appena pronte le sue dipendenze, senza attendere una barriera globale. Una sola coda di richieste alimenta il parallelismo scelto; evitare pool annidati indipendenti che accodano lavoro non osservabile.

## 7. Storia clinica cronologica come proiezione ricostruibile

Per ogni evidenza distinguere tempo dell’evento, data del documento e momento di acquisizione/estrazione. Conservare l’espressione originale, la precisione e l’ancora. Una data sconosciuta rimane sconosciuta. Un intervallo non diventa arbitrariamente un giorno. In assenza di data assoluta, conservare relazioni supportate come prima/dopo, evitando un ordine clinico totale inventato.

Esempio inventato: una visita di giugno ricorda un intervento di marzo. L’intervento viene collocato a marzo; giugno è la data della fonte. Una sospensione e una ripresa dello stesso farmaco sono transizioni diverse. Un referto successivo che ricorda la sospensione aggiunge una fonte, non una seconda sospensione automaticamente.

Costruire eventi a partire da occorrenze, concetto, soggetto, stato, sito, tempo e identificatori documentali. Le relazioni possibili includono riferito_allo_stesso_evento, segue, risolve e contraddice. Una relazione causale richiede attribuzione esplicita nella fonte o revisione; la successione temporale non basta.

Mantenere le contraddizioni, le revisioni umane e gli atomi originari. Le correzioni producono nuove versioni e invalidano soltanto proiezioni dipendenti. La sintesi della storia deve poter essere rigenerata, con riferimenti agli eventi e da questi alle fonti. Una similarità elevata o un collegamento transitivo nel grafo non deve fondere eventi incompatibili.

## 8. Domande su tre canali di dati

1. **Dati strutturati:** laboratorio completo, date, farmaci e stati. Interrogazioni parametrizzate e operazioni deterministiche per valori, ordinamenti e conteggi.
2. **Evidenze ed eventi:** ricerca lessicale e semantica filtrata per paziente, concetto e tempo, con espansione delle fonti e dei collegamenti necessari.
3. **Testi clinici originali attivi:** recupero delle informazioni non indicizzate o dubbie. Per domande esaustive o assenze, una ricerca top-k non basta; ampliare il perimetro ed esplicitare documenti mancanti o non elaborati.

La risposta viene composta su un insieme di fonti identificabile e cita ogni affermazione sostanziale. Non usare una risposta precedente della chat come nuova prova clinica. “Non trovato” non equivale a “mai avvenuto”; “assente” e “non documentato” sono stati distinti.

Esempi di collaudo: prima sospensione di un farmaco e ragione documentata; valori TSH prima/dopo una terapia e successivo rientro nel range; differenza tra procedure programmate ed eseguite; progressioni con relativa sede; informazioni discordanti tra due referti. Il laboratorio completo risponde anche quando nell’indice atomico sono presenti solo anomalie.

Il benchmark [LongHealth](https://arxiv.org/abs/2401.14490) include estrazione, negazione e ordinamento e segnala difficoltà nell’identificare informazioni mancanti sui modelli esaminati. Il più recente preprint [EHRNote-ChatQA](https://arxiv.org/abs/2606.15735) valuta domande su più dimissioni, con revisione esperta, e riporta che correttezza del contenuto e supporto delle evidenze non coincidono. Questi risultati motivano valutazioni separate di risposta e citazioni; non forniscono prestazioni garantite per la nostra app.

## 9. Modello, velocità e priorità

Tenere Qwen3-14B come baseline e confrontare i candidati già installati sullo stesso contratto, senza cambiare simultaneamente segmentazione e modello. Il più grande non è automaticamente il più accurato nel formato richiesto; la variante quantizzata effettivamente usata va misurata.

Ottimizzare in questo ordine: eliminare generazione duplicata e recuperi inutili; checkpoint per gruppo; raggruppamento adattivo; memoria di contesto; modello o eventuale specializzazione. La distillazione o il fine-tuning di un modello piccolo può essere una fase futura dopo annotazioni cliniche e separazione dei pazienti fra train e test. Non addestrare automaticamente sugli output non revisionati del modello grande.

Il costo totale comprende prefill, generazione, attesa, recuperi e persistenza. La percentuale di chiamate eliminate non è la percentuale di tempo risparmiato. Registrare anche i fallimenti: confrontare soltanto le richieste riuscite favorirebbe artificialmente la variante meno completa.

## 10. Verifica e implementazione progressiva

**Misurazione prima della modifica:** catturare una baseline versionata con token, tempi, motivi di recupero, troncamenti e checkpoint. La run corrente deve rimanere riconoscibile e non mescolare versioni.

**Prototipo compatibile:** migliorare segmentazione e contesto citabile, mantenere lo schema di evidenza esistente tramite adattatore, unificare il recupero e implementare checkpoint per gruppo. Non cancellare vecchi risultati; confrontare in un archivio separato.

**Valutazione:** usare un insieme annotato indipendentemente, con più pazienti e referti rappresentativi. Il workspace corrente è utile per sviluppo e prestazioni, ma non basta per dimostrare generalizzazione. Separare pazienti e referti quasi duplicati fra sviluppo e valutazione; riportare incertezza statistica rispettando la dipendenza tra documenti dello stesso paziente.

Misurare: richiamo dei fatti, precisione, esattezza di negazione/soggetto/tempo/stato/misure, token per fatto corretto, latenza mediana e p95, tempo per dossier, memoria, recuperi, fallimenti e sensibilità nei sottogruppi. Per la storia misurare fusioni indebite, eventi duplicati, ordine e transizioni; per le domande misurare correttezza, completezza, citazioni e gestione dell’assenza di dati.

Confrontare le modifiche una alla volta. Definire preventivamente con il revisore clinico margini accettabili e categorie di errore critiche. Se il campione non distingue le varianti, dichiarare il risultato inconclusivo. Promuovere il metodo solo dopo aver verificato il percorso completo estrazione → eventi → domande, non soltanto un incremento di F1 sugli atomi.

## Cosa cambierebbe nel repository

- `clinical/atomic_evidence.py`: separare segmentazione, pianificazione, generazione e validazione; contesto di supporto citabile e recupero unificato.
- `clinical/registry_builder.py`: orchestrazione per gruppi persistenti, coda unica, finalizzazione locale delle dipendenze e metriche per chiamata.
- `clinical/block_reuse.py`: separare identità della richiesta e riuso di menzioni tra fonti.
- Modelli/repository: eventuali campi mancanti per soggetto, ancoraggio e provenienza per attributo; usare migrazioni additive e preservare le evidenze già presenti.
- `clinical/query_service.py` e `clinical/dossier_query.py`: percorso comune per interrogare dati strutturati, eventi e fonti; nessuna perdita della possibilità di analizzare il dossier completo.
- GUI: avanzamento per gruppi, problemi irrisolti, anteprima della citazione e visualizzazione distinta di data clinica/data del referto.

La parte nuova è questa composizione concreta, non un metodo già dimostrato dalla letteratura. I criteri verificabili sono: nessuna fonte eliminata, nessun attributo clinico scartato per comprimere l’output, nessuna fusione irreversibile, nessun successo silenzioso su estrazioni incomplete, nessuna promessa di accuratezza basata soltanto su JSON valido o frasi marcate come elaborate.
