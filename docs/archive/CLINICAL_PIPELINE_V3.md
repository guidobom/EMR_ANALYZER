# Pipeline clinica end-to-end v3

Questa specifica descrive il contratto autorevole della pipeline clinica a
valle dei testi normalizzati. I documenti e le osservazioni di laboratorio
strutturate sono sorgenti immutabili; SQLite è la fonte di verità. JSON e
JSONL sono formati di esportazione versionati, non database paralleli.

## Invarianti

1. Nessun fatto, claim, evento o relazione può citare un ID sorgente
   inesistente.
2. Ogni claim clinico possiede almeno una fonte verificabile.
3. Data dell'osservazione e data del referto sono campi distinti. Una data
   dell'osservazione assente resta `null`.
4. Il modello seleziona `source_refs` numerici; testo, pagina e coordinate
   sono risolti deterministicamente dall'applicazione.
5. I laboratori sono estratti dal livello strutturato, mai dall'LLM. Soltanto
   le anomalie diventano evidenze atomiche; una normalizzazione successiva
   può essere citata direttamente tramite la relativa osservazione
   laboratoristica.
6. Il contenuto amministrativo, metodologico e boilerplate viene conservato
   come escluso, con motivazione, ma non raggiunge eventi, sintesi o RAG.
7. Una evidenza può contribuire a più eventi, con un ruolo esplicito per ogni
   collegamento.
8. Le ipotesi esplorative sono separate dalle relazioni validate e non sono
   utilizzabili dal RAG prima della revisione umana.
9. Le correzioni umane sono versionate e non vengono sovrascritte da un
   rebuild.
10. Prompt e risposte LLM grezzi non vengono persistiti. Si conservano solo
    digest, parametri, metriche e codici di validazione.

## Fasi

1. **Segmentazione citabile**: frasi numerate, geometria PDF e classificazione
   delle zone boilerplate senza alterare gli offset originali.
2. **Estrazione atomica adattiva**: una chiamata multi-tipo per segmento; retry
   specializzato soltanto dopo un errore di schema, provenienza, completezza o
   payload tipizzato.
3. **Laboratorio deterministico**: osservazioni strutturate sempre conservate;
   evidenze atomiche solo secondo la politica delle anomalie configurata.
4. **Validazione ed esclusione**: classificazione in clinico, bassa rilevanza,
   amministrativo, metodologico, boilerplate, non informativo, non verificabile
   o da revisionare.
5. **Normalizzazione**: resolver locale per etichetta canonica e terminologie
   ATC/LOINC/SNOMED; conversioni di unità deterministiche con conservazione dei
   valori originali.
6. **Deduplicazione**: una occorrenza canonica, tutte le fonti e una coda per i
   casi semantici incerti.
7. **Candidate generation a due scale**: contatto clinico e collegamenti
   longitudinali; blocking deterministico e top-k semantico evitano il
   confronto quadratico.
8. **Grafo delle evidenze**: archi tipizzati e pesati con effetto
   `must_link`, `cohesive`, `context_only`, `cannot_link` o `uncertain`.
9. **Clustering vincolato**: costruzione dei candidati evento, rilevamento
   degli archi-ponte, split esplicito e verifica di coerenza.
10. **Formalizzazione**: categorie estensibili, appartenenze con ruolo, fasi e
    claim citati individualmente.
11. **Relazioni tra eventi**: distinzione tra documentate, inferite
    clinicamente e meramente temporali.
12. **Scoperta esplorativa**: comando separato che produce soltanto ipotesi
    `needs_review`.
13. **Indicizzazione e RAG**: recupero degli eventi, espansione delle relazioni
    validate, discesa a evidenze e documenti, fallback sulle sorgenti.

## Tipi di evidenza iniziali

`medication`, `laboratory_test`, `radiology_finding`,
`instrumental_finding`, `diagnosis`, `symptom`, `clinical_decision`,
`procedure`, `clinical_sign`, `vital_sign`, `histopathology`.

Il contratto usa un nucleo comune e payload tipizzati. Il resolver
terminologico opera dopo l'estrazione: l'LLM non assegna codici clinici.

Lo schema wire è costruito per ogni segmento: le categorie sono contenitori
separati, gli ID di citazione ammessi coincidono con le sole frasi presenti e
i campi opzionali assenti vengono omessi. Le difformità innocue (per esempio
`S2` al posto di `2`) sono normalizzate localmente; soltanto un oggetto che
perde informazione clinica genera un unico retry mirato. I valori di
laboratorio fuori range continuano a entrare per via deterministica, senza
chiamata LLM.

Anatomia normale incidentale, appuntamenti e boilerplate restano ispezionabili
come evidenze escluse e non raggiungono eventi o analisi. Le negazioni utili
alla stadiazione o al follow-up restano invece evidenze contestuali. Ogni
documento completato è un checkpoint; il comando **Interrompi** arresta il
processo al termine della chiamata attiva e consente la ripresa incrementale.

## Relazioni fra evidenze

`same_occurrence`, `same_process`, `manifestation_of`, `progression_of`,
`response_to`, `caused_by`, `complication_of`, `treats`, `evaluates`,
`decision_about`, `rules_out`, `contradicts`, `documents_resolution`,
`temporally_associated_with`.

Le relazioni causali, terapeutiche e valutative sono normalmente
`context_only`: collegano episodi autonomi senza fonderli. Solo archi coesivi
partecipano al clustering.

## Profili di consenso

- **Rapido**: una inferenza per candidato.
- **Selettivo**: una inferenza per candidato; è il profilo predefinito.
- **Robusto**: due inferenze per candidato e conservazione dei voti.
- **Ricerca**: tre inferenze per candidato e distribuzione completa dei voti.

## Gold set

Il gold set comprende annotazione atomica, esclusioni, duplicati, relazioni,
eventi, split/merge e claim. PDF e testo normalizzato sono affiancati. I
contratti e l'interfaccia non contengono riferimenti a uno specifico progetto,
paziente o patologia.

Metriche distinte: estrazione, tipo, polarità, data, citazioni,
deduplicazione, over-merge, over-split, clustering, copertura dei claim e
stabilità tra esecuzioni. Sono errori bloccanti: ID o citazioni inventati,
amministrativo negli eventi, claim senza fonte e ipotesi non validate nel
RAG.
