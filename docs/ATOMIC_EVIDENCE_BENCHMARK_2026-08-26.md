# Benchmark evidenze atomiche — 26 agosto 2026

## Scopo e metodo

Il benchmark usa 60 testi clinici normalizzati e anonimizzati: 20 per ciascuno
dei progetti MELANOMA, RENE e LUNG. Il campione comprende 57 pazienti distinti,
10 tipologie documentali per progetto, 196.350 caratteri e documenti brevi,
medi, lunghi e molto lunghi. I progetti e i database sono stati aperti in sola
lettura; nessun testo clinico è conservato nel repository.

Qwen3-4B-Instruct è stato confrontato con tre estrazioni indipendenti eseguite
con gpt-5.6-sol, seguite da revisione incrociata dei fatti, delle fonti e degli
errori. Il riferimento iniziale di 1.085 eventi è stato aumentato con 484 fatti
aggiuntivi validi individuati da Qwen e verificati sul testo, per un totale di
1.569 fatti. Questo evita di penalizzare un modello quando trova un fatto vero
omesso dal primo revisore.

## Baseline sui 60 documenti

| Metrica | Risultato |
|---|---:|
| Sensibilità | 90,44% (1.419/1.569) |
| Precisione sui claim deduplicati | 93,40% |
| F1 sui claim deduplicati | 91,90% |
| Precisione dell'output grezzo | 71,32% |
| Duplicati nell'output grezzo | 23,64% |
| Atomi prodotti | 2.797 |
| Chiamate LLM | 226 |
| Token locali Qwen | 687.158 |
| Tempo wall-clock | 59,6 minuti |

La sensibilità per progetto era 89,82% per MELANOMA, 92,05% per RENE e 89,04%
per LUNG. Il difetto dominante era il laboratorio narrativo: 49/92 fatti
riconosciuti (53,3%). Gli altri problemi principali erano procedure pianificate
scambiate per eseguite, fatti composti, duplicati semantici, date non ancorate
alla fonte e classificazioni instabili fra categorie vicine.

## Modifiche promosse nella pipeline v11

- Prompt istituzionale universale più compatto, con atomizzazione, controllo
  di copertura, stato dell'azione e vincoli di grounding espliciti.
- Fallback LLM per anomalie di laboratorio descritte in prosa; i valori
  strutturati restano autorevoli e le ripetizioni esatte vengono eliminate.
- Proiezione deterministica di esami, procedure, ricoveri e dimissioni soltanto
  pianificati verso `clinical_decision/planned`.
- Deduplicazione di formulazioni diverse dello stesso fatto quando coincidono
  concetto, stato, data, valore, sede e altri discriminanti; tutte le
  occorrenze restano disponibili per provenienza e Quick View.
- Contratto serializzato completato con stato clinico, sede, lateralità,
  gravità, precisione della data, intervallo, certezza e confidenza.
- Metriche separate per output iniziale, riparazioni, recupero di copertura e
  duplicati rimossi.
- Compatibilità dei bucket equivalenti nella guardia di copertura per evitare
  retry privi di nuovo contenuto clinico.

## A/B mirato v10 vs v11

Sei casi difficili del benchmark (due per progetto, inclusi documenti lunghi e
molto lunghi) sono stati rieseguiti con lo stesso modello e gli stessi testi.
Il confronto automatico usa il riferimento aumentato e costituisce una misura
di sviluppo, non una nuova validazione clinica manuale.

| Metrica sui 6 casi | v10 | v11 |
|---|---:|---:|
| Recall semantico proxy | 404/450 (89,78%) | 404/450 (89,78%) |
| Recall laboratorio | 12/46 (26,09%) | 34/46 (73,91%) |
| Duplicati semantici stimati | 154 | 133 |
| Evidenze considerate | 833 | 825 |
| Chiamate LLM | 59 | 61 |
| Token prompt | 136.483 | 126.689 |
| Token risposta | 57.015 | 70.349 |

La v11 recupera il principale deficit clinico, quello di laboratorio, e riduce
i duplicati senza ridurre la copertura complessiva del sottoinsieme. Non è però
dimostrato un vantaggio netto di velocità: i token di input diminuiscono del
7,2%, ma l'output aumenta del 23,4%. Il tempo osservato è migliorato, ma è
influenzato dalla variabilità del server e non viene usato come prova isolata.

È stata inoltre provata una variante v12 che separava il laboratorio in una
chiamata mirata. Su tre casi usava meno token della v11 ma riduceva il recall da
120/132 a 118/132; la variante è stata quindi respinta e non è attiva.

## Interpretazione e validazione successiva

La pipeline v11 è una correzione sostanziale e coperta da test, ma non equivale
alla validazione clinica definitiva. Il prossimo gold set dovrebbe sovracampionare
anomalie ematochimiche narrative, emogasanalisi, transizioni farmacologiche,
azioni pianificate/eseguite, date retrospettive e reperti negativi rilevanti.
Sensibilità e precisione definitive devono essere calcolate su documenti non
usati per modificare il prompt, con revisori clinici ciechi alla versione.

## Percorso A — estrazione SNOMED-vincolata (branch `snomed`)

La sezione documenta il tooling di benchmark del Percorso A e le metriche;
i valori vengono compilati dopo la prima run reale sul release licenziato
(Fase 0 del piano: acquisizione via MLDS). Con la generazione vincolata l'LLM
può emettere solo codici del set candidato per chunk: **il retrieval è quindi
il tetto del recall** e la metrica che lo isola è `retrieval_recall`.

Comandi:

```bash
# Run Percorso A (stessa infrastruttura sample/manifest del baseline).
python tools/benchmark_atomic_round.py snomed \
  --output /tmp/bench --release-dir <RELEASE> \
  --cap 32 --min-score 0.2

# Scoring contro l'annotazione SNOMED utente (gold esterno, stile
# sol/reference_all.json con snomed_code sugli eventi).
python tools/score_snomed_benchmark.py \
  --root /tmp/bench \
  --release-dir <RELEASE> \
  --case-ids <TC1> ... \
  --runs qwen snomed
```

In attesa della licenza MLDS, le run possono usare il **Global Patient Set**
come pilota intermedio: stesso loader manager con `kind="gps"` (flag `--gps`
nei comandi `snomed`/`events`), sorgente un TSV piatto
`ConceptID | Active | FSN | USPreferredTerm` (CC BY-ND, file fuori repo).
Limiti attesi rispetto al release RF2: nessun IS-A → `code_hierarchical`
collassa su `code_exact`; solo US-English → `retrieval_recall` su corpus
italiano degradato; nessun sinonimo.

Definizione delle metriche (per caso, aggregate sul totale dei codici gold):

| Metrica | Definizione |
|---|---|
| `retrieval_recall` | Quota dei codici gold presenti nell'unione `retrieved_codes` del caso (dal blocco `snomed` del payload). Diagnostico chiave: isola il retrieval dal LLM. |
| `code_exact` | Quota dei codici gold emessi verbatim. |
| `code_hierarchical` | Quota dei codici gold emessi verbatim o come antenato/discente IS-A del codice emesso (navigazione sull'indice del release). |
| `code_mismatch` | Quota dei codici gold senza corrispondenza gerarchica. |
| `snomed_coverage` | Quota degli eventi gold con `snomed_code` catturati come atomo codificato (match semantico, indipendente dall'identità del codice). |
| `semantic_proxy` | Le proxy semantiche a livello atomo (`score_run`) sugli stessi casi, per A/B comparabile baseline-vs-SNOMED. |

Il payload per caso in `snomed/<case>.json` espone anche `release_digest`,
`candidate_chunks`, `empty_candidate_chunks`, `candidate_set_sizes(_avg)`,
`coded_atoms`/`unmapped_atoms` e `mapped_by_fact_type`. Il fallback
abbreviazioni (dizionario curato in `resources/snomed_abbreviations.json`)
alza `retrieval_recall` per token come `HT`, `DM`, `SO2` assenti come token
letterali del release.

Valori della prima run reale (da compilare): *pending*.

### Percorso B — eventi diretti da RAG (sperimentale)

`DirectSnomedEventBuilder` salta lo strato atomico: per ogni chunk produce
eventi a livello documento con lo stesso contratto closed-set (`snomed_code`
`enum` sui candidati *event-like*, `observed_date` `enum` sulle date
grounded nel testo, `source_refs` di frasi esatte). Gli eventi di laboratorio
sono deterministici (righe fuori range mappate tramite l'indice), mai via LLM.
Il percorso è marcatamente sperimentale e non è mai il default.

```bash
# Run Percorso B (stessa infrastruttura di `snomed`).
python tools/benchmark_atomic_round.py events \
  --output /tmp/bench --release-dir <RELEASE>

# Confronto event-level: Percorso A (`snomed/`) vs Percorso B (`events/`)
# vs reference (stesso gold esterno).
python tools/score_direct_events.py \
  --root /tmp/bench \
  --release-dir <RELEASE> \
  --case-ids <TC1> ...
```

Metriche (per caso, aggregate): recall/precision di A (atomi codificati) e di B
(eventi diretti) contro gli eventi di reference (match semantico `_is_match`);
accordo codice di A e di B (exact / gerarchico IS-A) sui gold con `snomed_code`;
`B_date_agreement` (date uguali sulle coppie matchate); `B_grounded` (quota di
eventi B con passaggio sorgente esatto). Valori della prima run reale (da
compilare): *pending*.
