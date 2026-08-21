# Registro clinico v2

## Scopo e vincoli

Il registro v2 trasforma i testi clinici già anonimizzati e normalizzati in una
sequenza verificabile di eventi ed episodi. I testi normalizzati e le righe di
laboratorio sono sorgenti immutabili: correzioni, decisioni e annotazioni
manuali vengono registrate come overlay o revisioni a valle.

L'output è destinato alla revisione clinica e alla ricerca. Non sostituisce il
giudizio medico e non deve essere usato automaticamente per decisioni
diagnostiche o terapeutiche.

## Architettura

```text
documenti e laboratorio immutabili
             │
             ▼
  evidenze cliniche atomiche, citate
             │
             ├── normalizzazione temporale
             ├── clustering per entità e finestre specifiche
             └── riconoscimento di risoluzione/recidiva
             ▼
      episodi ed eventi clinici
             │
             ├── sintesi breve + dettaglio espandibile
             ├── correlazioni multimodali prudenti
             ├── trend di laboratorio
             ├── corsi farmacologici
             └── linee oncologiche
             ▼
 registro cronologico, revisione, query ed export
```

### Evidenze atomiche

Ogni documento viene analizzato indipendentemente. Un'evidenza conserva:

- categoria, entità normalizzata e formulazione originale;
- affermazione presente, assente, possibile o storica;
- stato clinico, certezza, gravità, sede e significatività;
- valore, operatore, unità e intervallo di riferimento, se applicabili;
- data clinica, data del documento e precisione temporale;
- documento, pagina, passaggio letterale e coordinate, quando disponibili;
- modello, prompt, versione dell'estrattore e confidenza.

Le negazioni vengono mantenute quando rappresentano un cambiamento clinico
rilevante, per esempio la risoluzione di un sintomo precedentemente presente.
La deduplicazione non avviene durante l'estrazione, salvo copie letteralmente
identiche nello stesso documento.

### Datazione

Il sistema separa sempre:

- `first_evidence_date`: prima data clinica ricostruibile, anche retrospettiva;
- `first_documented_date`: prima data del documento che ne fornisce evidenza;
- precisione `day`, `month`, `year`, `interval`, `approximate` o `unknown`;
- fonte della data e testo temporale originale.

Un documento successivo può retrodatare la prima evidenza senza perdere la
provenienza. Date incomplete non vengono trasformate in date giornaliere
fittizie.

### Episodi, fusione e conflitti

Le finestre di prossimità dipendono dalla categoria: sono brevi per eventi
acuti, più ampie per tossicità e patologie croniche. Una risoluzione seguita da
una nuova manifestazione apre un episodio distinto.

Tutte le evidenze di uno stesso episodio vengono passate insieme al motore di
fusione. La sintesi deve coprire ogni evidenza mediante il suo identificativo;
se il modello non produce JSON valido o perde citazioni, viene usata una
sintesi deterministica completa. Per gruppi molto grandi si applica una fusione
gerarchica, senza troncare silenziosamente le fonti.

Valori, stati o date discordanti sono conservati, marcati e inviati alla coda
di revisione umana. Nessuna fonte contraddittoria viene cancellata.

### Correlazioni cliniche

Il correlatore può associare sintomi, parametri vitali, laboratorio, imaging e
terapie provenienti da documenti diversi, entro finestre configurabili. Le
correlazioni inferite:

- hanno certezza `inferred` e revisione `pending`;
- usano formule come «quadro compatibile con»;
- dichiarano che il nesso causale non è dimostrato;
- riportano tutte le evidenze favorevoli e contrarie.

Sono inclusi test specifici per i quadri tiroideo (TSH e valutazione
specialistica) e respiratorio (tosse/dispnea, ipossiemia e reperti TC).

### Proiezioni specialistiche

Il registro mantiene, oltre ai singoli eventi:

- ogni valore di laboratorio anomalo e i relativi trend;
- farmaco originale e normalizzato, inizio, sospensione, ripresa, dose, via,
  frequenza, indicazione, aderenza e stato della prescrizione/assunzione;
- una scheda per ogni linea oncologica con schema, cicli, modifiche, tossicità,
  risposta e progressione.

Non vengono introdotte terminologie ATC, RxNorm, ICD-10, SNOMED CT, LOINC o
MedDRA, conformemente ai requisiti del progetto.

## Incrementalità e prestazioni

Un manifesto con hash e versioni consente di riesaminare soltanto i documenti
nuovi o modificati. Ogni documento completato viene salvato immediatamente:
una chiusura accidentale non annulla quindi il lavoro già concluso e la nuova
esecuzione marca il run precedente come interrotto prima di riprendere i soli
documenti mancanti.

Il flusso ad alto volume applica inoltre:

- documenti più lunghi assegnati per primi agli slot, per ridurre la coda
  dell'ultimo worker;
- prompt statico compatto e riferimenti numerici a frasi verificate, senza far
  ricopiare al modello i passaggi clinici nel JSON;
- limite di output calcolato per la singola richiesta; solo una risposta
  realmente troncata viene divisa e ritentata su confini di frase;
- riuso di blocchi testuali identici soltanto fra documenti dello stesso tipo e
  della stessa sezione, mantenendo una fonte distinta per ogni documento;
- estrazione completa automatica se anche una sola evidenza del blocco non può
  essere ricostruita: il riuso non può ridurre silenziosamente il richiamo;
- fusione di cluster indipendenti in parallelo e bypass del LLM quando le
  evidenze cliniche sono semanticamente identiche;
- telemetria per documento (tempo, chiamate, retry, token di prompt e risposta)
  inclusa nel risultato del run e riepilogata nell'audit.

Se il modello non è disponibile, laboratorio ed evidenze già estratte vengono
comunque consolidate e i documenti narrativi restano in coda per una successiva
elaborazione.

### Coda multi-paziente

Il comando **Coda registri** nella barra principale (disponibile anche in
**Strumenti → Genera registri multi-paziente**) permette di selezionare tutti i
pazienti da elaborare. Per impostazione predefinita usa la modalità incrementale:
i manifest correnti vengono verificati e i documenti già completati non generano
nuove chiamate LLM. La rigenerazione integrale è disponibile come scelta
esplicita.

I pazienti avanzano in sequenza, mentre ogni singolo paziente usa tutti gli slot
LLM configurati per i suoi documenti. Questo evita che più registri competano per
gli stessi slot e per la memoria del modello. Un errore viene isolato al paziente
interessato e la coda prosegue; l'annullamento diventa effettivo tra un paziente e
il successivo, dopo che i risultati del paziente in corso sono stati salvati. Al
termine viene mostrato un riepilogo di documenti elaborati o saltati, evidenze,
voci finali, errori e durata.

Il benchmark sintetico si avvia con:

```bash
python tools/benchmark_registry.py --evidence 500
```

Il tempo LLM, dipendente da modello e hardware, va misurato separatamente sui
progetti reali.

Una simulazione PHI-free dell'estrazione atomica reale si avvia con:

```bash
python tools/simulate_atomic_extraction.py --model qwen3-14b
```

## Revisione e provenienza

Una nota può essere accettata, differita, corretta o rifiutata. Le correzioni
non modificano la sorgente e sopravvivono alle ricostruzioni automatiche. Il
dettaglio espandibile mostra evidenze incluse, escluse e contraddittorie,
aggiornamenti, relazioni e decisioni di revisione.

L'audit è pseudonimizzato, concatenato tramite hash e append-only. Registra
attore, run, modello, hash dell'input e versione del prompt senza conservare
payload identificativi in chiaro.

## Query ed esportazione

Le query recuperano prima il registro strutturato e l'indice FTS5. Ogni risposta
deve citare coppie evento-documento valide; richieste che eccedono il contesto
sono risolte per blocchi e riconciliate. È sempre disponibile un fallback
deterministico citato.

Gli export JSON e XLSX preservano le collezioni separate. CSV usa righe tipizzate
(`record_type`) per non perdere episodi, evidenze, revisioni, laboratorio,
farmaci, oncologia e trend. Sono inoltre disponibili Markdown, TXT, PDF e DOCX.

## Valutazione

La scheda **Gold Set** implementa un flusso a livello di paziente:

1. inclusione e assegnazione a `pilot`, `development` o `test`;
2. annotazione indipendente del revisore A;
3. annotazione indipendente del revisore B;
4. confronto sbloccato soltanto dopo entrambe le consegne;
5. adjudication di ogni annotazione come inclusa o esclusa;
6. creazione degli eventi finali e blocco immutabile del caso;
7. confronto con il registro automatico ed export JSONL.

I revisori selezionano direttamente i passaggi dei testi normalizzati. Le fonti
manuali sono persistite anche in forma relazionale, così un documento citato non
può essere cancellato o attribuito a un altro paziente accidentalmente. La
riapertura di una consegna invalida l'intera adjudication derivata.

La separazione A/B è una protezione dell'interfaccia, non un sistema di
autenticazione: per studi formali le sessioni devono essere assegnate e
supervisionate dal coordinatore.

Il gold set bloccato è un JSONL a livello di paziente. Il runner calcola
precisione, richiamo e F1 micro/macro, compatibilità temporale, stato, certezza
e completezza delle citazioni:

```bash
python -m emr_analyzer.evaluation gold_set_clinico.jsonl \
  --output rapporto_validazione.json
```

Ogni riga esportata contiene già `gold_events` e `predicted_events`; il test
deve rimanere bloccato e separato dai casi usati per modificare prompt o modelli.

Prima dell'uso su una nuova coorte è necessario validare almeno: richiamo degli
eventi clinicamente rilevanti, correttezza della prima evidenza, tasso di
fusione errata, separazione delle recidive, completezza delle citazioni e
discordanze inviate a revisione.

## Bonifica di progetti preesistenti

Lo strumento seguente analizza soltanto metadati legacy e opera in modalità
simulazione per impostazione predefinita:

```bash
python tools/redact_legacy_identity_payloads.py /percorso/progetto
python tools/redact_legacy_identity_payloads.py /percorso/progetto --apply
```

Con `--apply` crea prima un backup SQLite e un archivio ZIP dei JSON interessati.
Non scrive mai nei testi clinici normalizzati o nelle righe di laboratorio.
