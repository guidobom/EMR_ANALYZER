# EMR Analyzer

Applicazione desktop locale e offline che legge i referti anonimizzati dei
pazienti di un progetto, ne estrae gli eventi clinici ancorati al testo, li
codifica in SNOMED CT e li esporta in FHIR R4. Ogni evento resta collegato al
frammento del referto da cui proviene e può essere revisionato.

> **Stato del progetto:** sviluppo sperimentale. Non è un dispositivo medico e
> non deve essere utilizzato per decisioni cliniche senza verifica umana.

## Flusso di lavoro

1. **Progetto e pazienti.** Ogni progetto è una cartella con un database
   `emr_registry.db` e una cartella per paziente.
2. **Importazione.** I documenti (PDF, immagini, TXT/MD/HL7, CSV, XML/CDA, JSON,
   DOCX, XLSX) vengono attribuiti al paziente; le attribuzioni incerte si
   risolvono nella scheda **✓ Attribuzioni**.
3. **Anonimizzazione e testo clinico.** Estrazione PDF (`pdfplumber → PyMuPDF →
   OCR locale`), pseudonimizzazione deterministica e filtro del testo clinico.
   Il risultato è `extraction/DOC_….md`; le correzioni manuali del testo sono
   salvate come versioni (overlay) senza modificare l'originale.
4. **Laboratorio.** Il parser deterministico legge i risultati tabellari (valore,
   comparatore, unità, range, flag, campione, data) nella scheda **🔬 Laboratorio**.
5. **Estrazione degli eventi.** Un modello locale annota gli eventi dei referti
   narrativi: ogni evento cita il proprio frammento tramite indirizzi di parole
   verificati dal programma, con asserzione, certezza, soggetto, temporalità,
   date e attributi. I referti di laboratorio già letti dal parser non vengono
   inviati al modello.
6. **Codifica SNOMED CT.** Ogni concetto (etichetta normalizzata + tipo di evento)
   viene codificato una sola volta, scegliendo fra candidati del catalogo
   International filtrati per gerarchia; la codifica vale per tutte le sue
   occorrenze, in tutti i pazienti. I risultati tabellari restano su LOINC.
7. **Revisione.** Nella scheda **🧬 Eventi SNOMED** si ispezionano eventi e
   frammenti e si modifica tutto: codice (per occorrenza o per concetto),
   frammento, attributi, eventi mancanti o errati, testo del referto. La finestra
   **🧬 Concetti SNOMED** rivede le codifiche dell'intero progetto per frequenza.
8. **FHIR.** Un Bundle per paziente (`<paziente>/clinical_events.fhir.json`) e un
   export di progetto in NDJSON (**File → Esporta FHIR del progetto**).

**▶ Elabora pazienti** esegue i punti 5–8 per i pazienti selezionati con un'unica
coda: i documenti di tutti i pazienti condividono gli slot del modello e ogni
paziente viene chiuso, con il suo file FHIR, appena finiscono i suoi documenti.
Un'elaborazione interrotta riprende dai documenti e dai gruppi già completati.

## Dati e tracciabilità

| Livello | Dove | Contenuto |
|---|---|---|
| Occorrenze estratte | `clinical_evidence` (progetto) | Evento, citazione e intervallo esatto nel testo, qualificatori, date con provenienza |
| Codifiche per concetto | `concept_mappings` (Lessico condiviso) | Codice SNOMED proposto o confermato, candidati, esempi; una conferma non viene mai sovrascritta |
| Decisioni del revisore | `event_overrides` (progetto) | Conferma, correzione, scarto, evento aggiunto, codice della singola occorrenza; istantanea completa dell'evento |
| FHIR | file per paziente e NDJSON | Proiezione ricostruibile degli eventi revisionati e dei risultati di laboratorio |

Le decisioni sono legate all'occorrenza sorgente e sopravvivono a una nuova
estrazione. Se l'occorrenza cambia contenuto la conferma decade; se il testo del
referto cambia, il frammento viene ricollocato quando è univoco, altrimenti
l'evento è segnalato «da ricontrollare». Gli eventi scartati non entrano nel FHIR;
le estensioni `review-status` e `coding-status` indicano lo stato di revisione.
Dettagli in [docs/FHIR_EVENT_REGISTRY.md](docs/FHIR_EVENT_REGISTRY.md).

## Terminologie

- **SNOMED CT International** (RF2 Snapshot): importare il pacchetto da
  **Lessico condiviso → Catalogo SNOMED CT…**. La distribuzione richiede una
  licenza SNOMED (Affiliate o tramite il National Release Center) e non è inclusa
  nel repository. Vedi [docs/SNOMED_INTERNATIONAL_RF2.md](docs/SNOMED_INTERNATIONAL_RF2.md).
- **LOINC**: importare `Loinc.csv` o lo ZIP ufficiale da **Lessico condiviso →
  Catalogo LOINC…**; associazioni confermate legate ad analita, campione e unità.

Il **Lessico condiviso** raccoglie esempi e controesempi usati come guida
dell'estrazione: [guida all'annotazione](docs/LOCAL_LEXICON.md).

## Principi di sicurezza

L'applicazione funziona senza servizi cloud: PDF, database, testi estratti,
chiavi di identità e modelli rimangono sul computer. Il processo Python blocca le
connessioni di rete non locali; i modelli girano con llama.cpp (oppure vLLM su
Linux/NVIDIA) gestiti dall'applicazione.

Il repository non deve contenere dati sanitari reali. La `.gitignore` esclude
PDF, immagini, database, workspace, cache, log, chiavi, pesi dei modelli, le
distribuzioni SNOMED/LOINC e la cartella `.clinical-evaluation/`. Nei test si
usano solo dati sintetici.

## Requisiti

- macOS o Linux; Python 3.12; ambiente Conda consigliato;
- [llama.cpp](https://github.com/ggml-org/llama.cpp) con modelli GGUF e un
  runtime Metal/CUDA verificato dall'applicazione, oppure vLLM con modelli
  Hugging Face locali su Linux/NVIDIA.

Due ruoli LLM si configurano da **Configura LLM**: *documenti* (normalizzazione
dei referti) ed *estrazione e codifica*. Per l'estrazione il preset è
temperatura 0,7 con top_p 0,8 e top_k 20 — i valori raccomandati per i modelli
Qwen3, misurati sul campione revisionato: la decodifica greedy produce
ripetizioni e violazioni dello schema. Più slot paralleli aiutano il
throughput; l'app chiede conferma sopra 0,8.

## Installazione

```bash
conda create -n emr-analyzer python=3.12
conda activate emr-analyzer
pip install -r requirements.txt
```

Su DGX OS/Ubuntu possono servire le librerie di sistema Qt/XCB:

```bash
sudo apt install -y libegl1 libgl1 libxcb-cursor0 libxkbcommon-x11-0
```

Runtime `llama-server` gestito: [docs/LLAMA_SERVER_RUNTIME.md](docs/LLAMA_SERVER_RUNTIME.md).
I GGUF vanno registrati in `~/.emr_analyzer/models/` (lo script
`python tools/setup_llama_backend.py` importa quelli già presenti nell'archivio
Ollama). vLLM su DGX Spark: [docs/VLLM_DGX_SPARK.md](docs/VLLM_DGX_SPARK.md).

Avvio:

```bash
python run.py        # oppure ./run.sh
```

Per elaborazioni lunghe su macOS conviene impedire lo stop del sistema, ad
esempio avviando l'app con `caffeinate -di python run.py`.

I dati di configurazione e i cataloghi condivisi sono in `~/.emr_analyzer/`.

## Struttura

```text
emr_analyzer/
├── clinical/    # estrazione eventi, codifica SNOMED, revisione, FHIR
├── database/    # SQLite, migrazioni e repository
├── extraction/  # client LLM, laboratorio, testo clinico
├── gui/         # interfaccia desktop PyQt5
├── llm_backend/ # runtime e gestione dei modelli locali
├── models/      # modelli del dominio
├── pipeline/    # parsing PDF, attribuzione e pseudonimizzazione
├── security/    # vincoli offline
└── utils/
```

## Test

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests
```

La suite usa progetti temporanei, un catalogo SNOMED sintetico e modelli
simulati: verifica orchestrazione, codifica, revisione, FHIR e interfaccia, non
la qualità clinica dei modelli reali. Per quella esiste una valutazione su un
campione revisionato, tenuta fuori dal repository.

## Limiti noti

- La qualità dell'estrazione dipende dal modello locale e va misurata su
  referti revisionati; nessuna configurazione è validata clinicamente.
- La ricerca dei candidati SNOMED parte da etichette italiane su un catalogo
  inglese (traduzione della query con il modello); i concetti senza candidato
  adatto restano da codificare a mano.
- Il FHIR è controllato strutturalmente in locale; non è dichiarata conformità
  a un Implementation Guide.

I documenti delle versioni precedenti dell'applicazione sono in
[docs/archive/](docs/archive/).
