# EMR Analyzer

Applicazione desktop locale e offline per trasformare dossier clinici PDF in
un database longitudinale interrogabile a scopo di ricerca, audit e revisione
della pratica clinica.

> **Stato del progetto:** sviluppo sperimentale. Non è un dispositivo medico e
> non deve essere utilizzato per decisioni cliniche senza verifica umana.

## Obiettivi

- una workspace indipendente per ciascun paziente;
- importazione e attribuzione automatica dei documenti al paziente;
- estrazione PDF tramite `pdfplumber → PyMuPDF → OCR locale`;
- normalizzazione conservativa del testo clinico con un modello locale
  (llama.cpp);
- pseudonimizzazione deterministica prima e dopo l’elaborazione LLM;
- estrazione strutturata dei valori di laboratorio con unità, range, flag,
  data e pagina sorgente;
- registro clinico evidence-first con prima evidenza, prima documentazione,
  precisione temporale e provenienza verificabile;
- fusione delle fonti duplicate, separazione delle recidive e conservazione
  delle informazioni contraddittorie;
- trend di laboratorio, corsi farmacologici e linee di terapia oncologica;
- correlazioni cliniche prudenti fra sintomi, laboratorio, imaging e terapie;
- interrogazione individuale o di coorti mediante modelli locali configurabili;
- revisione persistente, audit append-only ed esportazione JSON, CSV, XLSX,
  Markdown, TXT, PDF e DOCX.
- gold set clinico con doppia annotazione cieca, adjudication, blocco del test,
  confronto con le predizioni ed export JSONL.

## Principi di sicurezza

L’applicazione è progettata per funzionare senza servizi cloud. PDF, database,
testi estratti, chiavi di identità e modelli locali rimangono sul computer.
I motori LLM locali sono llama.cpp e, opzionalmente su Linux/NVIDIA, vLLM:
l’applicazione avvia e ferma da sé i propri processi, senza alcun servizio
esterno da tenere in vita.

Il repository non deve contenere dati sanitari reali. La `.gitignore` esclude
per impostazione predefinita:

- PDF, immagini e file DICOM;
- database SQLite ed esportazioni Excel/CSV;
- workspace, cache, log e livelli di estrazione;
- chiavi, file `.env` e pesi dei modelli.

Usare nei test pubblicabili esclusivamente documenti sintetici.

## Requisiti

- macOS o Linux;
- Python 3.12;
- ambiente Conda consigliato;
- [llama.cpp](https://github.com/ggml-org/llama.cpp) con modelli GGUF e un
  runtime Metal/CUDA importato e verificato dall'applicazione, oppure vLLM con
  modelli Hugging Face già locali su Linux/NVIDIA.

La configurazione dei modelli, della temperatura, del contesto e dell’output
avviene dall’interfaccia tramite **Configura LLM**.

I ruoli condividono un solo processo e una sola copia dei pesi quando usano lo
stesso backend e modello con uguali parametri motore (contesto e concorrenza;
per vLLM anche precisione, quota GPU e tensor parallel). Temperatura,
top-p/top-k, seed e limite di output restano
indipendenti perché sono parametri della singola richiesta. La finestra mostra
se i server fisici sono condivisi o distinti, gli slot realmente caricati e
una stima complessiva della memoria; cambiando contesto o slot, **Salva e
applica** sostituisce il runtime precedente.

## Installazione

```bash
conda create -n emr-analyzer python=3.12
conda activate emr-analyzer
pip install -r requirements.txt
```

Su DGX OS/Ubuntu possono servire anche le librerie di sistema Qt/XCB:

```bash
sudo apt update
sudo apt install -y libegl1 libgl1 libxcb-cursor0 libxkbcommon-x11-0
```

La compilazione e l'importazione del runtime `llama-server` gestito sono
descritte in
[docs/LLAMA_SERVER_RUNTIME.md](docs/LLAMA_SERVER_RUNTIME.md). Non affidarsi a
una build Homebrew senza verificare che esponga realmente Metal.

Evitare di mescolare nella stessa environment i runtime Qt forniti da Conda e
quelli installati da `pip`: scegliere una sola distribuzione PyQt5. L'app prova
comunque a risolvere automaticamente il percorso dei plugin Qt di Conda.

Registrazione dei modelli: se Ollama è (stato) installato, lo script di setup
copia i GGUF già presenti in `~/.ollama/models/blobs` in
`~/.emr_analyzer/models/` senza scaricare nulla:

```bash
python tools/setup_llama_backend.py
```

In alternativa, scarica un GGUF (ad es. `qwen3-14b`) e registralo
manualmente in `~/.emr_analyzer/models/` con `index.json`.

Avvio:

```bash
python run.py
```

Anche `./run.sh` è multipiattaforma: usa il `python3` dell'ambiente attivo.
Per un launcher desktop è possibile fissare l'interprete con
`EMR_ANALYZER_PYTHON=/percorso/env/bin/python3 ./run.sh`.

### Linux / NVIDIA DGX Spark

Su DGX Spark è possibile scegliere per ogni ruolo sia llama.cpp sia vLLM.
Per vLLM, eseguire prima la diagnostica non invasiva:

```bash
python tools/setup_vllm_backend.py
```

L'installazione esplicita (`--install`), la preparazione offline dei modelli e
i parametri consigliati sono descritti in
[docs/VLLM_DGX_SPARK.md](docs/VLLM_DGX_SPARK.md).

Il backend llama.cpp dell’app è indipendente dalla piattaforma: la gestione
dei processi, le porte, il parallelismo multi-slot e i modelli GGUF sono
identici su macOS e Linux. Su DGX Spark (DGX OS, Linux `aarch64`, Grace
Blackwell GB10 / compute capability 12.1, `sm_121`, 128 GB di memoria
coerente unificata):

1. compila una build CUDA autosufficiente di llama.cpp:

   ```bash
   git clone --depth 1 https://github.com/ggml-org/llama.cpp
   cd llama.cpp
   cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
     -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF
   cmake --build build --config Release -j
   python tools/setup_llama_backend.py --server-binary build/bin/llama-server
   ```

   oppure usa un container NGC con llama.cpp già compilato;

2. registra i GGUF in `~/.emr_analyzer/models/` (su DGX Spark non esiste
   lo storage Ollama: scarica direttamente i file GGUF);

3. `python tools/setup_llama_backend.py` stampa le stesse istruzioni quando
   il binario manca.

Il dimensionamento dei worker riconosce DGX Spark e usa la RAM di sistema,
non il contatore VRAM di `nvidia-smi` (che su GB10 può risultare non
supportato). Conserva almeno il 10%/12 GiB per sistema, applicazione e CUDA;
il limite reale va confermato con il benchmark e con dossier rappresentativi,
perché contesto, quantizzazione e cache del modello incidono sulla memoria per
slot.

I dati runtime vengono salvati fuori dal repository in:

```text
~/.emr_analyzer/
```

## Domande generiche su pazienti e coorti

Da **Strumenti → Interroga referti: paziente o coorte...** puoi selezionare
uno o più pazienti e applicare lo stesso prompt ai Markdown clinici attivi (`DOC_….md`),
anche senza aver costruito il registro clinico. Il modello di analisi locale
legge i referti per blocchi; le citazioni testuali restituite vengono verificate
contro il testo sorgente. Documenti mancanti, pagine non verificabili ed errori
sono indicati nella copertura del report.

Report individuali JSON/Markdown e report di coorte vengono salvati in
`<workspace>/query_reports/<timestamp_id>/`. La copertura di elaborazione non
misura la sensibilità clinica e le sintesi richiedono revisione. I file `*_raw.md`, `*_source.txt` e `*_cleaned_source.md` non vengono usati
come ripiego. Se manca il Markdown attivo, completare la preparazione del referto.
Le citazioni sono verificate sul Markdown attivo; ciò non certifica la sua
completezza rispetto al PDF né la sua anonimizzazione.

La [revisione del codice e della pipeline](docs/CODE_AUDIT_2026-09-09.md)
descrive correzioni, limiti, duplicazioni e priorità successive.

## Test

```bash
conda activate emr-analyzer
python -m pip install -r requirements-dev.txt
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

`tests/conftest.py` è un file di configurazione caricato automaticamente da
pytest: non deve essere eseguito direttamente con Python.

La suite include inoltre datazione retrospettiva, deduplicazione, recidive,
correlazioni tiroidee e respiratorie, provenienza, revisioni persistenti,
formati di input, export, audit, flusso gold set e metriche di valutazione.

## Struttura

```text
emr_analyzer/
├── clinical/    # evidenze, episodi, registro, correlazioni e query
├── database/    # SQLite, migrazioni e repository
├── evaluation/  # metriche e runner per gold set a livello paziente
├── export/      # export completi e verificabili del registro
├── extraction/  # LLM locale, laboratorio e normalizzazione
├── gui/         # interfaccia desktop PyQt5
├── models/      # modelli del dominio
├── pipeline/    # parsing PDF, routing e pseudonimizzazione
├── security/    # vincoli offline
└── utils/
```

## Modelli e riproducibilità

I modelli locali (GGUF tramite llama.cpp oppure checkpoint Hugging Face tramite
vLLM) sono configurabili
separatamente per:

1. isolamento del testo clinico dai singoli documenti;
2. estrazione delle evidenze atomiche;
3. fusione e assemblaggio degli eventi/episodi clinici;
4. analisi e interrogazione longitudinale del registro.

Le impostazioni precedenti a quattro ruoli vengono migrate automaticamente:
il vecchio modello Clinical State è assegnato inizialmente a evidenze, eventi
e analisi, senza invalidare i checkpoint atomici già compatibili.

Da **Configura LLM → Scarica o importa modelli** è possibile:

- importare selettivamente un GGUF già presente nell'archivio Ollama;
- chiedere a Ollama di scaricare un nuovo tag e registrarlo;
- scaricare direttamente un singolo file GGUF da un URL HTTPS, verificando
  facoltativamente la checksum SHA-256.

Download e copia mostrano l'avanzamento, possono essere interrotti e usano un
file temporaneo che viene rinominato soltanto dopo la validazione. I modelli
sono registrati in `~/.emr_analyzer/models/index.json` e diventano subito
selezionabili. Il processo clinico resta offline: il processo separato di
installazione riceve soltanto identificativo/URL del modello e non accede ai
workspace. I GGUF suddivisi in più file non sono ancora installabili dalla
finestra.

Prompt, modello e parametri di generazione vengono versionati nei metadati e
nei log di audit per favorire la riproducibilità delle analisi. La matrice dei
modelli e il protocollo di benchmark sono in
[docs/MODEL_SELECTION.md](docs/MODEL_SELECTION.md).

## Registro clinico v2

La descrizione di schema, datazione, fusione, revisione, query, valutazione e
bonifica dei metadati legacy è in
[docs/CLINICAL_REGISTRY_V2.md](docs/CLINICAL_REGISTRY_V2.md).

Il registro viene costruito a valle dell'estrazione: i testi normalizzati e le
righe di laboratorio preesistenti non vengono modificati. Le decisioni manuali
sono overlay tracciati e sopravvivono alle ricostruzioni automatiche.

I nuovi Markdown normalizzati riportano una testata deterministica con la data
del referto (`document_date`), distinta dalle date degli eventi. Date assenti o
non valide sono indicate come non disponibili. L’interrogazione riceve questo
metadato dal database per ogni frammento, anche per i Markdown precedenti.
Per aggiornare i file esistenti: `python tools/add_report_dates.py /percorso/progetto`
mostra il numero di file; aggiungere `--apply` per applicare la modifica con backup
in `metadata_backups/`. Sono inclusi solo i documenti con metadati di anonimizzazione.


## Estrazione essenziale: PDF originale e Markdown clinico

L’estrazione legge ogni originale una volta per elaborazione e mantiene testo,
tabelle e geometria in memoria. Salva esclusivamente `extraction/DOC_….md`,
con data del referto e riferimenti alle pagine quando il parser li fornisce.
Non produce più `_raw.md`, `_source.txt`, `_cleaned_source.md`, `.json`,
`_pages.jsonl`, `_words.jsonl` o `_tables.json` per ciascun documento.
I metadati di elaborazione rimangono nel database del progetto.

Il testo viene anonimizzato e filtrato per pagina con regole deterministiche:
le porzioni conservate sono copiate dal testo anonimizzato senza riscrittura LLM.
Righe amministrative riconosciute (anagrafica, recapiti, intestazioni istituzionali,
firme digitali, informative privacy) vengono eliminate. Le righe miste con
contenuto clinico o ambiguo vengono conservate e segnalate per revisione: non è
possibile garantire automaticamente l’eliminazione di ogni amministrativo senza
rischiare omissioni cliniche. Risultati e righe numeriche cliniche sono conservati.
Il database registra intervalli conservati/rimossi e hash del testo anonimizzato
per pagina (`clinical_text.retention_audit`), senza copie del testo identificativo.
Il filtro v2 riconosce anche celle amministrative affiancate a note o terapie:
un recapito nella colonna del personale non deve eliminare la posologia nella
colonna accanto. Rimuove margini vuoti e righe di spazi, conservando la spaziatura
interna delle tabelle; l'audit distingue esclusioni amministrative e formattazione.
L'anonimizzazione v2 protegge le date complete dalle false identificazioni come
numeri telefonici e non unisce numeri su righe diverse.
Il filtro v3 aggiunge il riconoscimento di avvertenze amministrative su più righe,
firme nel loro contesto e moduli generici italiani (anche ASL, ASST, IRCCS e case
di cura). Nei PDF con una colonna del personale chiaramente identificabile,
usa le coordinate delle parole per separarla dalla colonna clinica, preservando
le celle della terapia e registrando gli indici esclusi nell'audit. Non applica
un ritaglio fisso a tutti i documenti. L'anonimizzazione v3 riconosce anche le
varianti della data di nascita nota, ad esempio ISO e giorno/mese senza zeri.
Docling rimane un parser di ripiego; la normalizzazione ne conserva ora la
provenienza per pagina. Le sue intestazioni sono esaminate dal filtro, perché
possono contenere date cliniche. Il livello BODY di Docling non equivale a
contenuto clinico: può contenere segreterie e altre informazioni amministrative.
Il filtro v4 produce Markdown continuo, senza separatori di pagina, e rimuove
timestamp isolati ripetuti in testa alle pagine. I nomi dei sanitari introdotti
da titoli o campi espliciti sono sostituiti con `[MEDICO]`; le indicazioni cliniche
che li accompagnano rimangono. La mappa pagina/intervalli del testo finale e il
suo hash restano in `clinical_text.retention_audit` nel database. Le interrogazioni
verificano l'hash prima di usare questa mappa per le citazioni, continuando a
supportare i vecchi Markdown con separatori. Non vengono creati file aggiuntivi.

Per ispezionare l'estrazione, selezionare un documento e premere **Confronta PDF
e Markdown** (disponibile anche nel menu contestuale). La finestra mostra PDF e
Markdown affiancati, con pannelli ridimensionabili, navigazione e zoom del PDF,
ricerca nel Markdown e ricaricamento del testo. L'ispezione è in sola lettura.
I Markdown prodotti con il filtro precedente devono essere riestratti dai PDF
per recuperare eventuali date oscurate o righe cliniche omesse. Le impaginazioni
con testo già sovrapposto tra colonne possono ancora richiedere revisione.

La coda **Non normalizzati** include anche i documenti di laboratorio. Parsing e
filtraggio sono un unico passaggio; la concorrenza usa il numero di worker
configurato per il ruolo documentale. La selezione del testo non richiede un modello.
L’eventuale verifica di attribuzione del documento può ancora usare il modello
configurato; interrogazioni e analisi successive continuano a usare i propri LLM.
La rielaborazione riparte dal PDF originale. Le geometrie per le evidenze sono
ricostruibili dal PDF senza salvare copie JSON. I vecchi prompt di riscrittura del
testo clinico non sono più esposti nella finestra dei prompt attivi.

I test automatici verificano file prodotti, anonimizzazione su esempi sintetici,
provenienza, errori e integrazione. Non dimostrano la sensibilità clinica del modello
locale sul corpus reale e non costituiscono un benchmark di velocità.
