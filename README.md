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
- [llama.cpp](https://github.com/ggml-org/llama.cpp) con modelli GGUF
  (`brew install llama.cpp` su macOS; build CUDA su Linux), oppure vLLM con
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
brew install llama.cpp    # solo macOS
```

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
identici su macOS e Linux. Su DGX Spark (DGX OS, arm64, Blackwell Ultra /
sm_100, 128 GB di memoria unificata):

1. installa una build CUDA di llama.cpp (nessun Homebrew):

   ```bash
   git clone --depth 1 https://github.com/ggml-org/llama.cpp
   cd llama.cpp
   cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON
   cmake --build build --config Release -j
   sudo cp build/bin/llama-server /usr/local/bin/
   ```

   oppure usa un container NGC con llama.cpp già compilato;

2. registra i GGUF in `~/.emr_analyzer/models/` (su DGX Spark non esiste
   lo storage Ollama: scarica direttamente i file GGUF);

3. `python tools/setup_llama_backend.py` stampa le stesse istruzioni quando
   il binario manca.

Il dimensionamento dei worker è automatico e basato sulla RAM. Il limite reale
va confermato con il benchmark e con dossier rappresentativi, perché contesto,
quantizzazione e cache del modello incidono sulla memoria per slot.

I dati runtime vengono salvati fuori dal repository in:

```text
~/.emr_analyzer/
```

## Test

```bash
conda activate emr-analyzer
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

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
├── rag/         # recupero delle evidenze per le query
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
