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
- ricostruzione temporale incrementale del Clinical State;
- interrogazione individuale o di coorti mediante modelli locali configurabili;
- revisione, validazione, audit trail ed esportazione Excel.

## Principi di sicurezza

L’applicazione è progettata per funzionare senza servizi cloud. PDF, database,
testi estratti, chiavi di identità e modelli locali rimangono sul computer.
Il motore LLM è llama.cpp: l’applicazione avvia e ferma da sé il proprio
processo `llama-server`, senza alcun servizio esterno da tenere in vita.

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
- [llama.cpp](https://github.com/ggml-org/llama.cpp) installato
  (`brew install llama.cpp` su macOS; build CUDA su Linux, vedi sotto);
- almeno un modello documentale e un modello per il Clinical State in
  formato GGUF.

La configurazione dei modelli, della temperatura, del contesto e dell’output
avviene dall’interfaccia tramite **Configura LLM**.

## Installazione

```bash
conda create -n emr-analyzer python=3.12
conda activate emr-analyzer
pip install -r requirements.txt
brew install llama.cpp    # solo macOS
```

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

Il dimensionamento dei worker è automatico e basato sulla RAM: con 128 GB
il numero di slot paralleli cresce fino al massimo configurato (8), il
limite pratico del pool di estrazione.

I dati runtime vengono salvati fuori dal repository in:

```text
~/.emr_analyzer/
```

## Test

```bash
conda activate emr-analyzer
python -m unittest discover -s tests -v
```

La suite include test per parsing PDF, identificazione del paziente,
pseudonimizzazione, estrazione del laboratorio, gestione dei workspace,
configurazione LLM e validazione del testo clinico.

## Struttura

```text
emr_analyzer/
├── clinical/    # Clinical State, eventi e cancellazione clinica
├── database/    # SQLite, migrazioni e repository
├── extraction/  # LLM locale, laboratorio e normalizzazione
├── gui/         # interfaccia desktop PyQt5
├── models/      # modelli del dominio
├── pipeline/    # parsing PDF, routing e pseudonimizzazione
├── rag/         # recupero delle evidenze per le query
├── security/    # vincoli offline
└── utils/
```

## Modelli e riproducibilità

I modelli locali (GGUF serviti da llama-server) sono configurabili
separatamente per:

1. isolamento del testo clinico dai singoli documenti;
2. costruzione e interrogazione del Clinical State.

Prompt, modello e parametri di generazione vengono versionati nei metadati e
nei log di audit per favorire la riproducibilità delle analisi.
