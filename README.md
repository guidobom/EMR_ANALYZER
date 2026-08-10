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
- normalizzazione conservativa del testo clinico con un modello Ollama locale;
- pseudonimizzazione deterministica prima e dopo l’elaborazione LLM;
- estrazione strutturata dei valori di laboratorio con unità, range, flag,
  data e pagina sorgente;
- ricostruzione temporale incrementale del Clinical State;
- interrogazione individuale o di coorti mediante modelli locali configurabili;
- revisione, validazione, audit trail ed esportazione Excel.

## Principi di sicurezza

L’applicazione è progettata per funzionare senza servizi cloud. PDF, database,
testi estratti, chiavi di identità e modelli Ollama rimangono sul computer
locale.

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
- [Ollama](https://ollama.com/) installato localmente;
- almeno un modello documentale e un modello per il Clinical State.

La configurazione dei modelli, della temperatura, del contesto e dell’output
avviene dall’interfaccia tramite **Configura LLM**.

## Installazione

```bash
conda create -n emr-analyzer python=3.12
conda activate emr-analyzer
pip install -r requirements.txt
```

Avvio:

```bash
python run.py
```

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

I modelli Ollama sono configurabili separatamente per:

1. isolamento del testo clinico dai singoli documenti;
2. costruzione e interrogazione del Clinical State.

Prompt, modello e parametri di generazione vengono versionati nei metadati e
nei log di audit per favorire la riproducibilità delle analisi.
