# Backend vLLM su DGX Spark

EMR Analyzer può assegnare indipendentemente a ciascun ruolo uno dei due
motori locali:

- `llama_cpp`: modelli GGUF, supportato sia su macOS/Metal sia su Linux/CUDA;
- `vllm`: checkpoint Hugging Face, supportato su Linux/NVIDIA CUDA e pensato
  per DGX Spark.

I processi sono gestiti dall'applicazione e ascoltano soltanto su
`127.0.0.1`. Quando avvia vLLM, l'app imposta `HF_HUB_OFFLINE=1` e
`TRANSFORMERS_OFFLINE=1`: un'inferenza non può scaricare pesi mancanti.

## Preparazione

Eseguire la diagnostica nell'ambiente Python con cui viene avviata l'app:

```bash
python tools/setup_vllm_backend.py
```

L'installazione è volontaria e separata dalle dipendenze macOS:

```bash
python tools/setup_vllm_backend.py --install
```

Questo crea l'ambiente isolato `~/.emr_analyzer/vllm-env` e usa il comando
raccomandato `uv pip install vllm --torch-backend=auto`. In questo modo le
dipendenze CUDA/PyTorch di vLLM non sostituiscono quelle dell'applicazione.
`uv` deve essere già installato: lo script non ripiega su una compilazione
sorgente implicita. L'app rileva automaticamente il comando `vllm`
nell'ambiente isolato. Se la combinazione di DGX OS, CUDA e architettura ARM64
richiede un wheel specifico, installarlo nello stesso ambiente isolato.

Se il comando `vllm` risiede in un ambiente separato ma compatibile, è
possibile indicarne il percorso assoluto prima di avviare l'app:

```bash
export EMR_ANALYZER_VLLM_BINARY=/percorso/ambiente/bin/vllm
```

Scaricare esplicitamente il modello desiderato nella cache Hugging Face,
oppure predisporre una directory locale completa:

```bash
python tools/setup_vllm_backend.py \
  --model Organizzazione/NomeModello --download-model
```

Il controllo successivo è offline e rifiuta cache parziali prive dei pesi:

```bash
python tools/setup_vllm_backend.py --model Organizzazione/NomeModello
```

## Configurazione nell'app

In **Configura LLM**, per ogni ruolo:

1. selezionare **vLLM · CUDA (DGX / Linux)**;
2. scegliere un modello rilevato nella cache locale;
3. impostare precisione, quota di memoria GPU, tensor parallel e, soltanto se
   necessario, quantizzazione;
4. premere **Carica e testa**.

Valori iniziali prudenti per una DGX Spark a GPU singola sono `bfloat16` o
`auto`, quota memoria GPU `0,85–0,90`, tensor parallel `1` e quantizzazione
lasciata al modello. Il numero di richieste parallele diventa
`--max-num-seqs`; aumentarlo solo dopo una prova reale con i documenti più
lunghi.

`Consenti codice remoto già locale` corrisponde a `--trust-remote-code` e va
abilitato solo per repository verificati. Non è necessario per la maggior
parte dei modelli standard.

## Contratto applicativo

Il backend espone `/v1/chat/completions`. Le estrazioni strutturate usano
`response_format.type=json_schema` con schema rigoroso; testo libero,
temperature, seed, top-p, top-k e limite di output rimangono parametri della
singola richiesta. Modello, contesto, concorrenza e parametri motore
identificano invece un processo fisico condivisibile fra ruoli.

llama.cpp e vLLM possono essere usati contemporaneamente da ruoli diversi.
**Libera tutti i modelli** arresta soltanto i processi figli avviati
dall'applicazione e non tocca container o servizi esterni.
