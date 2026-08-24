# Runtime llama-server gestito

EMR Analyzer può usare una copia privata e verificata di `llama-server`,
indipendente da Homebrew e dal `PATH`. Il runtime attivo è conservato in:

```text
~/.emr_analyzer/runtimes/llama.cpp/
```

L'app registra SHA-256, sistema operativo, architettura, versione, backend e
dispositivi esposti. La checksum viene ricontrollata prima della selezione. Su
macOS l'importazione fallisce se il processo non espone realmente un device
Metal; su Linux/NVIDIA è richiesto CUDA. Il semplice caricamento di una
libreria Metal/CUDA non è considerato sufficiente.

L'importazione è locale e non accede alla rete. Per ridurre i rischi della
catena di fornitura, clonare il repository ufficiale, scegliere esplicitamente
un tag/commit e annotarlo nel protocollo del progetto.

## macOS Apple Silicon — Metal

Prerequisiti una tantum:

```bash
xcode-select --install
brew install cmake git
```

Compilazione dal repository ufficiale:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git fetch --tags
git checkout <TAG_O_COMMIT_SCELTO>
cmake -S . -B build-emr \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DBUILD_SHARED_LIBS=OFF
cmake --build build-emr --config Release --target llama-server --parallel
./build-emr/bin/llama-server -v --list-devices
```

L'ultimo comando deve elencare un dispositivo `Metal`/`MTL`. Se mostra solo
`BLAS: Accelerate`, non importare quella build: userebbe la CPU.

## Ubuntu / NVIDIA, incluso DGX Spark — CUDA

Usare il CUDA Toolkit supportato dalla macchina; su DGX Spark è consigliabile
compilare direttamente sulla macchina così CMake seleziona l'architettura
nativa:

```bash
sudo apt update
sudo apt install -y build-essential cmake git
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git fetch --tags
git checkout <TAG_O_COMMIT_SCELTO>
cmake -S . -B build-emr \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DBUILD_SHARED_LIBS=OFF
cmake --build build-emr --config Release --target llama-server --parallel
./build-emr/bin/llama-server -v --list-devices
```

Il controllo finale deve elencare almeno un dispositivo `CUDA`. La build resta
dipendente dal driver e dal runtime CUDA installati sulla macchina, ma non da
una seconda installazione di llama.cpp.

## Importazione e utilizzo nell'app

Metodo grafico:

1. avvia EMR Analyzer;
2. apri **Configura LLM**;
3. nella sezione **llama-server gestito dall'applicazione**, premi
   **Importa llama-server…**;
4. seleziona `build-emr/bin/llama-server`;
5. inserisci la SHA-256 pubblicata, se il binario proviene da terzi; per una
   build compilata localmente puoi lasciare il campo vuoto;
6. conferma e attendi checksum, verifica e copia;
7. controlla che la sezione **Accelerazione hardware locale** sia verde;
8. scegli i modelli GGUF per i diversi ruoli e premi **Salva e applica**.

L'attivazione è immediata e i processi precedenti vengono fermati. Ai riavvii
successivi il runtime gestito ha precedenza su `LLAMA_SERVER_BINARY`, `PATH`,
Homebrew e `/usr/local/bin`.

Metodo da terminale:

```bash
python tools/setup_llama_backend.py \
  --server-binary /percorso/llama.cpp/build-emr/bin/llama-server
```

Per tornare temporaneamente al server di sistema, usare **Configura LLM → Usa
server di sistema**. La copia gestita rimane conservata e recuperabile; non
viene cancellata automaticamente.

## Controllo indipendente

```bash
~/.emr_analyzer/runtimes/llama.cpp/<piattaforma>/<installazione>/llama-server \
  -v --list-devices
```

Il percorso esatto e la checksum completa sono visibili in **Configura LLM**.
