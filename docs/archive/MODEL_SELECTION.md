# Matrice dei modelli

Queste sono candidature da confrontare sul gold set italiano del progetto, non
una graduatoria clinica generale. Il modello predefinito resta `qwen3-14b`
finché una valutazione riproducibile non dimostra un vantaggio sufficiente a
giustificare una modifica.

| Funzione | Candidato | Impostazione consigliata | Motivo e cautela |
|---|---|---|---|
| Estrazione atomica ad alto volume | Qwen3-14B GGUF | non-thinking, temperatura 0, richieste parallele per documento | Buon compromesso locale e multilingue; è il baseline già integrato. Verificare in particolare negazioni, italiano clinico e JSON. |
| Estrazione veloce su hardware limitato | Qwen3-8B GGUF | non-thinking, temperatura 0 | Da usare solo se il richiamo sul gold set resta accettabile. |
| Fusione, deduplicazione e query | Qwen3-30B-A3B-Instruct-2507 | contesto ampio, temperatura 0, parallelismo più basso | Il modello MoE ha 30B parametri totali ma 3B attivi; è un candidato efficiente per i passaggi più complessi, non un risultato già validato sul dominio. |
| Alternativa locale con output strutturato | gpt-oss-20b | ragionamento basso per estrazione, medio per fusione | Modello open-weight testuale, Apache 2.0, 131k di contesto e structured outputs. La compatibilità GGUF/backend e la qualità clinica italiana devono essere verificate prima di abilitarlo come preset. |
| Seconda lettura clinica | MedGemma 27B text-it | solo casi ambigui o discordanti | Specializzato su testo medico; non sostituisce la revisione e richiede una validazione specifica in italiano. |
| Immagini cliniche difficili, fase futura | MedGemma 1.5 4B multimodale | escalation dopo estrazione deterministica/OCR | Può aiutare su documenti e immagini, ma Google lo descrive come modello per sviluppatori non ancora clinical-grade. Non è collegato al flusso corrente. |
| Recupero semantico, fase futura | BGE-M3 + reranker | ricerca ibrida FTS/dense/sparse | Supporta oltre 100 lingue e testi fino a 8192 token. L'attuale registro usa FTS5; aggiungerlo solo se migliora richiamo e latenza sui quesiti reali. |
| Cloud opzionale | GPT-5.6 Luna / Terra / Sol | rispettivamente volume, bilanciamento, massima qualità | Non è abilitato: usarlo soltanto dopo valutazione privacy, base giuridica, accordi e de-identificazione adeguata. |

Fonti ufficiali consultate il 20 agosto 2026:

- [Qwen3: modelli, multilinguismo e supporto locale](https://github.com/QwenLM/Qwen3)
- [OpenAI gpt-oss-20b](https://developers.openai.com/api/docs/models/gpt-oss-20b)
- [OpenAI: confronto dei modelli correnti](https://developers.openai.com/api/docs/models/compare)
- [Google MedGemma](https://developers.google.com/health-ai-developer-foundations/medgemma)
- [BAAI BGE-M3](https://huggingface.co/BAAI/bge-m3)

## Protocollo di scelta

Per ogni candidato usare lo stesso set paziente, prompt versionato e schema.
Misurare separatamente:

1. richiamo e precisione delle evidenze atomiche;
2. accuratezza della prima data e della sua precisione;
3. fusioni corrette, fusioni indebite e separazione delle recidive;
4. copertura e validità delle citazioni;
5. aderenza alle formulazioni prudenti nelle correlazioni;
6. token al secondo, tempo/paziente, picco RAM/VRAM e fallimenti JSON.

La selezione può essere diversa per ogni ruolo. Il modello più grande va
riservato ai cluster ambigui o alla riconciliazione finale; usare lo stesso
modello per ogni fase spreca capacità e riduce il throughput.

Il backend espone anche la decodifica speculativa n-gram di llama.cpp. È
disattivata per impostazione predefinita: ogni token proposto viene verificato
dal modello principale, ma il guadagno dipende fortemente da build, modello e
tipo di JSON e va dimostrato con lo stesso dossier prima di abilitarla. Una
cascata 8B → 14B non viene attivata automaticamente finché il gold set non
dimostra che il filtro di escalation conserva il richiamo clinico.
