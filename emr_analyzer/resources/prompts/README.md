# Prompt LLM versionati

I file `.txt` di questa cartella sono le versioni **istituzionali** delle
istruzioni cliniche delle pipeline LLM. Non devono essere sovrascritti dalla
GUI: il Prompt Manager salva le modifiche come versioni personalizzate in
`~/.emr_analyzer/prompts/custom/<compito>/` e registra in
`~/.emr_analyzer/prompts/active_versions.json` quale versione usare per ogni
compito.

Il Prompt Manager è accessibile dal tasto `✎ Prompt` della barra principale o
da `Strumenti > Gestisci prompt LLM`. Permette anche una prova non distruttiva
su un documento importato: il risultato è mostrato nella finestra e non viene
salvato nel database.

Il codice continua a imporre schemi JSON, citazioni, provenienza, limiti e
validazioni: modificare un prompt non può disattivare tali controlli.

Riavviare l'applicazione dopo aver modificato o attivato una versione. L'hash
del contenuto invalida i checkpoint della sola fase interessata. Conservare i
nomi delle categorie e dei campi tecnici richiesti dal relativo schema e dal
manifest `manifest.json`.

I valori ematochimici fuori intervallo non usano un prompt LLM: sono prodotti
dalla pipeline deterministica `clinical/lab_evidence.py`, prioritariamente con
gli intervalli e i flag del singolo referto.
