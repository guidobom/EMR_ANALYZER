# Revisione del codice e della pipeline — 9 settembre 2026

## Esito e perimetro

Il progetto dispone di componenti utili per parsing locale, attribuzione dei documenti, evidenze, registro, revisione e valutazione. Il limite architetturale principale rispetto all'obiettivo richiesto era l'uso del registro derivato come sostituto della documentazione completa nelle domande generiche. Un'informazione omessa dalla normalizzazione, dall'estrazione atomica o dall'aggregazione non poteva essere recuperata dalla chat del registro.

La revisione comprende inventario AST di tutti i moduli Python dell'applicazione, controllo Pyflakes, ricerca di corpi duplicati e codice irraggiungibile, lettura mirata dei flussi di importazione, parsing, persistenza, estrazione, query, GUI, report e valutazione, ed esecuzione della suite. **Non equivale a una lettura manuale riga per riga né a una certificazione di assenza di bug.** L'inventario completo riproducibile è in [CODE_INVENTORY.json](CODE_INVENTORY.json); i rilievi statici residui sono in [PYFLAKES_AUDIT.txt](PYFLAKES_AUDIT.txt).

Sono state conservate le numerose modifiche locali già presenti all'inizio. Non sono stati aperti referti reali, database di produzione o gli Excel presenti nella radice; non sono state eseguite migrazioni sui dati utente. Nessuna modifica è stata pubblicata o committata.

## Problemi verificati e interventi

| Priorità | Difetto e impatto | Intervento |
|---|---|---|
| Alta | La query sulle sole evidenze importava `build_query_prompt` da un modulo che non lo definisce: errore quando il registro è vuoto. | Rimosso l'import errato in `gui/workers.py`; test sul percorso reale della cronologia. |
| Alta | Recupero limitato a 500 eventi, filtro per intenti e taglio della cronologia al primo superamento del budget: parti del dossier escluse dalla domanda. | Recupero completo del registro per impostazione predefinita; cronologia atomica interrogata integralmente per blocchi, senza filtro euristico di categoria. Il limite esplicito rimane disponibile per chiamanti che lo richiedono. |
| Alta | Riparazione citazioni concatenava tutti i blocchi; riduzione delle sintesi forzava coppie oltre budget; budget minimo imposto anche con contesto insufficiente. | Budget condiviso conservativo, nessuna ricomposizione forzata oltre soglia, fallback alle risposte separate o agli eventi citabili. Contesto insufficiente segnalato esplicitamente. |
| Alta | `DatabaseEngine.commit()` poteva confermare scritture di repository dentro una transazione esterna, compromettendo rollback e savepoint. | Il commit del repository viene differito fino all'uscita dalla transazione gestita esterna. Test con errore tardivo e savepoint annidati. |
| Alta | Il controllo citazioni verificava evento/documento ma non la pagina, e riconosceva solo una parte degli ID inventati. | Controllo esteso alle pagine fornite dal registro e agli ID estranei. Rimane un controllo di provenienza, non di correttezza semantica. |
| Alta | La cancellazione poteva raccogliere un originale o artefatto esterno alla directory del paziente tramite un percorso legacy o un link. | La raccolta ora blocca percorsi risolti esterni prima di spostare o eliminare file; test con originale esterno. |
| Media | Ordinamento lessicografico degli ID: dopo `P1000`, `P999` poteva tornare a essere il massimo e generare un duplicato. Analogo confine per documenti. | Ordinamento numerico del suffisso, con esclusione degli ID non conformi. **Non risolve l'allocazione simultanea in più thread**, indicata sotto. |
| Media | Decodifica UTF-16 tentata su testi Windows-1252 senza BOM: possibile testo apparentemente valido ma corrotto. | UTF-16 selezionato soltanto con BOM; test con testo accentato e numero pari di byte. |
| Media | `chunk_entries(overlap=0)` riutilizzava tutta la lista per effetto di `[-0:]`; anche overlap grandi potevano eccedere il budget. | Zero overlap corretto, overlap ridotto per rientrare nel budget, errore esplicito per una singola voce troppo grande. |
| Media | L'interfaccia offriva v4, ma `registry_builder` eseguiva sempre il grafo v3. Anche il messaggio di assemblaggio LLM descriveva un passaggio non eseguito. | Rimossa l'opzione non collegata; metadati distinguono motore effettivo v3 da eventuale richiesta legacy v4; messaggio di avanzamento corretto. Non è stato attivato un nuovo aggregatore senza verifica. |
| Bassa | Un secondo `return ExcludedEvidence(...)` dopo `return result` era irraggiungibile e conteneva nomi fuori scope. | Rimosso soltanto il blocco morto. La funzione valida `_excluded_record` è conservata. |
| Bassa | Normalizzazione Unicode identica nel grafo e nella terminologia; import inutilizzati nei moduli modificati. | Il grafo riutilizza `normalize_concept`; rimossi import sicuramente superflui. Sistemata anche la visibilità dell'annotazione `Path` in `documents_tab`. |

La query di registro senza fonti e senza evidenze di fallback ora restituisce indisponibilità dei dati senza chiedere al modello di generare una risposta da un vecchio profilo.

## Percorso nuovo: domande generiche sui referti

Disponibile dal menu **Strumenti → Interroga referti: paziente o coorte...**.

1. Selezionare un paziente, una selezione oppure tutti.
2. Scrivere qualsiasi domanda clinica relativa ai referti.
3. Avviare l'analisi. Lo stesso prompt viene applicato separatamente a ogni paziente.
4. Consultare report individuali, sintesi di coorte, citazioni e copertura di elaborazione.

Implementazione: [dossier_query.py](../emr_analyzer/clinical/dossier_query.py), [query_context.py](../emr_analyzer/clinical/query_context.py), [dossier_query_dialog.py](../emr_analyzer/gui/dossier_query_dialog.py).

Il servizio legge esclusivamente i Markdown clinici attivi (`DOC_….md`) nelle directory `extraction` o `docling` del paziente. I file grezzi e intermedi non vengono usati come ripiego: possono contenere dati identificativi e boilerplate. I documenti senza Markdown attivo sono segnalati come mancanti. La versione v2 registra questa politica nel report; non certifica retroattivamente anonimizzazione o completezza dei Markdown esistenti. Non dipende dal registro o dalle categorie irAE.

Ogni pagina viene suddivisa in frammenti con una piccola sovrapposizione, lasciando spazio per prompt e output. Il modello restituisce osservazioni e citazioni testuali in JSON. Le citazioni devono comparire nel frammento originale, tollerando solo differenze di spaziatura; gli errori o le citazioni inventate rendono incompleta la copertura del paziente. I riferimenti comprendono paziente implicitamente nell'identificativo hash, documento, pagina quando nota, numero del frammento e hash del Markdown attivo. Le query individuali non ricevono dati di altri pazienti; solo la successiva sintesi di coorte riceve i risultati distinti per paziente.

Le osservazioni e le citazioni sono mantenute anche nei report, indipendentemente dalla sintesi narrativa. I report sono salvati progressivamente per paziente in:

```text
<workspace>/query_reports/<timestamp_id>/
    manifest.json
    P001.json
    P001.md
    P002.json
    P002.md
    cohort.md
```

Il manifest contiene domanda, pazienti selezionati/completati, errori e stato. I JSON individuali conservano hash del prompt, modello, contesto, fonti, osservazioni e copertura. Un'interruzione cooperativa conserva i pazienti già salvati; la richiesta LLM in corso può dover terminare prima dell'arresto. I report sono snapshot: dopo modifiche ai documenti vanno rigenerati.

La chat precedente resta disponibile per il registro; il nuovo comando è il percorso da usare quando si vuole leggere la documentazione clinica in Markdown.

## Duplicazioni e codice non utilizzato

Dopo l'intervento restano cinque gruppi di corpi identici rilevati dall'inventario:

- normalizzazione in `clinical/concept_canonicalization.py` e `tools/duplicate_audit.py`;
- gestione selezione in `gui/irae_queue_dialog.py` e `gui/registry_queue_dialog.py`;
- apertura PDF in `gui/normalization_dialog.py` e `gui/validation_tab.py`;
- risoluzione Quick Look negli stessi due moduli;
- arresto runtime alternativi in `llm_backend/backend.py` e `llm_backend/vllm_backend.py`.

Sono candidati a helper condivisi piccoli, senza unificare indiscriminatamente le regole cliniche. Il rilevatore cerca corpi AST identici oltre una soglia di dimensione: non identifica tutte le duplicazioni concettuali.

`aggregate_v4.py` e `ClinicalEpisodeSynthesizer` hanno implementazioni e test, ma non erano chiamati dal percorso corrente di costruzione del registro; gli import non costituivano un collegamento operativo. Sono stati mantenuti come prototipi, rendendo l'interfaccia aderente al comportamento effettivo. `clinical_history_builder.py` e parti di `llm_client.py` conservano percorsi legacy effettivamente raggiungibili quando i servizi nuovi non vengono iniettati: non vanno eliminati soltanto perché non sono il percorso predefinito.

Il rapporto Pyflakes conserva anche import usati per verificare la disponibilità di dipendenze (`docling`, `pdfplumber`): **non tutti gli avvisi “unused” sono codice eliminabile**. Callback Qt, API pubbliche e import di compatibilità richiedono controllo dei chiamanti e dei test. Dopo la pulizia non risultano blocchi direttamente successivi a `return`/`raise` incondizionati nel corpo delle funzioni; questo non prova l'assenza di altri rami morti.

## Problemi residui e priorità architetturali

**1. Accuratezza da misurare, non da dedurre dalla copertura.** Un modello può omettere un fatto e restituire JSON valido, oppure associare una citazione vera a una conclusione errata. La verifica testuale non controlla implicazione logica, negazione o causalità. Anche una sintesi con ID validi può contenere affermazioni non sostenute. Occorrono annotazioni indipendenti su referti rappresentativi e una verifica per singola affermazione; il nuovo percorso non garantisce sensibilità massima.

**2. Evidenze distribuite su documenti diversi.** La lettura per frammenti può perdere relazioni riconoscibili soltanto unendo documenti lontani o conoscendo il trattamento precedente. La fase successiva consigliata è pianificazione della domanda in sottoquesiti, raccolta di osservazioni anche contestuali, ricostruzione temporale deterministica e seconda lettura mirata delle fonti. Non limitare questa ricerca ai soli top-k di un indice vettoriale.

**3. Tokenizzazione esatta e richieste grandi.** Il budget in caratteri è una stima prudente condivisa, non un limite matematico sui token. Va collegato alla tokenizzazione effettiva del backend per il modello caricato. La pipeline nuova registra errori di richiesta e preserva i risultati verificati; il registro usa fallback espliciti. Restano percorsi legacy da migrare sullo stesso budget, incluso il limite alle ultime 100 voci nella chat senza repository.

**4. Ripresa e cache delle domande.** I checkpoint atomici esistenti non sono una cache di risposta alle domande sui referti. Il nuovo percorso salva per paziente ma non riprende automaticamente il frammento interrotto. Per centinaia di pazienti con centinaia di referti servono checkpoint per documento/frammento indicizzati da hash sorgente, prompt, schema, pesi del modello e parametri. Prima di riusare un risultato occorre invalidarlo quando una qualunque dipendenza cambia. Il manifest permette di individuare i pazienti non completati, non realizza ancora la ripresa automatica.

**5. Snapshot e concorrenza.** Il nuovo dialogo cattura il workspace, è modale e blocca l'avvio se la GUI segnala elaborazione LLM in corso; il servizio verifica gli ID paziente. Per accessi concorrenti esterni serve anche uno snapshot/versione dell'insieme di documenti. Gli allocatori sequenziali `get_next_id()` rimangono separati da `insert()` e possono collidere con writer simultanei: sostituirli con allocazione transazionale o identificativi indipendenti dal contatore prima di parallelizzare ulteriormente l'importazione.

**6. Parsing e OCR.** Il fallback OCR locale è già implementato. Il rilevamento di pagine povere di testo tramite soglia di caratteri può non riconoscere un PDF misto con intestazione testuale e corpo scansionato. Servono qualità per pagina, confronto dei due estrattori nativi e segnalazione di testo sospetto/tabelle non recuperate. La presenza di testo e dei numeri di pagina non dimostra che tutto il PDF sia stato letto correttamente.

**7. Statistiche di coorte.** I conteggi prodotti automaticamente dal nuovo report riguardano pazienti e copertura di elaborazione. Prevalenze, incidenze, durate o confronti richiedono uno schema di estrazione esplicito: unità paziente/evento, intervallo temporale, criteri, gestione degli ignoti e denominatore. Calcolare questi numeri in Python/SQL su dati validati; usare il modello per estrarre e spiegare, non come unico calcolatore. La sintesi narrativa di coorte è ancora generativa e richiede revisione.

**8. File e report derivati.** La cancellazione ora blocca originali e artefatti esterni alla directory del paziente, risolvendo anche i link simbolici. I workspace legacy con percorsi esterni devono essere corretti prima della cancellazione. Resta da uniformare l’invalidazione dei report derivati dopo cancellazioni e riattribuzioni: non basta correggere le righe del database. Non sono state effettuate cancellazioni durante questa revisione.

**9. Valutazione.** Il matcher di eventi usa abbinamento greedy e somiglianza di stringhe, categoria e data; non misura da solo la qualità di risposte aperte. La metrica di completezza citazioni considera soddisfatto un gold senza fonti obbligatorie: non equivale a una misura di grounding. Estendere la valutazione a fatti omessi, negazioni, date retrospettive, valori normali, contraddizioni, precisione della pagina, contaminazione fra pazienti e denominatori delle coorti.

**10. Modularità.** La GUI ospita ancora molta orchestrazione in `documents_tab`, `clinical_history_tab` e `workers`. Migrare progressivamente verso servizi indipendenti da Qt e job persistenti. Il nuovo servizio di query è già indipendente da Qt; la gestione della coda e del salvataggio potrà essere estratta dal worker GUI per una CLI identica. Anche i prompt nuovi, attualmente versionati nel modulo, andrebbero inseriti nel catalogo modificabile con gli stessi controlli di schema e digest.

## Hardware e protocollo di verifica

Le due configurazioni indicate sono MacBook Pro M5 Pro con 48 GB e DGX Spark con 128 GB. Il nuovo percorso parte da **un paziente e una richiesta per volta**, usa il modello locale del ruolo di analisi già configurato e non impone nuovi pesi o download. Questo contiene la concorrenza ma non garantisce che qualsiasi modello/contesto entri nella RAM.

Non sono stati effettuati benchmark su queste macchine né confronti reali fra modelli. Dimensionare modello, quantizzazione e contesto con dossier rappresentativi, poi aumentare gradualmente la concorrenza sulla Spark. Misurare memoria di picco, token effettivi, durata per documento/paziente, errori di generazione e qualità. I meccanismi di runtime llama.cpp/vLLM già presenti rimangono in uso; l'OCR rimane un fallback, coerentemente con i PDF prevalentemente nativi.

Per un gold iniziale usare domande di categorie diverse, ad esempio:

- Ricostruisci diagnosi, sedi e date, separando sospetti e conferme.
- Elenca le terapie con inizio, fine, dose e motivi documentati delle variazioni.
- Ricostruisci ricoveri, accessi urgenti e motivazioni riportate.
- Riporta l'andamento di un parametro con tutti i valori, unità e date disponibili.
- Individua referti discordanti e cita entrambe le fonti.
- Distingui date dell'evento clinico da date di documentazione retrospettiva.
- Cerca una condizione anche quando è negata o soltanto sospettata.
- Sulla coorte, applica criteri espliciti e distingui positivi, negativi documentati e non valutabili.

Separare i pazienti fra sviluppo e valutazione; includere dossier lunghi, copie ripetute, referti scansionati, documenti non classificati e fatti presenti soltanto nell'ultima pagina. La precisione/sensibilità va confrontata sia con la pipeline precedente sia con annotazioni umane delle fonti.

## Verifica riproducibile

L'interprete Python di sistema non disponeva di pytest; l'ambiente Conda base mancava di PyMuPDF. È stato creato `/tmp/emr-audit-venv` con accesso ai pacchetti Conda, installando lì le dipendenze mancanti per i test, senza modificare gli ambienti applicativi.

```bash
/tmp/emr-audit-venv/bin/python -m pytest -q
python3 tools/audit_code.py > docs/CODE_INVENTORY.json
/tmp/emr-audit-venv/bin/python -m pyflakes emr_analyzer
```

**Risultato finale: 1.107 test superati in 53,35 secondi**, con cinque warning di deprecazione dei binding PyMuPDF/SWIG. Anche compilazione Python e `git diff --check` completati senza errori.

I test nuovi usano soltanto dati sintetici e modelli simulati. Verificano copertura dei frammenti, ultima osservazione, citazioni inventate, pagine mancanti, documenti mancanti, isolamento paziente, cancellazione, report persistiti, transazioni, ID, decodifica e budget. Non misurano inferenza clinica di un modello reale, capacità del modello sulla memoria disponibile o resa dell'OCR su referti reali.


## Aggiornamento: pipeline essenziale da originale a Markdown

Eliminati i salvataggi intermedi per documento e la dipendenza della rielaborazione
LLM dai file parser. L’ingresso della GUI ora esegue parsing e normalizzazione
in un solo passaggio. Il percorso parallelo usa una coda limitata ai worker attivi
con controllo periodico dell’interruzione; quello sequenziale è iterativo, senza
crescita dello stack Python sui dossier grandi. Solo l’output completo viene
pubblicato atomicamente, con data del referto e marcatori di pagina deterministici.

Il laboratorio produce anch’esso un Markdown anonimizzato e compare nella coda:
la conservazione di materiale biologico e righe numeriche prevale sulla rimozione
aggressiva del boilerplate. Anteprime dei prompt e geometria delle evidenze leggono
l’originale quando necessario. Il progetto MELANOMA non viene elaborato dai test.


### Rimozione dei percorsi obsoleti dell’estrazione

Rimosse le opzioni interne `parse_only`, `llm_only` e `skip_llm`, ora senza
chiamanti funzionali: resta un solo ingresso alla pipeline completa. Eliminato
il parametro inutilizzato del pulitore nel normalizzatore documentale. Le
anteprime non ripiegano più su `_cleaned_source.md`; l’importazione dei pazienti
copia soltanto il Markdown finale, e lo strumento di riclassificazione legge
l’originale invece di dipendere da `_source.txt`. Restano intenzionalmente i
riferimenti ai vecchi file nelle procedure di cancellazione e nei test che
verificano che le interrogazioni non li leggano.


### Aggiornamento: Markdown senza riscrittura generativa

Il servizio attivo utilizza `ClinicalTextFilter`: anonimizzazione deterministica
con conservazione della spaziatura e selezione di porzioni sorgenti. Le regole
amministrative sono ancorate a blocchi completi, con precedenza alle indicazioni
cliniche nelle righe miste. Le decisioni e gli intervalli sono registrati per
pagina nel database. L’anonimizzazione dei campi separati da punto e virgola
protegge le note cliniche adiacenti dalla rimozione di un indirizzo amministrativo.
Il filtro non effettua chiamate al modello; il controllo di attribuzione rimane
una fase distinta. Il vecchio normalizzatore generativo non è più istanziato
nei servizi dell’app. Non si dichiara una copertura universale del boilerplate:
i casi misti o dubbi sono conservati e segnalati.
