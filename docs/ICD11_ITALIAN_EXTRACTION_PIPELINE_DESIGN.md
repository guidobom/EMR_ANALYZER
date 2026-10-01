# Progetto: estrazione clinica italiana con ICD-11 e anomalie di laboratorio

Stato: proposta di implementazione, 14 settembre 2026. Non implementa né attiva una nuova estrazione. Il progetto conserva fonti e osservazioni indipendentemente dalla disponibilità di codici standard.

## Obiettivo e contratto

Estrarre affermazioni italiane con soggetto, negazione, certezza, stato, data e fonte; associare concetti ICD-11 verificati; produrre evidenze atomiche soltanto per le anomalie laboratoristiche documentate. Tutte le misurazioni, anche normali, rimangono nel database laboratoristico.

ICD-11 identifica il concetto clinico e non sostituisce il risultato analitico. Per ogni anomalia si tenterà una codifica ICD-11, ma non si assegneranno diagnosi o categorie non giustificate pur di ottenere copertura del 100%. Il fallimento della codifica non equivale al fallimento dell'estrazione.

Sono separati: versione del documento e sue occorrenze; osservazione estratta; interpretazione dello stato dell'affermazione; associazione terminologica versionata; eventi ed episodi derivati. La disponibilità di un codice non stabilisce presenza, attualità o soggetto della condizione.

## Catalogo ICD-11 e ponte linguistico

Baseline proposta: release ICD-11 2026-01, da fissare esplicitamente nel manifest di ogni esecuzione. Catalogo inglese ufficiale; italiano per documenti e interfaccia. La tabella WHO non elenca l'italiano per ICD-11 MMS 2026-01: l'italiano nell'ultima colonna riguarda ICF.

Un importer terminologico, separato dall'estrazione, acquisisce dati ufficiali secondo i termini WHO applicabili e costruisce un indice locale. Catalogo e versione devono essere disponibili prima dell'attivazione della codifica; nessuna chiamata remota con testi dei pazienti. Le modalità di distribuzione del catalogo e le dipendenze dell'installazione locale saranno definite verificando i termini e il formato della release scelta.

Per ogni voce: Foundation URI, eventuale entità e codice MMS della release, titolo ufficiale, definizione, termini inclusi, esclusioni, stato della voce, gerarchia e vincoli di postcoordinazione. Un Foundation URI non è un codice MMS. Non tutti i concetti Foundation sono categorie MMS direttamente codificabili.

Lo strato italiano conserva etichetta locale, sinonimi, abbreviazioni, ambito, provenienza, revisore e versione. Normalizzazione ortografica solo sull'indice di ricerca: il testo originale non viene riscritto. Sinonimi generati automaticamente sono proposte; non diventano traduzioni ufficiali WHO né equivalenze validate. Un sinonimo ambiguo conserva più candidati. Le nuove proposte non vengono promosse automaticamente dal medesimo LLM che le ha generate.

Ricerca locale: corrispondenze esatte e varianti controllate, poi ricerca lessicale bilingue. Se necessario, retrieval semantico multilingue locale come componente separata da misurare. Se il lessico non recupera candidati, il LLM può estrarre il termine italiano e proporre una breve denominazione inglese per la successiva risoluzione: nessuna traduzione integrale preventiva dei referti. Questa proposta bilingue rimane non verificata finché non è confrontata con il catalogo.

## Ramo narrativo

1. Leggere la versione effettiva del testo, comprese correzioni approvate, preservando pagina, offset e hash della fonte.
2. Segmentare frasi e sezioni mantenendo intestazioni temporali, soggetti e antecedenti utili. Riparazioni di encoding/OCR solo su una vista derivata con corrispondenza agli offset originali.
3. Individuare menzioni candidate senza escludere le frasi prive di termini riconosciuti. Recuperare pochi candidati ICD pertinenti per menzione; una soglia/top-k non deve impedire l'astensione.
4. Costruire gruppi entro il budget del prompt completo: istruzioni, fonti, contesto, candidati e risposta attesa. La densità clinica influisce sul budget di output. Cache dei conteggi di token per testo e tokenizer.
5. Una chiamata LLM per gruppo. Output: elenco di fatti, non tutte le categorie vuote; concept_ref locale oppure etichetta italiana non codificata; tipo di fatto, soggetto, asserzione, certezza, stato temporale, date documentate, attributi e riferimenti alla fonte. Etichette concise, senza duplicare le citazioni nel JSON.
6. Risolvere concept_ref in identificativi ufficiali via codice. Il modello non genera codici a memoria. Verificare compatibilità con la fonte e precisione consentita; distinguere scelta più generale giustificata da scelta eccessivamente specifica.
7. Validare oggetti e fonti. Una citazione valida dimostra la provenienza del passaggio, non automaticamente la correttezza dell'interpretazione. I casi ambigui restano revisionabili.
8. Un recupero selettivo per gruppo ordinario, limitato alle frasi problematiche e corredato dai fatti validi già presenti. Risposta come aggiunte/sostituzioni esplicite o problema irrisolto; limite totale anche per suddivisioni. Un problema solo terminologico non fa riestrarre l'intero gruppo.

La postcoordinazione non è libera concatenazione di codici. Nella prima versione gli attributi restano strutturati; si emette un'espressione MMS soltanto quando gli assi e gli eventuali requisiti obbligatori possono essere validati. Altrimenti lo stato è incompleto/candidato e il codice semplice non viene presentato come una codifica formalmente completa se richiede altre componenti.

## Negazione, incertezza e soggetto

Campi separati: assertion (present, absent, unknown); certainty (confirmed, suspected, possible, unknown); experiencer (patient, family, other, unknown); temporality (current, historical, hypothetical, unknown); clinical_status (ongoing, resolved, suspended, ecc., secondo il tipo).

Conservare riferimenti al bersaglio e agli indizi di negazione/incertezza. I valori descrivono quanto è affermato nella fonte, non una conferma indipendente della verità clinica.

Casi obbligatori: negazione coordinata; cambio di ambito con ma/tuttavia; non si può escludere; esame richiesto per escludere; negazione della progressione senza negazione della neoplasia; familiarità; malattia pregressa; sintomo risolto; assenza di febbre ma presenza di dispnea. Le regole deterministiche identificano incongruenze ma non ribaltano alla cieca un'asserzione LLM sulla sola parola “non”.

Le affermazioni negative conservano un'associazione al concetto per la ricerca ma non costituiscono diagnosi presenti. “Non documentato” resta diverso da “assente”. Una misura numerica negativa o un risultato qualitativo negativo non equivale automaticamente a negazione dell'esistenza dell'osservazione.

## Ramo laboratoristico: doppia codifica

### Misurazione

Ogni riga resta un'osservazione con identificativo persistente: analita originale e normalizzato, valore originale, operatore, unità, intervallo e sue disuguaglianze, flag, campione, metodo quando disponibile, data del prelievo e del referto, laboratorio, fonte. Codifica LOINC solo quando component/property/time/system/scale/method sono compatibili; un semplice alias di analita non basta. Il risultato non viene perso se LOINC non è risolvibile.

### Determinazione dell'anomalia

Regole deterministiche, senza LLM per ogni riga:

- Confrontare valore e range del referto con unità compatibili, rispettando limiti aperti/chiusi.
- Preservare unità originali; conversioni solo validate dimensionalmente e per analita. Non confondere concentrazione massica e molare.
- Usare il range applicabile riportato per quel prelievo; non sostituirlo con un intervallo “universale”. Se sono stampati più intervalli e manca il contesto necessario, segnalare l'ambiguità.
- Gestire <, >, <=, >= come intervalli di valori possibili. “<5” rispetto a un limite superiore di 4 non prova che il valore superi 4. Se la direzione non è dimostrabile, non dedurla dalla sola soglia numerica.
- Con flag H/L valido e range assente: registrare l'anomalia come segnalata dal laboratorio, distinta da quella calcolata. Flag e confronto numerico discordanti richiedono verifica, non una scelta silenziosa.
- Range e flag entrambi assenti: non inventare l'anomalia. Un giudizio testuale esplicito può costituire una fonte distinta di anomalia.
- Una variazione ancora nel range non crea un atomo di laboratorio.

### Associazione ICD-11 dell'anomalia

Una tabella di regole approvate è indicizzata almeno da analita, campione, proprietà, direzione dell'anomalia e release ICD; comprende prerequisiti, esclusioni, rapporto di mapping (exact/broader), stato e provenienza. Non creare una mappa universale LOINC→ICD: lo stesso esame può essere alto, basso o normale e lo stesso codice ICD può comprendere analiti differenti.

Se la regola è applicabile, assegnazione deterministica del concetto di reperto anomalo e del codice/espressione MMS verificati. L'unità, il valore e la direzione restano attributi dell'osservazione anche quando la categoria ICD è più generale.

Se esiste soltanto una categoria più ampia, accettarla solo se semanticamente corretta e mostrarla come broader; conservare l'analita preciso. Se non esiste un'associazione sicura, mantenere un identificativo locale esplicitamente non ICD e mapping_status=unmapped/pending_review. Proporre una nuova regola una sola volta per combinazione terminologica, non per ogni misurazione; nessuna autoapprovazione basata soltanto sulla fiducia dichiarata dal modello.

Esempi sintetici: TSH sopra range→reperto di funzione tiroidea anomalo se supportato dal catalogo, non ipotiroidismo automatico; ALT sopra range→reperto enzimatico pertinente, non epatite automatica; emoglobina sotto range→riduzione documentata, senza dedurre la causa. I codici alfanumerici esatti saranno inseriti dopo verifica della release, non ricostruiti a memoria in questo progetto.

Ogni anomalia ammissibile genera un'evidenza anche se ICD non è disponibile. Tipo dell'evidenza laboratory_test/laboratory_finding: l'attribuzione di ICD non la trasforma in diagnosis. Una diagnosi esplicita nel referto narrativo costituisce un'evidenza separata eventualmente collegata.

### Esempio di contratto di un'anomalia

```json
{
  "observation_id": "OBS_DEMO_1",
  "fact_type": "laboratory_test",
  "analyte_local": "ormone_tireostimolante",
  "value": 8.2,
  "operator": "=",
  "unit": "mIU/L",
  "reference_low": 0.4,
  "reference_high": 4.0,
  "direction": "high",
  "abnormality_basis": "reported_reference_range",
  "assertion": "present",
  "experiencer": "patient",
  "coding": {
    "icd_release": "2026-01",
    "foundation_uri": null,
    "mms_code": null,
    "expression": null,
    "status": "pending_catalog_validation"
  },
  "source_ref": "SRC_DEMO_1"
}
```

L'esempio descrive dati sintetici e uno stato precedente alla validazione della codifica. Non certifica che esista un codice specifico per TSH alto. LOINC dell'esame e ICD del reperto sono associazioni con ruoli distinti.

## Persistenza, ripresa e cronologia

Tabelle proposte:

- terminology_releases: sistema, release, lingua, impronta e provenienza;
- icd_entities: Foundation/MMS, definizioni e vincoli;
- terminology_aliases: alias italiano, ambito, provenienza e stato;
- evidence_codings: evidence_id, coding_role, sistema/release, URI/codice/espressione, relazione di mapping, metodo e revisione;
- lab_abnormality_mappings: condizioni della regola e concetto ICD verificato;
- extraction_issues: problemi identificati, riferimenti, stato e tentativi.

Riutilizzare i repository esistenti dove appropriato e le attuali osservazioni di laboratorio. La chiave corrente di terminology_mappings (concetto, tipo, unità) non è sufficiente per direzione, campione e release; evitare di sovrascrivere una mappa “alto” con una “basso”. Il campo di codifica principale in ClinicalEvidence resta una proiezione compatibile delle associazioni multiple.

Stati indipendenti di estrazione e codifica. Cache della generazione basata su testo, contesto, metadati temporali, paziente, modello/parametri, prompt e candidati forniti; cache della codifica basata anche su release e regole. Un aggiornamento delle mappe ricalcola le associazioni compatibili senza riestrarre automaticamente gli eventi. Se cambia l'interpretazione del fatto, resta una revisione esplicita.

Non fondere misurazioni soltanto perché condividono ICD. Conservare ogni prelievo e collegare duplicati solo quando identificano la medesima osservazione. Codici larghi possono essere condivisi da analiti diversi. Occorrenze della stessa storia clinica rimangono citabili separatamente; episodi e trend sono proiezioni.

Le query leggono osservazioni complete, evidenze e fonti: domande sul rientro del TSH nel range richiedono anche valori normali. Un rientro nel range può essere un risultato derivato del trend, senza creare un atomo per ciascun valore normale o dichiarare risolta una malattia. Negazioni, familiarità e diagnosi sospette sono interrogabili con filtri espliciti.

## Implementazione progressiva e accettazione

1. Correggere recuperi, copertura e dipendenze dagli integratori v12, mantenendo invariata la codifica, per una baseline misurabile.
2. Costruire catalogo ICD e ponte italiano; nuove tabelle e interfaccia di revisione delle mappe. Testare alias ambigui, versioni e concetti Foundation senza MMS.
3. Pilota sul laboratorio: inventario degli analiti del workspace; mappatura per direzione/campione; nessuna inferenza clinica dai soli valori. Controlli su unità, comparatori, limiti, flag, date e duplicati.
4. Estrattore narrativo con candidati ICD e contratto compatto; recuperi selettivi e checkpoint parziali.
5. Collegamento a cronologia e query; confronto controllato con la baseline prima di sostituire la pipeline attiva.

Interfaccia: stato di estrazione distinto da stato di codifica; titolo italiano locale accanto a codice e titolo ufficiale; indicatore di mapping specifico/generale/da verificare; filtri per soggetto e negazione; origine dell'anomalia calcolata/segnalata/discordante. Dashboard con documenti/gruppi conclusi, problemi residui, misurazioni anomale, quota codificata, chiamate e tempo di recupero.

Valutazione su documenti sintetici e campione reale annotato clinicamente. Misurare sensibilità e precisione dell'estrazione, accuratezza di negazione/soggetto/date, precisione e copertura della codifica separatamente, token per fatto corretto, tempo per documento e costo dei recuperi. La soglia di ammissibilità della nuova pipeline va concordata sul campione; nessuna promessa preventiva di incremento simultaneo di tutte le metriche.

## Fonti ufficiali

- WHO, versioni e lingue: https://icd.who.int/docs/icd-api/SupportedClassifications/
- WHO, API e distinzione Foundation/MMS: https://icd.who.int/docs/icd-api/APIDoc-Version2/
- WHO, vincoli di postcoordinazione: https://icd.who.int/docs/codingtool/en/Postcoordination/
- WHO, Reference Guide: https://icdcdn.who.int/icd11referenceguide/en/refguide.pdf
- LOINC, assi della misurazione: https://loinc.org/kb/users-guide/major-parts-of-a-loinc-term

Le associazioni specifiche tra analiti/anomalie e codici ICD-11 sono un deliverable della fase catalogo, non dati già validati da questo documento.
