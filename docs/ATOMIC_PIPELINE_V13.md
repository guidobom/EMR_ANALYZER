> Documento storico: questi metodi di estrazione sono stati rimossi dal branch di sviluppo. Vedere [pipeline ICD-11 attuale](ICD11_PIPELINE.md).

# Pipeline di estrazione atomica v13

## Attivazione

In **Configurazione pipeline → Estrazione → Metodo** selezionare **Gruppi di frasi con checkpoint (v13)**. È il valore predefinito quando non esiste una scelta salvata; le configurazioni che specificano un metodo precedente lo conservano. Riavviare l'applicazione per caricare il nuovo codice e la migrazione additiva dello schema (18). Le estrazioni già in corso non cambiano metodo.

Lanciare **Estrai evidenze atomiche** e, dopo il completamento, la creazione degli eventi. La compatibilità viene verificata su testo, metadati, modello, prompt e parametri: i risultati v12 non sono dichiarati automaticamente aggiornati per v13.

## Comportamento implementato

- Laboratorio: conversione deterministica dei risultati anomali; nella v13 una variazione ancora nel range non genera un atomo. Tutti i valori restano nelle tabelle e nei grafici di laboratorio.
- Narrativa: segmentazione senza filtro lessicale, con offset nel testo originale; gruppi da circa 1.000 token sorgente, configurabili da 256 a 2.400. Ogni frase è obiettivo una sola volta; frasi vicine e intestazioni forniscono contesto citabile.
- Il tokenizer locale conta il testo e il template chat completo. Prima della generazione si riservano output e un margine di 256 token. Un backend privo di tokenizer usa una stima conservativa basata sui byte.
- Una prima generazione strutturata e al massimo un recupero combinato per errori di formato/citazione e segnali di copertura mancanti. Il recupero è disattivabile. Non è una verifica semantica esaustiva della completezza.
- Contesto insufficiente o output troncato: suddivisione tra frasi, entro un budget di 8 operazioni/chiamate per gruppo iniziale. Una frase singola non viene tagliata a metà: se non elaborabile, viene segnalata come incompleta.
- I gruppi conclusi, anche senza fatti, vengono salvati subito. Alla ripresa vengono riutilizzati soltanto risultati compatibili; quelli falliti vengono ritentati. Le fonti vengono rimaterializzate per il documento corrente.
- SQLite conserva `atomic_group_results` e `atomic_group_calls`, con stato, tempi, token disponibili e fase della chiamata. Gli errori non diventano risultati vuoti riusciti. Le evidenze valide di documenti incompleti sono conservate; la creazione degli eventi richiede che l'estrazione sia aggiornata.
- Gli atomi conservano citazioni, offset, versione della fonte, data del fatto distinta dalla data del referto e soggetto (`patient`, `family`, `other`, `unknown`). Soggetti diversi non vengono deduplicati insieme; familiarità e soggetti incerti non possono costruire episodi del paziente. Gli atomi incerti restano revisionabili.
- Il formato esportabile e quello usato per la sintesi espongono il soggetto. La cronologia e le interrogazioni esistenti continuano a usare evidenze e fonti; questa modifica non introduce un nuovo motore di question answering.

Prompt: `emr_analyzer/resources/prompts/atomic_evidence_grouped_system.txt`, disponibile anche nel gestore dei prompt con anteprima v13.

## Limiti e verifica

Il riuso v13 è conservativo: richiede testo completo e richiesta identici, incluso il contesto. Il vecchio riuso di blocchi trasferiti tra documenti differenti rimane nel metodo v12. La prima estrazione v13 può quindi elaborare più testo di una v12 con molto riuso: il vantaggio netto di tempo deve essere misurato, non dedotto soltanto dal minor numero di prompt sequenziali.

I test usano documenti sintetici e LLM simulati per verificare contratti, ripresa, errori, citazioni, date, soggetti e integrazione nel registro. Non misurano sensibilità o accuratezza clinica del modello. Per il confronto reale occorrono gli stessi documenti e modello, tempi/token registrati e un campione annotato manualmente; le elaborazioni cliniche attive non sono state riavviate né usate come test.
