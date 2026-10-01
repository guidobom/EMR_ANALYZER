# Registro eventi FHIR R4

Ogni paziente ha un Bundle `collection` in
`<progetto>/<paziente>/clinical_events.fhir.json`; **File → Esporta FHIR del
progetto** scrive inoltre `<progetto>/exports/fhir/progetto-<data>.ndjson`, con
una risorsa FHIR per riga e ogni risorsa una sola volta.

## Da dove vengono i dati

Il file è una proiezione ricostruibile del database, mai modificata a mano:

- **eventi clinici**: occorrenze estratte dai referti narrativi, con le decisioni
  del revisore applicate (scheda *Eventi SNOMED*). Gli eventi scartati sono
  esclusi; gli eventi aggiunti a mano sono inclusi;
- **codici SNOMED CT**: codifica del concetto (etichetta + tipo di evento) o, se
  il revisore l'ha scelto, codice della singola occorrenza;
- **laboratorio**: tutti i risultati letti dal parser (`lab_values`), normali e
  anomali, con LOINC confermato o proposto quando disponibile.

Il file viene riscritto alla fine dell'elaborazione del paziente, qualche secondo
dopo ogni modifica di revisione e a ogni export. Cancellazione, riattribuzione o
merge di documenti rimuovono il file dei pazienti coinvolti, che va rigenerato.

## Risorse

| Evento | Risorsa |
|---|---|
| Diagnosi affermata | `Condition` (`verificationStatus` da certezza) |
| Prescrizione esplicita | `MedicationRequest` |
| Somministrazione esplicita e datata | `MedicationAdministration` |
| Altre menzioni di farmaci | `MedicationStatement` (anche `not-taken` se negate) |
| Procedura | `Procedure` (`not-done` se negata) |
| Familiarità positiva | `FamilyMemberHistory` |
| Sintomi, segni, reperti, esami narrativi, negazioni | `Observation` |
| Risultato di laboratorio | `Observation` categoria *laboratory*, con `referenceRange` e `interpretation` |

Ogni risorsa clinica ha una `Provenance` verso la `DocumentReference` del referto,
con la citazione letterale e il contesto (intervallo nel testo, provenienza della
data). Sono presenti anche `Patient` pseudonimo, `Device` (estrattore) e una
risorsa `Basic` con la copertura dell'elaborazione.

## Identificativi e stati

- L'id di un evento dipende dall'occorrenza sorgente (documento, intervallo,
  etichetta, tipo): resta stabile dopo una revisione o una nuova estrazione.
- Le occorrenze di uno stesso enunciato — la frase ripetuta in più referti —
  condividono una risorsa con una `Provenance` per referto, purché tutti i campi
  clinici concordino. Un fatto ripetuto la cui data non è nella frase resta una
  risorsa a sé: parole identiche possono descrivere episodi diversi. Una copia
  corretta dal revisore esce dal gruppo e diventa un fatto proprio; una conferma
  senza modifiche vi rientra all'esportazione successiva.
- L'id di un risultato di laboratorio dipende dal contenuto grezzo della misura
  nel suo documento, non da revisioni, interpretazioni o altri referti.
- Estensioni applicative (`urn:emr-analyzer:fhir:StructureDefinition:`):
  `review-status` (`proposed`, `needs-review`, `confirmed`, `corrected`, `manual`),
  `coding-status` (`proposed`, `confirmed`, `needs_review`) ed `extraction-context`
  (asserzione, certezza, soggetto, temporalità, attributi, provenienza della data).

## Validazione

Prima di scrivere si verificano riferimenti interni, identità, campi obbligatori
e valori JSON finiti; se `fhirclient` è installato anche i modelli R4. Non è
dichiarata conformità a un Implementation Guide né validazione terminologica su
un server esterno.
