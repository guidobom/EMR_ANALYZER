> Documentazione della pipeline precedente. Per l’estrazione narrativa attuale
> vedere [Estrazione dal Lessico condiviso](LEXICON_EXTRACTION.md).

# Pipeline ICD-11 per referti italiani

Questo branch usa un unico estrattore: `Icd11Extractor`, versione
`icd11_examples_v2`. Sono stati rimossi gli estrattori a chunk, frase singola,
v13 e i fallback diretti testo→timeline. I dati storici restano leggibili;
le nuove estrazioni sostituiscono gli output automatici precedenti per documento.
Le precedenti modifiche locali sono state salvate prima della sostituzione in
`/tmp/emr-before-icd11.patch` e `/tmp/emr-before-icd11-untracked.tar.gz`.
Questi backup temporanei non equivalgono a un commit in `main`.

## Uso

Riavviare l'applicazione. Aprire **Strumenti → Configura pipeline clinica**:
non occorre selezionare un metodo. Configurare il modello in **Configura LLM**,
poi usare **Storia clinica → Estrai evidenze**. I gruppi hanno un obiettivo di
1.000 token; il budget predefinito è 6 operazioni/chiamate per gruppo, incluse
le suddivisioni. I limiti sono configurabili.

**Senza catalogo WHO:** l'estrazione funziona, ma gli eventi hanno stato
`catalog_missing` e nessun codice ICD-11. Il riepilogo e il tooltip dell'evidenza
lo mostrano. Non sono distribuiti codici o traduzioni inventati.
Un catalogo configurato ma non più leggibile produce un errore esplicito.

## Flusso effettivamente implementato

1. I valori strutturati di laboratorio producono evidenze soltanto se anomali
   rispetto a range o flag. Sono conservati valore, unità, campione, data,
   limiti e fonte. I valori censurati (`<`, `>`, ecc.) sono interpretati come
   intervalli; flag e range incompatibili richiedono revisione.
2. I testi narrativi sono suddivisi in gruppi di frasi obiettivo, con contesto
   adiacente. Una frase può produrre diversi fatti.
3. Il catalogo locale propone fino a 16 candidati mediante termini e sinonimi,
   inclusi alias italiani curati. Il modello restituisce fatti e copertura delle
   frasi in un'unica risposta JSON. Non sceglie codici fuori dai candidati.
4. Il validatore verifica schema, riferimenti, date supportate e alcune
   incoerenze. Un recupero mirato corregge output incompleti; in caso di
   troncamento il gruppo viene suddiviso entro il budget. Problemi irrisolti
   restano espliciti, conservando i fatti validi.
5. Checkpoint SQLite e metriche per chiamata consentono ripresa e misura di
   token, durata, recuperi e gruppi riutilizzati. Testo, modello/configurazione,
   prompt e catalogo contribuiscono all'identità del checkpoint.
6. Gli eventi cronologici vengono costruiti dalle evidenze persistite con i
   passaggi dedicati nell'interfaccia. Asserzione, certezza, soggetto, data del
   fatto e data del documento restano distinti; fonte e coordinate permettono
   di ritrovare il passaggio originale. L'export atomico include ICD-11 e LOINC.

Prompt attivo: `emr_analyzer/resources/prompts/icd11_extraction_system.txt`,
modificabile da **Gestisci prompt LLM**. Le negazioni complesse, i quesiti e
l'anamnesi familiare sono istruiti esplicitamente nel prompt; non esiste una
validazione deterministica completa del loro significato.

## Catalogo locale

**Importa catalogo ICD-11…** seleziona un JSON locale normalizzato. Non è un
client WHO né un importatore generico di qualunque formato WHO. Il file deve
essere preparato da contenuti ufficiali della medesima release. La validazione
locale controlla struttura e coerenza, non certifica l'autenticità del file.
La Foundation identifica i concetti; i codici richiedono la linearizzazione
MMS. Per ottenere i contenuti dalle API servono credenziali WHO o un servizio
locale: [documentazione WHO](https://icd.who.int/docs/icd-api/APIDoc-Version2/).
Nessun referto viene inviato a WHO.

Campi del JSON (segnaposto descrittivi, non un catalogo da importare):

```json
{
  "release": "2026-01",
  "entities": [{
    "foundation_uri": "URI Foundation ufficiale del concetto",
    "code": "codice MMS ufficiale, oppure null",
    "title": "titolo ufficiale",
    "synonyms": [],
    "aliases_it": ["sinonimi italiani verificati"],
    "definition": "definizione ufficiale",
    "requires_postcoordination": false
  }],
  "lab_rules": [{
    "analyte": "nome normalizzato dell'analita",
    "specimen": "siero",
    "unit": "U/L",
    "direction": "high",
    "foundation_uri": "URI del reperto anomalo presente nelle entities",
    "relation": "exact",
    "review_status": "accepted"
  }]
}
```

I titoli/definizioni JSON-LD `@value` e i campi WHO `source`, `synonym`,
`requiredPostcoordination` sono supportati. Se occorre postcoordinazione,
si conserva soltanto il riferimento Foundation: non si dichiara un codice
MMS completo. Le espressioni postcoordinate non sono generate automaticamente.

Le regole di laboratorio devono essere revisionate: corrispondenza esatta
(case-insensitive) di analita, campione, unità e direzione (`high`, `low`,
`abnormal`). Regole assenti o ambigue non assegnano codici. Un dato fuori range
non autorizza a dedurre una diagnosi eziologica. LOINC, quando risolvibile, resta
separato in `data.loinc`; ICD-11 in `data.icd11`.

## Limiti da misurare

Il catalogo ufficiale non è ancora disponibile nel progetto. Senza alias
italiani adeguati la ricerca lessicale può perdere corrispondenze: il fatto
resta estratto ma non codificato. Non sono ancora implementati ricerca semantica
multilingue, postcoordinazione automatica o ricodifica indipendente dal modello.
Cambiare catalogo invalida i checkpoint e richiede una nuova estrazione.

La copertura dichiarata dal modello non dimostra che ogni evento sia stato
riconosciuto. Sensibilità, precisione delle negazioni e velocità devono essere
misurate su referti annotati e hardware reale. I test automatici usano dati
sintetici e non rappresentano una validazione clinica.

Il registro annotato ora guida l’estrazione: vedere [schede evento ed esempi](LOCAL_LEXICON.md#usare-le-annotazioni-per-guidare-il-llm).
