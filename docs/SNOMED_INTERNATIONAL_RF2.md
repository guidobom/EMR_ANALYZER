# SNOMED CT International Edition — RF2

Il catalogo operativo è `~/.emr_analyzer/snomed_ct.db`. L'importatore GPS e il
precedente database `snomed_gps.db` non sono utilizzati dal nuovo codice.
Le vecchie estrazioni conservano la propria provenienza e release.

## Importazione

Nella UI: **Lessico condiviso → Catalogo SNOMED CT… → Importa catalogo SNOMED CT…**.
Selezionare la cartella estratta `SnomedCT_InternationalRF2_PRODUCTION_…`, che
contiene `release_package_information.json` e `Snapshot/`. Non selezionare un
singolo file, lo ZIP o la sottocartella `Full`.

Da terminale, usando l'ambiente Python dell'applicazione:

```sh
python tools/import_snomed_rf2.py /percorso/SnomedCT_InternationalRF2_PRODUCTION_… \
  --output /percorso/snomed_ct.db
```

L'importazione mantiene lo Snapshot, inclusi componenti inattivi, di:

- concetti e stato di definizione;
- descrizioni inglesi (FSN e sinonimi);
- language refset US e GB;
- relazioni inferite con relationshipGroup, tipo e caratteristica;
- relazioni con valori concreti;
- assiomi OWL;
- associazioni storiche;
- definizioni testuali.

I file Full, StatedRelationship e gli altri refset rimangono nella distribuzione
originale; non vengono importati in questo catalogo operativo. Non è implementato
un classificatore OWL, un motore ECL completo o una validazione MRCM delle
espressioni postcoordinate. Le gerarchie interrogate sono quelle inferite e
pubblicate nella release, non nuove inferenze cliniche sul paziente.

Ogni importazione verifica intestazioni e release, mantiene gli identificativi
come testo e registra hash SHA-256 dei componenti. L'importazione è transazionale:
errori e annullamenti conservano il catalogo precedente. Gli indici vettoriali
preesistenti sono invalidati; le associazioni italiane sono conservate soltanto
se il concetto di destinazione resta attivo.

## Ricerca e uso nella pipeline

FSN e preferred term sono campi distinti. Il termine preferito è selezionato dal
refset US English, con fallback GB English e descrizione attiva se necessario.
La ricerca lessicale include FSN e sinonimi attivi accettati in US/GB; dà priorità
alle corrispondenze esatte e alle associazioni italiane approvate. I candidati
sono deduplicati per conceptId, limitati alle gerarchie ammesse per il tipo di
evento (per esempio disorder/finding per le diagnosi, substance/product per i
farmaci, observable entity/procedure per gli esami) e includono i genitori
inferiti usati per disambiguare. Il modello può scegliere solo tra i codici
candidati attivi oppure astenersi. Non si attribuiscono al paziente
caratteristiche cliniche soltanto perché appartengono alla gerarchia di un codice.

La codifica avviene per concetto (etichetta normalizzata + tipo di evento), una
sola volta: il risultato è salvato in `concept_mappings` nel Lessico condiviso e
vale per tutte le occorrenze. Una codifica confermata dal revisore diventa anche
associazione italiana e non viene mai sostituita da una proposta automatica.

Sono disponibili API locali `lookup`, `synonyms`, `parents`, `relationships` e
`is_descendant_of`. La ricerca vettoriale opzionale resta basata su un modello
multilingue locale; l'importazione RF2 non scarica o costruisce automaticamente
quel modello. Senza indice multilingue, la pipeline usa le query inglesi generate
nella fase di mapping e le associazioni italiane già approvate.

I codici del nuovo export FHIR riportano la URI della versione International:
`http://snomed.info/sct/900000000000207008/version/YYYYMMDD`.
Le cache dell'estrazione/mapping cambiano con catalogo e prompt. Importare RF2 non
corregge retroattivamente gli errori di citazione e data della baseline P068,
né trasforma i candidati in codici clinicamente validati.

La distribuzione originale e i database generati sono esclusi da Git. Per una
nuova valutazione preparare un nuovo campione: il precedente P068-baseline
conserva intenzionalmente lo snapshot del catalogo che aveva al momento della
preparazione e non acquisisce automaticamente il nuovo database.

Riferimenti di implementazione: [language reference sets](https://docs.snomed.org/snomed-ct-specifications/snomed-ct-release-file-specification/reference-set-release-file-specification/5.2-reference-set-types/5.2.2.1-language-reference-set),
[FSN](https://docs.snomed.org/snomed-international-documents/snomed-ct-glossary/f/fully-specified-name),
[URI delle edizioni e versioni](https://docs.snomed.org/snomed-ct-specifications/snomed-ct-uri-standard/2-snomed-ct-uri-space).

## Indice vettoriale multilingue

Per cercare i concetti direttamente dalle etichette italiane si usa SapBERT
XLM-R (`cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR`, revisione
`47b6bd04`), un modello di entity linking biomedico multilingue addestrato su
UMLS 2020AB. La scheda del modello non dichiara una licenza propria (il codice
del progetto è MIT): va usato localmente e non ridistribuito.

Il modello è salvato una sola volta in
`~/.emr_analyzer/embeddings/sapbert-xlmr-cls` come modello sentence-transformers
con rappresentazione del token [CLS], come indicano gli autori. Dalla finestra
**Catalogo SNOMED CT** si costruisce l'indice scegliendo quella cartella:
un vettore per concetto attivo, dal termine preferito, limitato alle gerarchie
usate per gli eventi; l'indice (`snomed_ct.vectors.npz`, float16) è legato
all'impronta del catalogo e va ricostruito dopo una nuova importazione RF2.
Con l'indice presente la codifica non chiede più al modello linguistico le query
inglesi: i candidati vengono dalla ricerca vettoriale e lessicale e il modello
sceglie fra loro.

**Stato e aggiornamento.** Il dialogo del catalogo dichiara sempre lo stato
dell'indice: non creato, pronto, oppure da ricostruire. L'indice è legato
all'impronta del catalogo, quindi importare una nuova release lo rende
obsoleto; in quel caso la codifica non si ferma — torna a tradurre le etichette
e a cercare in inglese — e il pulsante diventa «Aggiorna indice vettoriale…».
Ricostruirlo richiede qualche minuto di calcolo locale (270.281 concetti attivi
delle gerarchie usate: 3 min 44 s su MPS, 414 MB) e non tocca codifiche o
associazioni già salvate.

**Limite noto.** La ricerca vettoriale non riproduce sempre gli stessi candidati
della via con traduzione: su un campione di 150 concetti già codificati, il
codice scelto in precedenza compare fra i primi dieci candidati dell'indice nel
36% dei casi. Non è una misura di qualità — in diversi confronti l'indice
propone concetti migliori di quelli scelti col percorso precedente — ma dice che
le due vie non sono equivalenti e che la scelta resta del modello, da confermare
in revisione.

