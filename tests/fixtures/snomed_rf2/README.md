# Fixture RF2 sintetica — SNOMED CT

Fixture **sintetica** di un release SNOMED CT (edizione International) usata
dai test della pipeline `snomed`.  Non è un release reale: i codici sono
rappresentativi (alcuni coincidono con codici SNOMED veri, altri sono
approssimazioni) e **nessun valore va trattato come clinicamente
autorevole**.  Per una run reale usare la guida Fase 0 (acquisto release via
MLDS) e puntare `EMR_ANALYZER_SNOMED_RELEASE_DIR` alla directory del release.

Struttura generata da `generate_fixture.py` (eseguilo per rigenerare):

```
Snapshot_20260101/
  Snapshot/
    Content/
      sct2_Concept_Snapshot_INT_20260101.txt
      sct2_Description_Snapshot-en_INT_20260101.txt
      sct2_Description_Snapshot-it_INT_20260101.txt
      sct2_Relationship_Snapshot_INT_20260101.txt
    Refset/Language/
      der2_cRefset_LanguageSnapshot-en_INT_20260101.txt
      der2_cRefset_LanguageSnapshot-it_INT_20260101.txt
concepts.csv     # manifest di asserzione letto dai test
README.md
```

I file RF2 hanno header standard; il loader applica la semantica snapshot
"max effectiveTime + active==1 per id".  `concepts.csv` è l'oracolo dei test:
colonna `expected_fact_types` = bucket atomici attesi da `fact_types_for_concept`.

58 concetti: ancore semantiche, diagnosi, sintomi, segni,
reperti radiologici, procedure, parametri vitali, esami di laboratorio,
farmaci, istologia e biomarcatori.
