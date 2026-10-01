# Annotazione del lessico condiviso

Dopo il riavvio dell'applicazione, selezionare un paziente e aprire la scheda
**Lessico condiviso**. Non occorre configurare modelli LLM o cataloghi terminologici.

1. Scegliere il documento nel menu superiore. Il testo estratto è in sola lettura.
2. Selezionare con il mouse un frammento significativo.
3. Scrivere un termine sintetico o scegliere un termine suggerito.
4. Facoltativamente specificare il contesto: Presente, Negato, Possibile,
   Familiare o Altro. Il valore iniziale è sempre Non specificato.
5. Premere Invio nel campo del termine oppure Salva.

Il frammento viene evidenziato. Un clic sull'evidenziazione o sull'elenco
**Nel documento** apre l'annotazione: si può cambiare termine, contesto o
selezione e salvare. **Nuova annotazione** esce dalla modifica e permette di
annotare un altro frammento. Le sovrapposizioni sono ammesse; per scegliere tra
più annotazioni sovrapposte usare l'elenco laterale.

**Lessico condiviso** nel pannello laterale mostra tutti i termini condivisi fra progetti,
con conteggio degli esempi e ricerca. Cliccare un termine per riutilizzarlo e
vederne gli esempi; cliccare un esempio per aprire il documento sorgente, anche
di un altro paziente (il paziente viene indicato nell'elenco e nel selettore).
Rinomina conserva i collegamenti. Unisci termini trasferisce gli esempi al
termine scelto. Se entrambi annotano lo stesso frammento, la fusione viene
bloccata per consentire la risoluzione esplicita di eventuali contesti discordanti.

**Altre occorrenze → Trova altre occorrenze identiche** cerca la selezione nei
testi estratti di tutto il workspace, rispettando maiuscole e spazi. I frammenti
con una annotazione già presente sulla stessa versione del testo sono esclusi.
I risultati grigi sono proposte non salvate (Ignora proposta selezionata le rimuove
dall’elenco corrente): aprire la fonte, verificare il
contesto e premere Salva. Una nuova ricerca sostituisce quella precedente.
Non vengono propagati automaticamente negazioni o altri contesti.

**Altre occorrenze → Trova espressioni simili** confronta la selezione con
frasi e brevi passaggi dei testi estratti del workspace corrente. La ricerca
lavora in background; **Interrompi ricerca** la annulla. Cambiare progetto o
paziente invalida i risultati ancora in elaborazione.

Se il modello multilingue locale è disponibile, la ricerca usa gli embedding
per proporre anche formulazioni diverse. In sua assenza usa un confronto
**testuale approssimato**, esplicitamente indicato nel risultato: trova variazioni
di scrittura ma non garantisce il riconoscimento di sinonimi. Non viene
scaricato un modello automaticamente. Un errore di elaborazione viene segnalato.

La soglia predefinita è 0,55; abbassarla amplia le proposte. Il punteggio di
somiglianza non è una probabilità clinica. Vengono mostrati fino a 100 risultati
ordinati, con documento, paziente e contesto. Sono esclusi il passaggio di partenza
e i passaggi sovrapposti a un'annotazione già assegnata allo stesso termine sulla
stessa versione della fonte. Le finestre sovrapposte dello stesso documento
vengono accorpate mantenendo la proposta meglio classificata.

Cliccando un risultato si apre il testo e si seleziona il passaggio: è possibile
correggere la selezione, scegliere il termine e specificare il contesto prima di
salvare. La ricerca non elimina negazioni o ipotesi e non crea annotazioni da sola.

**Annulla ultima modifica** annulla fino alle ultime 20 modifiche effettuate
nella sessione della scheda, incluse eliminazioni, rinomine e fusioni. La
cronologia di annullamento non sopravvive al riavvio o al cambio di workspace.
Se il lessico è stato modificato altrove, l'annullamento viene bloccato per non
sovrascrivere quel lavoro.

Il registro comune viene creato automaticamente in
`~/.emr_analyzer/shared_lexicon.db`, fuori dalle cartelle dei progetti.
È condiviso da tutti i workspace aperti con lo stesso account del computer;
non è una sincronizzazione cloud o fra computer diversi. Il backup dei singoli
progetti non include questo file: per conservare il lessico condiviso occorre
salvare anche il registro comune (a applicazione chiusa).

All'apertura di ciascun progetto vengono importati automaticamente i termini e
gli esempi delle precedenti annotazioni locali, senza duplicarli. Aprire una volta
i progetti già annotati per raccoglierne gli esempi. Le vecchie tabelle rimangono
nel progetto come copia precedente alla migrazione; i nuovi salvataggi e le
modifiche agiscono soltanto sul registro comune. Un esempio eliminato dal
registro non viene ricreato alla successiva apertura del progetto.

Ogni esempio conserva un'identità del progetto, documento, paziente, frammento,
contesto e impronta del testo. Per le nuove annotazioni viene archiviata anche
la versione completa del testo clinico visualizzato: più annotazioni della
stessa versione usano una sola copia, nella tabella `annotated_source_texts`.
Il registro è quindi autonomo: la lettura della copia non richiede il progetto
originale. Non viene copiato il PDF, ma il testo estratto utilizzato per annotare.

Per gli esempi precedenti alla migrazione, frammento e contesto vengono sempre
conservati; il testo completo viene acquisito solo se la fonte ancora disponibile
ha la stessa impronta della versione annotata. In caso contrario l'interfaccia
segnala che la versione completa non è disponibile, senza sostituirla con un testo
più recente. **Apri copia del testo annotato** visualizza la versione archiviata,
con il frammento selezionato, anche per gli esempi del progetto corrente.
 Documenti omonimi con gli stessi identificatori
in progetti diversi non vengono confusi. La data mostrata è quella del documento,
non una data dell'evento inferita automaticamente.

Nel pannello degli esempi è indicato anche il progetto sorgente. Cliccando un
esempio di un altro progetto si apre la copia del testo annotato (oppure frammento e contesto per gli esempi precedenti senza copia completa), senza
cambiare il progetto corrente. **Usa questo termine nel documento corrente**
riutilizza l'etichetta; non copia l'annotazione o il suo contesto. Per modificare
la selezione originale aprire il progetto di origine. Rinomine e fusioni dei
termini hanno effetto sull'intero registro condiviso.

Se il testo viene rielaborato, le vecchie annotazioni restano leggibili ma sono
segnalate **TESTO CAMBIATO** e non vengono evidenziate su coordinate obsolete.
Per riallinearle, aprire l'annotazione, selezionare il passaggio aggiornato e
salvare. Un controllo prima del salvataggio intercetta modifiche al file
avvenute mentre la scheda era aperta.

Queste annotazioni non modificano gold set, evidenze automatiche o storia
clinica. Sono disponibili come esempi didattici per l’estrazione LLM, secondo il flusso
descritto qui sotto. Eliminando un documento o un progetto, gli esempi già condivisi restano
disponibili come frammenti salvati nel registro; il documento originale potrebbe
non essere più consultabile.


## Usare le annotazioni per guidare il LLM

Nel pannello Lessico condiviso, selezionare un termine e aprire **Scheda evento**.
Scrivere cosa riconoscere e, una per riga, le informazioni da estrarre (per
esempio farmaco, dose, motivo). Non serve compilare tutti i termini subito:
anche il solo frammento annotato è utilizzabile come esempio.

Selezionare un esempio e usare **Imposta esempio / controesempio** per indicare
passaggi simili che non attestano quel tipo di evento. Un esempio negato può
restare un esempio del concetto con contesto Negato; un controesempio insegna
invece a non attribuire quel tipo di evento. Il modello deve comunque conservare
altri fatti espliciti eventualmente presenti nella stessa frase.

Durante l'estrazione degli eventi (**Eventi SNOMED → Elabora paziente** o
**▶ Elabora pazienti**) la pipeline legge il catalogo confermato e, per ogni
referto, invia come guida fino a dodici termini, scelti per sovrapposizione
lessicale con il testo, con al massimo due esempi ciascuno e un limite di 6.000
caratteri: il testo sorgente e la risposta hanno precedenza. Nessun documento
viene saltato perché non ha esempi corrispondenti, e gli esempi non limitano gli
eventi estraibili.

Un referto viene rielaborato solo se cambia la guida effettivamente inviata per
quel testo: modificare un termine o un esempio non pertinente non invalida le
estrazioni già fatte.

Gli esempi sono guida, non fonti per il paziente corrente. Date e valori devono
provenire dal referto corrente. I controlli strutturali non garantiscono la
correttezza clinica: gli effetti su sensibilità, precisione e velocità devono
essere misurati su un campione annotato, tenendo separati gli esempi di test da
quelli usati per guidare il modello. Non è stato effettuato fine-tuning.


## Proposte LLM nel documento attivo

Il pulsante **Analizza documento con LLM**, sopra il testo, analizza l'intero
documento attivo usando il modello configurato per estrazione e codifica. Se il
modello non è disponibile, un messaggio indica dove configurarlo: non viene
scelto silenziosamente un altro modello. Il nome del modello, il documento e
l'avanzamento sono mostrati durante l'analisi. **Interrompi analisi** richiede
l'arresto; l'eventuale chiamata LLM in corso deve prima terminare.

I risultati sono proposte temporanee nel pannello **Proposte LLM**, evidenziate
in blu sul testo. Il lessico fornisce esempi di guida con ricerca lessicale, ma
non limita il riconoscimento: sono proposti anche termini nuovi. Gli stati sono
**Presente**, **Assente**, **Non determinato**. Casi dubbi, ipotetici o relativi a
soggetti diversi dal paziente non sono marcati automaticamente come presenti;
nel tooltip si trovano soggetto, certezza e data proposta.

Selezionare una proposta nell'elenco o cliccare l'evidenziazione. Correggere il
termine, il contesto o la selezione del testo e premere **Salva** per confermare
solo quell'annotazione. **Scarta proposta selezionata** la rimuove dalle proposte.
Le evidenziazioni gialle identificano le annotazioni già salvate. Per eventi
sovrapposti sullo stesso passaggio utilizzare l'elenco delle proposte.

Nessuna proposta crea automaticamente termini, esempi, evidenze o eventi nella
storia clinica. La conferma salva il termine, il frammento corretto e il contesto
scelto nel registro condiviso, insieme alla copia del testo. I dettagli estratti
nella proposta non vengono automaticamente trasferiti alla storia clinica.

Le proposte non confermate restano in memoria durante la sessione, separate per
documento e versione del testo; chiudendo l'app o cambiando progetto si perdono.
Le annotazioni confermate sono invece persistenti. Una nuova versione del testo
non riceve le evidenziazioni della vecchia. Gli errori e le analisi parziali sono
segnalati esplicitamente; le proposte valide di un'analisi parziale restano
revisionabili. L'annullamento di una conferma ripristina la proposta pendente.


### Modello scelto all’avvio

Il pulsante **Analizza documento con LLM** apre una finestra dedicata al Lessico
condiviso: scegli modello, backend e parametri, poi premi **Avvia analisi**.
**Annulla** chiude la finestra senza avviare l’analisi né salvare la selezione.
Il modello deve essere disponibile localmente; dalla stessa finestra si possono
scaricare o importare modelli e aprire la diagnostica del motore.

Le preferenze sono salvate nel file applicativo `settings.json`, nella sezione
`pipeline_llm`, separatamente per ogni funzione. La configurazione del Lessico
non sostituisce quella ricordata per l’estrazione degli eventi.
Le preferenze sopravvivono al riavvio e sono condivise tra workspace; al primo
utilizzo vengono proposte le precedenti impostazioni del ruolo corrispondente.

Lo stesso meccanismo è disponibile per la normalizzazione dei documenti, per
l'estrazione e codifica degli eventi e per le prove dei singoli prompt. La coda
multipaziente chiede il modello una sola volta per tutta la coda.


Gli esempi del catalogo confermato guidano l'estrazione degli eventi
(**Eventi SNOMED → Elabora paziente** e **▶ Elabora pazienti**): per ogni referto
vengono inviati soltanto gli esempi dei termini più pertinenti. Modificare un
termine non pertinente a un referto non lo fa rielaborare.


### Esempi scritti senza documento

Nella lista **Lessico condiviso**, seleziona una voce e premi **Aggiungi esempio
scritto…**. Inserisci liberamente una frase, scegli lo stato (Presente, Assente,
Non determinato ecc.) ed eventualmente **Controesempio**, poi **Salva esempio**.
Puoi aggiungerne più di uno per termine, anche senza un documento attivo.
Questi testi sono salvati nel registro comune, senza documenti o pazienti fittizi.
Non sono evidenze cliniche: servono soltanto a guidare il riconoscimento dell’LLM.

La lista mostra direttamente le frasi, sia quelle selezionate nei documenti sia
quelle scritte a mano. Cliccando un esempio selezionato si legge soltanto il testo
salvato, senza aprire la fonte; cliccando un esempio scritto si può modificarlo.
**Elimina esempio selezionato** lo rimuove; **Annulla ultima modifica** consente di
ripristinarlo. Gli esempi originari con i loro ancoraggi restano conservati per
compatibilità e per le evidenziazioni nel testo, ma non occorre recuperarne il
documento per usarli nel Lessico.

Tutti gli esempi alimentano la selezione dei testi didattici pertinenti per ogni
termine. Le modifiche aggiornano l’impronta del catalogo: la successiva estrazione
non riutilizza risultati calcolati con una versione diversa degli esempi.

## Scheda evento strutturata

In **Lessico condiviso → seleziona un termine → Scheda evento** puoi configurare:
- categoria (anche personalizzata), definizione e criteri di esclusione;
- tipo di valore: nessuno, numero, testo, scelta da elenco o composto;
- campi con nome, tipo, unità e opzioni ammesse;
- regola di deduplicazione per condizione, misurazione, azione o terapia.

Il termine è il concetto stabile: ad esempio “Pressione arteriosa”, categoria
“Parametri vitali”. Per un valore composto configura “sistolica” e “diastolica”,
entrambe numeriche in mmHg. I valori 150 e 80 appartengono all'occorrenza estratta,
non diventano nuovi termini del lessico. Per un valore semplice il primo campo
è il valore principale. I campi assenti nel documento restano mancanti.

Per le terapie sono consigliati i campi `farmaco`, `dose`, `ambito` e `stato`.
La regola terapia richiede farmaco, dose e stato noti e concordanti; `ambito` può
avere opzioni come oncologico, domiciliare cronico e altro. Le condizioni compatibili
si raggruppano entro 15 giorni dalla prima osservazione. Misurazioni e azioni
richiedono stessa data e identificatore esplicito dell'osservazione/episodio.
Le vecchie schede mantengono i campi testuali e la deduplicazione automatica.

Le schede sono salvate nel registro condiviso e modificabili con annullamento.
Categoria e valori estratti compaiono nella storia clinica. Cambiare una scheda
invalida la cache di estrazione; per aggiornare eventi già estratti occorre
rieseguire l'estrazione. Nessun dato clinico precedente viene riscritto all'apertura
 della finestra di configurazione.
