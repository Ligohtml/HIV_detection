# 🧬 Predizione di inibitori dell'HIV con tre approcci di Machine Learning

## In breve

Questo progetto affronta un problema di **classificazione binaria** in chimica
computazionale: data una molecola, prevedere se è in grado di **inibire il
virus HIV** oppure no. È un classico problema di *drug discovery* — vagliare
in silico migliaia di composti per individuare i candidati promettenti prima
di passare ai test di laboratorio, molto più costosi.

La sfida non è solo costruire un classificatore, ma confrontare **tre filosofie
diverse** per rappresentare e modellare le molecole, e capire quale regge
meglio in presenza di un dataset fortemente sbilanciato.

---

## Il dataset

Si usa il dataset **HIV di MoleculeNet** (accessibile via PyTorch Geometric).
Ogni elemento è una molecola descritta da una stringa **SMILES** (la notazione
testuale standard per le strutture chimiche) con un'etichetta binaria:

- `1` → **inibitore** (la molecola blocca la replicazione dell'HIV)
- `0` → **non inibitore**

Il problema centrale del dataset è lo **sbilanciamento estremo**: solo una
piccola frazione delle molecole (intorno al 3–4%) è un inibitore. In pratica
ci sono circa **27 non inibitori per ogni inibitore**. Un modello "pigro" che
predice sempre "non inibitore" otterrebbe oltre il 96% di accuratezza pur
essendo inutile. Per questo l'accuratezza grezza è fuorviante e la metrica di
riferimento del progetto è il **ROC-AUC**, che misura la capacità di
distinguere le due classi indipendentemente dalla soglia.

---

## Come si rappresenta una molecola

Il progetto sfrutta **due rappresentazioni diverse** della stessa molecola, ed
è questa scelta a definire le famiglie di modelli:

1. **Grafo molecolare.** Una molecola è naturalmente un grafo: gli atomi sono i
   nodi, i legami chimici sono gli archi. Con `from_smiles` di PyTorch Geometric
   ogni atomo viene descritto da **9 feature** (numero atomico, carica, ibridazione,
   ecc.) e i legami diventano la mappa delle connessioni. Questa rappresentazione
   alimenta le reti neurali a grafo (GNN).

2. **Stringa di testo.** Lo SMILES è di fatto una sequenza di caratteri, quindi
   può essere trattato come "linguaggio" da un modello di tipo Transformer
   pre-addestrato sulla chimica. Questa rappresentazione alimenta ChemBERTa.

---

## I tre modelli

### 1. SimpleGNN — la baseline a grafo

Una **Graph Convolutional Network** semplice: tre strati di convoluzione su
grafo (`GCNConv`) che fanno *message passing* tra atomi vicini, seguiti da un
*global mean pooling* che condensa l'intera molecola in un unico vettore, e
infine uno strato lineare che produce la predizione.

Per gestire lo sbilanciamento, due accorgimenti:

- **Loss pesata** (`BCEWithLogitsLoss` con `pos_weight ≈ 27`): l'errore sugli
  inibitori "pesa" 27 volte di più, costringendo il modello a non ignorarli.
- **SMILES enumeration** (data augmentation): la stessa molecola può essere
  scritta con SMILES diversi a seconda dell'ordine di partenza. Durante il
  training, con una certa probabilità, la molecola viene riscritta in una forma
  equivalente. Così il modello impara le **proprietà chimiche reali** invece di
  memorizzare un particolare ordine di scrittura.

È il punto di riferimento rispetto a cui valutare gli altri due approcci.

### 2. Majority — ensemble con undersampling e voto di maggioranza

Questo approccio attacca lo sbilanciamento dal lato dei **dati** anziché della
loss. L'idea:

1. Si dividono i (tanti) **non inibitori** in **27 sottoinsiemi disgiunti**, ognuno
   di dimensione paragonabile al numero di inibitori.
2. Si crea così un **dataset bilanciato** per ciascun sottoinsieme, accoppiandolo
   con gli inibitori.
3. Si addestrano **27 GNN indipendenti**, una per ogni dataset bilanciato. Ogni
   rete è una variante della GCN (due strati convoluzionali, con pooling sia
   *mean* che *max* concatenati, più due strati lineari).
4. In predizione, le 27 reti votano e si prende la **maggioranza**.

L'intuizione è quella del *bagging*: tanti classificatori "deboli" ma diversi,
combinati, sono più robusti di uno singolo, e ciascuno vede tutti gli inibitori
senza essere sommerso dai non inibitori.

> **Limite noto.** I sottoinsiemi dei non inibitori sono disgiunti (bene), ma gli
> inibitori usati sono gli **stessi** in tutti i 27 modelli. Questo riduce la
> diversità dell'ensemble sul lato positivo: i 27 modelli "vedono" esattamente le
> stesse molecole attive. È un punto migliorabile (es. bootstrap anche sui
> positivi) ma non invalida il voto di maggioranza.

### 3. ChemBERTa — transfer learning

Qui si cambia paradigma. Invece di addestrare da zero, si parte da
**ChemBERTa** (`DeepChem/ChemBERTa-77M-MLM`), un Transformer già pre-addestrato
su milioni di SMILES e che quindi "conosce" già la grammatica delle molecole.

La strategia è il **transfer learning con backbone congelato**:

- Si **congelano** tutti i pesi del Transformer pre-addestrato (non vengono più
  aggiornati): fanno da estrattore di feature.
- Si addestra **solo una piccola testa di classificazione** (due strati lineari
  con ReLU e dropout) sopra la rappresentazione del token `[CLS]`.

Il vantaggio è che si riutilizza la conoscenza chimica già appresa da un modello
enorme, allenando solo una manciata di parametri: veloce, stabile e tipicamente
efficace anche con pochi dati etichettati. Anche qui la loss è pesata in base al
rapporto reale tra negativi e positivi del training.

---

## Come vengono valutati i modelli

Tutti e tre i modelli sono validati con la stessa logica:

- **ROC-AUC** come metrica principale (robusta allo sbilanciamento).
- **Matrice di confusione** e **classification report** (precision, recall,
  f1) per capire *dove* il modello sbaglia — in questo dominio i **falsi
  negativi** (un vero inibitore scartato) sono spesso più gravi dei falsi
  positivi.
- Curve di **loss** e di **AUC** lungo le epoche per diagnosticare overfitting.

---

## L'app

A corredo del progetto c'è un'**app Streamlit** che rende tutto interattivo:
si inserisce uno SMILES, l'app lo **visualizza** (struttura 2D e 3D), e lancia i
**tre modelli separatamente**, mostrando per ciascuno la **confidence** della
predizione. In fondo confronta i verdetti e fa la **conta** di quanti modelli
classificano la molecola come inibitore e quanti no — un piccolo "consiglio di
esperti" in cui tre approcci diversi si esprimono sullo stesso composto.

---

## In sintesi: l'idea del progetto

> Lo stesso problema — *questa molecola inibisce l'HIV?* — affrontato da tre
> angolazioni complementari: una **GNN baseline** che gestisce lo sbilanciamento
> con loss pesata e augmentation; un **ensemble** che lo gestisce ribilanciando i
> dati e votando a maggioranza; e un **Transformer pre-addestrato** che porta
> conoscenza chimica esterna via transfer learning. Confrontarli, e farli votare
> insieme, è il cuore del lavoro.
