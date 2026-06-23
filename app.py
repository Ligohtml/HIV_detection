"""
HIV Inhibitor Predictor — App Streamlit
=======================================

Passa uno SMILES, lo visualizza (2D + 3D opzionale) e lancia i TRE modelli
separatamente per stimare se la molecola inibisce o meno l'HIV:

  1. SimpleGNN          (una GCN a 3 strati)
  2. Majority           (ensemble di 27 GCN, voto di maggioranza)
  3. ChemBERTa          (transfer learning su SMILES)

Per ogni modello mostra la CONFIDENCE di predizione, e in fondo fa la CONTA
di quanti dei tre modelli dicono "inibitore" e quanti no.

Avvio:
    streamlit run app.py
"""

import os
import io

import streamlit as st
import torch
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from torch_geometric.utils import from_smiles

from models_def import (
    SimpleGCN,
    MajorityGCN,
    ChemBERTaClassifier,
    CHEMBERTA_MODEL_NAME,
)

# ---------------------------------------------------------------------------
# CONFIGURAZIONE — qui dici all'app dove sono i pesi salvati
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# File attesi dentro MODELS_DIR:
SIMPLE_GCN_WEIGHTS = os.path.join(MODELS_DIR, "simple_gcn.pth")
CHEMBERTA_WEIGHTS = os.path.join(MODELS_DIR, "chemberta.pth")
# i 27 majority: modello_pesi_lista_0.pth ... modello_pesi_lista_26.pth
N_MAJORITY = 27
MAJORITY_PATTERN = os.path.join(MODELS_DIR, "modello_pesi_lista_{}.pth")

THRESHOLD = 0.5  # soglia di decisione (prob > 0.5 -> inibitore)
DEVICE = torch.device("cpu")  # in locale la CPU va benissimo per 1 molecola

st.set_page_config(page_title="HIV Inhibitor Predictor", page_icon="🧬", layout="wide")


# ===========================================================================
# CARICAMENTO MODELLI (cache: si caricano una sola volta)
# ===========================================================================
@st.cache_resource(show_spinner=False)
def load_simple_gcn():
    if not os.path.exists(SIMPLE_GCN_WEIGHTS):
        return None
    model = SimpleGCN()
    state = torch.load(SIMPLE_GCN_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


@st.cache_resource(show_spinner=False)
def load_majority_models():
    models = []
    for i in range(N_MAJORITY):
        path = MAJORITY_PATTERN.format(i)
        if not os.path.exists(path):
            continue
        m = MajorityGCN()
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.to(DEVICE).eval()
        models.append(m)
    return models  # lista (può essere vuota)


@st.cache_resource(show_spinner=True)
def load_chemberta():
    if not os.path.exists(CHEMBERTA_WEIGHTS):
        return None, None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL_NAME)
    model = ChemBERTaClassifier(freeze_backbone=True)
    state = torch.load(CHEMBERTA_WEIGHTS, map_location=DEVICE)
    # strict=False: funziona sia se hai salvato tutto il modello sia solo la testa
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model, tokenizer


# ===========================================================================
# FEATURIZZAZIONE / INFERENZA
# ===========================================================================
def smiles_to_graph(smiles):
    """SMILES -> oggetto grafo PyG pronto per le GCN (singola molecola)."""
    data = from_smiles(smiles)
    x = data.x.float().to(DEVICE)
    edge_index = data.edge_index.long().to(DEVICE)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=DEVICE)
    return x, edge_index, batch


@torch.no_grad()
def predict_simple_gcn(model, smiles):
    x, edge_index, batch = smiles_to_graph(smiles)
    logit = model(x, edge_index, batch).squeeze()
    prob = torch.sigmoid(logit).item()
    return prob


@torch.no_grad()
def predict_majority(models, smiles):
    """Ritorna (prob_media, n_voti_inibitore, n_modelli)."""
    x, edge_index, batch = smiles_to_graph(smiles)
    probs = []
    for m in models:
        logit = m(x, edge_index, batch).squeeze()
        probs.append(torch.sigmoid(logit).item())
    probs = np.array(probs)
    votes_inhibitor = int((probs > THRESHOLD).sum())
    return float(probs.mean()), votes_inhibitor, len(models)


@torch.no_grad()
def predict_chemberta(model, tokenizer, smiles):
    enc = tokenizer(
        smiles,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    logit = model(input_ids, attention_mask).squeeze()
    prob = torch.sigmoid(logit).item()
    return prob


# ===========================================================================
# VISUALIZZAZIONE MOLECOLA
# ===========================================================================
def draw_2d(mol):
    img = Draw.MolToImage(mol, size=(420, 420))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_3d_html(mol):
    """Vista 3D interattiva con py3Dmol (se installato)."""
    try:
        import py3Dmol
    except ImportError:
        return None
    m = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(m, randomSeed=42, useRandomCoords=True) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(m)
    block = Chem.MolToMolBlock(m)
    view = py3Dmol.view(width=420, height=420)
    view.addModel(block, "mol")
    view.setStyle({"stick": {}})
    view.setBackgroundColor("white")
    view.zoomTo()
    return view._make_html()


# ===========================================================================
# UI
# ===========================================================================
st.title("🧬 HIV Inhibitor Predictor")
st.caption(
    "Inserisci uno SMILES: l'app lo rappresenta e fa girare i tre modelli "
    "(**SimpleGNN**, **Majority**, **ChemBERTa**) per stimare l'efficacia "
    "come inibitore dell'HIV."
)

# --- caricamento modelli + avvisi sui pesi mancanti ---
simple_model = load_simple_gcn()
majority_models = load_majority_models()
chemberta_model, chemberta_tokenizer = load_chemberta()

with st.sidebar:
    st.header("Stato modelli")
    st.write("✅ SimpleGNN" if simple_model else "⚠️ SimpleGNN — pesi mancanti")
    if majority_models:
        st.write(f"✅ Majority — {len(majority_models)}/{N_MAJORITY} sotto-modelli")
    else:
        st.write("⚠️ Majority — pesi mancanti")
    st.write("✅ ChemBERTa" if chemberta_model else "⚠️ ChemBERTa — pesi mancanti")
    st.divider()
    st.caption(
        f"I pesi vengono cercati in:\n`{MODELS_DIR}`\n\n"
        "Vedi il README per i nomi file attesi e lo snippet per esportare "
        "SimpleGNN e ChemBERTa dal notebook."
    )
    show_3d = st.checkbox("Mostra anche la struttura 3D", value=False)

# --- input SMILES ---
examples = {
    "— scrivi il tuo —": "",
    "Aspirina": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeina": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Azidotimidina (AZT, anti-HIV)": "Cc1cn(C2CC(N=[N+]=[N-])C(CO)O2)c(=O)[nH]c1=O",
}
col_a, col_b = st.columns([1, 2])
with col_a:
    choice = st.selectbox("Esempi", list(examples.keys()))
with col_b:
    smiles = st.text_input("SMILES", value=examples[choice], placeholder="Es. CC(=O)Oc1ccccc1C(=O)O")

run = st.button("Analizza molecola", type="primary", use_container_width=True)

if run:
    smiles = (smiles or "").strip()
    if not smiles:
        st.warning("Inserisci uno SMILES.")
        st.stop()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("SMILES non valido — RDKit non riesce a interpretarlo. Controlla la sintassi.")
        st.stop()

    canonical = Chem.MolToSmiles(mol)
    st.success(f"SMILES valido. Forma canonica: `{canonical}`")

    # --- struttura ---
    st.subheader("Struttura")
    cols = st.columns(2 if show_3d else 1)
    with cols[0]:
        st.image(draw_2d(mol), caption="2D", use_container_width=False)
    if show_3d:
        with cols[1]:
            html = render_3d_html(mol)
            if html:
                st.components.v1.html(html, height=440)
            else:
                st.info("Vista 3D non disponibile (installa `py3Dmol` o embedding fallito).")

    # --- predizioni dei tre modelli ---
    st.subheader("Predizioni dei modelli")
    results = []  # (nome, verdetto_inibitore: bool|None, confidence: float|None, extra: str)

    # 1) SimpleGNN
    if simple_model is not None:
        p = predict_simple_gcn(simple_model, smiles)
        results.append(("SimpleGNN", p > THRESHOLD, p, ""))
    else:
        results.append(("SimpleGNN", None, None, "pesi mancanti"))

    # 2) Majority
    if majority_models:
        mean_p, votes, n = predict_majority(majority_models, smiles)
        extra = f"{votes}/{n} sotto-modelli votano inibitore"
        results.append(("Majority", mean_p > THRESHOLD, mean_p, extra))
    else:
        results.append(("Majority", None, None, "pesi mancanti"))

    # 3) ChemBERTa
    if chemberta_model is not None:
        p = predict_chemberta(chemberta_model, chemberta_tokenizer, smiles)
        results.append(("ChemBERTa", p > THRESHOLD, p, ""))
    else:
        results.append(("ChemBERTa", None, None, "pesi mancanti"))

    cols = st.columns(3)
    for col, (name, verdict, prob, extra) in zip(cols, results):
        with col:
            st.markdown(f"### {name}")
            if verdict is None:
                st.info("Non disponibile\n\n" + extra)
                continue
            label = "🟢 Inibitore" if verdict else "🔴 Non inibitore"
            # la confidence è la probabilità verso la classe predetta
            conf = prob if verdict else (1 - prob)
            st.metric(label="Verdetto", value=label)
            st.metric(label="Confidence", value=f"{conf*100:.1f}%")
            st.progress(prob, text=f"P(inibitore) = {prob*100:.1f}%")
            if extra:
                st.caption(extra)

    # --- conta finale ---
    available = [r for r in results if r[1] is not None]
    if available:
        n_inhibitor = sum(1 for r in available if r[1])
        n_total = len(available)
        n_non = n_total - n_inhibitor

        st.subheader("Verdetto collettivo")
        c1, c2, c3 = st.columns(3)
        c1.metric("Modelli che dicono INIBITORE", f"{n_inhibitor}/{n_total}")
        c2.metric("Modelli che dicono NON inibitore", f"{n_non}/{n_total}")
        consensus = "INIBITORE" if n_inhibitor > n_non else (
            "NON INIBITORE" if n_non > n_inhibitor else "PARITÀ"
        )
        c3.metric("Consenso", consensus)

        if n_total < 3:
            st.caption(
                f"Nota: solo {n_total} modello/i disponibile/i. "
                "Carica i pesi mancanti per il voto completo a 3."
            )
    else:
        st.error("Nessun modello disponibile: carica almeno un set di pesi nella cartella `models/`.")
