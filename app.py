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

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

SIMPLE_GCN_WEIGHTS = os.path.join(MODELS_DIR, "simple_gcn.pth")
CHEMBERTA_WEIGHTS = os.path.join(MODELS_DIR, "chemberta.pth")
# modello_pesi_lista_0.pth … modello_pesi_lista_26.pth
N_MAJORITY = 27
MAJORITY_PATTERN = os.path.join(MODELS_DIR, "modello_pesi_lista_{}.pth")

THRESHOLD = 0.5
DEVICE = torch.device("cpu")  # CPU is fine for a single molecule

st.set_page_config(page_title="HIV Inhibitor Predictor", page_icon="🧬", layout="wide")


# MODEL LOADING (cached: loaded only once)

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
    return models


@st.cache_resource(show_spinner=True)
def load_chemberta():
    if not os.path.exists(CHEMBERTA_WEIGHTS):
        return None, None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL_NAME)
    model = ChemBERTaClassifier(freeze_backbone=True)
    state = torch.load(CHEMBERTA_WEIGHTS, map_location=DEVICE)
    # strict=False: works whether the full model or just the head was saved
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model, tokenizer


# FEATURISATION / INFERENCE
def smiles_to_graph(smiles):
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
    """Returns (mean_prob, n_inhibitor_votes, n_models)."""
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


# MOLECULE VISUALISATION

def draw_2d(mol):
    img = Draw.MolToImage(mol, size=(420, 420))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_3d_html(mol):
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



# UI
st.title("🧬 HIV Inhibitor Predictor")
st.caption(
    "Enter a SMILES string: the app will render it and run the three models "
    "(**SimpleGNN**, **Majority**, **ChemBERTa**) to estimate its effectiveness "
    "as an HIV inhibitor."
)

simple_model = load_simple_gcn()
majority_models = load_majority_models()
chemberta_model, chemberta_tokenizer = load_chemberta()

with st.sidebar:
    st.header("Model status")
    st.write("✅ SimpleGNN" if simple_model else "⚠️ SimpleGNN — weights missing")
    if majority_models:
        st.write(f"✅ Majority — {len(majority_models)}/{N_MAJORITY} sub-models")
    else:
        st.write("⚠️ Majority — weights missing")
    st.write("✅ ChemBERTa" if chemberta_model else "⚠️ ChemBERTa — weights missing")
    st.divider()
    st.caption(
        f"Weights are looked up in:\n`{MODELS_DIR}`\n\n"
        "See the README for the expected file names and the snippet to export "
        "SimpleGNN and ChemBERTa from the notebook."
    )
    show_3d = st.checkbox("Also show 3D structure", value=False)

examples = {
    "— write your own —": "",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Zidovudine (AZT, anti-HIV)": "Cc1cn(C2CC(N=[N+]=[N-])C(CO)O2)c(=O)[nH]c1=O",
}
col_a, col_b = st.columns([1, 2])
with col_a:
    choice = st.selectbox("Examples", list(examples.keys()))
with col_b:
    smiles = st.text_input("SMILES", value=examples[choice], placeholder="E.g. CC(=O)Oc1ccccc1C(=O)O")

run = st.button("Analyse molecule", type="primary", use_container_width=True)

if run:
    smiles = (smiles or "").strip()
    if not smiles:
        st.warning("Please enter a SMILES string.")
        st.stop()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES — RDKit could not parse it. Please check the syntax.")
        st.stop()

    canonical = Chem.MolToSmiles(mol)
    st.success(f"Valid SMILES. Canonical form: `{canonical}`")

    st.subheader("Structure")
    cols = st.columns(2 if show_3d else 1)
    with cols[0]:
        st.image(draw_2d(mol), caption="2D", use_column_width=False)
    if show_3d:
        with cols[1]:
            html = render_3d_html(mol)
            if html:
                st.components.v1.html(html, height=440)
            else:
                st.info("3D view not available (install `py3Dmol` or embedding failed).")

    st.subheader("Model predictions")
    results = []  # (name, inhibitor_verdict: bool|None, confidence: float|None, extra: str)

    # 1) SimpleGNN
    if simple_model is not None:
        p = predict_simple_gcn(simple_model, smiles)
        results.append(("SimpleGNN", p > THRESHOLD, p, ""))
    else:
        results.append(("SimpleGNN", None, None, "weights missing"))

    # 2) Majority
    if majority_models:
        mean_p, votes, n = predict_majority(majority_models, smiles)
        extra = f"{votes}/{n} sub-models vote inhibitor"
        results.append(("Majority", mean_p > THRESHOLD, mean_p, extra))
    else:
        results.append(("Majority", None, None, "weights missing"))

    # 3) ChemBERTa
    if chemberta_model is not None:
        p = predict_chemberta(chemberta_model, chemberta_tokenizer, smiles)
        results.append(("ChemBERTa", p > THRESHOLD, p, ""))
    else:
        results.append(("ChemBERTa", None, None, "weights missing"))

    cols = st.columns(3)
    for col, (name, verdict, prob, extra) in zip(cols, results):
        with col:
            st.markdown(f"### {name}")
            if verdict is None:
                st.info("Not available\n\n" + extra)
                continue
            label = "🟢 Inhibitor" if verdict else "🔴 Non-inhibitor"
            # confidence is the probability toward the predicted class
            conf = prob if verdict else (1 - prob)
            st.metric(label="Verdict", value=label)
            st.metric(label="Confidence", value=f"{conf*100:.1f}%")
            st.progress(prob, text=f"P(inhibitor) = {prob*100:.1f}%")
            if extra:
                st.caption(extra)

    available = [r for r in results if r[1] is not None]
    if available:
        n_inhibitor = sum(1 for r in available if r[1])
        n_total = len(available)
        n_non = n_total - n_inhibitor

        st.subheader("Collective verdict")
        c1, c2, c3 = st.columns(3)
        c1.metric("Models saying INHIBITOR", f"{n_inhibitor}/{n_total}")
        c2.metric("Models saying NON-INHIBITOR", f"{n_non}/{n_total}")
        consensus = "INHIBITOR" if n_inhibitor > n_non else (
            "NON-INHIBITOR" if n_non > n_inhibitor else "TIE"
        )
        c3.metric("Consensus", consensus)

        if n_total < 3:
            st.caption(
                f"Note: only {n_total} model(s) available. "
                "Load the missing weights for a full 3-model vote."
            )
    else:
        st.error("No models available: load at least one set of weights into the `models/` folder.")
