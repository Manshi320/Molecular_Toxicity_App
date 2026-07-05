import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Molecular Toxicity Prediction",
    page_icon="🧪",
    layout="centered"
)

st.title("🧪 AI-Based Molecular Toxicity Prediction")
st.write("Predict whether a molecule is Toxic or Non-Toxic using a Random Forest model.")
st.info("""
### Sample SMILES

- Ethanol → CCO
- Benzene → c1ccccc1
- Acetic Acid → CC(=O)O
- Aspirin → CC(=O)OC1=CC=CC=C1C(=O)O
""")
st.sidebar.title("About")


st.sidebar.info("""
This application predicts whether a molecule is Toxic or Non-Toxic using a Random Forest Machine Learning model.

Developed using:
- RDKit
- Scikit-learn
- Streamlit
""")
# -------------------------------
# LOAD MODEL
# -------------------------------

@st.cache_resource
def load_model():
    return joblib.load("RandomForest_model.pkl")

model = load_model()

# -------------------------------
# LOAD FEATURE TEMPLATE
# -------------------------------

feature_template = pd.read_csv("master_features.csv")

feature_columns = feature_template.columns.tolist()

if "Label" in feature_columns:
    feature_columns.remove("Label")

# -------------------------------
# FEATURE EXTRACTION FUNCTION
# -------------------------------

def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol)
    ]

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )

    features = descriptors + list(fp)

    return pd.DataFrame([features], columns=feature_columns)
        


        # -------------------------------
# USER INPUT
# -------------------------------

smiles = st.text_input(
    "Enter SMILES String",
    placeholder="Example: CCO (Ethanol)"
)
st.caption(
    "ℹ️ Note: Predictions are generated using a Random Forest model trained on a specific molecular toxicity dataset. "
    "Predictions are most reliable for compounds similar to those in the training dataset and should be used for screening purposes only."
)
if st.button("Predict"):

    if smiles.strip() == "":
        st.warning("Please enter a SMILES string.")
        st.stop()

    X = smiles_to_features(smiles)

    if X is None:
        st.error("Invalid SMILES string.")
        st.stop()


mol = Chem.MolFromSmiles(smiles)

if mol:
    #st.subheader("Molecule Structure")
    #img = Draw.MolToImage(mol, size=(300, 300))
    #st.image(img)

    st.subheader("Molecular Properties")
    st.write("Molecular Weight:", Descriptors.MolWt(mol))
    st.write("LogP:", Descriptors.MolLogP(mol))
    st.write("H-Bond Donors:", Descriptors.NumHDonors(mol))
    st.write("H-Bond Acceptors:", Descriptors.NumHAcceptors(mol))
    st.write("TPSA:", Descriptors.TPSA(mol))
    st.write("Rotatable Bonds:", Descriptors.NumRotatableBonds(mol))
        # Make prediction
prediction = model.predict(X)[0]
probability = model.predict_proba(X)[0]

st.subheader("Prediction")


if prediction == 1:
    st.error("☠️ Toxic")
    confidence = probability[1]
else:
    st.success("✅ Non-Toxic")
    confidence = probability[0]

st.subheader("Prediction Confidence")
st.progress(float(confidence))
st.write(f"Confidence: {confidence*100:.2f}%")
st.subheader("Class Probabilities")

prob_df = pd.DataFrame({
    "Class": ["Non-Toxic", "Toxic"],
    "Probability": [
        probability[0]*100,
        probability[1]*100
    ]
})

st.dataframe(prob_df, hide_index=True)


st.write("🕒 Prediction Time:", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))

st.markdown("---")
st.caption("Developed by MANSHI")
