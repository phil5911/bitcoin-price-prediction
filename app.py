import streamlit as st
import pandas as pd
import numpy as np
import tensorflow_decision_forests as tfdf
import matplotlib.pyplot as plt
import math
import os

# Chemin vers les données
DATA_PATH = "data/main.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["target"] = df["Close"].shift(-1)
    df = df.dropna()
    return df

@st.cache_resource
def train_model(train_df):
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="target", task=tfdf.keras.Task.REGRESSION)
    model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
    model.fit(train_ds)
    model.compile(metrics=["mse", "mae"])
    return model

def main():
    st.title("Prédiction prix Bitcoin avec TensorFlow Decision Forests")

    if not os.path.exists(DATA_PATH):
        st.error(f"Le fichier {DATA_PATH} est introuvable.")
        return

    df = load_data(DATA_PATH)
    st.write("Aperçu des données :", df.head())

    # Découpage train/test
    train_df = df.iloc[:150000].copy()
    test_df = df.iloc[150000:].copy()

    st.write(f"Taille train: {train_df.shape}, taille test: {test_df.shape}")

    # Entraînement
    with st.spinner("Entraînement du modèle..."):
        model = train_model(train_df)

    # Préparation dataset test
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="target", task=tfdf.keras.Task.REGRESSION)

    # Évaluation
    evaluation = model.evaluate(test_ds, return_dict=True)
    st.write("Évaluation du modèle :", evaluation)
    st.write(f"RMSE: {math.sqrt(evaluation['mse']):.4f}")

    # Prédictions
    predictions = model.predict(test_ds)
    y_pred = np.array([pred[0] for pred in predictions])
    test_df = test_df.copy()
    test_df["pred"] = y_pred

    st.write("Exemple de prédiction:")
    i = st.slider("Choisir un index de prédiction", 0, len(test_df)-1, 1)
    vraie_val = test_df['target'].iloc[i]
    prediction = test_df['pred'].iloc[i]
    erreur_pourcent = abs(vraie_val - prediction) / vraie_val * 100

    st.write(f"Valeur réelle : {vraie_val}")
    st.write(f"Valeur prédite : {prediction}")
    st.write(f"Erreur relative : {erreur_pourcent:.2f}%")

    # Graphique Prédictions vs Réel
    st.subheader("Prédictions vs Valeurs réelles")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(test_df["target"], test_df["pred"], alpha=0.5)
    ax.plot([test_df["target"].min(), test_df["target"].max()],
            [test_df["target"].min(), test_df["target"].max()],
            color='red', linestyle='--')
    ax.set_xlabel("Valeurs réelles")
    ax.set_ylabel("Valeurs prédites")
    ax.set_title("Scatter plot Prédictions vs Réel")
    st.pyplot(fig)

if __name__ == "__main__":
    main()





