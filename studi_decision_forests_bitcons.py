#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import os

path = "/mnt/c/Users/phili/OneDrive/Images/Bitcoin-studi/archive/main.csv"
print(os.path.exists(path))  # doit afficher True si le fichier est accessible


# In[3]:


import pandas as pd

df = pd.read_csv(path)
print(df.head())


# In[4]:


df["target"] = df["Close"].shift(-1)


# In[5]:


df["target_2"] = df["Close"].shift(-2)  # J+2


# In[6]:


df["target_3"] = df["Close"].shift(-3)  # J+3


# In[7]:


df.tail()


# In[8]:


df = df.dropna()


# In[9]:


df.isnull().sum()


# In[10]:


df.head()


# In[11]:


df.shape


# In[12]:


import tensorflow_decision_forests as tfdf


# In[13]:


import tensorflow_decision_forests as tfdf
print(tfdf.__version__)


# In[14]:


train_df = df.iloc[:150000].copy()


# In[16]:


train_df.shape


# In[17]:


test_df = df.iloc[150000:].copy()


# In[19]:


print(train_df.columns)


# In[20]:


test_df.shape


# In[21]:


train_df.head()


# In[22]:


train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="target", task=tfdf.keras.Task.REGRESSION)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="target", task=tfdf.keras.Task.REGRESSION)


# In[23]:


model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)


# In[55]:


model.fit(train_ds)


# In[28]:


import math


# In[60]:


model.compile(metrics=["mse", "mae"])
evaluation = model.evaluate(test_ds, return_dict=True)

print(evaluation)
print()
print(f"MSE: {evaluation['mae']}")
print(f"MSE: {evaluation['mse']}")
print(f"RMSE: {math.sqrt(evaluation['mse'])}")


# In[30]:


y_pred = model.predict(test_ds)


# In[61]:


# Prédire
predictions = model.predict(test_ds)
y_pred = np.array([pred[0] for pred in predictions])

# Ajouter la colonne
test_df = test_df.copy()
test_df["pred"] = y_pred

# Accéder à la prédiction n°1
print("Valeur prédite :", test_df["pred"].iloc[1])
print("Valeur réelle :", test_df["target"].iloc[1])


# In[62]:


i = 1  # position (0 = première ligne)
vraie_val = test_df['target'].iloc[i]
prediction = test_df['pred'].iloc[i]
erreur_pourcent = abs(vraie_val - prediction) / vraie_val * 100

print(f"Exemple #{i}")
print(f"Valeur réelle   : {vraie_val}")
print(f"Valeur prédite  : {prediction}")
print(f"Erreur relative : {erreur_pourcent:.2f}%")


# In[58]:


y_true = test_df["target"].values


# In[56]:


get_ipython().system('pip install matplotlib')



# In[32]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Prédictions vs Réel (prix Bitcoin)")
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')  # ligne idéale
plt.grid(True)
plt.show()


# In[41]:


plt.figure(figsize=(12, 6))
plt.plot(results["true"][:100], label="Valeur réelle (J+3)")
plt.plot(results["predicted"][:100], label="Valeur prédite (J+3)")
plt.title("Prédiction du Bitcoin à J+3")
plt.xlabel("Exemple")
plt.ylabel("Prix")
plt.legend()
plt.grid(True)
plt.show()


# In[34]:


plt.figure(figsize=(8, 6))
plt.scatter(results["true"], results["predicted"], alpha=0.5)
plt.plot([results["true"].min(), results["true"].max()],
         [results["true"].min(), results["true"].max()],
         color='red', linestyle='--', label="Idéal")
plt.xlabel("Valeur réelle")
plt.ylabel("Valeur prédite")
plt.title("Valeur réelle vs prédite (Scatter Plot)")
plt.legend()
plt.grid(True)
plt.show()


# In[35]:


results["abs_error"] = abs(results["true"] - results["predicted"])

plt.figure(figsize=(12, 6))
plt.plot(results["abs_error"][:200])
plt.title("Erreur absolue sur les 200 premières prédictions")
plt.xlabel("Exemple")
plt.ylabel("Erreur absolue")
plt.grid(True)
plt.show()


# In[36]:


results["error"] = results["predicted"] - results["true"]

plt.figure(figsize=(10, 5))
plt.hist(results["error"], bins=50, edgecolor='black')
plt.title("Distribution des erreurs de prédiction")
plt.xlabel("Erreur (prédite - vraie)")
plt.ylabel("Fréquence")
plt.grid(True)
plt.show()


# In[77]:


pip install -U scikit-learn


# In[42]:


get_ipython().system('pip install streamlit')


# In[63]:


tfdf.model_plotter.plot_model_in_colab(model, tree_idx=0)


# In[64]:


inspector = model.make_inspector()


# In[66]:


print(f"Available variable importances:")
for importance in inspector.variable_importances().keys():
    print(f"\t{importance}")


# In[67]:


inspector.evaluation()


# In[69]:


inspector.variable_importances()["INV_MEAN_MIN_DEPTH"]


# In[70]:


inspector.variable_importances()["SUM_SCORE"]


# In[71]:


inspector.variable_importances()["NUM_NODES"]


# In[72]:


import matplotlib.pyplot as plt

importances = {
    "Close": 365800.0,
    "High": 268594.0,
    "Low": 263764.0,
    "Open_Time": 251016.0,
    "Close_Time": 250823.0,
    "Open": 241831.0,
    "Number_of_trades": 225568.0,
    "Volume": 162969.0,
    "Taker_buy_quote_asset_volume": 162926.0,
    "Quote_asset_volume": 162795.0,
    "Taker_buy_base_asset_volume": 162623.0
}

# Trier par importance décroissante
importances_sorted = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

plt.figure(figsize=(12, 6))
plt.bar(importances_sorted.keys(), importances_sorted.values(), color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Importance')
plt.title('Importance des variables dans la prédiction Bitcoin')
plt.tight_layout()
plt.show()


# In[ ]:




