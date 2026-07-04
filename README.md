# Détection de Tweets Suspects - Alassane SOMA

**Université Virtuelle du Burkina Faso (UV-BF) | Examen Final Machine Learning | 2026**  
**Encadrant : Dr. Abdoul Kader KABORE**

---

##  Résultats obtenus (dataset réel - 60 000 tweets)

| Modèle | Accuracy | F1 | AUC |
|---|---|---|---|
| Random Forest | 0.909 | 0.844 | 0.932 |
| SVM Linéaire | 0.881 | 0.820 | 0.925 |
| Régression Logistique | 0.863 | 0.787 | 0.905 |
| Naive Bayes | 0.823 | 0.753 | 0.895 |

 Modèle retenu pour le déploiement

---

## Structure du projet

```
tweet-suspect-detection/
├── data/
│   └── tweets_suspect.csv          # Dataset réel (60 000 tweets)
├── src/
│   ├── preprocess.py               # Nettoyage et prétraitement
│   ├── train.py                    # Entraînement des 4 modèles
│   └── evaluate.py                 # Rapport de performance
├── models/
│   ├── best_model.pkl              # Meilleur modèle sérialisé
│   ├── tfidf_vectorizer.pkl        # Vectoriseur TF-IDF
│   └── results.json                # Métriques JSON
├── reports/
│   └── fig1 à fig6.png             # Visualisations
├── app/
│   └── streamlit_app.py            # Application de déploiement
├── notebook_corrige_alassane_soma.ipynb   # Notebook Colab complet
├── dvc.yaml                        # Pipeline DVC
├── requirements.txt                # Dépendances Python
└── README.md
```

---

## Installation et utilisation

### 1. Cloner le dépôt
```bash
git clone https://github.com/a-soma/tweet-suspect-detection.git
cd tweet-suspect-detection
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Reproduire le pipeline DVC
```bash
dvc pull        # Télécharger données et modèles
dvc repro       # Relancer le pipeline complet
```

### 4. Lancer l'application Streamlit
```bash
streamlit run app/streamlit_app.py
```

### 5. Ou utiliser directement le notebook Colab
Ouvrir `notebook.ipynb` sur [Google Colab](https://colab.research.google.com)

---

##  Pipeline DVC

```
tweets_suspect.csv
       │
       ▼
  [preprocess]  →  tweets_preprocessed.csv
       │
       ▼
    [train]     →  best_model.pkl + results.json
       │
       ▼
  [evaluate]    →  Rapport de performance
```

---

##  Auteur

**SOMA Alassane**  
Statisticien, chercheur, étudiant - Université Virtuelle du Burkina Faso (UV-BF)  
GitHub : [a-soma](https://github.com/a-soma)  
Encadrant : Dr. Abdoul Kader KABORE
