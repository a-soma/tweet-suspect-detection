# Détection de Tweets Suspects avec Machine Learning

**Université Virtuelle du Burkina Faso (UV-BF)**  
**Examen Final – Machine Learning & MLOps (Git + DVC)**  
**Année académique : 2025–2026**

**Auteur : Alassane SOMA**  
**Encadrant : Dr. Abdoul Kader KABORE**

---

# Présentation du projet

Ce projet consiste à développer une solution de **classification automatique de tweets suspects** à partir d'un jeu de données de **60 000 tweets**.

L'objectif est de mettre en œuvre un pipeline complet de Machine Learning comprenant :

- l'exploration et le prétraitement des données textuelles ;
- la vectorisation des tweets avec **TF-IDF** ;
- la gestion du déséquilibre des classes avec **SMOTE** ;
- la comparaison de plusieurs algorithmes de classification ;
- l'évaluation des performances ;
- le déploiement d'une application **Streamlit** ;
- la reproductibilité du projet grâce à **Git** et **DVC**.

---

# Résultats obtenus

Les modèles suivants ont été entraînés et comparés.

| Modèle | Accuracy | Precision | Recall | F1-score | Validation croisée |
|---------|---------:|----------:|--------:|---------:|-------------------:|
| Régression Logistique | 0.850 | 0.782 | 0.923 | 0.846 | 0.863 ± 0.005 |
| Naive Bayes | 0.815 | 0.809 | 0.769 | 0.788 | 0.806 ± 0.007 |
| SVM Linéaire | **0.895** | 0.848 | 0.933 | **0.889** | 0.901 ± 0.005 |
| **Random Forest** | 0.893 | 0.834 | **0.951** | **0.889** | **0.904 ± 0.003** |

## Modèle retenu

Le modèle **Random Forest** a été retenu pour le déploiement, car il présente les meilleures performances globales en validation croisée tout en offrant un excellent compromis entre précision, rappel et F1-score.

---

# Structure du projet

```text
tweet-suspect-detection/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── tweets_suspect.csv
│   └── tweets_preprocessed.csv
│
├── models/
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── results.json
│
├── reports/
│   ├── fig1_distribution.png
│   ├── fig2_longueur.png
│   ├── fig3_top_mots.png
│   ├── fig4_comparaison.png
│   ├── fig5_confusion.png
│   └── fig6_roc.png
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── Notebook.ipynb
├── dvc.yaml
├── requirements.txt
└── README.md
```

---

# Technologies utilisées

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Imbalanced-learn (SMOTE)
- Matplotlib
- Streamlit
- Git
- DVC

---

# Installation

## 1. Cloner le dépôt

```bash
git clone https://github.com/a-soma/tweet-suspect-detection.git

cd tweet-suspect-detection
```

## 2. Créer un environnement virtuel

```bash
python -m venv .venv
```

Activation sous Windows :

```bash
.\.venv\Scripts\Activate
```

## 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

# Exécution du projet

## Prétraitement

```bash
python src/preprocess.py
```

Cette étape produit :

```
data/tweets_preprocessed.csv
```

---

## Entraînement

```bash
python src/train.py
```

Cette étape génère :

- best_model.pkl
- tfidf_vectorizer.pkl
- results.json

---

## Évaluation

```bash
python src/evaluate.py
```

Le script affiche un rapport comparatif des performances des modèles.

---

## Déploiement Streamlit

```bash
streamlit run app/streamlit_app.py
```

L'application est ensuite accessible à l'adresse :

```
http://localhost:8501
```

Elle permet :

- de saisir un tweet ;
- d'obtenir la prédiction ;
- d'afficher la probabilité associée.

---

# Reproductibilité avec DVC

Le pipeline est organisé selon les étapes suivantes :

```text
tweets_suspect.csv
        │
        ▼
Prétraitement
(preprocess.py)
        │
        ▼
tweets_preprocessed.csv
        │
        ▼
Entraînement
(train.py)
        │
        ▼
best_model.pkl
results.json
        │
        ▼
Évaluation
(evaluate.py)
```

Pour reproduire le projet :

```bash
dvc pull

dvc repro
```

---

# Visualisations

Le dossier `reports/` contient notamment :

- Distribution des classes
- Longueur des tweets
- Nuage des mots les plus fréquents
- Comparaison des modèles
- Matrice de confusion
- Courbe ROC

---

# Auteur

**Alassane SOMA**

Statisticien • Data Scientist • Chercheur en statistique appliquée et économétrie

Université Virtuelle du Burkina Faso (UV-BF)

GitHub : https://github.com/a-soma

---

## Remerciements

Mes remerciements vont à **Dr. Abdoul Kader KABORE** pour son encadrement et ses conseils dans le cadre de cet exercice.
