# Détection de Tweets Suspects avec Machine Learning

**Université Virtuelle du Burkina Faso (UV-BF)**  
**Examen Final - Machine Learning & MLOps (Git + DVC)**  
**Année académique : 2025–2026**

**Auteur : Alassane SOMA**  
**Encadrant : Dr. Abdoul Kader KABORE**

---

## Démonstration en ligne

**Application Streamlit :**

**https://tweetsuspectdetection.streamlit.app**

L'application permet :

- d'entrer un tweet en anglais ;
- d'obtenir une prédiction (*Suspect* ou *Non suspect*) ;
- d'afficher la probabilité associée à la prédiction.

---

## Présentation du projet

Ce projet consiste à développer une solution de **classification automatique de tweets suspects** à partir d'un jeu de données de **60 000 tweets**.

L'objectif est de mettre en œuvre un pipeline complet de Machine Learning comprenant :

- l'analyse exploratoire des données ;
- le prétraitement des tweets ;
- la représentation TF-IDF ;
- la gestion du déséquilibre des classes avec SMOTE ;
- la comparaison de plusieurs modèles de classification ;
- l'évaluation des performances ;
- le déploiement d'une application web avec Streamlit ;
- la reproductibilité avec Git et DVC.

---

## Résultats obtenus

| Modèle | Accuracy | Precision | Recall | F1-score | Validation croisée |
|---------|---------:|----------:|--------:|---------:|-------------------:|
| Régression Logistique | 0.850 | 0.782 | 0.923 | 0.846 | 0.863 ± 0.005 |
| Naive Bayes | 0.815 | 0.809 | 0.769 | 0.788 | 0.806 ± 0.007 |
| SVM Linéaire | **0.895** | 0.848 | 0.933 | **0.889** | 0.901 ± 0.005 |
| **Random Forest** | 0.893 | 0.834 | **0.951** | **0.889** | **0.904 ± 0.003** |

### Modèle retenu

Le modèle **Random Forest** a été retenu pour le déploiement grâce à ses excellentes performances globales.

---

## Structure du projet

```text
tweet-suspect-detection/
├── .dvc/
│   ├── .gitignore
│   └── config
│
├── app/
│   └── streamlit_app.py
├── data/
│   ├── tweets_suspect.csv
│   └── tweets_preprocessed.csv
├── models/
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── results.json
├── reports/
│   ├── fig1_distribution.png
│   ├── fig2_longueur.png
│   ├── fig3_top_mots.png
│   ├── fig4_comparaison.png
│   ├── fig5_confusion.png
│   └── fig6_roc.png
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── Notebook.ipynb
├── dvc.yaml
├── requirements.txt
└── README.md
└── .dvcignore

```

---

## Technologies utilisées

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Imbalanced-Learn (SMOTE)
- Matplotlib
- Streamlit
- Git
- DVC

---

## Installation

```bash
git clone https://github.com/a-soma/tweet-suspect-detection.git

cd tweet-suspect-detection

py -m venv .venv

or

python -m venv .venv 

# Windows
.\.venv\Scripts\Activate

or

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## Exécution

Prétraitement

```bash
python src/preprocess.py
```

Entraînement

```bash
python src/train.py
```

Évaluation

```bash
python src/evaluate.py
```

Déploiement local

```bash
streamlit run app/streamlit_app.py
```

---

## Reproductibilité avec DVC

```bash
dvc pull

dvc repro
```

---

## Visualisations

Le dossier **reports/** contient :

- Distribution des classes
- Longueur des tweets
- Mots les plus fréquents
- Comparaison des modèles
- Matrice de confusion
- Courbe ROC

---

## Auteur

**Alassane SOMA**

Statisticien • Data Scientist • Chercheur en Statistique Appliquée et Économétrie

GitHub : https://github.com/a-soma

Application Streamlit : https://tweetsuspectdetection.streamlit.app

Université Virtuelle du Burkina Faso (UV-BF)

---

## Remerciements

Je remercie **Dr. Abdoul Kader KABORE** pour son encadrement dans le cadre de cet examen final de Machine Learning.
