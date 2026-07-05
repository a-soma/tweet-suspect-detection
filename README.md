# Détection de tweets suspects avec Machine Learning

**Université Virtuelle du Burkina Faso (UV-BF)**  
**Examen Final – Machine Learning & MLOps (Git + DVC)**  
**Année académique : 2025–2026**

**Auteur : Alassane SOMA**  
**Encadrant : Dr. Abdoul Kader KABORE**

---

# Démonstration en ligne

## Application Streamlit

**https://tweetsuspectdetection.streamlit.app**

L'application permet de :

- saisir un tweet en anglais ;
- prédire automatiquement s'il est **Suspect** ou **Non suspect** ;
- afficher la probabilité associée à la prédiction.

---

# Présentation du projet

Ce projet met en œuvre une chaîne complète de **Machine Learning** pour la **détection automatique de tweets suspects** à partir d'un jeu de données réel contenant **60 000 tweets**.

Le pipeline couvre toutes les étapes d'un projet de science des données :

- exploration et analyse des données ;
- prétraitement des tweets ;
- vectorisation TF-IDF ;
- équilibrage des classes avec SMOTE ;
- entraînement et comparaison de plusieurs modèles ;
- évaluation des performances ;
- déploiement d'une application web avec Streamlit ;
- gestion de versions avec Git ;
- reproductibilité des expériences avec DVC.

---

# Résultats obtenus

| Modèle                |  Accuracy | Precision |    Recall |  F1-score | Validation croisée |
| --------------------- | --------: | --------: | --------: | --------: | -----------------: |
| Régression Logistique |     0.872 |     0.849 |     0.749 |     0.796 |      0.880 ± 0.005 |
| Naive Bayes           |     0.824 |     0.706 |     0.806 |     0.753 |      0.868 ± 0.003 |
| SVM Linéaire          |     0.886 |     0.835 | **0.820** |     0.827 |      0.919 ± 0.004 |
| **Random Forest**     | **0.917** | **0.977** |     0.769 | **0.861** |  **0.938 ± 0.004** |

## Modèle retenu

Le modèle **Random Forest** a été retenu pour le déploiement en raison de ses excellentes performances globales et de sa meilleure validation croisée.

---

# Structure du projet

```text
tweet-suspect-detection/
│
├── .dvc/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── tweets_suspect.csv.dvc
│   ├── tweets_preprocessed.csv
│   └── .gitignore
├── models/
│   ├── results.json
│   └── .gitignore
├── reports/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── Notebook.ipynb
├── dvc.lock
├── dvc.yaml
├── README.md
├── requirements.txt
└── .dvcignore
```

> Les fichiers volumineux (`tweets_suspect.csv`, `best_model.pkl` et `tfidf_vectorizer.pkl`) sont gérés par **DVC** afin de conserver un dépôt Git léger et reproductible.

---

# Technologies utilisées

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

# Installation

Clonage du projet depuis github
```bash
git clone https://github.com/a-soma/tweet-suspect-detection.git
```
Paramettrage de chemin d'acces
```bash
cd tweet-suspect-detection
```

Creation d'environnement virtuel
```bash
python -m venv .venv
```
oubien
```bash
py -m venv .venv
```

Sous Windows :

```bash
.venv\Scripts\activate
```
ou bien
```bash
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

# Reproduire entièrement le projet

Télécharger les fichiers suivis par DVC :

```bash
dvc pull
```

Reconstruire automatiquement le pipeline :

```bash
dvc repro
```

---

# Exécution manuelle

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

# Pipeline DVC

```text
tweets_suspect.csv
        │
        ▼
Preprocessing
        │
        ▼
tweets_preprocessed.csv
        │
        ▼
Training
        │
        ├── best_model.pkl
        ├── tfidf_vectorizer.pkl
        └── results.json
        │
        ▼
Evaluation
```

---

# Visualisations

Le dossier **reports/** contient notamment :

- Distribution des classes ;
- Longueur des tweets ;
- Mots les plus fréquents ;
- Comparaison des modèles ;
- Matrice de confusion ;
- Courbe ROC.

---

# Démonstration

**Application Streamlit**

https://tweetsuspectdetection.streamlit.app

---

# Auteur

**Alassane SOMA**

Statisticien • Data Scientist • Chercheur en Statistique Appliquée et Économétrie

GitHub : https://github.com/a-soma

Projet GitHub : https://github.com/a-soma/tweet-suspect-detection

Application Streamlit : https://tweetsuspectdetection.streamlit.app

Université Virtuelle du Burkina Faso (UV-BF)

---

# Remerciements

Mes sincères remerciements à **Dr. Abdoul Kader KABORE** pour son accompagnement et ses conseils dans le cadre de cet examen final de Machine Learning et MLOps.
