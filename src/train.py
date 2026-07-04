"""
Étape 2 — Entraînement des modèles de classification
Auteur : Alassane SOMA | UV-BF | 2026
"""
import pandas as pd
import numpy as np
import pickle
import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

def entrainer_modeles(data_path: str, models_dir: str) -> dict:
    """Pipeline complet d'entraînement avec 4 modèles."""
    df = pd.read_csv(data_path)
    print(f"[train] {len(df)} tweets chargés")
    print(f"[train] Distribution : {df['label'].value_counts().to_dict()}")

    # Sous-échantillonnage stratifié pour gérer le déséquilibre
    df_sus = df[df['label'] == 1].sample(min(5000, (df['label']==1).sum()), random_state=42)
    df_nor = df[df['label'] == 0].sample(min(10000, (df['label']==0).sum()), random_state=42)
    df_s = pd.concat([df_sus, df_nor]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[train] Échantillon : {len(df_s)} tweets — {df_s['label'].value_counts().to_dict()}")

    X = df_s['tweet_clean'].fillna('')
    y = df_s['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[train] Train={len(X_train)} | Test={len(X_test)}")

    # Vectorisation TF-IDF (unigrammes + bigrammes)
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    X_tr_tfidf = tfidf.fit_transform(X_train)
    X_te_tfidf = tfidf.transform(X_test)

    # SMOTE pour équilibrer les classes
    smote = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_tfidf, y_train)
    print(f"[train] Après SMOTE : {dict(zip(*np.unique(y_tr_sm, return_counts=True)))}")

    modeles = {
        'Regression Logistique': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        'Naive Bayes':           MultinomialNB(alpha=0.5),
        'SVM Lineaire':          LinearSVC(random_state=42, max_iter=3000),
        'Random Forest':         RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultats = {}

    for nom, modele in modeles.items():
        cv_scores = cross_val_score(modele, X_tr_sm, y_tr_sm, cv=cv, scoring='f1')
        modele.fit(X_tr_sm, y_tr_sm)
        y_pred = modele.predict(X_te_tfidf)

        resultats[nom] = {
            'accuracy':    float(accuracy_score(y_test, y_pred)),
            'precision':   float(precision_score(y_test, y_pred)),
            'recall':      float(recall_score(y_test, y_pred)),
            'f1':          float(f1_score(y_test, y_pred)),
            'cv_f1_mean':  float(cv_scores.mean()),
            'cv_f1_std':   float(cv_scores.std()),
        }
        r = resultats[nom]
        print(f"[train] {nom}: Acc={r['accuracy']:.3f} | F1={r['f1']:.3f} "
              f"| CV-F1={r['cv_f1_mean']:.3f}±{r['cv_f1_std']:.3f}")

    os.makedirs(models_dir, exist_ok=True)
    best_nom = max(resultats, key=lambda n: resultats[n]['f1'])
    print(f"[train] Meilleur modèle : {best_nom} (F1={resultats[best_nom]['f1']:.3f})")

    with open(f'{models_dir}/best_model.pkl', 'wb') as f:
        pickle.dump(modeles[best_nom], f)
    with open(f'{models_dir}/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    resultats['_best'] = best_nom
    resultats['_test_size'] = len(y_test)
    with open(f'{models_dir}/results.json', 'w', encoding='utf-8') as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)

    print(f"[train] Modèles sauvegardés dans {models_dir}/")
    return resultats

if __name__ == '__main__':
    data = sys.argv[1] if len(sys.argv) > 1 else 'data/tweets_preprocessed.csv'
    mdir = sys.argv[2] if len(sys.argv) > 2 else 'models'
    entrainer_modeles(data, mdir)
