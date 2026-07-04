"""
Étape 1 - Prétraitement des tweets
Auteur : Alassane SOMA | UV-BF | 2026
"""
import pandas as pd
import re
import sys
import os

STOP_EN = set([
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'is','are','was','were','be','been','have','has','do','does','did',
    'i','you','he','she','it','we','they','me','him','her','us','them',
    'my','your','his','its','our','their','this','that','these','those',
    'not','no','so','as','if','just','from','by','rt','am','im','got','get',
    'will','would','could','should','may','might','shall','what','which',
    'who','how','when','where','why','very','too','more','most','also',
])

def nettoyer_tweet(text: str) -> str:
    """Nettoyage complet d'un tweet anglais."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)           # URLs
    text = re.sub(r'@\w+|#\w+', '', text)                # Mentions et hashtags
    text = re.sub(r'[^a-z\s]', ' ', text)                # Caractères spéciaux
    text = re.sub(r'\s+', ' ', text).strip()              # Espaces multiples
    tokens = [w for w in text.split() if w not in STOP_EN and len(w) > 2]
    return ' '.join(tokens)

def preprocess_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """Charge, nettoie et sauvegarde le dataset."""
    df = pd.read_csv(input_path)
    print(f"[preprocess] Dataset chargé : {len(df)} tweets")
    print(f"[preprocess] Colonnes : {list(df.columns)}")

    assert 'message' in df.columns, "Colonne 'message' manquante"
    assert 'label' in df.columns,   "Colonne 'label' manquante"

    n_missing = df['message'].isna().sum()
    if n_missing > 0:
        print(f"[preprocess] {n_missing} valeurs manquantes supprimées")
        df = df.dropna(subset=['message'])

    df['tweet_clean'] = df['message'].apply(nettoyer_tweet)
    df['longueur']    = df['message'].str.len()
    df['nb_mots']     = df['message'].str.split().str.len()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[preprocess] Sauvegardé : {output_path}")
    print(f"[preprocess] Distribution labels :\n{df['label'].value_counts().to_string()}")
    return df

if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else 'data/tweets_suspect.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else 'data/tweets_preprocessed.csv'
    preprocess_dataset(inp, out)
