"""
Application Streamlit - Détection de Tweets Suspects
Auteur : Alassane SOMA | Université Virtuelle du Burkina Faso (UV-BF) | 2026
"""
import streamlit as st
import pickle
import re
import os

st.set_page_config(page_title="Detection de Tweets Suspects", page_icon=" ", layout="centered")

STOP_EN = set(['the','a','an','and','or','but','in','on','at','to','for','of','with',
               'is','are','was','were','be','been','have','has','do','does','did',
               'i','you','he','she','it','we','they','me','him','her','us','them',
               'my','your','his','its','our','their','this','that','these','those',
               'not','no','so','as','if','just','from','by','rt','am','im','got','get'])

def nettoyer_tweet(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join([w for w in text.split() if w not in STOP_EN and len(w) > 2])

@st.cache_resource
def charger_modeles():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, 'models', 'best_model.pkl'), 'rb') as f: model = pickle.load(f)
    with open(os.path.join(base, 'models', 'tfidf_vectorizer.pkl'), 'rb') as f: tfidf = pickle.load(f)
    return model, tfidf

st.title(" Détection de Tweets Suspects")
st.caption("Alassane SOMA | Université Virtuelle du Burkina Faso (UV-BF) | ML 2026")
st.markdown("---")
st.markdown("Entrez un tweet en **anglais** pour analyser s'il est suspect ou normal.")

tweet_input = st.text_area(" Tweet à analyser :",
    placeholder="Ex: I hate everything and everyone around me",
    height=120)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyser = st.button(" Analyser", use_container_width=True, type="primary")

if analyser:
    if not tweet_input.strip():
        st.warning(" Veuillez saisir un tweet.")
    else:
        try:
            model, tfidf = charger_modeles()
            clean = nettoyer_tweet(tweet_input)
            X = tfidf.transform([clean])
            pred = model.predict(X)[0]
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                prob_s, prob_n = proba[1], proba[0]
            elif hasattr(model, 'decision_function'):
                score = model.decision_function(X)[0]
                prob_s = min(max((score + 2) / 4, 0), 1)
                prob_n = 1 - prob_s
            else:
                prob_s = float(pred); prob_n = 1 - prob_s

            st.markdown("---")
            if pred == 1:
                st.error(f" **TWEET SUSPECT** - Score : {prob_s:.1%}")
                st.progress(prob_s)
            else:
                st.success(f" **TWEET NORMAL** - Score de normalité : {prob_n:.1%}")
                st.progress(prob_n)

            with st.expander(" Détails"):
                st.write(f"**Original :** {tweet_input}")
                st.write(f"**Nettoyé :** {clean}")
                st.write(f"**Tokens :** {len(clean.split())}")
        except FileNotFoundError:
            st.error(" Modèle non trouvé. Lancez d'abord : `python src/train.py`")

st.markdown("---")
st.caption("UV-BF | Mr Alassan SOMA - asomabf@gmail.com | 2026")
