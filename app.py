 
import streamlit as st
 
from src.predict import load_artifacts, predict_sentiment
 
st.set_page_config(page_title="Analyseur de sentiment", page_icon="🎬")
 
st.title("Analyseur de sentiment de critiques de films")
st.write("Écris une critique de film en français : le modèle prédit si elle est positive ou négative.")
 
try:
    load_artifacts()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()
 
text = st.text_area("Ta critique", height=150, placeholder="Un film magnifique, je recommande !")
 
if st.button("Analyser"):
    if not text.strip():
        st.warning("Écris d'abord une critique.")
    else:
        result = predict_sentiment(text)
 
        if result["label"] == "positif":
            st.success(f"Critique positive ({result['confidence']:.0%} de confiance)")
        else:
            st.error(f"Critique négative ({result['confidence']:.0%} de confiance)")
 
        st.progress(result["proba_positive"], text=f"Probabilité positive : {result['proba_positive']:.0%}")
 