from functools import lru_cache
from pathlib import Path
 
import joblib
 
try:
    from src.preprocessing import transform_texts
except ImportError:  
    from preprocessing import transform_texts
 
 
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LABELS = {0: "négatif", 1: "positif"}
 
 
@lru_cache(maxsize=1)
def load_artifacts():
    """Charge le processeur et le modèle (une seule fois)."""
    processor_path = MODELS_DIR / "processor.joblib"
    model_path = MODELS_DIR / "model.joblib"
 
    if not processor_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Modèle introuvable. Lance d'abord l'entraînement : python -m src.train"
        )
 
    return joblib.load(processor_path), joblib.load(model_path)
 
 
def predict_sentiment(text: str) -> dict:
    """Renvoie le sentiment prédit et les probabilités pour une critique."""
    if not text or not text.strip():
        raise ValueError("Le texte est vide.")
 
    processor, model = load_artifacts()
 
    X = transform_texts(processor, [text])
    proba = model.predict_proba(X)[0]
 
    classes = list(model.classes_)
    proba_positive = float(proba[classes.index(1)])
    proba_negative = float(proba[classes.index(0)])
 
    label_id = 1 if proba_positive >= proba_negative else 0
 
    return {
        "label": LABELS[label_id],
        "confidence": max(proba_positive, proba_negative),
        "proba_positive": proba_positive,
        "proba_negative": proba_negative,
    }
 