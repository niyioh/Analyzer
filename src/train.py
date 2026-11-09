"""
Entraîne le modèle d'analyse de sentiment sur Allociné.
 
Lancement (depuis la racine du projet) :
    python -m src.train
"""
 
import json
from pathlib import Path
 
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
 
try:
    from src.preprocessing import clean_data, preprocess_data, dataset 
except ImportError:  # lancé directement depuis le dossier src
    from preprocessing import clean_data, preprocess_data, dataset
 
 
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
PROCESSOR_PATH = MODELS_DIR / "processor.joblib"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
 
 
def load_allocine():
    from datasets import load_dataset
    return load_dataset("tblard/allocine")
 
 
def evaluate(model, X, y, name):
    pred = model.predict(X)
    accuracy = accuracy_score(y, pred)
 
    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(
        classification_report(
            y, pred, target_names=["négatif", "positif"], digits=4
        )
    )
 
    return accuracy
 
 
def train(dataset=None):
    if dataset is None:
        dataset = load_allocine()
 
    df_train = clean_data(dataset, "train")
    df_val = clean_data(dataset, "validation")
    df_test = clean_data(dataset, "test")
 
    print(
        f"Critiques : train={len(df_train)}, "
        f"validation={len(df_val)}, test={len(df_test)}"
    )
 
    # 2. Vectorisation : le vocabulaire est appris sur le train uniquement
    X_train, processor = preprocess_data(df_train[["review"]], "train")
    X_test, _ = preprocess_data(df_test[["review"]], "test", processor)
 
    print(f"Formes : train={X_train.shape}, test={X_test.shape}")
 
    # 3. Entraînement
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, df_train["label"])
 
    # 4. Évaluation
    test_accuracy = evaluate(model, X_test, df_test["label"], "Test")
 
    # 5. Sauvegarde
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(processor, PROCESSOR_PATH)
    joblib.dump(model, MODEL_PATH)
 
    METRICS_PATH.write_text(
        json.dumps(
            {
                "test_accuracy": test_accuracy,
                "train_size": len(df_train),
                "vocabulary_size": int(X_train.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
 
    print(f"\nModèle sauvegardé dans : {MODELS_DIR}")
 
    return model, processor
 
 
if __name__ == "__main__":
    train(dataset)
 