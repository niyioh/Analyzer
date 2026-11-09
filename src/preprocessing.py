from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset, DatasetDict
import pandas as pd
dataset = load_dataset("tblard/allocine")

def build_processor() -> ColumnTransformer:
    tfidf = TfidfVectorizer(
        lowercase=True,
        max_features=50000,   
        ngram_range=(1, 2),  
        min_df=2,             
        sublinear_tf=True,    
    )
 
    return ColumnTransformer(
        transformers=[("tfidf", tfidf, "review")]
    )

def preprocess_data(X, split: str, processor=None):
    """Prépare les données pour l'entraînement du modèle."""
    available = list(dataset.keys())
    if split not in available:
        raise ValueError(
            f"Le set {split} n'existe pas. Les sets disponibles sont : {available}"
        )

    if isinstance(X, pd.Series):
        X = X.to_frame(name="review")

    X = X.copy()
    X["review"] = X["review"].fillna("")

    if split == "train":
        processor = build_processor()
        X_processed = processor.fit_transform(X)
    else:
        if processor is None:
            raise ValueError("Passe le processeur appris sur le train.")
        X_processed = processor.transform(X)

    return X_processed, processor


def clean_data(dataset: DatasetDict, split: str) -> pd.DataFrame:
    """
    Nettoie le dataset en supprimant les doublons et les valeurs manquantes.
    """
    if split not in dataset:
        raise ValueError(f"Le set {split} n'existe pas dans le dataset. Les sets disponibles sont : {list(dataset.keys())}")
    else:
        df = dataset[split].to_pandas()
        df = df.drop_duplicates()
        df = df.dropna()
    return df

def transform_texts(processor, texts):
    """Vectorise une liste de textes avec un processeur déjà appris."""
    df = pd.DataFrame({"review": list(texts)})
    df["review"] = df["review"].fillna("")
    return processor.transform(df)
