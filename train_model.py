import os, joblib
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    ds = load_dataset("glue", "sst2")
    X = ds["train"]["sentence"]
    y = ds["train"]["label"]
    
    N = 15000
    X, y = X[:N], y[:N]

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=25000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )

    model.fit(Xtr, ytr)
    acc = accuracy_score(yva, model.predict(Xva))
    print("val_acc =", round(acc, 4))

    os.makedirs("backend", exist_ok=True)
    joblib.dump(model, "backend/model.pkl")
    print("Saved backend/model.pkl")


if __name__ == "__main__":
    main()