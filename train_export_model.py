import os
import re
import math
import pickle
from collections import Counter
from datasets import load_dataset

_token_re = re.compile(r"[a-zA-Z']+")


def tokenize(s: str):
    return _token_re.findall(s.lower())


def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def train_logreg_sgd(texts, labels, vocab, epochs=2, lr=0.08, l2=1e-6):
    V = len(vocab)
    w = [0.0] * V
    b = 0.0

    for ep in range(epochs):
        for x, y in zip(texts, labels):
            counts = Counter()
            for tok in tokenize(x):
                idx = vocab.get(tok)
                if idx is not None:
                    counts[idx] += 1

            z = b
            for idx, c in counts.items():
                z += w[idx] * c

            p = sigmoid(z)
            g = p - y  # gradient

            b -= lr * g

            for idx, c in counts.items():
                w[idx] -= lr * (g * c + l2 * w[idx])

        print(f"epoch {ep+1}/{epochs} done")

    return w, b


def main():
    ds = load_dataset("glue", "sst2")

    X = ds["train"]["sentence"]
    y = ds["train"]["label"]

    vocab_counter = Counter()
    for s in X:
        vocab_counter.update(tokenize(s))

    MAX_VOCAB = 12000
    most_common = vocab_counter.most_common(MAX_VOCAB)
    vocab = {tok: i for i, (tok, _) in enumerate(most_common)}

    w, b = train_logreg_sgd(X, y, vocab, epochs=2, lr=0.08)

    model = {"vocab": vocab, "weights": w, "bias": b, "threshold": 0.5}

    os.makedirs("backend", exist_ok=True)
    out_path = os.path.join("backend", "model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(model, f)

    print("Saved", out_path)
    print("vocab size:", len(vocab))


if __name__ == "__main__":
    main()