import json
import os
import math
import pickle
import re

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

VOCAB = MODEL["vocab"]  # dict: token -> index
W = MODEL["weights"]  # list[float] length = |vocab|
B = MODEL["bias"]  # float
THRESH = MODEL.get("threshold", 0.5)

_token_re = re.compile(r"[a-zA-Z']+")


def _sigmoid(z: float) -> float:
    # stable-ish sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def _predict_proba(text: str) -> float:
    # simple bag-of-words counts
    counts = {}
    for tok in _token_re.findall(text.lower()):
        idx = VOCAB.get(tok)
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1

    z = B
    for idx, c in counts.items():
        z += W[idx] * c
    return _sigmoid(z)


def handler(event, context):
    try:
        body_raw = event.get("body") or "{}"
        body = json.loads(body_raw) if isinstance(body_raw, str) else body_raw
        text = body.get("text", "")

        if not isinstance(text, str) or not text.strip():
            return _resp(400, {"error": "Missing or empty 'text' field"})

        p_pos = _predict_proba(text)
        pred = 1 if p_pos >= THRESH else 0
        label = "Positive" if pred == 1 else "Negative"

        return _resp(200, {"prediction": label, "p_positive": round(p_pos, 4)})

    except Exception as e:
        return _resp(500, {"error": str(e)})


def _resp(status_code, payload):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "POST,OPTIONS",
        },
        "body": json.dumps(payload),
    }