import json
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)


def handler(event, context):
    """
    Expected API Gateway HTTP API event with JSON body:
    { "text": "your sentence" }
    """
    try:
        body_raw = event.get("body") or "{}"
        body = json.loads(body_raw) if isinstance(body_raw, str) else body_raw
        text = body.get("text", "")

        if not isinstance(text, str) or not text.strip():
            return _resp(400, {"error": "Missing or empty 'text' field"})

        pred = int(model.predict([text])[0])  # 1=positive, 0=negative
        label = "Positive" if pred == 1 else "Negative"

        return _resp(200, {"prediction": label})

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