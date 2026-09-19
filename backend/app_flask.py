"""Flask variant of the inference API for teams standardised on Flask."""
from __future__ import annotations

from flask import Flask, jsonify, request

from ml.inference import predict

MAX_BYTES = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.post("/api/predict")
def predict_route():
    if "file" not in request.files:
        return jsonify(error="No file part named 'file'"), 400

    payload = request.files["file"].read()
    try:
        return jsonify(predict(payload))
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 503
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
