import numpy as np
import onnxruntime as ort
import joblib
import os


BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../model_nn/fraud_mlp.onnx")
SCALER_PATH = os.path.join(BASE_DIR, "../model_nn/scaler.joblib")


session = ort.InferenceSession(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


FEATURE_ORDER = [
    "Time", "Amount",
    "V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16",
    "V17","V18","V19","V20","V21","V22","V23",
    "V24","V25","V26","V27","V28"
]


def predict(input_dict):
    x = np.array([[input_dict[f] for f in FEATURE_ORDER]], dtype=np.float32)

    x_scaled = scaler.transform(x)

    logits = session.run(None, {"input": x_scaled})[0][0]

    prob = 1 / (1 + np.exp(-logits))
    return float(prob)
