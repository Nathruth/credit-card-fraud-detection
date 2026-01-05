import torch
import numpy as np

# Load ONNX
import onnxruntime as ort
session = ort.InferenceSession("artifacts/fraud_mlp.onnx")

def predict(input_dict):
    # Ensure features are in the correct order
    features = np.array([list(input_dict.values())], dtype=np.float32)
    output = session.run(None, {"input": features})
    prob = 1 / (1 + np.exp(-output[0][0]))  # sigmoid
    return float(prob)

# Example
sample = {f"V{i}": 0.1 for i in range(1, 10)}
print("Predicted fraud probability:", predict(sample))
