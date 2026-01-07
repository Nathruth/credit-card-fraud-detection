from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict
import uvicorn

app = FastAPI()

# Define request schema
class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_endpoint(transaction: Transaction):
    input_dict = transaction.dict()
    prob = predict(input_dict)
    return {"fraud_probability": prob}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
