from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict
import uvicorn

app = FastAPI()

# Define request schema
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float

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
