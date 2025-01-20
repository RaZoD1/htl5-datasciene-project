from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()
# Load your pre-trained model
class PredictionRequest(BaseModel):
    input: list  # Adjust based on your model's input structure

@app.post("/predict")
async def predict(request: PredictionRequest):
    input_data = request.input

    # Perform prediction
    prediction = []

    return {"prediction": prediction.tolist()}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
