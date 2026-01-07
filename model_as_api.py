from fastapi import FastAPI 
import pickle 
import numpy as np 

# Initialize FastAPI app 
app = FastAPI() 

# Load saved model 
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f) 

@app.post("/predict") 
def predict(data: dict):
    """
    Accepts JSON like:
    {
        "features": [3,22,1,0]
    }
    """
    features = np.array(data["features"]).reshape(1,-1)
    prediction = models.predict(features) 
    return {"prediction": int(prediction[0])}