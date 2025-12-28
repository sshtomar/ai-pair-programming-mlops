# Lesson 3.2: Building an API (FastAPI)

Read the lesson content from `lesson-modules/3-deployment/3-2-fastapi-basics.md` and guide the student through it.

## Lesson Flow

### 1. Opener (2 min)
"Your model can make predictions. But it's trapped in a Python script. Let's free it."

### 2. Socratic Question
Ask: "What does a prediction API need to handle besides just calling model.predict()?"

Expected answers: Input validation, error handling, authentication, logging, health checks, response formatting. Build on this to motivate proper API design.

### 3. FastAPI Fundamentals (10 min)
Cover the basics:
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}
```

Key concepts:
- Decorators define routes
- Type hints enable validation
- Automatic OpenAPI docs at `/docs`

### 4. Create the Prediction Endpoint (20 min)
Guide them to build `project/api/main.py`:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI(title="Sentiment Classifier API")

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

# Load model at startup
model = joblib.load("models/sentiment_model.pkl")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    prediction = model.predict([request.text])[0]
    confidence = model.predict_proba([request.text]).max()
    return PredictionResponse(
        text=request.text,
        sentiment=prediction,
        confidence=float(confidence)
    )
```

### 5. Run and Test Locally (10 min)
```bash
pip install fastapi uvicorn
uvicorn api.main:app --reload
```

Have them:
1. Visit `http://localhost:8000/docs`
2. Try the interactive Swagger UI
3. Make a curl request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'
```

### 6. Error Handling (10 min)
Add robust error handling:
```python
@app.post("/predict")
def predict(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        # prediction logic
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")
```

Ask: "What other errors should we handle?"

### 7. Wrap Up
- API is running locally with validation and docs
- Pydantic models ensure type safety
- Preview: Lesson 3.3 containerizes this API
- Next: `/start-3-3`

## Teaching Notes
- Let them struggle with typos—debugging is learning
- Point out the automatic docs as a huge productivity win
- Emphasize: model loading happens once at startup, not per request
- If they have experience with Flask, contrast the approaches
