from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ Allow React frontend on localhost and deployed Vercel app
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://salary-predictor-frontend.vercel.app"  # ✅ Vercel domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Root route for health check
@app.get("/")
def read_root():
    return {"message": "API is live and working fine ✅"}

# ✅ Load model and label encoder
try:
    model = joblib.load("xgb_model.pkl")
    le = joblib.load("label_encoder.pkl")
except Exception as e:
    print("Error loading model or encoder:", e)
    model = None
    le = None

# ✅ Input schema
class InputData(BaseModel):
    experience_level: int
    salary_in_usd: float
    company_size: int
    employment_type_FL: int
    employment_type_FT: int
    employment_type_PT: int
    job_title_Data_Analyst: int
    job_title_Data_Engineer: int
    job_title_Data_Scientist: int
    job_title_Machine_Learning_Engineer: int
    employee_residence_ES: int
    employee_residence_GB: int
    employee_residence_Other: int
    employee_residence_US: int
    company_location_ES: int
    company_location_GB: int
    company_location_Other: int
    company_location_US: int
    remote_type_Remote: int
    is_recent: int
    seniority_score: float

# ✅ Prediction route
@app.post("/predict")
def predict(data: InputData):
    if model is None or le is None:
        raise HTTPException(status_code=500, detail="Model or encoder not loaded")

    try:
        input_features = np.array([[  
            data.experience_level,
            data.salary_in_usd,
            data.company_size,
            data.employment_type_FL,
            data.employment_type_FT,
            data.employment_type_PT,
            data.job_title_Data_Analyst,
            data.job_title_Data_Engineer,
            data.job_title_Data_Scientist,
            data.job_title_Machine_Learning_Engineer,
            data.employee_residence_ES,
            data.employee_residence_GB,
            data.employee_residence_Other,
            data.employee_residence_US,
            data.company_location_ES,
            data.company_location_GB,
            data.company_location_Other,
            data.company_location_US,
            data.remote_type_Remote,
            data.is_recent,
            data.seniority_score
        ]])

        prediction_encoded = model.predict(input_features)
        prediction_label = le.inverse_transform(prediction_encoded)

        return {"prediction": prediction_label[0]}
    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Run app if run as script (optional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
