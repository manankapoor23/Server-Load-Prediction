from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = joblib.load("server_overload_xgb.pkl")
feature_names = joblib.load("feature_names.pkl")

# Load test data for random sample mode
X_test = pd.read_csv("cleaned_data/X_test.csv")
y_test = pd.read_csv("cleaned_data/y_test.csv").squeeze()

@app.get("/")
def serve_landing(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard")
def serve_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/random_sample")
def random_sample():
    """Return a random real sample from the test set with its true label."""
    idx = np.random.randint(0, len(X_test))
    row = X_test.iloc[idx]
    true_label = int(y_test.iloc[idx])

    df = pd.DataFrame([row])
    df = df[feature_names]
    prob = model.predict_proba(df)[0][1]
    prediction = int(prob >= 0.3)

    return {
        "features": row.to_dict(),
        "overload_probability": round(float(prob), 4),
        "early_warning": prediction,
        "true_label": true_label,
        "sample_index": int(idx)
    }

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    df = df[feature_names]

    prob = model.predict_proba(df)[0][1]
    prediction = int(prob >= 0.3)

    return {
        "overload_probability": round(float(prob), 4),
        "early_warning": prediction
    }

@app.post("/simulate")
def simulate(data: dict):
    """Take a few slider values and auto-compute all 38 features."""
    cpu = data.get("CPU_Utilization_%", 50)
    temp = data.get("CPU1_Temp_C", 45)
    power = data.get("System_Power_W", 250)
    fan = data.get("Avg_Fan_Speed_RPM", 3500)
    hour = data.get("hour", 12)

    # Auto-derive realistic features
    row = {
        "CPU_Utilization_%": cpu,
        "Memory_Utilization_%": 30 + cpu * 0.35 + np.random.normal(0, 2),
        "CPU1_Temp_C": temp,
        "CPU2_Temp_C": temp + np.random.normal(0.5, 0.3),
        "Inlet_Temp_C": 18 + np.random.normal(0, 1),
        "Exhaust_Temp_C": temp + 10 + np.random.normal(0, 1),
        "System_Power_W": power,
        "Avg_Fan_Speed_RPM": fan,
        "Air_Flow_CFM": 15 + fan / 300 + np.random.normal(0, 1),
        "Voltage_V": 12 + np.random.normal(0, 0.05),
        "Clock_Speed_GHz": 2.5 + (cpu / 100) * 0.5 + np.random.normal(0, 0.05),
        "hour": hour,
        "day_of_week": 2,
    }

    # Generate lag features (simulate gradual trend toward current value)
    for base, val in [("CPU_Utilization_%", cpu), ("CPU1_Temp_C", temp),
                      ("CPU2_Temp_C", row["CPU2_Temp_C"]),
                      ("System_Power_W", power), ("Avg_Fan_Speed_RPM", fan)]:
        noise = [np.random.normal(0, 1) for _ in range(3)]
        lags = [val * 0.95 + noise[0], val * 0.90 + noise[1], val * 0.85 + noise[2]]
        for i in range(1, 4):
            row[f"{base}_lag{i}"] = round(lags[i-1], 4)
        vals = [val, lags[0], lags[1]]
        row[f"{base}_roll_mean_3"] = round(float(np.mean(vals)), 4)
        row[f"{base}_roll_std_3"] = round(float(np.std(vals, ddof=1)) if np.std(vals) > 0 else 0.01, 4)

    df = pd.DataFrame([row])
    df = df[feature_names]
    prob = model.predict_proba(df)[0][1]
    prediction = int(prob >= 0.3)

    return {
        "features": row,
        "overload_probability": round(float(prob), 4),
        "early_warning": prediction
    }

app.mount("/static", StaticFiles(directory="static"), name="static")
