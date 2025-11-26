import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# --- Load ML Model ---
try:
    MODEL = joblib.load("house_model.pkl") 
    print("--- INFO: house_model.pkl loaded successfully. ---")
except FileNotFoundError:
    print("--- WARNING: house_model.pkl not found. Using placeholder prediction logic. ---")
    MODEL = None
except Exception as e:
    print(f"--- WARNING: Error loading model: {e}. Using placeholder prediction logic. ---")
    MODEL = None


# --- Standard FastAPI Setup ---
app = FastAPI(title="UrbanEstimator App - Pro ML")
templates = Jinja2Templates(directory="templates") 
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Helper Function: Indian Currency Formatting ---
def format_currency(amount):
    """Formats a number into Indian text format (Crores, Lakhs, Thousands)."""
    try:
        amount = int(round(amount))
        original_amount = amount
        
        crores = amount // 10000000
        amount = amount % 10000000
        
        lakhs = amount // 100000
        amount = amount % 100000
        
        thousands = amount // 1000
        remainder = amount % 1000
        
        parts = []
        if crores > 0:
            parts.append(f"{crores} Crore")
        if lakhs > 0:
            parts.append(f"{lakhs} Lakhs")
        if thousands > 0:
            parts.append(f"{thousands} Thousand")
        if remainder > 0 and (crores == 0 and lakhs == 0): 
            parts.append(f"{remainder}")
            
        if not parts:
            return "Rs. 0"
            
        return "Rs. " + " ".join(parts) + " only"
    except Exception as e:
        return f"Rs. {original_amount}"


# --- Frontend Page Routes ---
@app.get("/", response_class=HTMLResponse)
async def get_predictor_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# --- API Endpoints ---
@app.post("/predict")
async def predict_house_price(
    latitude: float = Form(...),
    longitude: float = Form(...),
    Area: float = Form(...),
    Bedrooms: int = Form(...), 
    Bathrooms: int = Form(...),
):
    """Accepts house features and returns a price prediction."""
    
    # --- CRITICAL UPDATE FOR PRO MODEL ---
    # We must calculate the 6th feature (Bath_Bed_Ratio) that the model expects.
    # Avoid division by zero if bedrooms is 0.
    bath_bed_ratio = Bathrooms / Bedrooms if Bedrooms > 0 else 0
    
    # Input array now has 6 columns
    input_data = np.array([[
        latitude, 
        longitude, 
        Area, 
        Bedrooms, 
        Bathrooms,
        bath_bed_ratio  # <--- The new engineered feature
    ]])

    raw_price = 0

    if MODEL:
        try:
            # Predict raw price (in Rupees)
            raw_price = MODEL.predict(input_data)[0]
        except Exception as e:
            print(f"--- ML Prediction Error: {e} ---")
            # Fallback logic
            raw_price = (Area * 5000) + (Bedrooms * 500000) + (Bathrooms * 200000)
    else:
        # Fallback Placeholder Logic
        raw_price = (Area * 5000) + (Bedrooms * 500000) + (Bathrooms * 200000)

    # Format the price using the helper function
    formatted_text = format_currency(raw_price)

    return {
        "status": "success",
        "input_features": {
            "latitude": latitude,
            "longitude": longitude,
            "Area": Area,
            "Bedrooms": Bedrooms,
            "Bathrooms": Bathrooms
        },
        "predicted_price_text": formatted_text
    }