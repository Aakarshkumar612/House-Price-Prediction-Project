import os
import numpy as np
import joblib
from fastapi import FastAPI, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Attempt to import Mistral client libraries
try:
    from mistralai.client import MistralClient
    from mistralai.models.chat import ChatMessage
except ImportError:
    print("--- WARNING: Cannot import Mistral AI Client. Mistral features disabled. ---")
    MistralClient = None
    ChatMessage = None


# --- Load API Key and ML Model ---
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") 

MISTRAL_CLIENT = None
if not MISTRAL_API_KEY:
    print("--- FATAL ERROR: MISTRAL_API_KEY not found. Mistral features disabled. ---")
else:
    # Initialize Mistral Client
    try:
        MISTRAL_CLIENT = MistralClient(api_key=MISTRAL_API_KEY)
    except Exception as e:
        print(f"--- FATAL ERROR: Mistral Client initialization failed: {e} ---")
        MISTRAL_CLIENT = None

# Placeholder for ML Model Load. This will fail if house_model.pkl is missing.
try:
    # Load your ML model (replace with your actual model file if needed)
    MODEL = joblib.load("house_model.pkl") 
    print("--- INFO: house_model.pkl loaded successfully. ---")
except FileNotFoundError:
    print("--- WARNING: house_model.pkl not found. Using placeholder prediction logic. ---")
    MODEL = None
except Exception as e:
    print(f"--- WARNING: Error loading model: {e}. Using placeholder prediction logic. ---")
    MODEL = None


# --- Standard FastAPI Setup ---
app = FastAPI(title="UrbanEstimator App")
templates = Jinja2Templates(directory="templates") 
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Pydantic Models for Data Validation ---

class ChatInput(BaseModel):
    message: str
    context_price: str | None = None
    context_features: dict | None = None

class PropertyAnalysisRequest(BaseModel):
    price: float
    features: dict
    formatted_price: str


# --- Frontend Page Routes ---

# 1. Home Page (Index)
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_home_page(request: Request):
    """Serves the main landing page (index.html)."""
    return templates.TemplateResponse("index.html", {"request": request})


# 2. Predictor Page (Map)
@app.get("/map", response_class=HTMLResponse, include_in_schema=False)
async def get_predictor_page(request: Request):
    """Serves the map-based predictor interface (predictor.html)."""
    return templates.TemplateResponse("predictor.html", {"request": request})


# 3. Chat Page
@app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
async def get_chat_page(request: Request):
    """Serves the voice-enabled chat interface."""
    return templates.TemplateResponse("chat.html", {"request": request})


# --- API Endpoints ---

# 4. Prediction API Route (Uses Form data for compatibility with HTML fetch)
@app.post("/predict")
async def predict_house_price(
    latitude: float = Form(...),
    longitude: float = Form(...),
    Area: float = Form(...),
    Bedrooms: int = Form(...), 
    Bathrooms: int = Form(...),
):
    """Accepts house features and returns a price prediction."""
    
    input_data = np.array([[
        latitude, 
        longitude, 
        Area, 
        Bedrooms, 
        Bathrooms
    ]])

    if MODEL:
        try:
            # Use the loaded model for prediction
            prediction_inr = MODEL.predict(input_data)[0]
            prediction_in_lakhs = prediction_inr / 100000.0
        except Exception as e:
            print(f"--- ML Prediction Error: {e} ---")
            prediction_in_lakhs = (Area * 0.01) + (Bedrooms * 5) + (Bathrooms * 2) # Fallback
    else:
        # Fallback Placeholder Logic (used if model failed to load)
        prediction_in_lakhs = (Area * 0.01) + (Bedrooms * 5) + (Bathrooms * 2) 

    return {
        "status": "success",
        "input_features": {
            "latitude": latitude,
            "longitude": longitude,
            "Area": Area,
            "Bedrooms": Bedrooms,
            "Bathrooms": Bathrooms
        },
        "predicted_price_in_lakhs": round(prediction_in_lakhs, 2)
    }


# 5. Chat Processing Route (Mistral AI)
@app.post("/chat")
def handle_chat(chat_request: ChatInput):
    """Handles chat messages with RAG context using Mistral AI."""
    
    system_prompt = f"""
    You are a polite, expert real estate AI assistant for the Indian market.
    Your knowledge comes from a machine learning model.
    
    Here is the CURRENT CONTEXT from the user's prediction (if any):
    - Predicted Price: {chat_request.context_price}
    - Property Features: {chat_request.context_features}
    
    Use this context to answer the user's question concisely. If the context is empty, respond with friendly, general information about local real estate trends or invite them to use the Predictor page.
    """
    
    if not MISTRAL_CLIENT:
        return {"response": "Sorry, the AI assistant is offline. Please check the server's MISTRAL_API_KEY."}

    # Prepare messages
    messages = []
    if ChatMessage:
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=chat_request.message)
        ]
    else:
        # Fallback to dictionary format if ChatMessage class is unavailable
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chat_request.message}
        ]

    # Use Mistral
    try:
        response = MISTRAL_CLIENT.chat(
            model="mistral-tiny", # Fast model for conversation
            messages=messages
        )
        return {"response": response.choices[0].message.content}
        
    except Exception as e:
        print(f"--- Mistral Chat API Error ---: {e}")
        return {"response": "Sorry, I'm having trouble connecting to my brain right now. Please check my API key or try again in a moment."}


# 6. Analysis API Route (Mistral AI)
@app.post("/analyze_property")
def analyze_property(req: PropertyAnalysisRequest):
    """Analyzes property investment potential using Mistral AI."""

    if not MISTRAL_CLIENT:
        return {"analysis": "Investment analysis is unavailable. Please check the server's MISTRAL_API_KEY."}

    features = req.features
    
    # Use Mistral for detailed analysis
    system_prompt = """
    You are a professional real estate investment analyst. Your task is to provide a concise, balanced analysis (in exactly 3 bullet points) of the property data provided below. Focus on market valuation, potential risks, and investment class (e.g., luxury, starter home). Do not use placeholders (N/A) in the final response.
    """
    
    user_prompt = f"""
    Analyze this property for investment potential:
    - Predicted Price: {req.formatted_price}
    - Area: {features.get('Area', 'Unknown')} sq. ft.
    - Bedrooms: {features.get('Bedrooms', 'Unknown')} BHK
    - Bathrooms: {features.get('Bathrooms', 'Unknown')}
    - Location: Latitude {features.get('latitude', 'Unknown')}, Longitude {features.get('longitude', 'Unknown')}
    
    Provide your analysis now using exactly three bullet points.
    """
    
    # Prepare messages
    if ChatMessage:
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    try:
        response = MISTRAL_CLIENT.chat(
            model="mistral-medium", # More capable model for detailed analysis
            messages=messages
        )
        return {"analysis": response.choices[0].message.content}
        
    except Exception as e:
        print(f"--- Mistral Analysis API Error ---: {e}")
        return {"analysis": "Error generating investment insight."}