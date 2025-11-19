import os
import numpy as np
import joblib
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from mistralai.client import MistralClient 

# Attempt to import ChatMessage from the two most common locations.
try:
    from mistralai.models.chat import ChatMessage
except ImportError:
    # Fallback structure if the preferred path is wrong (older SDKs)
    try:
        from mistralai.models import ChatMessage 
    except ImportError:
        # If both fail, we will use a basic dictionary structure for messages, 
        # but ChatMessage is strongly preferred. We proceed with the knowledge 
        # the model might be less reliable if ChatMessage fails to import.
        ChatMessage = None 
        print("--- WARNING: Cannot import ChatMessage model. AI reliability may be affected. ---")

# --- Load API Key from .env file ---
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") 

# --- AI Setup (MISTRAL ONLY) ---
MISTRAL_CLIENT = None
if not MISTRAL_API_KEY:
    print("--- FATAL ERROR: MISTRAL_API_KEY not found. Mistral features disabled. ---")
else:
    # Initialize Mistral Client
    MISTRAL_CLIENT = MistralClient(api_key=MISTRAL_API_KEY)


# --- Standard FastAPI Setup (Same as before) ---
app = FastAPI()
templates = Jinja2Templates(directory="templates") 
model = joblib.load("house_model.pkl") # Load your ML model

# --- Pydantic Models (Same as before) ---
class HouseData(BaseModel):
    latitude: float
    longitude: float
    Area: float
    Bedrooms: int
    Bathrooms: int

class ChatRequest(BaseModel):
    message: str
    context_price: str = "" 
    context_features: dict = {}

class PropertyAnalysisRequest(BaseModel):
    price: float
    features: dict
    formatted_price: str

# --- Frontend Page Routes (Same as before) ---
@app.get("/", response_class=HTMLResponse)
def serve_predictor(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
def serve_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# --- API Endpoints ---
@app.post("/predict")
def predict_price(data: HouseData):
    """Predicts house price using the loaded ML model."""
    input_data = np.array([[
        data.latitude, 
        data.longitude, 
        data.Area, 
        data.Bedrooms, 
        data.Bathrooms
    ]])
    prediction_inr = model.predict(input_data)[0]
    prediction_in_lakhs = prediction_inr / 100000.0
    return {"predicted_price_in_lakhs": prediction_in_lakhs}


@app.post("/chat")
def handle_chat(chat_request: ChatRequest):
    """Handles chat messages with RAG context using Mistral AI."""
    
    system_prompt = f"""
    You are a polite, expert real estate AI assistant for the Delhi market.
    Your knowledge comes from a machine learning model.
    
    Here is the CURRENT CONTEXT from the user's prediction (if any):
    - Predicted Price: {chat_request.context_price}
    - Property Features: {chat_request.context_features}
    
    Use this context to answer the user's question.
    If the context is empty, just have a friendly, general conversation about Delhi real estate trends.
    """
    
    if not MISTRAL_CLIENT:
        return {"response": "Sorry, the AI assistant is offline. Please check the server's MISTRAL_API_KEY."}

    # Prepare messages using ChatMessage if available, otherwise use dictionary (standard for older SDKs)
    if ChatMessage:
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=chat_request.message)
        ]
    else:
        # Fallback to dictionary format for older/inconsistent SDK versions
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
        return {"response": "Sorry, I'm having a trouble connecting to my brain right now. Please check my API key or try again in a moment."}


@app.post("/analyze_property")
def analyze_property(req: PropertyAnalysisRequest):
    """Analyzes property investment potential using Mistral AI."""

    if not MISTRAL_CLIENT:
        return {"analysis": "Investment analysis is unavailable. Please check the server's MISTRAL_API_KEY."}

    features = req.features
    
    # Use Mistral for detailed analysis
    system_prompt = """
    You are a professional real estate investment analyst. Your task is to provide a concise, balanced analysis (in 3-4 bullet points) of the property data provided below. Focus on market value, rental potential, and any immediate risks based on the price and features.
    """
    
    user_prompt = f"""
    Analyze this property for investment potential:
    - Predicted Price: {req.formatted_price}
    - Area: {features.get('Area', 'N/A')} sq. ft.
    - Bedrooms: {features.get('Bedrooms', 'N/A')} BHK
    - Bathrooms: {features.get('Bathrooms', 'N/A')}
    - Location: Latitude {features.get('latitude', 'N/A')}, Longitude {features.get('longitude', 'N/A')}
    
    Provide your analysis now using bullet points.
    """
    
    # Prepare messages using ChatMessage if available, otherwise use dictionary
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