import os
import numpy as np
import joblib
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# --- NEW: Gemini AI Setup ---
import google.generativeai as genai

# --- Load API Key from .env file ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("--- ERROR: GEMINI_API_KEY not found. Please check your .env file. ---")
else:
    # --- 3. Configure the Gemini "Brain" ---
    genai.configure(api_key=GEMINI_API_KEY)

# --- Standard FastAPI Setup ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")
model = joblib.load("house_model.pkl")

# --- Pydantic Models (Data Contracts) ---
# These are the same as before
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

# --- Frontend Page Routes ---
# These are the same as before
@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chatpage")
def get_chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# --- API Endpoints ---
# This is the same as before
@app.post("/predict")
def predict_price(data: HouseData):
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


# --- NEW: This is the "Gemini Brain" ---
@app.post("/chat")
def handle_chat(chat_request: ChatRequest):
    
    # 1. Create the "Augmentation" (the RAG part)
    system_prompt = f"""
    You are a polite, expert real estate AI assistant for the Delhi market.
    Your knowledge comes from a machine learning model.
    
    Here is the CURRENT CONTEXT from the user's prediction (if any):
    - Predicted Price: {chat_request.context_price}
    - Property Features: {chat_request.context_features}
    
    Use this context to answer the user's question.
    If the context is empty, just have a friendly, general conversation.
    """
    
    # 2. Initialize the Gemini Model
    # We use 'gemini-1.5-flash' for speed
    model = genai.GenerativeModel('gemini-pro-latest')
    
    # 3. Create the chat session with our RAG prompt
    chat = model.start_chat(history=[
        {'role': 'user', 'parts': [system_prompt]},
        {'role': 'model', 'parts': ["Understood. I am a helpful real estate AI. I will use the context provided."]},
    ])
    
    # 4. Send the user's message to Gemini
    try:
        response = chat.send_message(chat_request.message)
        bot_response = response.text
    except Exception as e:
        print(f"--- Gemini API Error ---: {e}")
        bot_response = "Sorry, I'm having a little trouble connecting to my brain right now. Please check my API key or try again in a moment."

    return {"response": bot_response}