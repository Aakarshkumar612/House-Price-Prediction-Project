from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np

# Create a new FastAPI app
app = FastAPI()

# --- TEMPLATING SETUP ---
templates = Jinja2Templates(directory="templates")

# --- 1. Load your prediction model ---
model = joblib.load("house_model.pkl")

# --- 2. Define Pydantic Models (Data Structures) ---

# This is for your prediction model
class HouseData(BaseModel):
    latitude: float
    longitude: float
    Area: float
    Bedrooms: int
    Bathrooms: int

# --- NEW: This is for your chatbot ---
class ChatMessage(BaseModel):
    message: str

# --- 3. Define API Endpoints ---

# Root route to serve the HTML page
@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Add this new route to your app.py
@app.get("/chatpage")
def get_chat_page(request: Request):
    # This will find chat.html in the 'templates' folder
    return templates.TemplateResponse("chat.html", {"request": request})

# Prediction route (same as before)
@app.post("/predict")
def predict_price(data: HouseData):
    
    # Convert input data to a numpy array
    input_data = np.array([[
        data.latitude, 
        data.longitude, 
        data.Area, 
        data.Bedrooms, 
        data.Bathrooms
    ]])
    
    # Make a prediction (this is in raw INR)
    prediction_inr = model.predict(input_data)[0]
    
    # Convert INR to Lakhs
    prediction_in_lakhs = prediction_inr / 100000.0
    
    # Return the converted prediction
    return {"predicted_price_in_lakhs": prediction_in_lakhs}

# --- NEW: Chatbot route ---
@app.post("/chat")
def handle_chat(chat_message: ChatMessage):
    user_message = chat_message.message.lower() # Convert to lowercase
    
    # --- THE FIX ---
    # Split the message into a list of individual words
    words = user_message.split()
    
    # Now, check if the *word* "hi" or "hello" is in the list.
    # This won't be fooled by "delhi" anymore.
    if "hello" in words or "hi" in words:
        bot_response = "Hi there! How can I help you with your house price prediction?"
    
    # --- The rest of the rules ---
    # We use 'in user_message' here, which is fine for longer keywords
    elif "how are you" in user_message:
        bot_response = "I'm just a set of rules, but I'm happy to help!"
    elif "price" in user_message or "predict" in user_message:
        bot_response = "To get a price, please drag the marker on the map and fill in the form with the area, bedrooms, and bathrooms. Then click 'Predict Price'!"
    elif "data" in user_message or "model" in user_message:
        bot_response = "This model is trained on a dataset of real house listings from Delhi, using a Random Forest algorithm."
    elif "bye" in user_message or "thanks" in user_message:
        bot_response = "Goodbye! Happy to help."
    else:
        bot_response = "Sorry, I don't understand that. You can ask me about the price, the data, or the model."
    
    return {"response": bot_response}
   