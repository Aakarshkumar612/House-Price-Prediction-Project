<<<<<<< HEAD
# House-Price-Prediction-Project
"AI-Powered House Price Predictor"
=======
# 🏠 AI-Powered House Price Predictor

This is a full-stack web application that predicts house prices in Delhi using a machine learning model and features an interactive map and an AI chatbot assistant.

## ✨ Features

* **Interactive Map:** Uses Leaflet.js to select a location and get its latitude/longitude.
* **ML Model:** A Random Forest model (trained on Delhi housing data) predicts the price in Lakhs.
* **FastAPI Backend:** A high-performance Python backend serves the model and chat logic.
* **RAG AI Chatbot:** A dedicated chat page powered by the Mistral AI API provides context-aware answers about predictions and the market.

## 🛠️ Tech Stack

* **Frontend:** HTML, CSS, JavaScript, Leaflet.js
* **Backend:** Python, FastAPI, Uvicorn
* **Machine Learning:** Scikit-learn, Pandas, Joblib
* **AI:** Mistral AI, Pydantic, python-dotenv

## 🚀 How to Run This Project

1.  **Clone the repository (or download the ZIP):**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/House-Price-Project.git](https://github.com/YOUR_USERNAME/House-Price-Project.git)
    cd House-Price-Project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create your `.env` file:**
    Create a file named `.env` and add your Mistral API key:
    ```
    MISTRAL_API_KEY=YOUR_API_KEY_GOES_HERE
    ```

5.  **Run the server:**
    ```bash
    uvicorn app:app --reload
    ```

6.  **Open the app:**
    Go to `http://127.0.0.1:8000` in your browser.
>>>>>>> 1b6fb42 (Initial commit: Full project setup with ML model, FastAPI, and AI chatbot)
