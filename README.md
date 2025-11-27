🏡 UrbanEstimator - AI-Powered Real Estate Valuation

UrbanEstimator is a machine learning-powered web application that predicts real estate prices in New Delhi with 91% accuracy. It uses a Random Forest Regressor trained on 7,000+ property listings to provide instant valuations based on location, area, and amenities.

🚀 Live Application: https://house-price-prediction-project-sxyf.onrender.com

🌟 Features

High-Accuracy Model: Trained on real market data using Random Forest Regression.

Interactive Map: Drag-and-drop pin selection using Leaflet.js to capture precise Latitude/Longitude.

Smart Feature Engineering: Automatically calculates Bath-to-Bed Ratio to assess luxury levels.

Indian Currency Formatting: Automatically formats large numbers into Crores and Lakhs (e.g., "Rs. 2 Crore 50 Lakhs").

Modern UI: Dark-themed, responsive interface.

🛠️ Tech Stack

Frontend: HTML5, CSS3, JavaScript, Leaflet.js (Maps)

Backend: Python, FastAPI, Uvicorn

Machine Learning: Scikit-Learn, Pandas, NumPy, Joblib

DevOps: Docker

Algorithm: Random Forest Regressor (R2 Score: 0.91)

🚀 Installation & Setup

Option 1: Standard Installation

1. Clone the Repository

git clone [https://github.com/Aakarshkumar612/House-Price-Prediction-Project.git](https://github.com/Aakarshkumar612/House-Price-Prediction-Project.git)
cd House-Price-Prediction-Project


2. Create Virtual Environment

python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


4. Run the Application

uvicorn app:app --reload


Open your browser at http://127.0.0.1:8000

Option 2: Run with Docker 🐳

If you have Docker installed, you can run the app without installing Python or libraries manually.

# 1. Build the image
docker build -t urban-estimator .

# 2. Run the container
docker run -p 8000:8000 urban-estimator


The app will be live at http://localhost:8000.

🧠 Model Training

To retrain the model with new data:

Place your dataset as house_data.csv in the root folder.

Run the training pipeline:

python train_model.py


This performs Data Cleaning, Outlier Removal, Feature Engineering, and Grid Search tuning before saving the new house_model.pkl.

🤝 Contributing

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📜 License

Distributed under the MIT License. See LICENSE for more information.