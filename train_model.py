import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# --- 1. DATA INGESTION ---
def load_data(filepath):
    print("--- 1. Loading Data ---")
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # CLEANUP: Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        return df
    except FileNotFoundError:
        print("ERROR: 'house_data.csv' not found. Please place your dataset in this folder.")
        return None

# --- 2. EDA & CLEANING ---
def clean_data(df):
    print("\n--- 2. Cleaning Data & EDA ---")
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Handle Missing Values (Imputation)
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
        
    # --- OUTLIER REMOVAL ---
    if 'Area' in df.columns and 'Bedrooms' in df.columns:
        # Remove unrealistic houses (Area/Bedrooms < 200 sqft)
        df = df[~((df['Bedrooms'] > 0) & ((df['Area'] / df['Bedrooms']) < 200))]
    
    # Remove price outliers (Top 1% and Bottom 1%)
    if 'Price' in df.columns:
        q_low = df["Price"].quantile(0.01)
        q_hi  = df["Price"].quantile(0.99)
        df = df[(df["Price"] < q_hi) & (df["Price"] > q_low)]
    
    print(f"Removed outliers. New Shape: {df.shape}")
    return df

# --- 3. FEATURE ENGINEERING ---
def engineer_features(df):
    print("\n--- 3. Feature Engineering ---")
    
    # Ratio of Bathrooms to Bedrooms
    if 'Bathrooms' in df.columns and 'Bedrooms' in df.columns:
        df['Bath_Bed_Ratio'] = df['Bathrooms'] / df['Bedrooms']
        df['Bath_Bed_Ratio'] = df['Bath_Bed_Ratio'].fillna(0)
        df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df

# --- 4. MODEL TRAINING PIPELINE ---
def train_model(df):
    print("\n--- 4. Model Training & Tuning ---")
    
    # Define Inputs (X) and Output (y)
    feature_cols = ['latitude', 'longitude', 'Area', 'Bedrooms', 'Bathrooms', 'Bath_Bed_Ratio']
    target_col = 'Price'
    
    # Check if columns exist
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print(f"CRITICAL ERROR: Features missing after processing: {missing}")
        return None

    X = df[feature_cols]
    y = df[target_col]
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    # Hyperparameter Tuning
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5]
    }
    
    print("Starting Grid Search (this may take a minute)...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Evaluation
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nModel Accuracy (R2 Score): {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    
    return best_model

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    csv_file = "house_data.csv"
    
    df = load_data(csv_file)
    
    if df is not None:
        print("\n--- DEBUGGING COLUMNS ---")
        print("Your CSV Columns are:", df.columns.tolist())
        print("-------------------------\n")

        # --- UPDATED MAPPING FOR YOUR DATASET ---
        column_mapping = {
            'price': 'Price',          # Maps 'price' -> 'Price'
            'area': 'Area',            # Maps 'area'  -> 'Area'
            'Bedrooms': 'Bedrooms',    # Matches
            'Bathrooms': 'Bathrooms',  # Matches
            'latitude': 'latitude',    # Matches
            'longitude': 'longitude'   # Matches
        }
        
        # Apply renaming
        df = df.rename(columns=column_mapping)
        
        # Check required columns
        required_cols = ['Price', 'latitude', 'longitude', 'Area', 'Bedrooms', 'Bathrooms']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if not missing_cols:
            df = clean_data(df)
            df = engineer_features(df)
            model = train_model(df)
            
            if model:
                joblib.dump(model, "house_model.pkl")
                print("\n--- SUCCESS: Model saved as 'house_model.pkl' ---")
                print("Don't forget to restart your 'app.py' server to use the new model!")
        else:
            print(f"❌ ERROR: Dataset is still missing columns: {missing_cols}")