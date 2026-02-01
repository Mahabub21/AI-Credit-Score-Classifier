#!/usr/bin/env python3
"""
Model Training and Saving Script for Credit Score Prediction Web App

This script trains the machine learning model and saves the necessary components
for the Flask web application to use for predictions.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def clean_data(df):
    """Clean and preprocess the dataset"""
    df_clean = df.copy()
    
    # Clean Age column
    df_clean['Age'] = pd.to_numeric(df_clean['Age'], errors='coerce')
    df_clean['Age'] = df_clean['Age'].apply(lambda x: x if x > 0 and x < 100 else np.nan)
    
    # Clean Annual_Income
    df_clean['Annual_Income'] = pd.to_numeric(df_clean['Annual_Income'], errors='coerce')
    df_clean['Annual_Income'] = df_clean['Annual_Income'].apply(lambda x: x if x > 0 else np.nan)
    
    # Clean Monthly_Inhand_Salary
    df_clean['Monthly_Inhand_Salary'] = pd.to_numeric(df_clean['Monthly_Inhand_Salary'], errors='coerce')
    
    # Clean numerical columns
    numerical_columns = ['Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 
                        'Num_of_Loan', 'Delay_from_due_date', 'Changed_Credit_Limit',
                        'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
                        'Total_EMI_per_month', 'Monthly_Balance']
    
    for col in numerical_columns:
        if col in df_clean.columns:
            # Remove string artifacts
            df_clean[col] = df_clean[col].astype(str).str.replace('_', '').str.replace('#F%$D@*&8', '')
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Clean Amount_invested_monthly
    if 'Amount_invested_monthly' in df_clean.columns:
        df_clean['Amount_invested_monthly'] = df_clean['Amount_invested_monthly'].astype(str)
        df_clean['Amount_invested_monthly'] = df_clean['Amount_invested_monthly'].str.replace('__', '').str.replace('_', '')
        df_clean['Amount_invested_monthly'] = pd.to_numeric(df_clean['Amount_invested_monthly'], errors='coerce')
    
    # Clean Num_of_Delayed_Payment
    if 'Num_of_Delayed_Payment' in df_clean.columns:
        df_clean['Num_of_Delayed_Payment'] = df_clean['Num_of_Delayed_Payment'].astype(str).str.replace('_', '')
        df_clean['Num_of_Delayed_Payment'] = pd.to_numeric(df_clean['Num_of_Delayed_Payment'], errors='coerce')
    
    # Clean categorical columns
    if 'Credit_Mix' in df_clean.columns:
        df_clean['Credit_Mix'] = df_clean['Credit_Mix'].replace('_', 'Bad').fillna('Standard')
    
    if 'Payment_of_Min_Amount' in df_clean.columns:
        df_clean['Payment_of_Min_Amount'] = df_clean['Payment_of_Min_Amount'].replace(['NM', 'nm'], 'No').fillna('No')
    
    if 'Payment_Behaviour' in df_clean.columns:
        df_clean['Payment_Behaviour'] = df_clean['Payment_Behaviour'].str.replace('!@9#%8', 'Low_spent_Small_value_payments')
    
    if 'Occupation' in df_clean.columns:
        df_clean['Occupation'] = df_clean['Occupation'].str.replace('_______', 'Unknown')
    
    # Extract Credit History Age in years
    if 'Credit_History_Age' in df_clean.columns:
        df_clean['Credit_History_Age_Years'] = df_clean['Credit_History_Age'].str.extract('(\d+)').astype(float)
    
    return df_clean

def prepare_features(df, is_training=True):
    """Prepare features for machine learning"""
    df_processed = df.copy()
    
    # Exclude non-predictive columns
    features_to_exclude = ['ID', 'Customer_ID', 'Name', 'SSN', 'Month', 'Credit_History_Age']
    if is_training:
        features_to_exclude.append('Credit_Score')
    
    feature_columns = [col for col in df_processed.columns if col not in features_to_exclude]
    X = df_processed[feature_columns].copy()
    
    # Handle missing values for numerical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        X[col] = X[col].fillna(X[col].median())
    
    # Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].fillna('Unknown')
    
    # Create binary features from categorical variables
    for col in categorical_cols:
        if col in X.columns:
            # Get top categories to avoid too many features
            top_categories = X[col].value_counts().head(10).index.tolist()
            for category in top_categories:
                X[f'{col}_{category}'] = (X[col] == category).astype(int)
    
    # Drop original categorical columns
    X = X.drop(columns=categorical_cols)
    
    # Feature engineering
    if 'Annual_Income' in X.columns and 'Monthly_Inhand_Salary' in X.columns:
        X['Income_Salary_Ratio'] = X['Annual_Income'] / (X['Monthly_Inhand_Salary'] * 12 + 1)
    
    if 'Outstanding_Debt' in X.columns and 'Annual_Income' in X.columns:
        X['Debt_Income_Ratio'] = X['Outstanding_Debt'] / (X['Annual_Income'] + 1)
    
    if 'Total_EMI_per_month' in X.columns and 'Monthly_Inhand_Salary' in X.columns:
        X['EMI_Salary_Ratio'] = X['Total_EMI_per_month'] / (X['Monthly_Inhand_Salary'] + 1)
    
    if 'Num_Credit_Card' in X.columns and 'Num_Bank_Accounts' in X.columns:
        X['Credit_Bank_Ratio'] = X['Num_Credit_Card'] / (X['Num_Bank_Accounts'] + 1)
    
    # Replace infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    return X

def train_and_save_model():
    """Train the best model and save all components"""
    print("Starting model training and saving process...")
    
    # Load and clean data
    print("Loading and cleaning data...")
    train_data = pd.read_csv('dataset/train.csv')
    train_clean = clean_data(train_data)
    
    # Prepare features
    print("Preparing features...")
    X_train_processed = prepare_features(train_clean, is_training=True)
    y_train = train_clean['Credit_Score'].copy()
    
    # Split data
    X_train, X_val, y_train_split, y_val = train_test_split(
        X_train_processed, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    }
    
    # Prepare label encoder and scaler
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_split)
    y_val_encoded = label_encoder.transform(y_val)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    best_model = None
    best_score = 0
    best_model_name = ""
    
    print("Training models...")
    for name, model in models.items():
        print(f"Training {name}...")
        
        if name in ['XGBoost', 'LightGBM']:
            model.fit(X_train, y_train_encoded)
            val_predictions_encoded = model.predict(X_val)
            val_predictions = label_encoder.inverse_transform(val_predictions_encoded)
        else:
            model.fit(X_train, y_train_split)
            val_predictions = model.predict(X_val)
        
        f1 = f1_score(y_val, val_predictions, average='weighted')
        accuracy = accuracy_score(y_val, val_predictions)
        
        print(f"{name} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} with F1-Score: {best_score:.4f}")
    
    # Create directory for model components
    os.makedirs('model_components', exist_ok=True)
    
    # Save model components
    print("Saving model components...")
    joblib.dump(best_model, 'model_components/best_model.pkl')
    joblib.dump(scaler, 'model_components/scaler.pkl')
    joblib.dump(label_encoder if best_model_name in ['XGBoost', 'LightGBM'] else None, 'model_components/label_encoder.pkl')
    joblib.dump(list(X_train.columns), 'model_components/feature_columns.pkl')
    
    # Save model metadata
    metadata = {
        'model_name': best_model_name,
        'f1_score': best_score,
        'accuracy': accuracy_score(y_val, val_predictions),
        'feature_count': len(X_train.columns),
        'training_samples': len(X_train)
    }
    
    joblib.dump(metadata, 'model_components/model_metadata.pkl')
    
    print("Model components saved successfully!")
    print(f"Files saved in 'model_components/' directory:")
    print("- best_model.pkl")
    print("- scaler.pkl")
    print("- label_encoder.pkl")
    print("- feature_columns.pkl")
    print("- model_metadata.pkl")
    
    return metadata

if __name__ == "__main__":
    try:
        metadata = train_and_save_model()
        print(f"\nTraining completed successfully!")
        print(f"Best model: {metadata['model_name']}")
        print(f"F1-Score: {metadata['f1_score']:.4f}")
        print(f"Accuracy: {metadata['accuracy']:.4f}")
        print("\nYou can now run the Flask app with: python app.py")
    except Exception as e:
        print(f"Error during training: {str(e)}")