from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Add some configuration for better debugging
app.config['DEBUG'] = True

@app.after_request
def after_request(response):
    """Add headers to allow CORS and improve debugging"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Global variables for model and preprocessing components
model = None
scaler = None
label_encoder = None
feature_columns = None

def load_model_components():
    """Load the trained model and preprocessing components"""
    global model, scaler, label_encoder, feature_columns
    
    try:
        # Try to load saved model components
        model = joblib.load('model_components/best_model.pkl')
        scaler = joblib.load('model_components/scaler.pkl')
        label_encoder = joblib.load('model_components/label_encoder.pkl')
        feature_columns = joblib.load('model_components/feature_columns.pkl')
        print("Model components loaded successfully!")
        return True
    except:
        print("Model components not found. Please train and save the model first.")
        return False

def prepare_input_features(input_data):
    """Prepare user input for model prediction"""
    # Create a DataFrame with user input
    df = pd.DataFrame([input_data])
    
    # Handle categorical variables - create binary features
    categorical_mapping = {
        'Occupation': ['Scientist', 'Teacher', 'Engineer', 'Manager', 'Doctor', 'Lawyer', 'Unknown'],
        'Credit_Mix': ['Good', 'Standard', 'Bad'],
        'Payment_of_Min_Amount': ['Yes', 'No'],
        'Payment_Behaviour': ['High_spent_Small_value_payments', 'Low_spent_Small_value_payments', 
                             'Low_spent_Medium_value_payments', 'Low_spent_Large_value_payments',
                             'High_spent_Medium_value_payments', 'High_spent_Large_value_payments']
    }
    
    # Create binary features for categorical variables
    for col, categories in categorical_mapping.items():
        for category in categories:
            df[f'{col}_{category}'] = (df[col] == category).astype(int)
    
    # Drop original categorical columns
    df = df.drop(columns=list(categorical_mapping.keys()))
    
    # Add engineered features
    if 'Annual_Income' in df.columns and 'Monthly_Inhand_Salary' in df.columns:
        df['Income_Salary_Ratio'] = df['Annual_Income'] / (df['Monthly_Inhand_Salary'] * 12 + 1)
    
    if 'Outstanding_Debt' in df.columns and 'Annual_Income' in df.columns:
        df['Debt_Income_Ratio'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1)
    
    if 'Total_EMI_per_month' in df.columns and 'Monthly_Inhand_Salary' in df.columns:
        df['EMI_Salary_Ratio'] = df['Total_EMI_per_month'] / (df['Monthly_Inhand_Salary'] + 1)
    
    if 'Num_Credit_Card' in df.columns and 'Num_Bank_Accounts' in df.columns:
        df['Credit_Bank_Ratio'] = df['Num_Credit_Card'] / (df['Num_Bank_Accounts'] + 1)
    
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], 0)
    
    # Ensure all required columns are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Select only the required columns in the correct order
    df = df[feature_columns]
    
    return df

@app.route('/')
def home():
    """Home page with credit score prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please run "python train_and_save_model.py" first to train and save the model.',
                'success': False
            }), 500
        
        # Validate that all required fields are present
        required_fields = ['age', 'annual_income', 'monthly_salary', 'num_bank_accounts', 
                          'num_credit_cards', 'interest_rate', 'num_loans', 'delay_days',
                          'delayed_payments', 'credit_limit_change', 'credit_inquiries',
                          'outstanding_debt', 'credit_utilization', 'credit_history_years',
                          'emi_per_month', 'monthly_investment', 'monthly_balance',
                          'occupation', 'credit_mix', 'payment_min_amount', 'payment_behaviour']
        
        missing_fields = [field for field in required_fields if field not in request.form or not request.form[field]]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'success': False
            }), 400
        
        # Get form data with validation
        try:
            input_data = {
                'Age': float(request.form['age']),
                'Annual_Income': float(request.form['annual_income']),
                'Monthly_Inhand_Salary': float(request.form['monthly_salary']),
                'Num_Bank_Accounts': int(request.form['num_bank_accounts']),
                'Num_Credit_Card': int(request.form['num_credit_cards']),
                'Interest_Rate': float(request.form['interest_rate']),
                'Num_of_Loan': int(request.form['num_loans']),
                'Delay_from_due_date': int(request.form['delay_days']),
                'Num_of_Delayed_Payment': int(request.form['delayed_payments']),
                'Changed_Credit_Limit': float(request.form['credit_limit_change']),
                'Num_Credit_Inquiries': int(request.form['credit_inquiries']),
                'Outstanding_Debt': float(request.form['outstanding_debt']),
                'Credit_Utilization_Ratio': float(request.form['credit_utilization']),
                'Credit_History_Age_Years': int(request.form['credit_history_years']),
                'Total_EMI_per_month': float(request.form['emi_per_month']),
                'Amount_invested_monthly': float(request.form['monthly_investment']),
                'Monthly_Balance': float(request.form['monthly_balance']),
                'Occupation': request.form['occupation'],
                'Credit_Mix': request.form['credit_mix'],
                'Payment_of_Min_Amount': request.form['payment_min_amount'],
                'Payment_Behaviour': request.form['payment_behaviour']
            }
        except (ValueError, KeyError) as e:
            return jsonify({
                'error': f'Invalid input data: {str(e)}. Please check that all numeric fields contain valid numbers.',
                'success': False
            }), 400
        
        # Prepare features
        features_df = prepare_input_features(input_data)
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            # Get prediction and probabilities
            prediction_encoded = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            # Convert back to original labels if using label encoder
            if label_encoder is not None:
                prediction = label_encoder.inverse_transform([prediction_encoded])[0]
                class_names = label_encoder.classes_
            else:
                prediction = prediction_encoded
                class_names = model.classes_
            
            # Create probability dictionary
            prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
            
        else:
            prediction_encoded = model.predict(features_scaled)[0]
            prediction = label_encoder.inverse_transform([prediction_encoded])[0] if label_encoder else prediction_encoded
            prob_dict = {}
        
        # Determine risk level and color
        risk_info = get_risk_info(prediction)
        
        return jsonify({
            'prediction': str(prediction),
            'probabilities': prob_dict,
            'risk_level': risk_info['level'],
            'risk_color': risk_info['color'],
            'risk_description': risk_info['description'],
            'success': True
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")  # For debugging
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }), 500

def get_risk_info(prediction):
    """Get risk level information based on prediction"""
    risk_mapping = {
        'Good': {
            'level': 'Low Risk',
            'color': 'success',
            'description': 'Excellent creditworthiness. Low default probability.'
        },
        'Standard': {
            'level': 'Medium Risk', 
            'color': 'warning',
            'description': 'Average creditworthiness. Moderate default probability.'
        },
        'Poor': {
            'level': 'High Risk',
            'color': 'danger', 
            'description': 'Poor creditworthiness. High default probability.'
        }
    }
    return risk_mapping.get(prediction, {
        'level': 'Unknown',
        'color': 'secondary',
        'description': 'Unable to determine risk level.'
    })

@app.route('/about')
def about():
    """About page with model information"""
    return render_template('about.html')

@app.route('/test', methods=['GET', 'POST'])
def test():
    """Test endpoint to check if routes are working"""
    if request.method == 'POST':
        return jsonify({'message': 'POST request received successfully', 'method': 'POST'})
    else:
        return jsonify({'message': 'GET request received successfully', 'method': 'GET'})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'feature_columns_loaded': feature_columns is not None
    })

if __name__ == '__main__':
    # Try to load model components
    model_loaded = load_model_components()
    
    if not model_loaded:
        print("Warning: Model not loaded. Please run the training script first.")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)