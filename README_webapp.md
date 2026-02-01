# Credit Score Predictor Web Application

## 🚀 Quick Start

### 1. Train and Save the Model
First, run the training script to create the model components:

```bash
python train_and_save_model.py
```

This will:
- Train multiple ML models on your credit data
- Select the best performing model
- Save all necessary components for the web app

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
python app.py
```

The application will be available at: `http://localhost:5000`

## 📁 Project Structure

```
├── app.py                      # Flask web application
├── train_and_save_model.py     # Model training and saving script
├── requirements.txt            # Python dependencies
├── dataset/
│   ├── train.csv              # Training data
│   └── test.csv               # Test data
├── templates/
│   ├── index.html             # Main prediction form
│   └── about.html             # About page
├── static/
│   └── style.css              # Custom CSS styles
├── model_components/           # Saved model files (created after training)
│   ├── best_model.pkl         # Trained ML model
│   ├── scaler.pkl             # Feature scaler
│   ├── label_encoder.pkl      # Label encoder
│   ├── feature_columns.pkl    # Feature column names
│   └── model_metadata.pkl     # Model metadata
└── README_webapp.md           # This file
```

## 🎯 Features

### Web Interface
- **Responsive Design**: Works on desktop, tablet, and mobile
- **User-Friendly Form**: Step-by-step credit information input
- **Real-Time Predictions**: Instant credit score classification
- **Probability Scores**: Shows confidence for each credit category
- **Risk Assessment**: Clear risk level indication with color coding

### Machine Learning
- **Multiple Algorithms**: Random Forest, XGBoost, LightGBM
- **Feature Engineering**: 25+ financial and behavioral indicators
- **Data Cleaning**: Handles missing values and data inconsistencies
- **Model Selection**: Automatically selects best performing model

### Credit Score Categories
- **🟢 Good**: Excellent creditworthiness, low default risk
- **🟡 Standard**: Average creditworthiness, moderate risk  
- **🔴 Poor**: High risk, requires careful evaluation

## 📊 Input Fields

### Personal Information
- Age
- Occupation

### Financial Information  
- Annual Income
- Monthly In-hand Salary
- Monthly Balance
- Amount Invested Monthly

### Banking & Credit
- Number of Bank Accounts
- Number of Credit Cards
- Interest Rate
- Credit Utilization Ratio

### Loan Information
- Number of Loans
- Total EMI per Month
- Outstanding Debt
- Changed Credit Limit

### Payment History
- Delay from Due Date
- Number of Delayed Payments
- Number of Credit Inquiries
- Credit History Age
- Credit Mix
- Payment of Minimum Amount
- Payment Behaviour

## 🔧 Technical Details

### Backend
- **Flask**: Web framework
- **scikit-learn**: Machine learning algorithms
- **XGBoost/LightGBM**: Gradient boosting models
- **pandas/numpy**: Data processing

### Frontend
- **Bootstrap 5**: Responsive CSS framework
- **Font Awesome**: Icons
- **JavaScript**: Form handling and AJAX

### Model Performance
- **Accuracy**: ~78%+ on validation data
- **F1-Score**: ~78%+ weighted average
- **Training Data**: 100,000+ credit records
- **Features**: 25+ engineered features

## 🚨 Important Notes

1. **Training Required**: You must run `train_and_save_model.py` before starting the web app
2. **Data Path**: Ensure `dataset/train.csv` exists in the project directory  
3. **Model Files**: The `model_components/` directory is created after training
4. **Browser Compatibility**: Works with modern browsers (Chrome, Firefox, Safari, Edge)
5. **Educational Use**: This is a demonstration tool, not for production credit decisions

## 🐛 Troubleshooting

### Model Not Found Error
```
Model components not found. Please train and save the model first.
```
**Solution**: Run `python train_and_save_model.py`

### Import Errors
```
ModuleNotFoundError: No module named 'flask'
```
**Solution**: Install requirements with `pip install -r requirements.txt`

### Data File Not Found
```
FileNotFoundError: dataset/train.csv
```
**Solution**: Ensure the dataset files are in the correct location

### Port Already in Use
```
OSError: [Errno 48] Address already in use
```
**Solution**: Kill the process or change the port in `app.py`

## 📈 Usage Examples

### Example Input (Good Credit)
- Age: 35
- Annual Income: $75,000
- Credit Utilization: 15%
- Payment Delays: 0 days
- Credit History: 10 years

### Example Input (Poor Credit)  
- Age: 25
- Annual Income: $30,000
- Credit Utilization: 85%
- Payment Delays: 45 days
- Credit History: 2 years

## 🔮 Future Enhancements

- [ ] API endpoint for external integrations
- [ ] Batch prediction capability
- [ ] Model retraining interface
- [ ] Credit improvement recommendations
- [ ] Historical prediction tracking
- [ ] Advanced visualization charts
- [ ] User authentication system
- [ ] Database integration for storing predictions

## 📞 Support

For issues or questions about the web application, please check:
1. Training data is available in `dataset/train.csv`
2. All requirements are installed correctly
3. Model training completed successfully
4. Port 5000 is available

---

Built with ❤️ for financial technology and machine learning education.