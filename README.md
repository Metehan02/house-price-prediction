# 🏠 End-to-End House Price Prediction Pipeline

This project builds a complete machine learning pipeline to predict house prices using the Ames Housing dataset.

It includes data analysis, preprocessing, model training, evaluation, tuning, and an interactive web app.

---

## 📌 Project Features

- 📊 Exploratory Data Analysis (EDA)
- ⚙️ Data preprocessing pipeline (handling missing values, encoding, scaling)
- 🤖 Multiple ML models (Ridge, Random Forest, Gradient Boosting)
- 🔁 Cross-validation (K-Fold)
- 📈 Model comparison (table + chart)
- 🔍 Feature importance analysis + visualization
- 🎯 Hyperparameter tuning (GridSearchCV)
- 💾 Model saving and loading
- 🧪 Single prediction evaluation script
- 🌐 Interactive Streamlit web app

---

## 🧠 Models Used

- Ridge Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor (best model)

### ✅ Best Model (after tuning)

Gradient Boosting with:

- n_estimators = 200  
- learning_rate = 0.1  
- max_depth = 3  
- subsample = 0.8  

**Best CV RMSE: ~0.1308**

---

## 📊 Model Comparison (Cross-Validation)

| Model | CV Mean RMSE |
|------|-------------|
| GradientBoosting | 0.134 |
| RandomForest | 0.144 |
| Ridge | 0.148 |

---

## 🔍 Feature Importance

Top features influencing house prices:

- OverallQual
- GrLivArea
- TotalBsmtSF
- GarageCars
- YearBuilt

(See `src/feature_importance.py` for visualization)

---

## 🚀 How to Run

### 1. Clone the repository
# 🏠 End-to-End House Price Prediction Pipeline

This project builds a complete machine learning pipeline to predict house prices using the Ames Housing dataset.

It includes data analysis, preprocessing, model training, evaluation, tuning, and an interactive web app.

---

## 📌 Project Features

- 📊 Exploratory Data Analysis (EDA)
- ⚙️ Data preprocessing pipeline (handling missing values, encoding, scaling)
- 🤖 Multiple ML models (Ridge, Random Forest, Gradient Boosting)
- 🔁 Cross-validation (K-Fold)
- 📈 Model comparison (table + chart)
- 🔍 Feature importance analysis + visualization
- 🎯 Hyperparameter tuning (GridSearchCV)
- 💾 Model saving and loading
- 🧪 Single prediction evaluation script
- 🌐 Interactive Streamlit web app

---

## 🧠 Models Used

- Ridge Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor (best model)

### ✅ Best Model (after tuning)

Gradient Boosting with:

- n_estimators = 200  
- learning_rate = 0.1  
- max_depth = 3  
- subsample = 0.8  

**Best CV RMSE: ~0.1308**

---

## 📊 Model Comparison (Cross-Validation)

| Model | CV Mean RMSE |
|------|-------------|
| GradientBoosting | 0.134 |
| RandomForest | 0.144 |
| Ridge | 0.148 |

---

## 🔍 Feature Importance

Top features influencing house prices:

- OverallQual
- GrLivArea
- TotalBsmtSF
- GarageCars
- YearBuilt

(See `src/feature_importance.py` for visualization)

---

## 🚀 How to Run

### 1. Clone the repository
git clone https://github.com/Metehan02/house-price-prediction.git
cd house-price-prediction


---

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate


---

### 3. Install dependencies
pip install -r requirements.txt


---

### 4. Train models
python -m src.train


---

### 5. Run prediction script
python -m src.predict


---

### 6. Run single evaluation
python src/evaluate_single.py


---

## 🌐 Interactive App

Run the Streamlit app:
streamlit run app/app.py


Then open the local URL (usually http://localhost:8501).

### Features:
- Input house characteristics manually
- Instant price prediction
- Uses tuned Gradient Boosting model

---

## 📂 Project Structure
house-price-prediction
│
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
├── app/
├── submissions/
├── requirements.txt
└── README.md


---

## 📊 Dataset

Ames Housing Dataset  
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

---

## 💡 Key Takeaways

- Tree-based models outperform linear models on tabular data
- Cross-validation provides more reliable evaluation than single splits
- Feature importance helps interpret model decisions
- Hyperparameter tuning improves model performance
- ML models can be turned into real applications using Streamlit

---

## 👨‍💻 Author

Metehan
