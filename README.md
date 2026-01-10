# Multi-model Classification App

This repository contains a Streamlit application to train and evaluate multiple classification models.

Included files:
- streamlit_app.py : main Streamlit application
- requirements.txt : Python dependencies
- model/ : Jupyter notebooks for training each model (and they save trained .joblib models when run locally)

Models implemented:
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors Classifier
- Naive Bayes (Gaussian or Multinomial)
- Random Forest
- XGBoost

Metrics computed for each model:
- Accuracy
- AUC (ROC AUC; multiclass uses OvR macro average)
- Precision (weighted)
- Recall (weighted)
- F1 score (weighted)
- Matthews Correlation Coefficient (MCC)

How to run the app:

1. Create a virtual environment

   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows

2. Install requirements

   pip install -r requirements.txt

3. Run Streamlit

   streamlit run streamlit_app.py

Notes:
- The app supports uploading your own CSV and selecting the target column.
- The notebooks in model/ provide example training scripts for each individual model and will save model files to model/ when executed.
