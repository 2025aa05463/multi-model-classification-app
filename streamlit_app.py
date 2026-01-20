# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import io
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Multi-model Classification App", layout="wide")
st.title("🤖 Multi-model Classification App")
st.markdown("Train and evaluate multiple classification models on Mobile Price Classification dataset")

# --- About Dataset ---
with st.expander("📖 About the Dataset", expanded=True):
    st.markdown("""
    **Mobile Price Classification Dataset**
    
    This dataset contains mobile phone specifications and their price ranges.
    - **Source**: Kaggle (iabhishekofficial/mobile-price-classification)
    - **Target**: `price_range` (0: low cost, 1: medium cost, 2: high cost, 3: very high cost)
    - **Features**: Battery power, RAM, internal memory, camera specs, etc.
    - **Train/Test Split**: Fixed 80/20 split with random_state=42
    """)

# Load fixed train/test data
@st.cache_data
def load_data():
    """Load pre-split train and test data"""
    try:
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
        return train_df, test_df
    except FileNotFoundError:
        st.error("data/train.csv or data/test.csv not found in repository!")
        st.stop()

train_df, test_df = load_data()

# Display dataset info
col1, col2 = st.columns(2)
with col1:
    st.info(f"📊 Training set: {len(train_df)} samples")
with col2:
    st.info(f"📊 Test set: {len(test_df)} samples")

# Show dataset preview
with st.expander("View Dataset Preview"):
    st.subheader("Training Data Sample")
    st.dataframe(train_df.head())
    st.subheader("Test Data Sample")
    st.dataframe(test_df.head())

# Download test.csv link
with open('data/test.csv', 'rb') as f:
    st.download_button(
        label="📥 Download test.csv",
        data=f,
        file_name="test.csv",
        mime="text/csv",
        help="Download the test dataset used for evaluation"
    )

# Prepare data
target_column = 'price_range'
X_test = test_df.drop(target_column, axis=1)
y_test = test_df[target_column]

# Load pre-trained models at startup
@st.cache_resource
def load_pretrained_models():
    """Load all pre-trained models from saved_models directory"""
    models_dir = 'model/saved_models'
    model_files = {
        'Logistic Regression': 'Logistic_Regression_model.joblib',
        'Decision Tree': 'Decision_Tree_model.joblib',
        'K-Nearest Neighbors': 'K-Nearest_Neighbors_model.joblib',
        'Naive Bayes': 'Naive_Bayes_model.joblib',
        'Random Forest': 'Random_Forest_model.joblib',
        'XGBoost': 'XGBoost_model.joblib'
    }
    
    loaded_models = {}
    missing_models = []
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            loaded_models[model_name] = joblib.load(model_path)
        else:
            missing_models.append(model_name)
    
    return loaded_models, missing_models

# Load models
with st.spinner('Loading pre-trained models...'):
    pretrained_models, missing = load_pretrained_models()

if missing:
    st.warning(f"⚠️ Missing pre-trained models: {', '.join(missing)}")
    st.info("Please run the training notebooks in the model/ folder to generate these models.")

# --- Sidebar: Model selection ---
st.sidebar.header("🎯 Model Selection")
available_models = list(pretrained_models.keys())
selected_model = st.sidebar.selectbox("Select a model to evaluate", available_models)

st.header(f"Evaluation: {selected_model}")

# Get pre-trained model
if selected_model not in pretrained_models:
    st.error(f'{selected_model} is not available. Please train it first.')
    st.stop()

model = pretrained_models[selected_model]

# Predict using pre-trained model
y_pred = model.predict(X_test)
try:
    y_proba = model.predict_proba(X_test)
except Exception:
    y_proba = None

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

# AUC
unique_labels = np.unique(y_test)
auc = None
if y_proba is not None:
    try:
        if len(unique_labels) == 2:
            auc = float(roc_auc_score(y_test, y_proba[:,1]))
        else:
            from sklearn.preprocessing import label_binarize
            y_test_b = label_binarize(y_test, classes=unique_labels)
            auc = float(roc_auc_score(y_test_b, y_proba, multi_class='ovr', average='macro'))
    except Exception:
        auc = None

# Display metrics
st.subheader('Evaluation Metrics')
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Accuracy", f"{acc:.4f}")
    st.metric("Precision", f"{prec:.4f}")
with col2:
    st.metric("Recall", f"{rec:.4f}")
    st.metric("F1 Score", f"{f1:.4f}")
with col3:
    st.metric("MCC", f"{mcc:.4f}")
    if auc is not None:
        st.metric("AUC", f"{auc:.4f}")
    else:
        st.metric("AUC", "N/A")

# Classification Report
st.subheader('Classification Report')
report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

# Confusion Matrix
st.subheader('Confusion Matrix')
cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels, 
           ax=ax, cmap='Blues')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix - {selected_model}')
st.pyplot(fig)

# Model info
with st.expander("ℹ️ Model Information"):
    st.markdown(f"""
    **Model**: {selected_model}  
    **Status**: Pre-trained  
    **Training Data**: 1600 samples  
    **Test Data**: 400 samples  
    """)
    
    # Download model button
    model_path = f'model/saved_models/{selected_model.replace(" ", "_")}_model.joblib'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            st.download_button(
                label='💾 Download trained model',
                data=f,
                file_name=f"{selected_model.replace(' ','_')}_model.joblib",
                mime='application/octet-stream'
            )

