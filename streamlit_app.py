# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, classification_report, confusion_matrix
import joblib
import io
import matplotlib.pyplot as plt
import seaborn as sns
import os

# optional xgboost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

st.set_page_config(page_title="Multi-model Classification App", layout="wide")
st.title("Multi-model Classification App")

# --- Sidebar: Data ---
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload dataset (CSV) - this will be used for training and evaluation", type=['csv'])

st.sidebar.markdown("---")
# test size slider
test_size = st.sidebar.slider("Test set proportion (when splitting uploaded data)", 0.05, 0.5, 0.2, 0.05)
random_state = int(st.sidebar.number_input("Random state", value=42, step=1))

# --- Sidebar: Model selection ---
st.sidebar.header("Model selection")
model_options = ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Naive Bayes", "Random Forest"]
if HAS_XGB:
    model_options.append("XGBoost")

selected_model = st.sidebar.selectbox("Select a model to train/evaluate", model_options)

nb_variant = None
if selected_model == 'Naive Bayes':
    nb_variant = st.sidebar.selectbox("Naive Bayes variant", ["GaussianNB", "MultinomialNB"]) 

# hyperparams
st.sidebar.markdown("---")
rf_n_estimators = st.sidebar.slider("RF: n_estimators", 10, 500, 100, 10)
knn_k = st.sidebar.slider("KNN: n_neighbors", 1, 50, 5, 1)

# --- Main area ---
if uploaded is None:
    st.info("Please upload a CSV dataset in the sidebar to train and evaluate models.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read uploaded CSV: {e}")
    st.stop()

st.subheader("Uploaded dataset preview")
st.dataframe(df.head())

all_columns = list(df.columns)
# default target: last column
target_column = st.selectbox("Select target column", options=all_columns, index=len(all_columns)-1)
feature_columns = [c for c in all_columns if c != target_column]
selected_features = st.multiselect("Select feature columns (empty = all features)", options=feature_columns, default=feature_columns)

if len(selected_features) == 0:
    st.warning("No features selected.")
    st.stop()

X = df[selected_features]
y = df[target_column]

# detect types
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

preprocess_numeric = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]) if len(numeric_cols)>0 else 'passthrough'
preprocess_categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]) if len(categorical_cols)>0 else 'passthrough'
transformers = []
if len(numeric_cols)>0:
    transformers.append(('num', preprocess_numeric, numeric_cols))
if len(categorical_cols)>0:
    transformers.append(('cat', preprocess_categorical, categorical_cols))
preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

st.header("Train and evaluate")
if st.button("Train & Evaluate"):
    # split uploaded data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y))>1 else None)

    # instantiate model
    if selected_model == 'Logistic Regression':
        clf = LogisticRegression(max_iter=1000)
    elif selected_model == 'Decision Tree':
        clf = DecisionTreeClassifier(random_state=random_state)
    elif selected_model == 'K-Nearest Neighbors':
        clf = KNeighborsClassifier(n_neighbors=knn_k)
    elif selected_model == 'Naive Bayes':
        clf = GaussianNB() if nb_variant == 'GaussianNB' else MultinomialNB()
    elif selected_model == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=random_state)
    elif selected_model == 'XGBoost' and HAS_XGB:
        clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        st.error('Selected model is not available')
        st.stop()

    pipeline = Pipeline([('preprocessor', preprocessor), ('clf', clf)])
    with st.spinner('Training model...'):
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    try:
        y_proba = pipeline.predict_proba(X_test)
    except Exception:
        y_proba = None

    # metrics
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

    st.subheader('Evaluation metrics')
    metrics = {
        'accuracy': acc,
        'auc': auc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'mcc': mcc
    }
    st.json(metrics)

    st.subheader('Classification report')
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=False)
    st.text(report)

    st.subheader('Confusion matrix')
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    buf = io.BytesIO()
    joblib.dump(pipeline, buf)
    buf.seek(0)
    st.download_button('Download trained model', data=buf, file_name=f"{selected_model.replace(' ','_')}_model.joblib")

    try:
        os.makedirs('model', exist_ok=True)
        joblib.dump(pipeline, os.path.join('model', f"{selected_model.replace(' ','_')}_model.joblib"))
        st.success(f"Model saved to model/{selected_model.replace(' ','_')}_model.joblib")
    except Exception as e:
        st.warning(f"Failed to save model locally: {e}")
