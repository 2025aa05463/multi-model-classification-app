# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
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

st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"]) 
use_example = st.sidebar.checkbox("Use example Iris dataset", value=True)

if use_example:
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    if 'target' not in df.columns:
        df['target'] = iris.target
else:
    df = None
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

if df is None:
    st.info("Upload a CSV or select the example dataset from the sidebar.")
    st.stop()

st.write("Preview of dataset (first 5 rows):")
st.dataframe(df.head())

# select target
all_columns = list(df.columns)
target_column = st.sidebar.selectbox("Select target column", options=all_columns, index=len(all_columns)-1)

# feature columns
feature_columns = [c for c in all_columns if c != target_column]
st.sidebar.write(f"{len(feature_columns)} feature columns detected.")
selected_features = st.sidebar.multiselect("Select features (empty = all features)", options=feature_columns, default=feature_columns)

if len(selected_features) == 0:
    st.warning("No features selected.")
    st.stop()

X = df[selected_features]
y = df[target_column]

# detect types
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

st.sidebar.write("Preprocessing")
impute_strategy = st.sidebar.selectbox("Numeric imputation", options=["mean", "median", "most_frequent"], index=0)
scale_numeric = st.sidebar.checkbox("Scale numeric features", value=True)
onehot_encode = st.sidebar.checkbox("One-hot encode categorical features", value=True)

test_size = st.sidebar.slider("Test set proportion", 0.1, 0.5, 0.25, 0.05)
random_state = int(st.sidebar.number_input("Random state", value=42, step=1))

# Model selection
st.sidebar.header("Models")
models_to_run = []
if st.sidebar.checkbox("Logistic Regression", value=True):
    models_to_run.append("Logistic Regression")
if st.sidebar.checkbox("Decision Tree", value=True):
    models_to_run.append("Decision Tree")
if st.sidebar.checkbox("K-Nearest Neighbors", value=True):
    models_to_run.append("KNN")
if st.sidebar.checkbox("Naive Bayes", value=True):
    models_to_run.append("Naive Bayes")
if st.sidebar.checkbox("Random Forest", value=True):
    models_to_run.append("Random Forest")
if HAS_XGB and st.sidebar.checkbox("XGBoost", value=True):
    models_to_run.append("XGBoost")

if len(models_to_run) == 0:
    st.warning("Select at least one model from the sidebar.")
    st.stop()

# NB variant
nb_variant = st.sidebar.selectbox("Naive Bayes variant (if selected)", options=["GaussianNB", "MultinomialNB"], index=0)

# hyperparams examples
st.sidebar.header("Hyperparameters")
rf_n_estimators = st.sidebar.slider("RF: n_estimators", 10, 500, 100, 10)
rf_max_depth = st.sidebar.slider("RF: max_depth (0 = None)", 0, 50, 0, 1)
knn_k = st.sidebar.slider("KNN: n_neighbors", 1, 50, 5, 1)

# Train and evaluate
st.header("Train and evaluate")
if st.button("Train models"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y))>1 else None)

    # preprocessors
    numeric_transformers = []
    numeric_transformers.append(('imputer', SimpleImputer(strategy=impute_strategy)))
    if scale_numeric:
        numeric_transformers.append(('scaler', StandardScaler()))
    numeric_pipeline = Pipeline(numeric_transformers)

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')) if onehot_encode else ('passthrough', 'passthrough')
    ]) if len(categorical_cols) > 0 else None

    transformers = []
    if len(numeric_cols) > 0:
        transformers.append(('num', numeric_pipeline, numeric_cols))
    if len(categorical_cols) > 0:
        transformers.append(('cat', categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    results = {}
    for name in models_to_run:
        st.subheader(name)
        if name == 'Logistic Regression':
            model = LogisticRegression(max_iter=1000)
        elif name == 'Decision Tree':
            model = DecisionTreeClassifier(random_state=random_state)
        elif name == 'KNN':
            model = KNeighborsClassifier(n_neighbors=knn_k)
        elif name == 'Naive Bayes':
            model = GaussianNB() if nb_variant == 'GaussianNB' else MultinomialNB()
        elif name == 'Random Forest':
            md = None if rf_max_depth == 0 else int(rf_max_depth)
            model = RandomForestClassifier(n_estimators=int(rf_n_estimators), max_depth=md, random_state=random_state)
        elif name == 'XGBoost' and HAS_XGB:
            model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
        else:
            st.write(f"Model {name} not available")
            continue

        pipeline = Pipeline([('preprocessor', preprocessor), ('clf', model)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        # probabilities for AUC
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

        # AUC handling
        unique_labels = np.unique(y)
        if y_proba is not None:
            if len(unique_labels) == 2:
                # binary
                try:
                    auc = float(roc_auc_score(y_test, y_proba[:, 1]))
                except Exception:
                    auc = float('nan')
            else:
                # multiclass
                try:
                    # binarize labels
                    from sklearn.preprocessing import label_binarize
                    y_test_b = label_binarize(y_test, classes=unique_labels)
                    auc = float(roc_auc_score(y_test_b, y_proba, multi_class='ovr', average='macro'))
                except Exception:
                    auc = float('nan')
        else:
            auc = float('nan')

        metrics = {
            'accuracy': float(acc),
            'auc': float(auc) if not np.isnan(auc) else None,
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'mcc': float(mcc)
        }

        st.write('Metrics:')
        st.json(metrics)

        # confusion matrix
        try:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        except Exception:
            pass

        # allow download
        buf = io.BytesIO()
        joblib.dump(pipeline, buf)
        buf.seek(0)
        st.download_button(label=f"Download trained {name}", data=buf, file_name=f"{name.replace(' ','_')}_model.joblib")

        # save to model/ directory in workspace if running locally
        try:
            os.makedirs('model', exist_ok=True)
            joblib.dump(pipeline, os.path.join('model', f"{name.replace(' ','_')}_model.joblib"))
        except Exception:
            pass

        results[name] = metrics

    st.success('Training complete')
    st.write('Summary of results:')
    st.table(pd.DataFrame(results).T)
