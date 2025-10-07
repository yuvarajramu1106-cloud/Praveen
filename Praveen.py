# participation_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="Student Participation Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center; color: gold;'>🎓 Student Participation Dashboard 🎓</h1>", unsafe_allow_html=True)

# -----------------------
# Helper function
# -----------------------
def sanitize_df(df):
    df = df.rename(columns={c: c.strip() for c in df.columns})
    expected = ["Department", "Year", "Event", "Role", "Individual_or_Team", "Previous_Participation_Count", "Skill domain interested in"]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    df["Year"] = df["Year"].astype(str)
    df["Department"] = df["Department"].fillna("Unknown").astype(str)
    df["Event"] = df["Event"].fillna("Unknown").astype(str)
    df["Role"] = df["Role"].fillna("Participant").astype(str)
    df["Individual_or_Team"] = df["Individual_or_Team"].fillna("Team").astype(str)
    df["Skill domain interested in"] = df["Skill domain interested in"].fillna("Coding / Technical").astype(str)
    return df

# -----------------------
# Upload CSV / sample
# -----------------------
uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV/Excel", type=["csv", "xlsx"])
use_sample = st.sidebar.checkbox("Use sample data", value=False)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    df = sanitize_df(df)
elif use_sample:
    np.random.seed(42)
    n = 200
    names = [f"Student {i}" for i in range(1, 51)]
    depts = ["CSE", "ECE", "MECH", "CIVIL", "EEE", "AIDS", "AIML"]
    events = ["Symposium", "Paper presentation", "Seminar", "Quiz", "Workshop", "Sports", "Cultural event"]
    roles = ["Presenter/Speaker", "Organizer/Volunteer", "Coordinator/Leader", "Winner/Acheiver"]
    years = ["1st Year", "2nd Year", "3rd Year", "4th Year"]
    skills = ["Coding / Technical", "Creativity and arts", "Soft skills", "Management", "Sports and fitness", "Research and innovation"]
    data = {
        "Timestamp": pd.date_range("2023-01-01", periods=n, freq="7D"),
        "Full Name": np.random.choice(names, size=n),
        "Roll Number": np.random.randint(1000, 2000, size=n).astype(str),
        "Department": np.random.choice(depts, size=n),
        "Year": np.random.choice(years, size=n),
        "Event": np.random.choice(events, size=n),
        "Role": np.random.choice(roles, size=n),
        "Individual_or_Team": np.random.choice(["Individual", "Team"], size=n),
        "Previous_Participation_Count": np.random.poisson(2, size=n),
        "Skill domain interested in": np.random.choice(skills, size=n)
    }
    df = pd.DataFrame(data)
    df = sanitize_df(df)
else:
    st.warning("📥 Please upload a CSV or tick 'Use sample data'.")
    st.stop()

st.subheader("📊 Dataset Preview")
st.write(df.head())

# -----------------------
# Task type
# -----------------------
task_type = st.sidebar.selectbox("🧩 Select Task Type", ["Classification", "Regression"])

# -----------------------
# Target and feature selection
# -----------------------
target_col = st.sidebar.selectbox("🎯 Select Target Column", df.columns)
feature_cols = st.sidebar.multiselect(
    "✨ Select Feature Columns (choose at least 3)", 
    [c for c in df.columns if c != target_col],
    default=[c for c in df.columns if c != target_col][:5]
)
if len(feature_cols) < 3:
    st.warning("⚠ Please select at least 3 features.")
    st.stop()

X = df[feature_cols].copy()
y_orig = df[target_col].copy()

# -----------------------
# Classification encoding
# -----------------------
if task_type == "Classification":
    le_target = LabelEncoder()
    y = le_target.fit_transform(y_orig.astype(str))
else:
    y = y_orig.copy()

# -----------------------
# Preprocessing
# -----------------------
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# -----------------------
# Model selection
# -----------------------
if task_type == "Classification":
    model_choice = st.sidebar.selectbox("🤖 Choose Classifier", ["SVM", "Logistic Regression", "Random Forest", "Decision Tree"])
else:
    model_choice = st.sidebar.selectbox("🤖 Choose Regressor", ["Linear Regression", "SVR", "Random Forest Regressor", "Decision Tree Regressor"])

# -----------------------
# Define model pipeline
# -----------------------
def get_model_pipeline(task, choice):
    if task == "Classification":
        if choice == "SVM":
            return Pipeline([('preprocessor', preprocessor), ('classifier', SVC())])
        elif choice == "Logistic Regression":
            return Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))])
        elif choice == "Random Forest":
            return Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))])
        else:
            return Pipeline([('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced'))])
    else:
        if choice == "Linear Regression":
            return Pipeline([('preprocessor', preprocessor), ('regressor', LinearRegression())])
        elif choice == "SVR":
            return Pipeline([('preprocessor', preprocessor), ('regressor', SVR())])
        elif choice == "Random Forest Regressor":
            return Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))])
        else:
            return Pipeline([('preprocessor', preprocessor), ('regressor', DecisionTreeRegressor(random_state=42))])

# -----------------------
# Predefined dropdown lists for prediction input
# -----------------------
depts = ["CSE", "ECE", "MECH", "CIVIL", "EEE", "AIDS", "AIML"]
events = ["Symposium", "Paper presentation", "Seminar", "Quiz", "Workshop", "Sports", "Cultural event"]
roles = ["Presenter/Speaker", "Organizer/Volunteer", "Coordinator/Leader", "Winner/Acheiver"]
years = ["1st Year", "2nd Year", "3rd Year", "4th Year"]
skills = ["Coding / Technical", "Creativity and arts", "Soft skills", "Management", "Sports and fitness", "Research and innovation"]

# -----------------------
# User input for prediction
# -----------------------
st.markdown("---")
st.subheader("📝 Enter Feature Values for Prediction")
input_data = {}
for col in feature_cols:
    if col in numeric_features:
        val = st.number_input(f"{col}", value=float(df[col].mean()))
    else:
        if col.lower() == "department":
            val = st.selectbox(f"{col}", depts)
        elif col.lower() == "event":
            val = st.selectbox(f"{col}", events)
        elif col.lower() == "role":
            val = st.selectbox(f"{col}", roles)
        elif col.lower() == "year":
            val = st.selectbox(f"{col}", years)
        elif col.lower() == "skill domain interested in":
            val = st.selectbox(f"{col}", skills)
        else:
            val = st.selectbox(f"{col}", sorted(df[col].dropna().unique()))
    input_data[col] = val
input_df = pd.DataFrame([input_data])

# -----------------------
# Predict button
# -----------------------
if st.button("🔮 Predict"):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = get_model_pipeline(task_type, model_choice)
    model.fit(X_train, y_train)
    
    # Predict test set
    y_pred = model.predict(X_test)

    # -----------------------
    # Display metrics
    # -----------------------
    st.subheader("✨ Model Performance")
    if task_type == "Classification":
        acc = accuracy_score(y_test, y_pred)
        st.write(f"🎯 Accuracy: {acc:.2f}")
        st.text("📑 Classification Report:")
        test_classes = np.unique(y_test)
        test_class_names = le_target.inverse_transform(test_classes)
        st.text(classification_report(y_test, y_pred, zero_division=0, labels=test_classes, target_names=test_class_names))
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"📏 MSE: {mse:.3f}")
        st.write(f"📈 R² Score: {r2:.3f}")

    # -----------------------
    # Predict user input
    # -----------------------
    prediction = model.predict(input_df)
    if task_type == "Classification":
        pred_label = le_target.inverse_transform(prediction)[0]
        st.subheader(f"✅ Predicted {target_col}: {pred_label}")
    else:
        st.subheader(f"✅ Predicted {target_col}: {prediction[0]:.2f}")
        st.info("ℹ Regression predictions are numeric values.")

    # Balloons only after prediction
    st.balloons()
