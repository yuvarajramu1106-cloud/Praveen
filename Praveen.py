# skillgap_analyzer.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------
# 🎨 Page Configuration
# ---------------------------------
st.set_page_config(page_title="Skill Gap Analyzer", layout="wide")
st.markdown("<h1 style='text-align:center; color:#00C9A7;'>💼 Skill Gap Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Identify and bridge your missing skills for your desired career goals</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------
# 📂 Dataset Section
# ---------------------------------
col1, col2 = st.columns([2, 1])

with col2:
    st.header("📥 Data Input")
    uploaded_file = st.file_uploader("Upload your student skill dataset", type=["csv", "xlsx"])
    use_sample = st.checkbox("Use sample dataset", value=False)

def create_sample_data():
    data = {
        "Year_of_Study": ["2nd", "3rd", "4th", "2nd", "3rd"],
        "Degree_Branch": ["CSE", "ECE", "AIML", "EEE", "CIVIL"],
        "Python_Skill(1-5)": [4, 3, 5, 2, 1],
        "Java_Skill(1-5)": [3, 4, 2, 3, 2],
        "C_C++_Skill(1-5)": [4, 2, 5, 1, 2],
        "SQL_Skill(1-5)": [3, 3, 4, 2, 1],
        "WebDev_Skill(1-5)": [4, 2, 5, 3, 2],
        "Communication_Skill(1-5)": [4, 5, 3, 4, 2],
        "ProblemSolving_Skill(1-5)": [5, 4, 4, 3, 2],
        "Leadership_Skill(1-5)": [3, 4, 3, 2, 1],
        "Teamwork_Skill(1-5)": [5, 4, 3, 4, 3],
        "Completed_Courses": [3, 5, 6, 2, 1],
        "Career_Goal": ["Software Engineer", "Data Scientist", "AI Engineer", "Developer", "Civil Engineer"],
        "Industry_Interest": ["IT", "AI", "AI", "Software", "Construction"],
        "Learning_Hours_per_Week": [10, 12, 8, 5, 6],
        "Learning_Method": ["Online", "Offline", "Online", "Hybrid", "Online"],
        "Last_Training": ["Python", "ML", "DL", "Java", "AutoCAD"],
        "Desired Role": ["Backend Developer", "ML Engineer", "AI Developer", "Frontend Dev", "Design Engineer"],
        "Missing_Skills": ["Cloud", "Deep Learning", "NLP", "Frontend", "Project Management"],
        "Confidence_Level(1-10)": [8, 7, 9, 6, 5],
        "Challenges": ["Time management", "Lack of resources", "Practical exposure", "Motivation", "Guidance"],
        "Need_Recommendations": ["Yes", "Yes", "No", "Yes", "Yes"]
    }
    return pd.DataFrame(data)

# Load dataset
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
elif use_sample:
    df = create_sample_data()
else:
    st.warning("⚠️ Please upload a dataset or use the sample dataset.")
    st.stop()

# Drop Name column if it exists
if "Name" in df.columns:
    df = df.drop(columns=["Name"])

with col1:
    st.header("📊 Dataset Preview")
    st.dataframe(df.head())

st.markdown("---")

# ---------------------------------
# 🧠 Model Training Section
# ---------------------------------
st.header("🧩 Model Training and Evaluation")

# Define target and features
target_col = "Missing_Skills"
feature_cols = [col for col in df.columns if col != target_col]

X = df[feature_cols]
y = df[target_col]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Identify numeric and categorical features
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

# Train model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluation
col3, col4 = st.columns(2)
with col3:
    st.subheader("📈 Model Accuracy")
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
with col4:
    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

st.markdown("---")

# ---------------------------------
# 🔮 Skill Gap Prediction
# ---------------------------------
st.header("🔮 Skill Gap Prediction")

col5, col6 = st.columns(2)
with col5:
    year = st.selectbox("Year of Study", sorted(df["Year_of_Study"].unique()))
    branch = st.selectbox("Degree Branch", sorted(df["Degree_Branch"].unique()))
    goal = st.selectbox("Career Goal", sorted(df["Career_Goal"].unique()))
    interest = st.selectbox("Industry Interest", sorted(df["Industry_Interest"].unique()))
    method = st.selectbox("Learning Method", sorted(df["Learning_Method"].unique()))
    training = st.selectbox("Last Training", sorted(df["Last_Training"].unique()))
    desired_role = st.selectbox("Desired Role", sorted(df["Desired Role"].unique()))

with col6:
    py = st.slider("Python Skill (1-5)", 1, 5, 3)
    java = st.slider("Java Skill (1-5)", 1, 5, 3)
    cpp = st.slider("C/C++ Skill (1-5)", 1, 5, 3)
    sql = st.slider("SQL Skill (1-5)", 1, 5, 3)
    web = st.slider("WebDev Skill (1-5)", 1, 5, 3)
    comm = st.slider("Communication Skill (1-5)", 1, 5, 3)
    prob = st.slider("Problem Solving Skill (1-5)", 1, 5, 3)
    lead = st.slider("Leadership Skill (1-5)", 1, 5, 3)
    team = st.slider("Teamwork Skill (1-5)", 1, 5, 3)
    confidence = st.slider("Confidence Level (1-10)", 1, 10, 7)
    completed_courses = st.number_input("Completed Courses", min_value=0, max_value=20, value=3)
    learning_hours = st.number_input("Learning Hours per Week", min_value=0, max_value=50, value=8)
    challenges = st.text_input("Challenges", "Time management")
    need_reco = st.selectbox("Need Recommendations?", ["Yes", "No"])

# Build prediction DataFrame
input_data = pd.DataFrame([{
    "Year_of_Study": year,
    "Degree_Branch": branch,
    "Python_Skill(1-5)": py,
    "Java_Skill(1-5)": java,
    "C_C++_Skill(1-5)": cpp,
    "SQL_Skill(1-5)": sql,
    "WebDev_Skill(1-5)": web,
    "Communication_Skill(1-5)": comm,
    "ProblemSolving_Skill(1-5)": prob,
    "Leadership_Skill(1-5)": lead,
    "Teamwork_Skill(1-5)": team,
    "Completed_Courses": completed_courses,
    "Career_Goal": goal,
    "Industry_Interest": interest,
    "Learning_Hours_per_Week": learning_hours,
    "Learning_Method": method,
    "Last_Training": training,
    "Desired Role": desired_role,
    "Confidence_Level(1-10)": confidence,
    "Challenges": challenges,
    "Need_Recommendations": need_reco
}])

if st.button("🔍 Analyze Skill Gap"):
    try:
        # --- Robust preprocessing for prediction ---
        for col in numeric_features:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)
        for col in categorical_features:
            input_data[col] = input_data[col].astype(str).fillna("Unknown")

        # Reorder columns exactly as training
        input_data = input_data[X.columns]

        # Predict
        pred = pipeline.predict(input_data)
        predicted_skill = le.inverse_transform(pred)[0]
        st.success(f"🎯 Predicted Missing Skill: **{predicted_skill}**")
        st.info(f"💡 Suggestion: Focus on improving your *{predicted_skill}* through targeted courses, workshops, or projects.")
        st.balloons()
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
