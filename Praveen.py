import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Streamlit Page Config ---
st.set_page_config(page_title="AI Participation Predictor", layout="wide")
st.title("🤖 AI Participation Tracker & Predictor")

st.markdown("""
Upload your **Participation CSV** (with columns like Username, Event name, Date of event, Level of participation, Hours invested, etc.)
and get instant insights + AI predictions on future participation!
""")

# --- CSV Upload ---
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Data Preview")
    st.dataframe(df.head())

    # --- Column Mapping ---
    st.write("### 🧩 Map Your Columns")
    name_col = st.selectbox("Select column for Participant Name", df.columns)
    event_col = st.selectbox("Select column for Event Name", df.columns)
    date_col = st.selectbox("Select column for Date of Event", df.columns)
    status_col = st.selectbox("Select column for Participation Level (High/Low/Absent)", df.columns)
    score_col = st.selectbox("Select column for Hours/Days Invested", df.columns)

    if st.button("🚀 Run Analysis + Train Model"):
        try:
            # --- Convert Participation Level to Numeric ---
            df['Status_Num'] = df[status_col].apply(lambda x: 1 if str(x).lower() in ['high', 'present', 'yes'] else 0)

            # --- Date Processing ---
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df['Month'] = df[date_col].dt.month
            df['DayOfWeek'] = df[date_col].dt.dayofweek

            # --- Encode Event Column ---
            le_event = LabelEncoder()
            df['Event_Enc'] = le_event.fit_transform(df[event_col].astype(str))

            # --- Features and Labels ---
            X = df[['Event_Enc', 'Month', 'DayOfWeek', score_col]]
            y = df['Status_Num']

            # --- Train/Test Split ---
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            # --- Train Logistic Regression ---
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            # --- Predictions ---
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ Model Trained Successfully — Accuracy: {acc*100:.2f}%")

            st.write("### 📈 Classification Report")
            st.text(classification_report(y_test, y_pred))

            # --- Predict Participation Probabilities ---
            df['Predicted_Prob'] = model.predict_proba(X)[:, 1]
            df['Predicted_Status'] = df['Predicted_Prob'].apply(
                lambda p: "Likely High Participation" if p > 0.5 else "Likely Low/Absent"
            )

            st.write("### 🔮 Predictions")
            st.dataframe(df[[name_col, event_col, score_col, date_col, 'Predicted_Prob', 'Predicted_Status']])

            # --- Visualization ---
            st.write("## 📊 Participation Probability by Participant")
            avg_prob = df.groupby(name_col)['Predicted_Prob'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=avg_prob.values, y=avg_prob.index, palette="viridis", ax=ax)
            ax.set_xlabel("Average Predicted Probability")
            ax.set_ylabel("Participant Name")
            st.pyplot(fig)

            st.success("🎉 Analysis Complete!")

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")
else:
    st.info("👆 Upload a CSV file to begin.")
