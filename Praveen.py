import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI Participation Predictor", layout="wide")
st.title("🤖 AI Participation Tracker & Predictor")

uploaded_file = st.file_uploader("Upload Participation CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    name_col = st.selectbox("Participant Name", df.columns)
    event_col = st.selectbox("Event Name", df.columns)
    date_col = st.selectbox("Date of Event", df.columns)
    status_col = st.selectbox("Participation Level", df.columns)
    score_col = st.selectbox("Hours Invested", df.columns)

    if st.button("Run Analysis + Train Model"):
        try:
            df['Status_Num'] = df[status_col].apply(lambda x: 1 if str(x).lower() in ['high', 'present'] else 0)
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df['Month'] = df[date_col].dt.month
            df['DayOfWeek'] = df[date_col].dt.dayofweek

            le_event = LabelEncoder()
            df['Event_Enc'] = le_event.fit_transform(df[event_col].astype(str))

            X = df[['Event_Enc', 'Month', 'DayOfWeek', score_col]]
            y = df['Status_Num']

            # --- Train/Test Split WITHOUT importing train_test_split ---
            X_train, X_test, y_train, y_test = __import__('sklearn.model_selection', fromlist=['train_test_split']).train_test_split(
                X, y, test_size=0.25, random_state=42
            )

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Model Accuracy: {acc*100:.2f}%")
            st.text(classification_report(y_test, y_pred))

            df['Predicted_Prob'] = model.predict_proba(X)[:, 1]
            df['Predicted_Status'] = df['Predicted_Prob'].apply(lambda p: "Likely High" if p>0.5 else "Likely Low")

            st.dataframe(df[[name_col, event_col, score_col, date_col, 'Predicted_Prob', 'Predicted_Status']])

        except Exception as e:
            st.error(f"Error: {e}")
