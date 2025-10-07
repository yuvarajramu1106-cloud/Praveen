import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="AI Participation Predictor", layout="wide")
st.title("🤖 AI Participation Tracker & Predictor")

# Upload CSV
uploaded_file = st.file_uploader("Upload Participation CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📂 Preview of your data")
    st.dataframe(df.head())

    # Map columns based on your CSV structure
    st.write("### 🔑 Map Your Columns")
    name_col = st.selectbox("Select column for Participant Name", df.columns, index=df.columns.get_loc('Username'))
    event_col = st.selectbox("Select column for Event Name", df.columns, index=df.columns.get_loc('Event name'))
    date_col = st.selectbox("Select column for Date of Event", df.columns, index=df.columns.get_loc('Date of event'))
    status_col = st.selectbox("Select column for Participation Level (High/Low/Absent)", df.columns, index=df.columns.get_loc('Level of participation'))
    score_col = st.selectbox("Select column for Hours/Days Invested", df.columns, index=df.columns.get_loc('Hours /days invested in preparation'))

    if st.button("Run Analysis + Train Model"):
        try:
            # Convert Level of Participation to numeric (1=High, 0=Low/Absent)
            df['Status_Num'] = df[status_col].apply(lambda x: 1 if str(x).lower() in ['high', 'present'] else 0)

            # Convert date column
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df['Month'] = df[date_col].dt.month
            df['DayOfWeek'] = df[date_col].dt.dayofweek

            # Encode categorical columns
            le_event = LabelEncoder()
            df['Event_Enc'] = le_event.fit_transform(df[event_col].astype(str))

            # Features and labels
            X = df[['Event_Enc', 'Month', 'DayOfWeek', score_col]]
            y = df['Status_Num']

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            # Train model
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.success(f"✅ Model Trained Successfully — Accuracy: {acc*100:.2f}%")
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))

            # Predict participation probabilities for all rows
            df['Predicted_Prob'] = model.predict_proba(X)[:, 1]
            df['Predicted_Status'] = df['Predicted_Prob'].apply(lambda p: "Likely High Participation" if p > 0.5 else "Likely Low/Absent")

            st.write("### 🔮 Predictions")
            st.dataframe(df[[name_col, event_col, score_col, date_col, 'Predicted_Prob', 'Predicted_Status']])

            # --- Visualization ---
            st.write("## 📊 Insights")
            st.bar_chart(df.groupby(name_col)['Predicted_Prob'].mean())

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")
