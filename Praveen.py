import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports inside try-except so app doesn't crash if missing
try:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
except ModuleNotFoundError as e:
    st.error(f"Required package missing: {e}. Make sure requirements.txt includes scikit-learn.")

st.set_page_config(page_title="AI Participation Predictor", layout="wide")
st.title("🤖 AI Participation Tracker & Predictor")

uploaded_file = st.file_uploader("Upload Participation CSV", type=["csv"])

def infer_column(possibles, columns):
    for poss in possibles:
        for col in columns:
            if poss.lower() in col.lower():
                return col
    return None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    columns = df.columns.tolist()

    # Auto-inference or user selection for each necessary column
    name_col = infer_column(['Name', 'Participant'], columns) or st.selectbox("Participant Name column", columns)
    event_col = infer_column(['Event', 'Competition'], columns) or st.selectbox("Event Name column", columns)
    date_col = infer_column(['Date'], columns) or st.selectbox("Date of Event column", columns)
    status_col = infer_column(['Level of participation', 'Participation Level', 'Status'], columns) or st.selectbox("Participation Level column", columns)
    score_col = infer_column(['Hours', 'Invested'], columns) or st.selectbox("Hours Invested column", columns)

    # Confirm detected columns with the user
    st.write(f"**Detected:** Name={name_col}, Event={event_col}, Date={date_col}, Level={status_col}, Hours={score_col}")

    if st.button("Run Analysis + Train Model"):
        try:
            # Clean/encode participation: label as 1 if 'high', 'present', or 'Winner / Achiever'
            def encode_status(val):
                v = str(val).lower()
                if 'high' in v or 'present' in v or 'winner' in v or 'achiever' in v or 'award' in v:
                    return 1
                else:
                    return 0
            df['Status_Num'] = df[status_col].apply(encode_status)

            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df['Month'] = df[date_col].dt.month.fillna(0).astype(int)
            df['DayOfWeek'] = df[date_col].dt.dayofweek.fillna(0).astype(int)

            le_event = LabelEncoder()
            df['Event_Enc'] = le_event.fit_transform(df[event_col].astype(str))

            # Clean score col, fallback to 0 for missing or malformed
            df[score_col] = pd.to_numeric(df[score_col], errors='coerce').fillna(0)

            X = df[['Event_Enc', 'Month', 'DayOfWeek', score_col]]
            y = df['Status_Num']

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Model Accuracy: {acc*100:.2f}%")
            st.text(classification_report(y_test, y_pred))

            df['Predicted_Prob'] = model.predict_proba(X)[:, 1]
            df['Predicted_Status'] = df['Predicted_Prob'].apply(lambda p: "Likely High" if p > 0.5 else "Likely Low")

            st.write("### Predictions")
            st.dataframe(df[[name_col, event_col, score_col, date_col, 'Predicted_Prob', 'Predicted_Status']])

            avg_prob = df.groupby(name_col)['Predicted_Prob'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10,6))
            sns.barplot(x=avg_prob.values, y=avg_prob.index, palette="viridis", ax=ax)
            ax.set_xlabel("Avg Predicted Probability")
            ax.set_ylabel("Participant Name")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during analysis: {e}")
