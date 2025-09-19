import streamlit as st
import pandas as pd

st.set_page_config(page_title="Participation Tracker", layout="wide")
st.title("📊 Participation Tracker")

# Upload CSV
uploaded_file = st.file_uploader("Upload Participation CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📂 Preview of your data")
    st.dataframe(df.head())

    # Ask user to map columns
    st.write("### 🔑 Map Your Columns")
    name_col = st.selectbox("Select column for Student Name", df.columns)
    roll_col = st.selectbox("Select column for Roll No", df.columns)
    event_col = st.selectbox("Select column for Event", df.columns)
    date_col = st.selectbox("Select column for Date", df.columns)
    status_col = st.selectbox("Select column for Status (Present/Absent)", df.columns)
    score_col = st.selectbox("Select column for Score/Points", df.columns)

    if st.button("Analyse Participation"):
        try:
            # Total Participation per Student
            participation_summary = df.groupby(name_col)[status_col].apply(lambda x: (x == "Present").sum())

            # Average Score per Student
            avg_scores = df.groupby(name_col)[score_col].mean()

            st.success("✅ Analysis Complete")
            st.write("### Participation Summary")
            st.dataframe(participation_summary)
            st.write("### Average Scores")
            st.dataframe(avg_scores)

            # --- Charts ---
            st.write("## 📊 Insights")

            st.subheader("Participation Count per Student")
            st.bar_chart(participation_summary)

            st.subheader("Average Score per Student")
            st.bar_chart(avg_scores)

            st.subheader("Event-wise Participation")
            st.bar_chart(df.groupby(event_col)[status_col].apply(lambda x: (x == "Present").sum()))

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")