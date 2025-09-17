import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection App")
st.write("Enter a news headline or article, and the app will predict if it's Real or Fake using Decision Tree.")

# User input
user_input = st.text_area("Paste the news content here:")

if st.button("Check News"):
    if user_input.strip():
        # Convert text into vector form
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]

        # Show result
        if prediction == 1:
            st.success("✅ This news seems **REAL**.")
        else:
            st.error("❌ This news seems **FAKE**.")
    else:
        st.warning("Please enter some text above.")
