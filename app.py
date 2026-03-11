import streamlit as st
import pickle

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# -----------------------------
# Load Model and Vectorizer
# -----------------------------
@st.cache_resource
def load_model():
    with open("fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

model, vectorizer = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("📰 Fake News Detection System")

st.write(
"""
This application uses **Machine Learning and Natural Language Processing (NLP)**  
to detect whether a news article is **Real or Fake**.

Enter the news text below and click **Check News**.
"""
)

# -----------------------------
# Input Box
# -----------------------------
news = st.text_area("Enter News Content", height=200)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Check News"):

    if news.strip() == "":
        st.warning("⚠ Please enter some news text")

    else:
        news_vec = vectorizer.transform([news])
        result = model.predict(news_vec)

        if result[0] == 1:
            st.success("✅ This News is REAL")

        else:
            st.error("❌ This News is FAKE")

# -----------------------------
# Footer
# -----------------------------
st.write("---")
st.write("Developed by **Sachin Sharma** | Data Science Project")
