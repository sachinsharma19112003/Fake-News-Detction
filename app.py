import streamlit as st
import pickle

# load model
model = pickle.load(open(r"C:\Users\sachi\Desktop\Fake news detecrion\fake_news_model.pkl","rb"))
vectorizer = pickle.load(open(r"C:\Users\sachi\Desktop\Fake news detecrion\vectorizer.pkl","rb"))

st.set_page_config(page_title="Fake News Detector")

st.title("📰 Fake News Detection System")

st.write("Enter news text below to check if it is Fake or Real")

news = st.text_area("Enter News Content")

if st.button("Check News"):

    if news.strip() == "":
        st.warning("Please enter news text")

    else:
        news_vec = vectorizer.transform([news])
        result = model.predict(news_vec)

        if result[0] == 1:
            st.success("✅ This News is REAL")
        else:
            st.error("❌ This News is FAKE")