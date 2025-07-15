import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

# Load model (pipeline or preprocessed-compatible model)
model = joblib.load('tweet_classifier.pkl')  # Replace with your actual filename

# Preprocessing function for single tweet
def clean_tweet(tweet):
    wordLemm = nltk.WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    tweet = str(tweet).lower()
    tweet = re.sub(urlPattern, ' ', tweet)
    tweet = re.sub(userPattern, ' ', tweet)
    tweet = re.sub(alphaPattern, " ", tweet)
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

    words = []
    for word in tweet.split():
        if word not in stop_words and len(word) > 1:
            word = wordLemm.lemmatize(word)
            words.append(word)
    return " ".join(words)

# Streamlit UI
st.title("Sentiment Classification App")
st.write("This app predicts the sentiment of a tweet or short message. (0 = Negative, 1 = Neutral, 2 = Positive)")

# Input box
user_input = st.text_area("Enter your tweet or sentence below:")

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        cleaned = clean_tweet(user_input)
        prediction = model.predict([cleaned])[0]
        probas = model.predict_proba([cleaned])[0]

        st.markdown(f"### 🔍 Predicted Sentiment: **{prediction}**")
        st.write("#### Class Probabilities:")
        for i, prob in enumerate(probas):
            st.write(f"- Class {i}: {prob:.2f}")
