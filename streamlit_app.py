import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model_path = "spam_classifier.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

# Load pre-trained model and vectorizer
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Streamlit UI
st.title("Spam Mail Detector")
st.write("Enter an email message to check if it's spam or not.")

# User input
user_input = st.text_area("Enter email content:")

if st.button("Predict"):  
    if user_input:
        processed_text = preprocess_text(user_input)
        input_vector = vectorizer.transform([processed_text])
        prediction = model.predict(input_vector)
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        st.subheader(f"Prediction: {result}")
    else:
        st.warning("Please enter some email content to classify.")

