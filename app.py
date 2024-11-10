import streamlit as st
import pickle
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load the model and vectorizer
with open('bookgenremodel.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfdifvector.pkl', 'rb') as file1:
    tfidf_vectorizer = pickle.load(file1)

# Preprocessing functions
def cleantext(text):
    text = re.sub("'\''", "", text)  # removing the "\"
    text = re.sub("[^a-zA-Z]", " ", text)  # removing special symbols
    text = ' '.join(text.split())  # removing whitespaces
    return text.lower()  # convert to lowercase

def removestopwords(text):
    stop_words = set(stopwords.words('english'))
    removedstopword = [word for word in text.split() if word not in stop_words]
    return ' '.join(removedstopword)

def lemmatizing(text):
    lemma = WordNetLemmatizer()
    return ' '.join(lemma.lemmatize(word) for word in text.split())

def stemming(text):
    stemmer = PorterStemmer()
    return ' '.join(stemmer.stem(word) for word in text.split())

def preprocess(text):
    text = cleantext(text)
    text = removestopwords(text)
    text = lemmatizing(text)
    text = stemming(text)
    return text

def predict_genre(text):
    processed_text = preprocess(text)
    text_vector = tfidf_vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)
    genre_mapper = {0: 'Fantasy', 1: 'Science Fiction', 2: 'Crime Fiction', 3: 'Historical novel', 4: 'Horror', 5: 'Thriller'}
    return genre_mapper[prediction[0]]

# Streamlit UI
st.title("Book Genre Classifier")
st.write("Enter a brief description of the book to predict its genre:")

# Input area
book_description = st.text_area("Book Description", height=150)

if st.button("Predict Genre"):
    genre = predict_genre(book_description)
    st.write(f"**Predicted Genre:** {genre}")
