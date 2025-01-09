import pandas as pd
import numpy as np
import pickle
import streamlit as st
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
stope_words = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

import string 
punc = string.punctuation

with open('vectorizer_tfidf.pkl','rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl','rb') as f:
    model = pickle.load(f)

st.title('Email/SMS Spam Classifier')

input_sms = st.text_input('Enter the message')

# Common with all NLP Models
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stope_words and i not in punc:
            y.append(i)
    
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

if st.button('Predict'):
    tranformed_sms = transform_text(input_sms)
    print(tranformed_sms)
    vectorized_sms = tfidf.transform([tranformed_sms])
    print('Vectorized SMS Shape: \n\n',vectorized_sms.shape)

    result = model.predict(vectorized_sms)[0]

    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")