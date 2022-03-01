from nltk.corpus import stopwords
import streamlit as st
import pickle
import nltk
import string
from nltk.stem import PorterStemmer 

st.title("SMS/Email Spam Classifier")
input_msg = st.text_area("Enter Your SMS Here")


model = pickle.load(open('model.pkl','rb'))

tdidf = pickle.load(open('vectorizer.pkl','rb'))

ps = PorterStemmer()

def text_processing(text):
    text = nltk.word_tokenize(text.lower())
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            stemming = ps.stem(i)
            y.append(stemming)
            
    return " ".join(y)

if st.button("Predict"):

    tranformed_sms = text_processing(input_msg)

    vector_sms = tdidf.transform([tranformed_sms]).toarray()

    print(vector_sms)

    result = model.predict(vector_sms)[0]

    print(result)

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

# if st.button('Predict'):
#     st.write(input_msg)