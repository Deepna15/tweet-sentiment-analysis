import pandas as pd
import streamlit as st
import re
import pickle
import spacy


def text_analyzer(my_text):
    nlp=spacy.load('en_core_web_sm')
    docx=nlp(my_text)

    tokens=[token.text for token in docx]
    allData=[('"Tokens":{},\n"Lemma":{}'.format(token.text,token.lemma_)) for token in docx]
    return allData


def clean(tweets):
    tweets=tweets.lower()
    tweets=re.sub(r'[0-9]',"",str(tweets)) 
    tweets=re.sub(r'#',"",str(tweets))
    tweets=re.sub(r"@[\w]*","",tweets) 
    tweets=re.sub(r"http\S+","",tweets) 
    tweets=re.sub(r'[^\w\s]',"",tweets)  
    return tweets


tfidf=pickle.load(open('tfidf_vectors.pkl','rb'))
model=pickle.load(open('tweet_analysis.pkl','rb'))   

st.header( 'TWEET SEMANTIC ANALYSIS')
st.write("This is a sentiment analysis app that analyses the tweet of a person and predict their sentiments")


st.sidebar.header("Select")
if st.sidebar.checkbox("Show tokens and lemma"):
    st.subheader("Tokenize your text")
    message=st.text_area("Enter text here","Type here")
    if st.button("Analyze"):
       nlp_result=text_analyzer(message)
       st.json(nlp_result)

if st.sidebar.checkbox("Analyze your tweet"):
    raw_text=st.text_area("Enter text here")
    input=[clean(raw_text)]
    if st.button('Predict'):
       x=tfidf.transform((input))
       pred=model.predict(x)
       if (int(pred)==0):
           st.write("The tweet is classified as FIGURATIVE class")
       if (int(pred)==1):
           st.write("The tweet is classified as IRONY class")
       if (int(pred)==2):
           st.write("The tweet is classified as REGULAR class")
       if (int(pred)==3):
           st.write("The tweet is classified as SARCASM class")


