import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.corpus import stopwords
stopWord = set(stopwords.words('english'))
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()

#remove_stopwords
def removeStop(txt, stopWord):
    words = [word for word in txt.split() if word.lower() not in stopWord]
    txtNew = " ".join(words)
    return str(txtNew)

def cleanStub(txt):
    regExp = re.compile('<.*?>')
    txtClean = re.sub(regExp, ' ', txt)
    txtClean = re.sub('[^A-Za-z_äöüÄÖÜùûüÿàâæéèêëïîôœÙÛÜŸÀÂÆÉÈÊËÏÎÔŒ]+',' ', txtClean)
    return txtClean

def wordTokenize(txt):
    return word_tokenize(txt)

def sentenceTokenize(txt):
    return sent_tokenize(txt)

def stem_text(txt):
    txt = cleanStub(txt)
    word_tok = wordTokenize(txt)
    word_tok = [porter.stem(w) for w in word_tok]
    txt_join = " ".join(word_tok)
    return txt_join

def word_lemmatize(txt):
    return [lemmatizer.lemmatize(w) for w in txt]

def clean_data(data):
    data = cleanStub(data)
    data = removeStop(data,stopWord)
    word_tokens = wordTokenize(data)
    lemmatize_word_token = word_lemmatize(word_tokens)
    return lemmatize_word_token