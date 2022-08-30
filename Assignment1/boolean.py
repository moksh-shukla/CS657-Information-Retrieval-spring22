from preprocess import *
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
from collections import defaultdict
import pickle

data_dir = r"D:\8th Semester 2021-22\CS657\IR-A1\english-corpora"
cur_dir = os.getcwd()


def postSave(postList, vocab, idxFile, doclen, avg_doclen):
    file_name = 'postings_list.pkl'
    with open(os.path.join(cur_dir, file_name), 'wb') as f:
        pickle.dump([postList, vocab, idxFile, doclen, avg_doclen], f)

def postLoad():
    file_name = 'files/postings_list.pkl'
    with open(file_name, 'rb') as f:
        postList, vocab, idxFile, doclen, avg_doclen = pickle.load(f)
    return postList, vocab, idxFile, doclen, avg_doclen

def postCreate(file_saved = 1):
    if file_saved:
          postList, vocab, idxFile, doclen, avg_doclen = postLoad()

    else:
        postList = defaultdict()
        doclen = defaultdict()
        avg_doclen = 0
        path = data_dir
        vocab = []
        idxFile = {}
        i = 0
        for filename in os.listdir(path):
            if i%10 ==0:
                print("Doc no: {}".format(i+1))
            fil = os.path.join(path, filename)
            if os.path.isfile(fil):
                txt_corpus = (open(fil,"r")).read()
                idxFile[i] = filename
                clean_corpus = clean_data(txt_corpus)
                doclen[i] = len(clean_corpus)
                avg_doclen += len(clean_corpus)
                for tok_word in clean_corpus:
                    if tok_word in postList:
                        postList[tok_word][i] = postList[tok_word].get(i,0) + 1
                    elif tok_word not in clean_corpus:                        
                        postList[tok_word] = {}
                        postList[tok_word][i] = 1
                        vocab.append(tok_word)
            i+=1
        avg_doclen = avg_doclen/len(idxFile)
        postSave(postList, vocab, idxFile, doclen, avg_doclen)
    
    return postList, vocab, idxFile, doclen, avg_doclen

#loading/creating post list
postList, vocab, idxFile, doclen, avg_doclen = postCreate(file_saved=1)

def qWords(query):
  words = []
  clean_query = clean_data(query)	  
  for i in range(len(clean_query)):
    words.append(clean_query[i])
  return words

def oneHot(words, postList, vocab, idxFile):
  wordB = []
  totalB = []
  for word in words:
    n = len(idxFile)
    wordB = [0] * n
    if word not in vocab:
      continue
    else:
      for id in postList[word].keys():
        wordB[id] = 1
      totalB.append(wordB)
  return totalB
  
#stub function for query search  
def qSearch(query, postList, vocab, idxFile):
  words = qWords(query)
  bitQ = oneHot(words, postList, vocab, idxFile)
  connector = ['AND'] * (len(bitQ) - 1 if len(bitQ) else 0)
  if len(bitQ) == 0:
    return []
  if len(bitQ) > 1:
    for word in connector:
      word_list1 = bitQ[0]
      word_list2 = bitQ[1]
      if word == "AND":
          bit_operation = [w1 & w2 for (w1,w2) in zip(word_list1,word_list2)]
          bitQ.remove(word_list1)
          bitQ.remove(word_list2)
          bitQ.insert(0, bit_operation);
  relvDocs = []
  i = 0
  for b in bitQ[0]:
    if b:
      relvDocs.append(idxFile[i])
    i+=1
  return relvDocs[:min(len(relvDocs), 5)]