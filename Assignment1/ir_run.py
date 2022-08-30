from boolean import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

#loading/creating post list
postList, vocab, idxFile, doclen, avg_doclen = postCreate(file_saved=1)
# print(qSearch("Machine Learning", postList, vocab, idxFile))
with open('queries.txt','r') as f:
    dataQuer = f.readlines()

for i in range(len(data)):
    dataQuer[i] = dataQuer[i].replace('\t','')
    dataQuer[i] = dataQuer[i].replace('\n','')
    dataQuer[i] = dataQuer[i][2:]

queryId = ['Q01', 'Q02', 'Q03', 'Q04', 'Q05', 'Q06', 'Q07', 'Q08', 'Q09', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15',
       'Q16', 'Q17', 'Q18', 'Q19', 'Q20','Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26', 'Q27', 'Q28', 'Q29', 'Q30', 
        'Q31', 'Q32', 'Q33', 'Q34', 'Q35', 'Q36', 'Q37', 'Q38', 'Q39', 'Q40']

with open("output.txt", "w") as output:
    for i in range(len(dataQuer)):
        search_res = qSearch(dataQuer[i], postList, vocab, idxFile)
        for j in range(len(search_res)):
            temp = search_res[j].split('.')
            docid = temp[0]
            output.write(queryId[i]+'\t1\t'+docid+'\t1')
            output.write('\n')