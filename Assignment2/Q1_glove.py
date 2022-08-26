import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm

from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from sklearn.metrics import accuracy_score


def load_glove_model(file):
    print("Loading Glove Model")
    glove_model = {}
    with open(file,'r',encoding='utf-8', errors='ignore') as f:
        for line in f:
      
          split_line = line.split()
          word = split_line[0]
          embedding = np.array(split_line[1:], dtype=np.float64)
          glove_model[word] = embedding

    print(f"{len(glove_model)} words loaded!")
    return glove_model

def cosine_similarity(A,B):
    return dot(A, B)/(norm(A)*norm(B))

#Glove
# model_gl = load_glove_model('/content/glove/hi-d50-glove.txt')


file = open('/content/hindi.txt', 'r')
Lines = file.readlines()

scr_lst = []
sim_lst = []
temp_lst = []

word1 = []
word2 = []
# for model in model_lst:
model = model_gl
for line in Lines:
  temp = line.strip()
  words = temp.split(',')
  #print(words)
  if words[0]!='':
    word1.append(words[0])
    word2.append(words[1])
    vector_m = model[words[0]]
    vector_s =  model[words[1]]
    scr_lst.append(words[2])
    sim = cosine_similarity(vector_m,vector_s)
    sim_lst.append(float(round(sim*10,2)))



thres_vals = [0.4,0.5,0.6,0.7,0.8]

#print("For Glove Model")

for val in thres_vals:
  true=[]
  pred=[]
  thres = val
  for i in range(len(scr_lst)):
    
    if sim_lst[i]>=thres*10:
      pred.append(1)
    else:
      pred.append(0)


  for i in range(len(scr_lst)):
    if float(scr_lst[i])>=thres*10:
      true.append(1)
    else:
      true.append(0)

  print("Threshold is: " + str(val) +  "  Accuracy Score: " + str(accuracy_score(true, pred)))
  df = pd.DataFrame()
  df['Word 1'] = word1
  df['Word 2'] = word2
  df['Similarity Score'] = sim_lst
  df['Ground Truth Score'] = scr_lst
  df['Predicted Label'] = pred

  df.to_csv("/content/" + "Q1_Glove_"+ "similarity_" + str(int(val*10)) + ".csv" , encoding='utf-8')


