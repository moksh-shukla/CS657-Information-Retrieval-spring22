import os
import time
import tarfile
import re
import shutil
import string
from nltk import ngrams
from collections import defaultdict
import matplotlib.pyplot as plt
from indicnlp.tokenize import indic_tokenize
wd = os.getcwd()
path_zip= os.path.join(wd,'hi.tar.xz')

#shutil.unpack_archive(path_to_zip_file, wd)
dat = 'hi/hi.txt'

import pickle
with open('/content/drive/MyDrive/PA2/vocab.pkl', 'rb') as f:
    vocab_unique = pickle.load(f)
vocab_unique = vocab_unique[0]

def char_n_gram(n, vocab_unique):
  top_100 = defaultdict()
  frequency = defaultdict()
  cnt = 0
  for word in vocab_unique.keys():
    if not word.strip() or len(word) < n-1:
      continue
    if cnt%10000==0:
      print("Line no {}".format(cnt))
    cnt+=1 
    try:
      n_grams = list(ngrams(word,n=n))
    except:
      print("error for word {}".format(word))
      continue
    if not n_grams:
      continue
    n_gram = ["".join(k1) for k1 in n_grams]
    for grams in n_gram:
      frequency[grams] = frequency.get(grams,0) + vocab_unique[word]
      top_100[grams] = frequency[grams]
      if len(top_100.keys()) > 100:
        top_100_list = sorted(top_100.items(), key = lambda kv:kv[1])
        top_100_list.remove(top_100_list[0])
        top_100 = dict(top_100_list)
  return frequency, sorted(top_100.items(), key = lambda kv:kv[1]) 

if os.path.isfile('/content/drive/MyDrive/PA2/char_n_grams.pkl'):
  with open('/content/drive/MyDrive/PA2/char_n_grams.pkl', 'rb') as f:
    a, unigramTop100, b, bigramTop100, c, trigramTop100, d, quadgramTop100 = pickle.load(f)
else:
  charUniFrequency, unigramTop100 = char_n_gram(1, vocab_unique)
  charBiFrequency, bigramTop100 = char_n_gram(2, vocab_unique)
  charTriFrequency, trigramTop100 = char_n_gram(3, vocab_unique)
  charQuadFrequency, quadgramTop100 = char_n_gram(4, vocab_unique)

def syllabii_gen(inputText):
  vowels = '\u0904-\u0914\u0960-\u0961\u0972-\u0977' 
  consonants = '\u0915-\u0939\u0958-\u095F\u0978-\u097C\u097E-\u097F' 
  glottal = '\u097D' 
  vowelSig = '\u093E-\u094C\u093A-\u093B\u094E-\u094F\u0955-\u0957\u1CF8-\u1CF9' 
  nasals = '\u0900-\u0902\u1CF2-\u1CF6' 
  visarga = '\u0903' 
  nukta = '\u093C' 
  avagraha = '\u093D' 
  virama = '\u094D' 
  vedic_signs = '\u0951-\u0952\u1CD0-\u1CE1\u1CED\u094D' 
  visarga_modifiers = '\u1CE2-\u1CE8' 
  combining = '\uA8E0-\uA8F1' 
  om = '\u0950' 
  accents = '\u0953-\u0954' 
  dandas = '\u0964-\u0965' 
  digits = '\u0966-\u096F\u0030-\u0039' 
  abbreviation = '\u0970' 
  spacing = '\u0971' 
  vedic_nasals = '\uA8F2-\uA8F7\u1CE9-\u1CEC\u1CEE-\u1CF1' 
  fillers = '\uA8F8-\uA8F9' 
  caret = '\uA8FA' 
  headstroke = '\uA8FB' 
  space = '\u0020' 
  joiners = '\u200C-\u200D'
  syllables = []
  curr = '' 
#  cycle through all of the characters in the input Add a char to the curr buffer if it belongs to a class that can be used to make a syllable. Otherwise, convert it to syllables as soon as possible.
  for char in inputText: 
    if re.match('[' + vowels + avagraha + glottal + om + ']', char): 

      if curr != '': 
        syllables.append(curr) 
        curr = char 
      else: 
        curr = curr + char
    elif re.match('[' + consonants + ']', char): 
    #   If the last character in curr is not virama, output it as a syllable; otherwise, add the current consonant to curr.
      if len(curr) > 0 and curr[-1] != virama:
        syllables.append(curr) 
        curr = char 
      else: 
        curr = curr + char
    elif re.match('[' + vowelSig + visarga + vedic_signs + ']', char): 
      curr = curr + char
    elif re.match('[' + visarga_modifiers + ']', char): 
      if len(curr) > 0 and curr[-1] == visarga: 
        curr = curr + char 
        syllables.append(curr) 
        curr = '' 
      else: 
        syllables.append(curr) 
        curr = '' 
    elif re.match('[' + nasals + vedic_nasals + ']', char): 
      vowelsign = re.match('[' + vowelSig + ']$', curr) 
      if vowelsign: 
        syllables.append(curr) 
        curr = '' 
      else: 
        curr = curr + char 
        syllables.append(curr) 
        curr = '' 
    elif re.match('[' + nukta + ']', char): 
      curr = curr + char 
    elif re.match('[' + virama + ']', char): 
      curr = curr + char 
    elif re.match('[' + digits + ']', char): 
      curr = curr + char 
    elif re.match('[' + fillers + headstroke + ']', char): 
      syllables.append(char) 
    elif re.match('[' + joiners + ']', char): 
      curr = curr + char 
    else:
      pass 
    # handle remaining curr 
  if curr != '': 
    syllables.append(curr) 
    curr = '' 
  return syllables
  # return each syllable as item in a list return syllables

def syllables_n_gram(n, vocab_unique):
  top_100 = defaultdict()
  frequency = defaultdict()
  cnt = 0
  for word in vocab_unique.keys():
    if cnt%10000==0:
      print("Line no {}".format(cnt))
    cnt+=1
    syllables = syllabii_gen(word)
    if not syllables or len(syllables) < n-1:
      continue
    n_gram = ngrams(syllables,n)
    try:
      for grams in n_gram:
        frequency[grams] = frequency.get(grams,0) + vocab_unique[word]
        top_100[grams] = frequency[grams]
        if len(top_100.keys()) > 100:
          top_100_list = sorted(top_100.items(), key = lambda kv:kv[1])
          top_100_list.remove(top_100_list[0])
          top_100 = dict(top_100_list)
    except:
      print("Some issue encountered for word {}".format(word))
      continue
  return frequency, sorted(top_100.items(), key = lambda kv:kv[1]) 

if os.path.isfile('/content/drive/MyDrive/PA2/syllables_n_grams.pkl'):
  with open('/content/drive/MyDrive/PA2/syllables_n_grams.pkl', 'rb') as f:
    a, uniSyllable_top_100, b, biSyllable_top_100, c, triSyllable_top_100, d, quadSyllable_top_100 = pickle.load(f)
else:
  emp = ""
  tmp = syllables_n_gram(1, vocab_unique)
  syll_uni_frequency, uniSyllable_top_100 = [(emp.join(i[0]), i[1]) for i in tmp[0].items()], [(emp.join(i[0]), i[1]) for i in tmp[1]]
  tmp = syllables_n_gram(2, vocab_unique)
  syll_bi_frequency, biSyllable_top_100 = [(emp.join(i[0]), i[1]) for i in tmp[0].items()], [(emp.join(i[0]), i[1]) for i in tmp[1]]
  tmp = syllables_n_gram(3, vocab_unique)
  syll_tri_frequency, triSyllable_top_100 = [(emp.join(i[0]), i[1]) for i in tmp[0].items()], [(emp.join(i[0]), i[1]) for i in tmp[1]]
  tmp = syllables_n_gram(4, vocab_unique)
  syll_quad_frequency, quadSyllable_top_100 = [(emp.join(i[0]), i[1]) for i in tmp[0].items()], [(emp.join(i[0]), i[1]) for i in tmp[1]]

def word_n_grams(n, dat):
  top_100 = defaultdict()
  frequency = defaultdict()
  cnt = 0
  prev = []
  table = str.maketrans(dict.fromkeys(string.punctuation))
  with open(dat) as myfile:
    for line in myfile:
      if cnt%10000==0:
        print("Line no {}".format(cnt))
      sentence = line.strip()
      cnt+=1
      s = sentence.translate(table)
      tokens = indic_tokenize.trivial_tokenize(s)
      if n == 3 or n == 2:
        if n == 2:
          if prev:
            tokens.insert(0,prev[-1])
        if n == 3:
          if prev:
            tokens.insert(0,prev[-1])
            if len(prev) >=2:
              tokens.insert(0,prev[-2])
        n_gram = ngrams(tokens,n)
        if len(tokens) < n-1:
          continue
      try:
        for grams in n_gram:
          if n == 1 or n==2 or n == 3:
            frequency[grams] = frequency.get(grams,0) + 1
            top_100[grams] = frequency[grams]
            if len(top_100.keys()) > 100:
              top_100_list = sorted(top_100.items(), key = lambda kv:kv[1])
              top_100_list.remove(top_100_list[0])
              top_100 = dict(top_100_list)
      except:
        print("Error im {}".format(n_gram))
        continue
      prev = tokens
  return frequency, sorted(top_100.items(), key = lambda kv:kv[1])

unigram_top_100 = sorted(vocab_unique.items(), key = lambda kv:kv[1])[-100:]

for i in range(99, -1, -1):
  print(unigramTop100[i][0])
for i in range(99, -1, -1):
  print(bigramTop100[i][0])
for i in range(99, -1, -1):
  print(trigramTop100[i][0])
for i in range(99, -1, -1):
  print(quadgramTop100[i][0])

for i in range(99, -1, -1):
  print(uniSyllable_top_100[i][0])
for i in range(99, -1, -1):
  print(biSyllable_top_100[i][0])
for i in range(99, -1, -1):
  print(triSyllable_top_100[i][0])
for i in range(99, -1, -1):
  print(quadSyllable_top_100[i][0])

for i in range(99, -1, -1):
  print(unigram_top_100[i][0])

lstComplete = [unigramTop100, bigramTop100, trigramTop100, quadgramTop100, uniSyllable_top_100, biSyllable_top_100, triSyllable_top_100, quadSyllable_top_100, unigram_top_100]
count = 0
idx = 0
for data in lstComplete:
  count = count+1
  idx = idx+1
  if count > 4:
    wch = 'syllables'
  elif count <=4:
    wch = 'characters'
  else:
    wch = 'words'
  if not isinstance(data, str):
    counts = [i[1] for i in data]
    tokens = [i[0] for i in data]
    rank = [i for i in range(101,1,-1)]
    mult = [counts[i] * rank[i] for i in range (len(counts))]
    plt.xlabel("frequency rank for the {}_gram of {}".format(idx%5,wch))
    plt.ylabel("frequency of Token for {}_gram for {}".format(idx%5, wch))
    plt.plot(rank, counts)
    plt.savefig('{}_{}_grams.png'.format(wch,idx%5))
    plt.show()