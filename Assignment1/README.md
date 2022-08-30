## Assignment 1

### Pre-Processing
The pre-processing step is to convert the raw text into a format that can be used by the model. The code is present in the file `preprocess.py`. Preprocessing involves stopword removal, tokenization, stemming, and lemmatization. I used the NLTK library for preprocessing.

### IR System
#### Boolean Retrieval
The Boolean Retrieval model is a simple implementation of the Boolean Retrieval algorithm. The Boolean Retrieval algorithm is a simple method for searching a collection of documents for a given query.
This simple implementation of Boolean Algorithm is present in the file `boolean_retrieval.py`. It imports cleaning functions from the file `preprocess.py`, generated posting list which has been saved a pickle binary file to save the running time required to generate this posting list. In the function postCreate(), the posting list is loaded from the pickle file if `file_saved=1`, else the posting list is generated and saved to the pickle file. The current state of this variable is set to 1 as saved posting list is present in the file `posting_list.pkl` stored in folder `files`.

**Set data directory** by changing the folder link in `boolean.py` file for the variable `data_dir`.

### Ground Truth Query Generation
The query generation for ground truth labeling is performed for Q3. The file `queries.txt` contains the 20 queries for documents across the corpora in the required format with `\t` delimiter. The file `qrels.txt` contains the 10 relevant documents corresponding to each query in `queries.txt`.

### IR System
The IR system is implemented. The file `ir_run.py` takes input a list of queries as an input text file (as `queries.txt`) and outputs the top 5 documents for each query in `output.txt`.
The IR system can be run from the makefile.