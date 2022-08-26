## Q1
The files `Q1_word2vec_fasttext.py` and `Q1_glove.py` are present in the code folder. In this the model needs to be loaded into the program and they output the accuracy of each model with corresponding threshold and exports a csv file with the results of word similarity.
The model trained with 50 iterations is used for word2vec and fasttext because of the hardware contraints.

## Q2
The file `Q2.py` is present in the code folder. This file contains the code for the Q2. For running this question 4 separate files were prepared containning the embeddings of the hindi language mapped with the enitity it represents for a particular word using the hindi ner dataset provided. These embeddings were prepared separately for the dataset using the model *AI for Bharat* provided. The provided model consists of 13 transformers  and calculates embedding corresponding to each. For me I averaged out the embeddings across all the transformers to get the final embedding.

## Q3
The file `Q3.py` is present in the code folder. This file contains the code for the Q3. The corresponding plots are present in the folder Q3_plots. First of all I prepared the unique vocabulary file.

The submission only contains the code files without the corresponding data. The complete code folder is uploaded on this link: [Complete File Link](https://iitk-my.sharepoint.com/:f:/g/personal/moksh_iitk_ac_in/EvO_0sr1SO1CvOloUJrlCc8Bo1hGapTKKivtqhyd4mFRFg?e=amPFuI)