# Abstractive Summarization

## Requirements:
+ tensorflow or tensorflow-gpu

+ Keras 2.1.0 (Newer versions do not work)

+ recurrentshop: https://github.com/farizrahman4u/recurrentshop

+ seq2seq: https://github.com/farizrahman4u/seq2seq

+ The preprocessed CNN-Dailymail dataset: https://github.com/abisee/cnn-dailymail

+ gensim to generate vectors https://radimrehurek.com/gensim/index.html

## Content

+ word2vec.py: Create word vectors from tokenized CNN-dailymail dataset

+ input_vectors.py: Converts the tokenized files to numpy arrays of vectors

+ train_model.py: Train a sequence to sequence model on CNN data
