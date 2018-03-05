# This script generates a vocabulary of word vectors by reading the CNN/Dailymail dataset.

from gensim import models
from sentence import SentenceReader
import os


CNN_TOKENS_PATH = "C:\\Users\\Nicolas\\Desktop\\NLP Data\\cnn-dailymail-master\\cnn_stories_tokenized"
DM_TOKENS_PATH = "C:\\Users\\Nicolas\\Desktop\\NLP Data\\cnn-dailymail-master\\dm_stories_tokenized"
VECTOR_LENGTH = 100
WINDOW = 5
MIN_COUNT = 5


# Train a word2vec model with low memory usage (less efficient)
def train_low_memory():
    cnn_dm_reader = SentenceReader([CNN_TOKENS_PATH, DM_TOKENS_PATH])
    model = models.Word2Vec(cnn_dm_reader, size=VECTOR_LENGTH, window=WINDOW, min_count=MIN_COUNT)
    model.save('models/cnn_dm_model_' + str(VECTOR_LENGTH) + "_" + str(WINDOW) + "_" + str(MIN_COUNT))
    model.wv.save('vectors/cnn_dm_vectors_' + str(VECTOR_LENGTH) + "_" + str(WINDOW) + "_" + str(MIN_COUNT))


# Train a word2vec model with high memory usage (more efficient with enough RAM. Memory usage depends on dataset size.)
# For cnn-dailymail dataset and dimension of 100, uses up to 14GB of Memory.
def train_high_memory():
    cnn_files = os.listdir(CNN_TOKENS_PATH)
    dm_files = os.listdir(DM_TOKENS_PATH)

    tokens = []

    for file in cnn_files:
        file_object = open(os.path.join(CNN_TOKENS_PATH, file), errors="ignore")
        for line in file_object:
            if line and line != "\n" and line != "@highlight\n":
                tokens.append(line.split(' '))
            print("Read: CNN " + file + " read.")

    for file in dm_files:
        file_object = open(os.path.join(DM_TOKENS_PATH, file), errors="ignore")
        for line in file_object:
            if line and line != "\n" and line != "@highlight\n":
                tokens.append(line.split(' '))
            print("Read: DM " + file)

    model = models.Word2Vec(tokens, size=VECTOR_LENGTH, window=WINDOW, min_count=MIN_COUNT)
    model.wv.save('vectors/cnn_dm_vectors_' + str(VECTOR_LENGTH) + "_" + str(WINDOW) + "_" + str(MIN_COUNT))


# train_low_memory()
train_high_memory()
