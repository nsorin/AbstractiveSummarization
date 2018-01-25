import numpy as np
from gensim import models as gm
import os

CNN_INPUT_DIR = "C:\\Users\\Nicolas\\Documents\\Passau\\Text Mining Project\\PROJECT\\cnn_stories_vectorized"
VECTORS_PATH = "vectors/cnn_dm_vectors_100_5_5"

vectors = gm.KeyedVectors.load(VECTORS_PATH)

input_files = os.listdir(CNN_INPUT_DIR)

first = np.load(os.path.join(CNN_INPUT_DIR, input_files[0]))

for v in first:
    print(vectors.most_similar(positive=[v], topn=1))
