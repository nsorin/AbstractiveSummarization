# This script is used to convert a list of vectors to tokens, using the most similar known vector at each step.

from gensim import models as gm

VECTORS_PATH = "vectors/cnn_dm_vectors_100_5_5"


def reverse_input(arr, file_name=None):
    vs = gm.KeyedVectors.load(VECTORS_PATH)
    result = ""
    print("\n")
    # prev = ""
    for v in arr:
        most_similar = vs.most_similar(positive=[v], topn=1)
        result += most_similar[0][0] + " "
    print(result)
    if file_name is not None:
        save_file = open(file_name, "w")
        save_file.write(result)
        save_file.close()

