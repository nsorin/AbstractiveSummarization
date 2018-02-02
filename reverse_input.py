from gensim import models as gm

VECTORS_PATH = "vectors/cnn_dm_vectors_100_5_5"


def reverse_input(arr):
    vs = gm.KeyedVectors.load(VECTORS_PATH)
    result = ""
    for v in arr:
        most_similar = vs.most_similar(positive=[v], topn=1)
        result += most_similar[0][0] + " "
    print(result)

