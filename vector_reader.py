from gensim import models


def test_similarity(vectors, word1, word2):
    print(word1 + ' - ' + word2)
    print(vectors.similarity(word1, word2))


vectors_100 = models.KeyedVectors.load('vectors/cnn_dm_vectors_100_5_5')
vectors_200 = models.KeyedVectors.load('vectors/cnn_dm_vectors_200_5_5')
vectors_300 = models.KeyedVectors.load('vectors/cnn_dm_vectors_200_5_5')

# Test 100
# test_similarity(vectors_100, 'man', 'woman')
# test_similarity(vectors_100, 'king', 'queen')
# test_similarity(vectors_100, 'king', 'dog')
# test_similarity(vectors_100, 'man', 'drink')
# test_similarity(vectors_100, 'eat', 'drink')

# Test 200
# test_similarity(vectors_200, 'man', 'woman')
test_similarity(vectors_200, 'king', 'queen')
# test_similarity(vectors_200, 'king', 'dog')
# test_similarity(vectors_200, 'man', 'drink')
# test_similarity(vectors_200, 'eat', 'drink')

# Test 300
# test_similarity(vectors_300, 'man', 'woman')
test_similarity(vectors_300, 'king', 'queen')
# test_similarity(vectors_300, 'king', 'dog')
# test_similarity(vectors_300, 'man', 'drink')
# test_similarity(vectors_300, 'eat', 'drink')
