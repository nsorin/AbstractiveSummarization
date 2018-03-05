# Script used to translate the raw tokenized text sources to word vectors using an existing vector vocabulary. Only CNN
# data (not dailymail) is used

import gensim.models as gm
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences

# Path to the saved word vectors obtained using word2vec.py.
VECTORS_SOURCE = "vectors/cnn_dm_vectors_100_5_5"

# Path to the tokenized CNN files.
CNN_SOURCE = "D:\\cnn_stories_tokenized"

# Maximum length of the source text and reference summaries as measured by the max_length.py script.
INPUT_MAX_LENGTH = 2412
OUTPUT_MAX_LENGTH = 107

# Path to where the converted files will be saved.
CNN_OUTPUT_DIR = "D:\\cnn_stories_vectorized_100"

# Load the vectors using gensim
vectors = gm.KeyedVectors.load(VECTORS_SOURCE)

cnn_dir = os.listdir(CNN_SOURCE)
for file in cnn_dir:
    is_summary = False
    raw = []
    summary = []
    source = open(os.path.join(CNN_SOURCE, file), errors="ignore")
    for line in source:
        # In the tokenized files, every summary part is preceded by "@highlight\n"
        if line == '@highlight\n':
            is_summary = True
        # Ignore line breaks
        elif line != "\n":
            tokens = line.split(' ')
            # Select the reference to the array to fill with vectors
            to_fill = summary if is_summary else raw
            # Convert each token to a vector and add it to the corresponding array.
            for token in tokens:
                if token in vectors:
                    to_fill.append(vectors[token])
    raw_array = np.array(raw)
    sum_array = np.array(summary)
    output_len = sum_array.shape[0]

    # Sample weight used in the model to indicate which part is the actual expected output (1) and which part is the
    # result of padding (0).
    sample_weights = np.array([([1] * output_len) + ([0] * (OUTPUT_MAX_LENGTH - output_len))])

    if raw_array.shape[0] > 0 and sum_array.shape[0] > 0:
        # Reshape arrays to correspond to a batch of 1 sample
        raw_reshaped = np.reshape(raw_array, (1, raw_array.shape[0], raw_array.shape[1]))
        sum_reshaped = np.reshape(sum_array, (1, sum_array.shape[0], sum_array.shape[1]))

        # Pad the source and summaries: vectors of 0s are added so the length remains constant across samples.
        raw_padded = pad_sequences(raw_reshaped, INPUT_MAX_LENGTH, np.float32)
        sum_padded = pad_sequences(sum_reshaped, OUTPUT_MAX_LENGTH, np.float32, 'post')

        # Save the result in 3 different files: one for the source, one for the summary and one for the sample weights.
        np.save(os.path.join(CNN_OUTPUT_DIR, file + '_raw'), raw_padded)
        np.save(os.path.join(CNN_OUTPUT_DIR, file + '_sum'), sum_padded)
        np.save(os.path.join(CNN_OUTPUT_DIR, file + '_weights'), sample_weights)

print('CNN has been vectorized')
