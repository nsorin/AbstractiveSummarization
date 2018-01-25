import os
import numpy as np
from keras.preprocessing import sequence


class SequenceFeeder:

    def __init__(self, path, end, max_length=None):
        self.path = path
        self.end = end
        self.max_length = max_length
        self.directory = os.listdir(self.path)

    def __iter__(self):
        for file in self.directory:
            if file.endswith(self.end):
                np_array = np.load(os.path.join(self.path, file))
                yield sequence.pad_sequences(np_array, self.max_length, np_array.dtype)
