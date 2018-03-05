# This script was used to determine the maximum length of input and output sequences for the CNN dataset. The values
# obtained after running the scipt are shown below:
# Max input CNN: 2412
# Max output CNN: 107

import os
import numpy as np

END_SUM = 'sum.npy'  # File ending for the summaries
END_RAW = 'raw.npy'  # File ending for the source texts

CNN_INPUT_DIR = "D:\\cnn_stories_vectorized_100_0"

directory = os.listdir(CNN_INPUT_DIR)

maximum_input = 0
maximum_output = 0

for file in directory:
    if file.endswith(END_SUM):
        np_array = np.load(os.path.join(CNN_INPUT_DIR, file))
        maximum_output = max(maximum_output, np_array.shape[0])
        print(file + "  --->  Max output=" + str(maximum_output))
    elif file.endswith(END_RAW):
        np_array = np.load(os.path.join(CNN_INPUT_DIR, file))
        maximum_input = max(maximum_input, np_array.shape[0])
        print(file + "  --->  Max input=" + str(maximum_input))
