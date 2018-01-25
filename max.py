import os
import numpy as np

# Max input CNN: 2412
# Max output CNN: 107
# Max input DM_0: 2309
# Max output DM_0: 1964
# Max input DM_1:
# Max output DM_1:
# Max input DM_2:
# Max output DM_2:
# Max input DM_3:
# Max output DM_3:


END_SUM = 'sum.npy'

CNN_INPUT_DIR = "D:\\dm_stories_vectorized_100_0"

directory = os.listdir(CNN_INPUT_DIR)

maximum_input = 0
maximum_output = 0

for file in directory:
    if file.endswith(END_SUM):
        np_array = np.load(os.path.join(CNN_INPUT_DIR, file))
        maximum_output = max(maximum_output, np_array.shape[0])
        print(file + "  --->  Max output=" + str(maximum_output))
    else:
        np_array = np.load(os.path.join(CNN_INPUT_DIR, file))
        maximum_input = max(maximum_input, np_array.shape[0])
        print(file + "  --->  Max input=" + str(maximum_input))
