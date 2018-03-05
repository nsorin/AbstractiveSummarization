# Basic Model

This model is a first attempt using LSTM and without beam search nor bucketing. It also contains some scripts used for preprocessing. Three different sets of hyperparameters have been tried, but the results were not usable. The weights are saved in the .h5 files.

## How to use

+ The tokenized CNN-DailyMail Dataset can be processed from here: https://github.com/abisee/cnn-dailymail.

+ The ```word2vec.py``` script can be used to generate a vocabulary of vectors from the tokenized dataset. Make sure you fix the path and use the most appropriate function (one uses a lot of memory and runs faster, the other uses less memory but runs slower).

+ The ```input_vectors.py``` script can be used to translate the tokenized dataset into vectors usable by the model. Make sur you fix the path.

+ The ```model.py``` file contains the main model. Edits might be necessary to fix the paths and choose weights to load.
