# Abstractive Summarization

## Requirements:

To install the requirements run `pip install -r requirements.txt`

## Dataset

Training and Evaluation data is available here: https://drive.google.com/open?id=0B6N7tANPyVeBNmlSX19Ld2xDU1E
Please download the files and put the data in a new folder `data/`.

## Training

+ word2vec.py: Create word vectors from tokenized CNN-dailymail dataset

+ input_vectors.py: Converts the tokenized files to numpy arrays of vectors

+ train_model.py: Train a sequence to sequence model on CNN data

To reproduce the results please run `python3 script/train.py`. This will train the system for 30,000 training iterations.

## Testing

To run the testing please run `python3 script/test.py`. 
