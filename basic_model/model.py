from seq2seq.models import AttentionSeq2Seq
import os
import numpy as np
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from first_model.reverse_input import reverse_input

CNN_INPUT_DIR = "D:\\cnn_stories_vectorized_100"
BATCH_SIZE = 1
INPUT_DIM = 100
OUTPUT_DIM = 100
HIDDEN_DIM = 40
DEPTH = 1
EPOCHS = 10

# First model hidden dim 40 depth 1
# Second model hidden dim 100 depth 1
# Third model hidden dim 50 depth 2

INPUT_MAX_LENGTH = 2412
OUTPUT_MAX_LENGTH = 107

TRAIN_SIZE = 0.8

SAVE_NAME = "model.h5"


def build_model():
    model = AttentionSeq2Seq(input_dim=INPUT_DIM, input_length=INPUT_MAX_LENGTH, hidden_dim=HIDDEN_DIM,
                             output_length=OUTPUT_MAX_LENGTH, output_dim=OUTPUT_DIM, depth=DEPTH, batch_size=BATCH_SIZE)
    model.compile(loss='mse', optimizer='sgd', sample_weight_mode="temporal")
    model.summary()
    return model


def create_input_array(path, size, start=0):
    inputs = np.empty((size, INPUT_MAX_LENGTH, INPUT_DIM))
    outputs = np.empty((size, OUTPUT_MAX_LENGTH, OUTPUT_DIM))
    weights = np.empty((size, OUTPUT_MAX_LENGTH))
    directory = os.listdir(path)
    index = start * 3
    total = min(len(directory), size * 3)
    array_index = 0
    while index < total:
        i = np.load(os.path.join(path, directory[index]))
        o = np.load(os.path.join(path, directory[index + 1]))
        w = np.load(os.path.join(path, directory[index + 2]))
        inputs[array_index] = i[0]
        outputs[array_index] = o[0]
        weights[array_index] = w[0]
        index += 3
        array_index += 1
    return inputs, outputs, weights


def test_model(path, i_test, model):
    directory = os.listdir(path)
    i = i_test
    while i < len(directory):
        # print("Testing on:", test_file)
        test_file = directory[i]
        input_array = np.load(os.path.join(path, test_file))
        # reverse_input(input_array[0])
        # print("Result:")
        prediction = model.predict(input_array)
        reverse_input(prediction[0])
        i += 3


# The generator function used to feed samples to the model. Since our dataset is too big to be loaded into memory, we
# must use an iterative approach.
def generate_input(path, batch_size):
    directory = os.listdir(path)
    index = 0
    total = len(directory)
    # The generator should yield samples indefinitely
    while True:
        # Get at least one sample
        i = 1
        input_array = np.load(os.path.join(path, directory[index]))
        output_array = np.load(os.path.join(path, directory[index + 1]))
        sample_weights = np.load(os.path.join(path, directory[index + 2]))
        # If all the data has been yielded, go back to the first sample.
        index = (index + 3) % total
        # Increase size until batch_size is reached
        while i < batch_size:
            input_array = np.append(input_array, np.load(os.path.join(path, directory[index])), 0)
            output_array = np.append(output_array, np.load(os.path.join(path, directory[index + 1])), 0)
            sample_weights = np.append(sample_weights, np.load(os.path.join(path, directory[index + 2])), 0)
            index = (index + 3) % total
            i += 1
        yield input_array, output_array, sample_weights


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        m = build_model()
        m.load_weights("model1.h5")
        # m.load_weights("model2.h5")
        # m.load_weights("model3.h5")

        dataset_size = len(os.listdir(CNN_INPUT_DIR)) / 3

        checkpoint = ModelCheckpoint("model1.h5", monitor='val_loss', verbose=1, save_best_only=False,
                                     save_weights_only=True, mode='auto', period=1)
        callbacks = [checkpoint]

        m.fit_generator(generate_input(CNN_INPUT_DIR, BATCH_SIZE), (TRAIN_SIZE*dataset_size) // BATCH_SIZE, EPOCHS,
                        validation_steps=((1 - TRAIN_SIZE)*dataset_size) // BATCH_SIZE, callbacks=callbacks,
                        max_queue_size=10)
