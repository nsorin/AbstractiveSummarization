from seq2seq.models import AttentionSeq2Seq
import os
import numpy as np
from keras.models import model_from_json

CNN_INPUT_DIR = "D:\\cnn_stories_vectorized_100"
BATCH_SIZE = 1
INPUT_DIM = 100
OUTPUT_DIM = 100
HIDDEN_DIM = 40
DEPTH = 1
EPOCHS = 1

INPUT_MAX_LENGTH = 2412
OUTPUT_MAX_LENGTH = 107

TRAIN_SIZE = 0.9

LOAD_NAME = "model_0.h5"
SAVE_NAME = "model_"


def build_model():
    model = AttentionSeq2Seq(input_dim=INPUT_DIM, input_length=INPUT_MAX_LENGTH, hidden_dim=HIDDEN_DIM,
                             output_length=OUTPUT_MAX_LENGTH, output_dim=OUTPUT_DIM, depth=DEPTH, batch_size=BATCH_SIZE)
    model.compile(loss='mse', optimizer='rmsprop', sample_weight_mode="temporal")
    model.summary()
    return model


def save_model(model, file_name):
    model_json = model.to_json()
    json_name = file_name + ".json"
    h5_name = file_name + ".h5"
    with open(json_name, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(h5_name)
    print("Saved model as " + file_name)


def load_model(file_name):
    json_name = file_name + ".json"
    h5_name = file_name + ".h5"
    json_file = open(json_name, "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(h5_name)
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


def use_fit(model, size, start=0, save=SAVE_NAME):
    input_array, output_array, weights_array = create_input_array(CNN_INPUT_DIR, size, start)
    print("Input is ready!")
    model.fit(x=input_array, y=output_array, batch_size=BATCH_SIZE, epochs=EPOCHS, sample_weight=weights_array)
    model.save_weights(save)


def generate_input(path):
    directory = os.listdir(path)
    index = 0
    total = len(directory)
    while index < total:
        input_array = np.load(os.path.join(path, directory[index]))
        output_array = np.load(os.path.join(path, directory[index + 1]))
        sample_weights = np.load(os.path.join(path, directory[index + 2]))
        index += 3
        yield (input_array, output_array, sample_weights)


if __name__ == '__main__':
    m = build_model()
    m.load_weights(LOAD_NAME)
    dataset_size = TRAIN_SIZE * (len(os.listdir(CNN_INPUT_DIR)) / 3)
    m.fit_generator(generate_input(CNN_INPUT_DIR), 20000, EPOCHS)

    m.save_weights("model_20k.h5")

    # m.load_weights(LOAD_NAME)
    #
    # part = 1
    #
    # while part < 5:
    #     use_fit(m, 5000, 5000*part, SAVE_NAME + str(part) + ".h5")
    #     part += 1
