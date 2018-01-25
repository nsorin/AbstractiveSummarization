from seq2seq.models import AttentionSeq2Seq
import os
import numpy as np
from keras.preprocessing import sequence
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

END_RAW = 'raw.npy'
END_SUM = 'sum.npy'

LOAD_NAME = "model_1.hp5"
SAVE_NAME = "model_1.hp5"


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


def generate_input(path):
    directory = os.listdir(path)
    index = 0
    total = len(directory)
    while index < total:
        input_final = []
        output_final = []
        sample_final = []

        i_batch = 0
        while i_batch < BATCH_SIZE:
            output_final.append(np.load(os.path.join(path, directory[index + i_batch + 1])))
            input_final.append(np.load(os.path.join(path, directory[index + i_batch])))
            output_len = output_final[i_batch].shape[0]
            sample_final.append(([1] * output_len) + ([0] * (OUTPUT_MAX_LENGTH - output_len)))
            i_batch += 1

        input_padded = sequence.pad_sequences(input_final, INPUT_MAX_LENGTH, np.float32)
        output_padded = sequence.pad_sequences(output_final, OUTPUT_MAX_LENGTH, np.float32, 'post')
        sample_weights = np.array(sample_final)

        index += 2 * BATCH_SIZE

        print("yield sample")
        yield (input_padded, output_padded, sample_weights)


if __name__ == '__main__':
    m = build_model()
    # m.load_weights(LOAD_NAME)
    # Use a proportion of the data for training
    dataset_size = TRAIN_SIZE * (len(os.listdir(CNN_INPUT_DIR)) / 2)
    m.fit_generator(generate_input(CNN_INPUT_DIR), dataset_size // BATCH_SIZE, EPOCHS, max_queue_size=20, workers=1)

    m.save_weights(SAVE_NAME)
