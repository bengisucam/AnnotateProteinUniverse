import pandas as pd
import os
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')  # https://realpython.com/python-keras-text-classification/

from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.utils import to_categorical  #.np_utils import to_categorical
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import BatchNormalization, Dropout, Conv1D, Activation, Add, Flatten, Dense
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import ZeroPadding1D, Input
from tensorflow.keras.models import Model

#tf.logging.set_verbosity(tf.logging.ERROR)
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


is_load_npy = True
max_classes = 100  # None (17929) or int -> limit dataset to this many classes
is_load_model = True
epochs = 10
batch_size = 256
test_only = False
is_colab = False
#%matplotlib inline


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def read_dataframe(path):
    df = []
    csv_files_list = os.listdir(path)

    for csv_file in csv_files_list:
        with open(os.path.join(path, csv_file)) as f:
            data = pd.read_csv(f, index_col=None)
            df.append(data)
    return pd.concat(df)


def representAminoAcids():
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    numbers = [i for i in range(1, 21)]
    amino_numbers = dict(zip(amino_acids, numbers))
    return amino_numbers


def encode_amino_seq(amino_acid_seq_list, amino_dict):
    vectors = []
    for amino_seq in amino_acid_seq_list:
        seq = []
        for code in amino_seq:
            seq.append(amino_dict.get(code, 0))
        vectors.append(seq)
    return vectors


def create_one_hot_max_lenght(vectors, max_length):
    # One hot encoding of list of amino acids
    padded_vector = pad_sequences(vectors, maxlen=max_length, padding='post', truncating='post')
    return to_categorical(padded_vector)


def load_data(data_filename, data, amino_dictionary, is_load_npy=True):
    if is_load_npy and Path(data_filename).is_file():
        onehot = np.load(data_filename)
    else:
        onehot = encode_amino_seq(list(data[:, 4]), amino_dictionary)
        onehot = create_one_hot_max_lenght(onehot, max_length=100)
        np.save(data_filename, onehot)
    return onehot


if __name__ == '__main__':
    if is_colab:
        dataframe_path = "./drive/MyDrive/shared/pfam_seed_dataset"
    else:
        dataframe_path = "C:/Users/bengi/Desktop/Y2.G/CMPE549/pfam_seed_dataset"
    train_data = read_dataframe(dataframe_path + "/train")
    val_data = read_dataframe(dataframe_path + "/dev")
    test_data = read_dataframe(dataframe_path + "/test")

    val_data_np = val_data.to_numpy()
    test_data_np = test_data.to_numpy()

    print(train_data.head())
    print(train_data.columns)
    train_data_np = train_data.to_numpy()
    print(train_data_np.shape)
    print("First 10 Sequence: ")
    for i in range(10):
        print(len(train_data_np[i][4]))

    '''dataframes = {"Train": train_data, "Validation": val_data, "Test": test_data}
    for key, value in dataframes.items():
        print("%s dataset has %d sequences." % (key, len(value)))'''

    # Protein families with the most sequences
    train_data.groupby("family_id").size().sort_values(ascending=False).head(8)

    # Number of distinct protein families in the Pfam Seed Dataset
    family_list = train_data["family_id"].value_counts()[:1000].index.tolist()
    len(family_list)

    amino_dictionary = representAminoAcids()
    print(amino_dictionary)
    # vectorize the third amino acid seq
    vectorized_amino_list = encode_amino_seq([train_data_np[0][4], train_data_np[0][4]], amino_dictionary)
    print(train_data_np[2][4])
    for i in range(len(vectorized_amino_list)):
        print(vectorized_amino_list[i])

    one_hot_vectors = create_one_hot_max_lenght(vectorized_amino_list, max_length=100)
    for i in range(len(one_hot_vectors)):
        print(one_hot_vectors[i])
    print(one_hot_vectors.shape)

    # Considering top n (e.g 1000) classes based on most observations because of limited computational power.

    num_classes = len(train_data['family_accession'].value_counts())
    if max_classes is None or max_classes > num_classes:
        max_classes = num_classes
    num_classes = max_classes
    classes = train_data['family_accession'].value_counts()[:max_classes].index.tolist()
    print(len(classes))

    # TODO reduce dataset by filtering the most common 1000 classes
    # Filtering data based on considered 1000 classes.
    train_data = train_data.loc[train_data['family_accession'].isin(classes)].reset_index()
    val_data = val_data.loc[val_data['family_accession'].isin(classes)].reset_index()
    test_data = test_data.loc[test_data['family_accession'].isin(classes)].reset_index()
    train_data_np = train_data.to_numpy()
    val_data_np = val_data.to_numpy()
    test_data_np = test_data.to_numpy()

    print('Data size after considering {0} classes for each data split:'.format(max_classes))
    print('Train size :', len(train_data))
    print('Val size :', len(val_data))
    print('Test size :', len(test_data))

    train_filename = 'train_{0}.npy'.format(num_classes)
    val_filename = 'val_{0}.npy'.format(num_classes)
    test_filename = 'test_{0}.npy'.format(num_classes)

    train_onehot = load_data(data_filename=train_filename, data=train_data_np, amino_dictionary=amino_dictionary,
                             is_load_npy=is_load_npy)
    val_onehot = load_data(data_filename=val_filename, data=val_data_np, amino_dictionary=amino_dictionary,
                           is_load_npy=is_load_npy)
    test_onehot = load_data(data_filename=test_filename, data=test_data_np, amino_dictionary=amino_dictionary,
                            is_load_npy=is_load_npy)

    print(train_onehot.shape, val_onehot.shape, test_onehot.shape)

    # label/integer encoding output variable: (y)
    label_encoder = LabelEncoder()

    y_train_le = label_encoder.fit_transform(train_data['family_accession'])
    y_val_le = label_encoder.transform(val_data['family_accession'])
    y_test_le = label_encoder.transform(test_data['family_accession'])

    print(y_train_le.shape, y_val_le.shape, y_test_le.shape)

    num_classes = len(label_encoder.classes_)
    print('Total classes: ', num_classes)
    # le.classes_

    # One hot encoding of outputs
    y_train = to_categorical(y_train_le)
    y_val = to_categorical(y_val_le)
    y_test = to_categorical(y_test_le)

    print(y_train.shape, y_val.shape, y_test.shape)

    print('data loaded')

    # model
    #x_input = keras.Input(shape=(100, 21))
    max_features = 100*21
    embed_dim = 256
    lstm_out = 196#64
    #input_shape = (100, 21)
    history2 = None
    Path.mkdir(Path('model'), parents=True, exist_ok=True)
    model_filename = 'model/model_resnet_c{0}_e{1}'.format(num_classes, epochs)
    history_filename = 'model_resnet_history_c{0}_e{1}'.format(num_classes, epochs)
    if is_load_model and Path(model_filename).is_file():
        model2 = keras.model.load_model(model_filename)
    else:
        model_input = Input(shape=(train_onehot.shape[1], train_onehot.shape[2]))
        x = Conv1D(64, 1, strides=1, padding='valid', name='conv1d_1', kernel_initializer=glorot_uniform(seed=0))(
            model_input)
        x = MaxPooling1D(pool_size=2)(x)
        x1 = BatchNormalization(axis=2, name='batch_normalization_1')(x)
        x2 = Activation('relu', name='activation_1')(x1)
        x3 = BatchNormalization(axis=2, name='batch_normalization_2')(x2)
        x4 = Activation('relu', name='activation_2')(x3)
        x5 = Conv1D(128, 1, strides=1, padding='valid', name='conv1d_3', kernel_initializer=glorot_uniform(seed=0))(x4)
        x6 = BatchNormalization(axis=2, name='batch_normalization_3')(x5)
        x7 = Activation('relu', name='activation_3')(x6)
        x8 = Conv1D(128, 1, strides=1, padding='valid', name='conv1d_4', kernel_initializer=glorot_uniform(seed=0))(x7)
        x8 = Dropout(0.01, name='dropout_1')(x8)
        x8 = MaxPooling1D(pool_size=2)(x8)
        x9 = Conv1D(128, 1, strides=1, padding='valid', name='conv1d_2', kernel_initializer=glorot_uniform(seed=0))(x2)
        x9 = Dropout(0.01, name='dropout_2')(x9)
        x9 = MaxPooling1D(pool_size=2)(x9)
        x10 = Add()([x8, x9])
        x11 = Activation('relu', name='activation_4')(x10)
        x11 = Dropout(0.02, name='dropout_3')(x11)
        x12 = BatchNormalization(axis=2, name='batch_normalization_4')(x11)
        x13 = Activation('relu', name='activation_5')(x12)
        x14 = Dropout(0.01, name='dropout_4')(x13)
        x15 = Flatten(name='flatten_1')(x14)
        x16 = Dense(y_train.shape[1], name='fc' + str(y_train.shape[1]),
                    kernel_initializer=glorot_uniform(seed=0))(x15)
        output = Activation('softmax', name='activation_6')(x16)

        print(y_train)

        # Create model
        model2 = Model(inputs=model_input, outputs=output)
        model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # model.summary() # sparse_categorical_crossentropy

        print(train_onehot.shape, train_onehot.shape)
        print(val_onehot.shape, y_val.shape)
        print(model_input)
        #hist = model2.fit(train_onehot, y_train, epochs=epochs, batch_size=batch_size, validation_data=(val_onehot, y_val))

    if not test_only:
        # Early Stopping
        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)

        history2 = model2.fit(
            train_onehot, y_train,
            epochs=epochs, batch_size=batch_size,
            validation_data=(val_onehot, y_val),
            callbacks=[es]
        )

        # saving model weights.
        #model2.save_weights('model/model_bdrnn_weights_{0}.h5'.format(max_classes))
        model2.save(model_filename)
        with open(history_filename, 'wb') as f:
            pickle.dump(history2, f)

    if history2 is None:
        with open(history_filename, 'rb') as f:
            history2 = pickle.load(f)

    plot_history(history2)

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model2.evaluate(test_onehot, y_test, batch_size=batch_size)
    print("test loss, test acc:", results)
    with open('results_c{0}_e{1}'.format(num_classes, epochs), 'w') as f:
        f.write("test loss: {0}, test acc: {1}".format(results[0], results[1]))

    print('done')
