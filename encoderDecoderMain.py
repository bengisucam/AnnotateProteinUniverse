import pandas as pd
import os
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from model import Encoder, Decoder, Model
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
import numpy as np

input_dim = 21
hidden_dim = 100


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(hidden_dim, 200)
        self.fc2 = nn.Linear(200, 300)

    def forward(self, x):
        # print("enc x : ", x.shape)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        # print("enc lstm out:  ", lstm_out.shape)
        fc1_out = self.fc1(lstm_out)
        fc2_out = self.fc2(fc1_out)
        # print("enc fc1 out:  ", fc1_out.shape)
        return fc2_out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.lstm = nn.LSTM(300, 200, 1, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(200, 100)

    def forward(self, x):
        # print("dec x : ", x.shape)
        lstm_out, _ = self.lstm(x)
        # print("dec lstm out:  ", lstm_out.shape)
        fc1_out = self.fc1(self.dropout(lstm_out))
        # print("dec fc1 out:  ", fc1_out.shape)
        prediction = torch.softmax(fc1_out, dim=2)
        return prediction


class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def forward(self, x):
        z = self.Encoder(x)
        prediction = self.Decoder(z)
        return prediction


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return lstm_out


class LSTMModel(nn.Module):
    def __init__(self, LSTM):
        super(LSTMModel, self).__init__()
        self.lstm = LSTM

    def forward(self, x):
        lstm_out = self.lstm(x)
        prediction = torch.softmax(lstm_out, dim=2)
        return prediction


def plotLoss(train_losses, val_losses):
    print(train_losses)
    print(val_losses)
    x_axis = [i for i in range(len(train_losses))]
    plt.plot(x_axis, val_losses, label="val_loss")
    plt.plot(x_axis, train_losses, label="train_loss")
    # naming the x axis
    plt.xlabel('epochs')
    # naming the y axis
    plt.ylabel('loss value')
    # show a legend on the plot
    plt.legend()
    # function to show the plot
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


def train(model, data_train, data_val, labels_train, labels_val, optimizer, num_epochs, batch_num_train, batch_num_val,
          path):
    train_losses = []  # epoch losses during training
    val_losses = []

    ############## TRAIN #################
    for epoch in range(num_epochs):
        print("training epoch #", epoch)
        epoch_loss = 0.0
        train_corrects = 0.0

        i = 0
        for batch in range(batch_num_train):
            train_sequences = data_train[i:i + batch_size]
            sequence_labels = labels_train[i:i + batch_size]
            amino_sequences = encode_amino_seq(train_sequences, amino_dictionary)
            one_hot_amino_sequences = create_one_hot_max_lenght(amino_sequences, 100)

            optimizer.zero_grad()

            one_hot_amino_sequences = torch.from_numpy(one_hot_amino_sequences).to(device)

            targets = []
            for family_accession in sequence_labels:
                one_hot = [0 for i in range(100)]  # 500
                label = classes_dictionary.get(family_accession)
                one_hot[label] = 1
                targets.append(one_hot)
            targets_np = np.array(targets, dtype='int64')

            targets_np = torch.from_numpy(targets_np).to(device)

            out = model(one_hot_amino_sequences)

            _, predicted = torch.max(out.data, 1)  # prediction class label

            loss = lossFunction(input=out, target=targets_np.to(device))
            epoch_loss += loss.item()

            train_corrects += (torch.max(predicted, 1).indices == torch.max(targets_np, 1).indices).sum().item()
            if batch % 50 == 0:
                print("corrrect : ", train_corrects)

            # Backpropagation based on the loss
            loss.backward()
            optimizer.step()
            i += batch_size

        train_loss = epoch_loss / batch_num_train
        train_losses.append(train_loss)
        print('Epoch #{} Train Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, train_loss,
                                                                100 * train_corrects / len(data_train)))
        print("schedular stepping")
        scheduler.step()
        print(optimizer.param_groups[0]['lr'])

        ####################### VALIDATION ##############
        val_loss = 0.0
        val_corrects = 0.0
        j = 0
        for batch in range(batch_num_val):
            val_sequences = data_val[j:j + batch_size]
            sequence_labels = labels_val[j:j + batch_size]
            amino_sequences = encode_amino_seq(val_sequences, amino_dictionary)
            one_hot_amino_sequences = create_one_hot_max_lenght(amino_sequences, 100)

            one_hot_amino_sequences = torch.from_numpy(one_hot_amino_sequences).to(device)

            targets = []
            for family_accession in sequence_labels:
                one_hot = [0 for i in range(100)]  # 500
                label = classes_dictionary.get(family_accession)
                one_hot[label] = 1
                targets.append(one_hot)
            targets_np = np.array(targets, dtype='int64')
            targets_np = torch.from_numpy(targets_np).to(device)

            out = model(one_hot_amino_sequences)

            _, predicted = torch.max(out.data, 1)  # prediction class label

            loss = lossFunction(input=out, target=targets_np.to(device))
            val_loss += loss.item()

            val_corrects += (torch.max(predicted, 1).indices == torch.max(targets_np, 1).indices).sum().item()
            j += batch_size

        val_loss = val_loss / batch_num_val
        val_losses.append(val_loss)
        print('Epoch #{} Val Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, val_loss,
                                                              100 * val_corrects / len(data_val)))
    print('Finished Training')
    torch.save(model.state_dict(), path)
    return train_losses, val_losses


def test(model, data, labels, batch_size):
    test_corrects = 0.0
    batch_num_test = len(data) // batch_size

    model.eval()  # deactivate train specific layers
    with torch.no_grad():  # deactivate autograd engine
        i = 0
        for batch in range(batch_num_test):
            test_sequences = data[i:i + batch_size]
            sequence_labels = labels[i:i + batch_size]
            amino_sequences = encode_amino_seq(test_sequences, amino_dictionary)
            one_hot_amino_sequences = create_one_hot_max_lenght(amino_sequences, 100)  # 500
            one_hot_amino_sequences = torch.from_numpy(one_hot_amino_sequences).to(device)

            targets = []
            for family_accession in sequence_labels:
                one_hot = [0 for i in range(100)]  # 500
                label = classes_dictionary.get(family_accession)
                one_hot[label] = 1
                targets.append(one_hot)
            targets_np = np.array(targets, dtype='int64')
            targets_np = torch.from_numpy(targets_np).to(device)

            out = model(one_hot_amino_sequences)

            _, predicted = torch.max(out.data, 1)  # prediction class label
            test_corrects += (torch.max(predicted, 1).indices == torch.max(targets_np, 1).indices).sum().item()

        print('Test Acc: {:.4f}'.format(100 * test_corrects / len(data)))


if __name__ == '__main__':
    # Parameters
    dataframe_path = "C:/Users/bengi/Desktop/Y2.G/CMPE549/pfam_seed_dataset"

    train_data = read_dataframe(dataframe_path + "/train")
    classes = train_data['family_accession'].value_counts()[:100].index.tolist()
    train_data_small = train_data.loc[train_data['family_accession'].isin(classes)].reset_index()

    test_data = read_dataframe(dataframe_path + "/test")
    test_data_small = test_data.loc[test_data['family_accession'].isin(classes)].reset_index()

    val_data = read_dataframe(dataframe_path + "/dev")
    val_data_small = val_data.loc[val_data['family_accession'].isin(classes)].reset_index()

    numbers = [i for i in range(len(classes))]
    classes_dictionary = dict(zip(classes, numbers))
    print(classes_dictionary)

    # Protein families with the most sequences
    train_data_small.groupby("family_id").size().sort_values(ascending=False).head(8)

    # Number of distinct protein families in the Pfam Seed Dataset
    family_list = train_data_small["family_id"].value_counts()[:100].index.tolist()  # 500
    print(family_list)

    amino_dictionary = representAminoAcids()
    print(amino_dictionary)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.is_available())

    MODEL_PATH = os.path.join(os.getcwd() + "\model.pth")

    batch_size = 256
    epoch_num = 20
    learning_rate = 0.05

    batch_num_train = len(train_data_small) // batch_size  # how many batches are needed in each epoch for trainset
    batch_num_val = len(val_data_small) // batch_size  # how many batches are needed in each epoch for validationset

    # encoder decoder model
    encoder = Encoder()
    decoder = Decoder()
    model = Model(encoder, decoder).to(device)  # get model from model.py

    # single LSTM model
    '''lstm = LSTM()
    model = LSTMModel(lstm).to(device)'''

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    lossFunction = nn.CrossEntropyLoss()

    train_data_labels_np = train_data_small['family_accession'].to_numpy()  # .reshape(439493, 1)
    train_sequences_small_np = train_data_small['sequence'].to_numpy()  # .reshape(439493, 1)

    val_data_labels_np = val_data_small['family_accession'].to_numpy()  # .reshape(439493, 1)
    val_sequences_small_np = val_data_small['sequence'].to_numpy()  # .reshape(439493, 1)

    test_data_labels_np = test_data_small['family_accession'].to_numpy()  # .reshape(, 1)
    test_sequences_small_np = test_data_small['sequence'].to_numpy()  # .reshape(, 1)

    print(model)
    train_loss_list, val_loss_list = train(model, train_sequences_small_np, val_sequences_small_np,
                                           train_data_labels_np, val_data_labels_np,
                                           optimizer, epoch_num, batch_num_train, batch_num_val, MODEL_PATH)

    ##load the trained model
    model.load_state_dict(torch.load(os.path.join(os.getcwd() + "\model.pth")))  # load the trained model
    test_losses = test(model, test_sequences_small_np, test_data_labels_np, batch_size)

    vectorized_amino_list = encode_amino_seq([train_sequences_small_np[0], train_sequences_small_np[1]],
                                             amino_dictionary)
    plotLoss(train_loss_list, val_loss_list)
