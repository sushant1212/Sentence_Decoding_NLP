import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import bcolz
import unicodedata
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import matplotlib.ticker as ticker
import numpy as np
import random
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from utils import (
    word_to_idx,
    sentence_to_matrix,
    sentence_to_index,
    asMinutes,
    timeSince,
    unicodeToAscii,
    normalizeString,
    preprocess,
    lemmatize,
    create_target_vocab,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, x, hidden):
        x = x.unsqueeze(0)
        output, hidden = self.gru(x, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        output = F.relu(x)
        output, hidden = self.gru(output, hidden)
        output_softmax = self.softmax(self.out(output[0]))
        return output, hidden, output_softmax

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
):

    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei].unsqueeze(0), encoder_hidden
        )

    decoder_input = torch.ones((1, 1, 50))

    decoder_hidden = encoder_hidden

    for i in range(target_tensor.shape[0]):
        decoder_input, decoder_hidden, decoder_output_softmax = decoder(
            decoder_input, decoder_hidden
        )
        loss += criterion(decoder_output_softmax, target_tensor[i].unsqueeze(0).long())

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(
    encoder,
    decoder,
    n_iters,
    df,
    learning_rate,
    embedding_matrix,
    target_vocab,
    input_size,
    print_every=1000,
):
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    nrows = len(df)

    for iter in range(1, n_iters + 1):
        i = random.randint(0, n_iters)
        i = i % nrows

        s = df.loc[i]["Content_Parsed_2"]

        input_tensor = sentence_to_matrix(s, input_size, embedding_matrix, target_vocab)

        output_sentence = (
            df.loc[i]["action"]
            + " "
            + df.loc[i]["object"]
            + " "
            + df.loc[i]["location"]
        )
        target_tensor = sentence_to_index(output_sentence, target_vocab)

        loss = train(
            input_tensor,
            target_tensor,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
        )
        writer.add_scalar("loss", loss, iter)
        writer.flush()
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(
                "%s (%d %d%%) %.4f"
                % (
                    timeSince(start, iter / n_iters),
                    iter,
                    iter / n_iters * 100,
                    print_loss_avg,
                )
            )
    writer.close()

def predict(encoder, decoder, input_sentence):
    encoder_hidden = encoder.initHidden()
    input_tensor = sentence_to_matrix(input_sentence)
    decoder_input = torch.ones((1, 1, 50))
    input_length = input_tensor.size(0)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei].unsqueeze(0), encoder_hidden
        )

    decoder_hidden = encoder_hidden

    for i in range(3):
        decoder_input, decoder_hidden, decoder_output_softmax = decoder(
            decoder_input, decoder_hidden
        )
        idx = torch.argmax(decoder_output_softmax)
        print(target_vocab[idx])


def evaluate(encoder, decoder, input_sentence, target_tensor, input_size, embedding_matrix, target_vocab):
    encoder_hidden = encoder.initHidden()
    input_tensor = sentence_to_matrix(input_sentence, input_size, embedding_matrix, target_vocab)
    decoder_input = torch.ones((1, 1, 50))
    input_length = input_tensor.size(0)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei].unsqueeze(0), encoder_hidden
        )

    decoder_hidden = encoder_hidden
    correct = 0
    for i in range(3):
        decoder_input, decoder_hidden, decoder_output_softmax = decoder(
            decoder_input, decoder_hidden
        )
        idx = torch.argmax(decoder_output_softmax)
        if idx == target_tensor[i]:
            correct += 1
    if correct == 3:
        return 1
    else:
        return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="path to config file")
    args = vars(ap.parse_args())

    path_to_config = args["config"]

    with open(path_to_config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    nltk.download("wordnet")

    path_df = config["paths"]["train_path"]
    with open(path_df, "rb") as data:
        df = pd.read_csv(data)

    path_df_val = config["paths"]["val_path"]
    with open(path_df, "rb") as data:
        df_val = pd.read_csv(data)

    # Preprocessing data
    preprocess(df)
    lemmatize(df)

    preprocess(df_val)
    lemmatize(df_val)

    # Getting glove embeddings
    glove_path = config["paths"]["glove_path"]
    vectors = bcolz.open(f"{glove_path}/6B.50.dat")[:]
    words = pickle.load(open(f"{glove_path}/6B.50_words.pkl", "rb"))
    word2idx = pickle.load(open(f"{glove_path}/6B.50_idx.pkl", "rb"))
    glove = {w: vectors[word2idx[w]] for w in words}

    target_vocab = create_target_vocab(df, df_val)

    ####### CREATING EMBEDDING MATRIX #############################
    vocab_size = len(target_vocab)
    input_size = 50

    embedding_matrix = torch.zeros((vocab_size, input_size))
    for w in target_vocab:
        i = word_to_idx(w, target_vocab)

        embedding_matrix[i, :] = torch.from_numpy(glove[w]).float()
    ###############################################################

    output_size = len(target_vocab)
    input_size = 50
    hidden_size = 50

    encoder = EncoderRNN(input_size, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, output_size).to(device)

    trainIters(
        encoder,
        decoder,
        config["training"]["n_iters"],
        df,
        config["training"]["lr"],
        embedding_matrix,
        target_vocab,
        input_size,
    )

    # Evaluating the model
    n = len(df_val)
    total = 0
    correct = 0
    for i in range(n):
        output_sentence = df_val.loc[i]["action"] + " "+ df_val.loc[i]["object"] + " " + df_val.loc[i]["location"]
        target_tensor = sentence_to_index(output_sentence, target_vocab)
        
        input_sentence = df_val.loc[i]["Content_Parsed_2"]
        correct += evaluate(encoder, decoder, input_sentence, target_tensor, input_size, embedding_matrix, target_vocab)
        total += 1
    print("total correct: ", correct)
    print("total: ", total)
    print(f"Accuracy on Val test : {(float(correct)/total)*100}")
