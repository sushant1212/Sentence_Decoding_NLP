import torch
import time
import math
import unicodedata
import re
from nltk.stem import WordNetLemmatizer

def word_to_idx(word, target_vocab):
    for i, w in enumerate(target_vocab):
        if w == word:
            return i
    return -1

def sentence_to_matrix(sentence, input_size, embedding_matrix, target_vocab):
    words = sentence.split(" ")
    n = len(words)
    m = torch.zeros((n, input_size))
    for i, w in enumerate(words):
        m[i] = embedding_matrix[word_to_idx(w, target_vocab)]
    return m

def sentence_to_index(sentence, target_vocab):
    w = sentence.split(" ")
    l = []
    for word in w:
        l.append(word_to_idx(word, target_vocab))
    t = torch.tensor(l, dtype=torch.float32)
    return t

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = s.replace("'","")
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def preprocess(df):
    nrows = len(df)
    real_preprocess = []
    df['Content_Parsed_1'] = df['transcription']
    for row in range(0, nrows):

        # Create an empty list containing preprocessed words
        real_preprocess = []

        # Save the text and its words into an object
        text = df.loc[row]['transcription']
        text = normalizeString(text)


        df.loc[row]['Content_Parsed_1'] = text

    df['action'] = df['action'].str.lower()
    df['object'] = df['object'].str.lower()
    df['location'] = df['location'].str.lower()


def lemmatize(df):
    wordnet_lemmatizer = WordNetLemmatizer()
    # Lemmatizing the content
    nrows = len(df)
    lemmatized_text_list = []
    for row in range(0, nrows):

        # Create an empty list containing lemmatized words
        lemmatized_list = []

        # Save the text and its words into an object
        text = df.loc[row]['Content_Parsed_1']
        text_words = text.split(" ")

        # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

        # Join the list
        lemmatized_text = " ".join(lemmatized_list)

        # Append to the list containing the texts
        lemmatized_text_list.append(lemmatized_text)
    df['Content_Parsed_2'] = lemmatized_text_list

def create_target_vocab(df, df_val):
    target_vocab = []
    for row in range(0, len(df)):
        text = df.loc[row]['Content_Parsed_2']
        text_words = text.split(" ")
        for word in text_words:
            if word not in target_vocab:
                target_vocab.append(word)
    
    for row in range(0, len(df_val)):
        text = df_val.loc[row]['Content_Parsed_2']
        text_words = text.split(" ")
        for word in text_words:
            if word not in target_vocab:
                target_vocab.append(word)
    
    for row in range(0, len(df)):
        text = df.loc[row]['action']
        text_words = text.split(" ")
        for word in text_words:
            if word not in target_vocab:
                target_vocab.append(word)

    for row in range(0, len(df_val)):
        text = df_val.loc[row]['action']
        text_words = text.split(" ")
        for word in text_words:
            if word not in target_vocab:
                target_vocab.append(word)

    for row in range(0, len(df)):
        text = df.loc[row]['object']
        text_words = text.split(" ")
        for word in text_words:
            if word not in target_vocab:
                target_vocab.append(word)

    for row in range(0, len(df_val)):
        text = df_val.loc[row]['object']
        text_words = text.split(" ")
        for word in text_words:
            if word not in target_vocab:
                target_vocab.append(word)
    
    for row in range(0, len(df)):
        text = df.loc[row]['location']
        text_words = text.split(" ")
        for word in text_words:
            if word not in target_vocab:
                target_vocab.append(word)

    for row in range(0, len(df_val)):
        text = df_val.loc[row]['location']
        text_words = text.split(" ")
        for word in text_words:
            if word not in target_vocab:
                target_vocab.append(word)
    return target_vocab