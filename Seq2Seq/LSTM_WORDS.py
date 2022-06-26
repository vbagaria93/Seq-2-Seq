
from __future__ import print_function
import pandas as pd
import numpy as np
import string
from string import digits
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

# Building a Sequence to Sequence converter(word wise)
lines = pd.read_table('dummy_words.txt', names=[
                      'input_seq', 'output_seq'], sep='|')

lines.input_seq = lines.input_seq.apply(lambda x: x.lower())
lines.output_seq = lines.output_seq.apply(lambda x: x.lower())

exclude = set(string.punctuation)
lines.input_seq = lines.input_seq.apply(lambda x: ''.join(
    ch for ch in x if ch not in exclude))
lines.output_seq = lines.output_seq.apply(lambda x: ''.join(
    ch for ch in x if ch not in exclude))

lines.output_seq = lines.output_seq.apply(lambda x: 'START_ ' + x + ' _END')

# Create vocabulary of all input words and all output words
max_words_in_input = 0
all_input_words = set()
for input_seq in lines.input_seq:
    # Set max no of words in input sequence.
    if len(input_seq.split()) > max_words_in_input:
        max_words_in_input = len(input_seq.split())
    for word in input_seq.split():
        if word not in all_input_words:
            all_input_words.add(word)

# Set max no of words in output sequence(output: input + START_ + _END).
max_words_in_output = max_words_in_input + 2

all_output_words = set()
for output_seq in lines.output_seq:
    for word in output_seq.split():
        if word not in all_output_words:
            all_output_words.add(word)

# Get number of all unique tokens in input and output
input_words = sorted(list(all_input_words))
target_words = sorted(list(all_output_words))
num_encoder_tokens = len(all_input_words)
num_decoder_tokens = len(all_output_words)

# Make a dictionary of all unique input_words and output_words.
input_token_index = dict(
    [(word, i) for i, word in enumerate(input_words)])
target_token_index = dict(
    [(word, i) for i, word in enumerate(target_words)])

# Prepare structure(with all 0's) of encoder_input_data, decoder_input_data, decoder_target_data
encoder_input_data = np.zeros(
    (len(lines.input_seq), max_words_in_input),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(lines.output_seq), max_words_in_output),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(lines.output_seq), max_words_in_output, num_decoder_tokens),
    dtype='float32')

# Prepare encoder_input_data, decoder_input_data, decoder_target_data
for i, (input_text, target_text) in enumerate(zip(lines.input_seq, lines.output_seq)):
    for t, word in enumerate(input_text.split()):
        encoder_input_data[i, t] = input_token_index[word]
    for t, word in enumerate(target_text.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = target_token_index[word]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[word]] = 1.

# Size of each word embedding.
embedding_size = 50

# ENCODER:

# Input Layer : Takes the Input Sequence and pass it to the embedding layer.
# This returns a tensor
encoder_inputs = Input(shape=(None,))

# Embedding Layer : Takes the Input sequemce and convert each word to fixed size vector(embedding size)
en_x = Embedding(input_dim=num_encoder_tokens,
                 output_dim=embedding_size)(encoder_inputs)

encoder = LSTM(50, return_state=True)
encoder_outputs, state_h, state_c = encoder(en_x)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# DECODER:

# Input Layer : Takes the Output Sequence and pass it to the embedding layer.
decoder_inputs = Input(shape=(None,))

# Hidden layer of the decoder :

# Embedding Layer : Takes the Output sequemce and convert each word to fixed  size vector(embedding size)
dex = Embedding(num_decoder_tokens, embedding_size)

final_dex = dex(decoder_inputs)

# LSTM Layer : Every time step, it takes a vector that represents a word and pass its output to the next layer
decoder_lstm = LSTM(50, return_sequences=True, return_state=True)

# Set up the decoder, using `encoder_states` as initial state.
decoder_outputs, _, _ = decoder_lstm(final_dex,
                                     initial_state=encoder_states)

# Output Layer : Takes the output from the previous layer and outputs a one hot vector representing the target output word
decoder_dense = Dense(num_decoder_tokens, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)

# Bringing the encoder and decoder together into one model :

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['acc'])
model.summary()

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=10,
          epochs=500,
          validation_split=0.05)

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models

encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.summary()
decoder_state_input_h = Input(shape=(50,))
decoder_state_input_c = Input(shape=(50,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

final_dex2 = dex(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    final_dex2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
                len(decoded_sentence) > 52):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(10):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', lines.input_seq[seq_index: seq_index + 1])
    print('Decoded sentence:', decoded_sentence)