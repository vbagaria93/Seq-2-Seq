#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
import numpy as np


# Remove non-alphabetic characters (Data Cleaning)
def clean_text(column):

    for row in column:
        row = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1',  str(row))).split()
        row = ' '.join(row)
        row = re.sub("(\\t)", " ", str(row)).lower()
        row = re.sub("(\\r)", " ", str(row)).lower()
        row = re.sub("(\\n)", " ", str(row)).lower()

    

        # Remove the characters - <>()|&©ø"',;?~*!
        row = re.sub(r"[<>()|&©ø\[\]\'\",.\}`$\{;@?~*!+=_\//1234567890]", " ", str(row)).lower()
        row = re.sub(r"\\b(\\w+)(?:\\W+\\1\\b)+", "", str(row)).lower()


#         # Replace INC nums to INC_NUM
#         row = re.sub("([iI][nN][cC]\d+)", "INC_NUM", str(row)).lower()

#         # Replace CM# and CHG# to CM_NUM
#         row = re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", "CM_NUM", str(row)).lower()

        # Remove punctuations at the end of a word
        row = re.sub("(\.\s+)", " ", str(row)).lower()
        row = re.sub("(\-\s+)", " ", str(row)).lower()
        row = re.sub("(\:\s+)", " ", str(row)).lower()

        # Replace any url to only the domain name
        try:
            url = re.search(r"((https*:\/*)([^\/\s]+))(.[^\s]+)", str(row))
            repl_url = url.group(3)
            row = re.sub(r"((https*:\/*)([^\/\s]+))(.[^\s]+)", repl_url, str(row))
        except:
            pass
        #row = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1',  str(row))).split()
        # Remove multiple spaces
        row = re.sub("(\s+)", " ", str(row)).lower()

        # Remove the single character hanging between any two spaces
        row = re.sub("(\s+.\s+)", " ", str(row)).lower()
        
        

        yield row


# In[2]:


df_code = pd.read_csv('javascript_train.csv')
df_code = df_code[:5000]


# In[3]:


df_code_p = df_code[["code","docstring"]]
#df_code_p=df_code_p[:3]


# In[4]:


print (df_code_p["docstring"][0])


# In[5]:


processed_code= clean_text(df_code_p['code'])
processed_summary = clean_text(df_code_p['docstring'])


# In[6]:


import spacy
from time import time

nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])

# Process text as batches and yield Doc objects in order
code = [str(doc) for doc in nlp.pipe(processed_code, batch_size=50)]
summary = [ str(doc)  for doc in nlp.pipe(processed_summary, batch_size=50)]

#summary = ['_START_ '+ str(doc) + ' _END_' for doc in nlp.pipe(processed_summary, batch_size=50)]


# In[7]:


#print (summary)


# In[8]:




df_code_p['code'] = code
df_code_p['docstring'] = summary
#print(df_code_p)


# In[9]:


# import matplotlib.pyplot as plt

# text_count = []
# summary_count = []


# print(df_code_p['code'])
# for sent in df_code_p['code']:
#     print (sent)
#     text_count.append(len(sent.split()))
    
# for sent in df_code_p['docstring']:
#     summary_count.append(len(sent.split()))

# graph_df = pd.DataFrame()

# graph_df['text'] = text_count
# graph_df['summary'] = summary_count

# graph_df.hist(bins = 10)
# plt.show()


# In[12]:


max_code_len = 100
max_summary_len =10


# Select the Summaries and Text which fall below max length 

import numpy as np

cleaned_code = np.array(df_code_p['code'])
cleaned_summary= np.array(df_code_p['docstring'])

short_text = []
short_summary = []

for i in range(len(cleaned_code)):
    if len(cleaned_summary[i].split()) <= max_summary_len and len(cleaned_code[i].split()) <= max_code_len:
        short_text.append(cleaned_code[i])
        short_summary.append(cleaned_summary[i])
        
post_code = pd.DataFrame({'code': short_text,'summary': short_summary})

post_code.head(100)


# In[13]:


post_code['summary'] = post_code['summary'].apply(lambda x: 'sostok ' + x         + ' eostok')

post_code.head(2)


# In[14]:


from sklearn.model_selection import train_test_split

x_train, x_validation, y_train, y_validation = train_test_split(
    np.array(post_code["code"]),
    np.array(post_code["summary"]),
    test_size=0.15,
    random_state=0,
    shuffle=True,
)


# In[15]:


# Tokenize the text to get the vocab count 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Prepare a tokenizer on training data
x_tokenizer = Tokenizer() 
x_tokenizer.fit_on_texts(list(x_train))

threshold = 5

cnt_infrequent = 0
total_cnt = 0

for key, value in x_tokenizer.word_counts.items():
    total_cnt = total_cnt + 1
    if value < threshold:
        cnt_infrequent = cnt_infrequent + 1

    
print("% of not frequent words in vocabulary: ", (cnt_infrequent / total_cnt) * 100)


# In[16]:


# Prepare a tokenizer, again -- by not considering the rare words
x_tokenizer = Tokenizer(num_words = total_cnt - cnt_infrequent) 
x_tokenizer.fit_on_texts(list(x_train))

# Convert text sequences to integer sequences 
x_train_seqs = x_tokenizer.texts_to_sequences(x_train) 
x_validation_seqs = x_tokenizer.texts_to_sequences(x_validation)
print (x_validation_seqs)
# Pad zero upto maximum length
x_train = pad_sequences(x_train_seqs,  maxlen=max_code_len, padding='post')
x_validation = pad_sequences(x_validation_seqs, maxlen=max_code_len, padding='post')

# Size of vocabulary (+1 for padding token)
x_voc = x_tokenizer.num_words + 1


print("Size of vocabulary in X = {}".format(x_voc))


# In[17]:


# Prepare a tokenizer on testing data
y_tokenizer = Tokenizer()   
y_tokenizer.fit_on_texts(list(y_train))

threshold = 5

cnt_infrequent = 0
total_cnt = 0

for key, value in y_tokenizer.word_counts.items():
    total_cnt = total_cnt + 1
    if value < threshold:
        cnt_infrequent = cnt_infrequent + 1
    
print("% of rare words in vocabulary:",(cnt_infrequent / total_cnt) * 100)

# Prepare a tokenizer, again -- by not considering the rare words
y_tokenizer = Tokenizer(num_words = total_cnt - cnt_infrequent) 
y_tokenizer.fit_on_texts(list(y_train))

# Convert text sequences to integer sequences 
y_train_seqs = y_tokenizer.texts_to_sequences(y_train) 
y_validation_seqs = y_tokenizer.texts_to_sequences(y_validation)

# Pad zero upto maximum length
y_train = pad_sequences(y_train_seqs,  maxlen=max_summary_len, padding='post')
y_validation = pad_sequences(y_validation_seqs, maxlen=max_summary_len, padding='post')

# Size of vocabulary (+1 for padding token)
y_voc = y_tokenizer.num_words + 1

print("Size of vocabulary in Y = {}".format(y_voc))


# In[18]:


ind = []

for i in range(len(y_train)):
    cnt = 0
    for j in y_train[i]:
        if j != 0:
            cnt = cnt + 1
    if cnt == 2:
        ind.append(i)

y_train = np.delete(y_train, ind, axis=0)
x_train = np.delete(x_train, ind, axis=0)


# In[19]:


ind = []
for i in range(len(y_validation)):
    cnt = 0
    for j in y_validation[i]:
        if j != 0:
            cnt = cnt + 1
    if cnt == 2:
        ind.append(i)

y_validation = np.delete(y_validation, ind, axis=0)
x_validation = np.delete(x_validation, ind, axis=0)


# In[20]:


print(len(x_train))
print(len(y_train))
print(len(x_validation))
print(len(y_validation))


# In[21]:


print((x_train[0]))
print((y_train[0]))


# In[22]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense,     Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


# In[23]:


latent_dim = 300
embedding_dim = 200

# Encoder
encoder_inputs = Input(shape=(max_code_len, ))

# Embedding layer
enc_emb = Embedding(x_voc, embedding_dim,
                    trainable=True)(encoder_inputs)

# Encoder LSTM 1
encoder_lstm1 = LSTM(latent_dim, return_sequences=True,
                     return_state=True, dropout=0.4,
                     recurrent_dropout=0.4)
(encoder_output1, state_h1, state_c1) = encoder_lstm1(enc_emb)

# Encoder LSTM 2
encoder_lstm2 = LSTM(latent_dim, return_sequences=True,
                     return_state=True, dropout=0.4,
                     recurrent_dropout=0.4)
(encoder_output2, state_h2, state_c2) = encoder_lstm2(encoder_output1)

# Encoder LSTM 3
encoder_lstm3 = LSTM(latent_dim, return_state=True,
                     return_sequences=True, dropout=0.4,
                     recurrent_dropout=0.4)
(encoder_outputs, state_h, state_c) = encoder_lstm3(encoder_output2)

# Set up the decoder, using encoder_states as the initial state
decoder_inputs = Input(shape=(None, ))

# Embedding layer
dec_emb_layer = Embedding(y_voc, embedding_dim, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

# Decoder LSTM
decoder_lstm = LSTM(latent_dim, return_sequences=True,
                    return_state=True, dropout=0.4,
                    recurrent_dropout=0.2)
(decoder_outputs, decoder_fwd_state, decoder_back_state) =     decoder_lstm(dec_emb, initial_state=[state_h, state_c])

# Dense layer
decoder_dense = TimeDistributed(Dense(y_voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()


# In[24]:


model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)


# In[25]:


history = model.fit(
    [x_train, y_train[:, :-1]],
    y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:, 1:],
    epochs=50,
    callbacks=[es],
    batch_size=128,
    validation_data=([x_validation, y_validation[:, :-1]],
                     y_validation.reshape(y_validation.shape[0], y_validation.shape[1], 1)[:
                     , 1:]),
    )


# In[26]:


from matplotlib import pyplot

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[27]:


reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index


# In[28]:


encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs,
                      state_h, state_c])

# Decoder setup

# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim, ))
decoder_state_input_c = Input(shape=(latent_dim, ))
decoder_hidden_state_input = Input(shape=(max_code_len, latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2 = dec_emb_layer(decoder_inputs)

# To predict the next word in the sequence, set the initial states to the states from the previous time step
(decoder_outputs2, state_h2, state_c2) = decoder_lstm(dec_emb2,
        initial_state=[decoder_state_input_h, decoder_state_input_c])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_outputs2)

# Final decoder model
decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input,
                      decoder_state_input_h, decoder_state_input_c],
                      [decoder_outputs2] + [state_h2, state_c2])


# In[29]:


def decode_sequence(input_seq):

    # Encode the input as state vectors.
    (e_out, e_h, e_c) = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        (output_tokens, h, c) = decoder_model.predict([target_seq]
                + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if sampled_token != 'eostok':
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find the stop word.
        if sampled_token == 'eostok' or len(decoded_sentence.split())             >= max_summary_len - 1:
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        (e_h, e_c) = (h, c)

    return decoded_sentence


# In[30]:


def seq2summary(input_seq):
    newString = ''
    for i in input_seq:
        if i != 0 and i != target_word_index['sostok'] and i             != target_word_index['eostok']:
            newString = newString + reverse_target_word_index[i] + ' '

    return newString


# To convert sequence to text
def seq2text(input_seq):
    newString = ''
    for i in input_seq:
        if i != 0:
            newString = newString + reverse_source_word_index[i] + ' '

    return newString


# In[31]:


for i in range(0, 19):
    print ('Code:', seq2text(x_train[i]))
    print ('Original summary:', seq2summary(y_train[i]))
    print ('Predicted summary:', decode_sequence(x_train[i].reshape(1,
           max_code_len)))
    print ('\n')


# In[ ]:




