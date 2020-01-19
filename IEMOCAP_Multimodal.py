#!/usr/bin/env python
# coding: utf-8

# # Speech Emotion Recognition: Audio + Text (IEMOCAP)
# ### Made by LuisFraga,
# ### January, 2020

# In[1]:


import gc
gc.collect()
get_ipython().run_line_magic('reset', '')


# In[2]:


import numpy as np
import librosa
import glob, os
import pandas as pd
import pickle # to save model after training


# ------------------------

# ## IEMOCAP Preprocessing

# ### Get Labels

# In[7]:


## Labels Pre-processing
label_dir = 'dataset/IEMOCAP/Labels'
label_list = []
for file in glob.glob(label_dir + '/*.txt'):  
    basename = os.path.basename(file)
    label_list.append(file)    
    
contents = []
for file in label_list:
    f = open(file, "r")
    for line in f:
        contents.append(line)

tmp = pd.Series(contents)
tmp = tmp.str.split(":")

file_name = []
label = []

for line in tmp:
    file_name.append(line[0])
    label.append(line[1])

## Create DataFrame
LABEL_DF = pd.DataFrame(file_name, columns = ["FILE_NAME"])
LABEL_DF['LABEL'] = label
LABEL_DF['FILE_NAME'] = LABEL_DF['FILE_NAME'].str.replace(' ','')
LABEL_DF['LABEL'] = LABEL_DF['LABEL'].apply(lambda x: x.split(';')[0])

selected_emotions = ['Anger', 'Happiness', 'Sadness', 'Neutral state', 'Other']

LABEL_DF = LABEL_DF[LABEL_DF['LABEL'].isin(selected_emotions)]
LABEL_DF = LABEL_DF.drop_duplicates(['FILE_NAME'])

LABEL_DF['LABEL'] = LABEL_DF['LABEL'].replace('Other', 'Neutral state')
print(LABEL_DF['LABEL'].value_counts())

print(LABEL_DF.shape)
LABEL_DF.head()


# ## Balance Labels

# In[8]:


from sklearn.utils import resample
anger = LABEL_DF.loc[LABEL_DF['LABEL'] == 'Anger']
sadness = LABEL_DF.loc[LABEL_DF['LABEL'] == 'Sadness']
happy = LABEL_DF.loc[LABEL_DF['LABEL'] == 'Happiness']
neutral = LABEL_DF.loc[LABEL_DF['LABEL'] == 'Neutral state']

anger = resample(anger, replace = False, n_samples = 2000)
sadness = resample(sadness, replace = True, n_samples = 2000)
happy = resample(happy, replace = False, n_samples = 2000)
neutral = resample(neutral, replace = False, n_samples = 2000)

frames = [anger, sadness, happy, neutral]

NEW_DF = pd.concat(frames)

print(NEW_DF['LABEL'].value_counts())
print(NEW_DF.shape)
NEW_DF = NEW_DF.sort_values(by=['FILE_NAME'])
NEW_DF = NEW_DF.reset_index()
NEW_DF = NEW_DF.drop(columns = ['index'])
LABEL_DF = NEW_DF
LABEL_DF.head()


# ### Get Text

# In[9]:


## Text Pre-processing
text_dir = 'dataset/IEMOCAP/Text_Files'
text_list = []
for file in glob.glob(text_dir + '/*.txt'):  
    basename = os.path.basename(file)
    text_list.append(file)       

contents = []
for file in text_list:
    f = open(file, "r")
    for line in f:
        contents.append(line)

tmp = pd.Series(contents)
tmp = tmp.str.split(":")

file_name = []
text = []

for line in tmp:
    name = line[0].split(" ")
    file_name.append(name[0])
    text.append(line[1])

## Create DataFrame
TEXT_DF = pd.DataFrame(file_name, columns = ["FILE_NAME"])
text = [t.replace('\n', '') for t in text]
TEXT_DF['TEXT'] = text

## Clean dataframe
remove_extras = ['M', 'F']
TEXT_DF = TEXT_DF[~TEXT_DF['FILE_NAME'].isin(remove_extras)]
TEXT_DF = TEXT_DF.drop_duplicates(['FILE_NAME'])

print(TEXT_DF.shape)
TEXT_DF.head()


# ### Get Audio Files

# In[10]:


## Get all wav paths
parent_dir = 'dataset/IEMOCAP/Wav_Files'
subject_dirs = [os.path.join(parent_dir, dir) for dir in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, dir))]

wav_list = []
for dir in subject_dirs:
    wav_files = [os.path.join(dir, wav) for wav in os.listdir(dir) if os.path.isfile(os.path.join(dir, wav)) and wav.endswith('.wav')]
    for file in wav_files:
        wav_list.append(file)
        
wav_list = pd.Series(wav_list)
tmp = wav_list.str.split("\\")

wav_names = []
for line in tmp:
    wav_names.append(line[2])
    
WAV_DF = pd.DataFrame(wav_list, columns = ["FILE_PATH"])
WAV_DF["FILE_NAME"] = wav_names
WAV_DF['FILE_NAME'] = WAV_DF['FILE_NAME'].str.replace('.wav','')


# ### Map (Audio x Labels x Text)

# In[11]:


AUDIO_LABEL = pd.DataFrame()
for index, row in LABEL_DF.iterrows():
    match = WAV_DF.loc[WAV_DF['FILE_NAME'] == row.FILE_NAME]
    match['LABEL'] = row.LABEL
    AUDIO_LABEL = AUDIO_LABEL.append(match, ignore_index = True)

IEMOCAP = pd.DataFrame()
for index, row in TEXT_DF.iterrows():
    match = AUDIO_LABEL.loc[AUDIO_LABEL['FILE_NAME'] == row.FILE_NAME]
    match['SENTENCE'] = row.TEXT
    IEMOCAP = IEMOCAP.append(match, ignore_index = True)
    
# IEMOCAP.to_csv("IEMOCAP.csv")

del WAV_DF, LABEL_DF, TEXT_DF, tmp

print(IEMOCAP.shape)
IEMOCAP.head()


# ---------------

# ## Text Processing

# In[12]:


from tensorflow.keras import  preprocessing, utils
import itertools

VOCAB_SIZE = 3000 # positive only

MAX_LEN = 30

### Encoder
input_lines = list()
labels = []
for index, row in IEMOCAP.iterrows():
    input_lines.append('<BOS> ' + row.SENTENCE + ' <EOS>') 
    labels.append(row.LABEL)

tokenizer = preprocessing.text.Tokenizer(filters='!"#$%&()*+,-.:;=?@[\\]^_`{|}~\t\n', num_words = VOCAB_SIZE)
tokenizer.fit_on_texts(input_lines) 
tokenized_input_lines = tokenizer.texts_to_sequences(input_lines) 

length_list = list()
for token_seq in tokenized_input_lines:
    length_list.append(len(token_seq))

max_input_length = max(length_list) # Gets the higher value in the list  
print('Input max length is:', max_input_length)

padded_input_lines = preprocessing.sequence.pad_sequences(tokenized_input_lines , maxlen= MAX_LEN , padding='post')
encoder_input_data = np.array(padded_input_lines)
print('Encoder input data shape:', encoder_input_data.shape)

input_word_dict = tokenizer.word_index
print("Maximum size of Vocab:", len(input_word_dict))
input_word_dict = dict(itertools.islice(input_word_dict.items(), VOCAB_SIZE-1))

num_input_tokens = len(input_word_dict)+1
input_word_dict['<unk>'] = num_input_tokens
print('Number of Input tokens:', num_input_tokens)

### Prepare Labels
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


print(encoder_input_data.shape)

X_text = encoder_input_data


# ## Load Glove

# In[13]:


### Load Glove (Word Embeddings)
embeddings_index = {}

with open(r'C:\Users\abdi\Desktop\2Meu\WordEmbeddings\glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

print("Glove Loaded!") 

def embedding_matrix_creator(embedding_dimension):
    embedding_matrix = np.zeros((VOCAB_SIZE, embedding_dimension))
    for word, i in input_word_dict.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

embedding_matrix = embedding_matrix_creator(100) # Change embedding dimensions


# -----------------------

# ## Audio Processing

# ## Extract Speech Features (MFCC)

# In[16]:


def extract_features(file_path):
    max_pad_len = 750
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')        
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    
        shape = mfccs.shape[1]
                     
        if max_pad_len > mfccs.shape[1]:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

        else:
            print(mfccs.shape)
            mfccs = mfccs[:, :max_pad_len]
            
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        mfccs = "error"
        return mfccs
        
    return mfccs


# ----------------------

# ## Load Features

# In[ ]:


def load_data():
    X, y = [], []
    i = 0
    for index, row in IEMOCAP.iterrows():
        print("File:", i+1)
        i+=1
        features = extract_features(row.FILE_PATH)
        emotion = row.LABEL
        X.append(features)
        y.append(emotion)

    return np.array(X), y

X_audio, y = load_data()


# ## Prepare Train/Test Data

# In[ ]:


from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

lb = LabelEncoder()
y_cat = np_utils.to_categorical(lb.fit_transform(y))

audio_train, audio_test, text_train, text_test, y_train, y_test = train_test_split(X_audio,
                                                                                   X_text,
                                                                                   y_cat, 
                                                                                   test_size = 0.25)


audio_train = np.expand_dims(audio_train, axis = 3)
audio_test = np.expand_dims(audio_test, axis = 3)

print("(>^.^)> audio_train shape:", audio_train.shape)
print("(>^.^)> audio_test shape:", audio_test.shape)
print("---")
print("(>^.^)> text_train shape:", text_train.shape)
print("(>^.^)> text_test shape:", text_test.shape)
print("---")
print("(>^.^)> y_train shape:", y_train.shape)
print("(>^.^)> y_test shape:", y_test.shape)


# ## CNN + LSTM -> Combine Text + Audio

# In[ ]:


from keras.layers import Concatenate
from keras.models import Model
from keras.layers.core import Flatten
from keras.layers import Input, Dense, Dropout, LSTM, Embedding, concatenate, RepeatVector, TimeDistributed, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.convolutional import MaxPooling2D, MaxPooling1D
from keras.callbacks import ModelCheckpoint

EPOCHS = 30
CONV_DIM = 128
BATCH_SIZE = 32
HIDDEN_DIM = 128
inputShape = audio_train.shape[1:]
chanDim = -1 

# Text Model
text_input = Input(shape=(MAX_LEN,))
text_model_1 = Embedding(VOCAB_SIZE, output_dim = 100, input_length = MAX_LEN, weights = [embedding_matrix], trainable =  True)(text_input)
text_model_2 = Dropout(0.25)(text_model_1)
text_model_3 = Conv1D(CONV_DIM, 5, padding='valid', activation='relu', strides=1)(text_model_2)
text_model_4 = MaxPooling1D(pool_size=4)(text_model_3)
text_model_5 = LSTM(HIDDEN_DIM, activation = 'relu')(text_model_4)

#Audio Model
audio_input = Input(shape=inputShape)
audio_model = Conv2D(CONV_DIM, (3, 3), padding="same", input_shape=inputShape)(audio_input)
audio_model = Activation("relu")(audio_model)
audio_model = BatchNormalization(axis=chanDim)(audio_model)
audio_model = MaxPooling2D(pool_size=(3, 3))(audio_model)
audio_model = Dropout(0.25)(audio_model)

audio_model = Flatten()(audio_model)
audio_model = Dense(HIDDEN_DIM)(audio_model)
audio_model = Activation("relu")(audio_model)
audio_model = BatchNormalization()(audio_model)
audio_model_2 = Dropout(0.25)(audio_model)

# Combined Model
final_model_1 = concatenate([text_model_5, audio_model_2])
final_model_2 = Dense(HIDDEN_DIM, activation='relu')(final_model_1)

final_model = Dense(4, activation='softmax')(final_model_1)

model = Model(inputs=[audio_input, text_input], outputs = final_model)
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics = ['accuracy'])    
model.summary()

# Checkpoint
filepath = "main_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Fit Data
history = model.fit([audio_train, text_train], y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, callbacks=callbacks_list, 
                     validation_data = ([audio_test, text_test], y_test))


# ------------------------

# ### Save Model

# In[ ]:


with open('main_model.json', 'w') as f:
    f.write(model.to_json())

import pickle
f = open("main_dic.pkl","wb")
pickle.dump(input_word_dict,f)
f.close()


# ### Load Model

# In[ ]:


import keras
from keras.models import model_from_json

Adam = keras.optimizers.Adam(learning_rate=0.0001)

### Main Model
# Model reconstruction from JSON file
with open('main_model.json', 'r') as f:
    main_model = model_from_json(f.read())


# ## Visualization

# In[ ]:


import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], linewidth=2)

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim(0,10)
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'], linewidth=2)

plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()


# ## Prediction

# In[ ]:


### Clean-text function
import re
def clean_text(text):
    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    return text


# In[ ]:


def load_vocab(vocab):
    with open('Load_Files/' + vocab + '.pkl', 'rb') as f:
        return pickle.load(f)


# In[ ]:


def encoding_input( sentence : str ):
    sentence = '<BOS> ' + sentence + ' <EOS>' # add bos and eos tokens
    words = sentence.lower().split()
    tokens_list = list()
    dictionary = load_vocab('main_dic')

    for word in words:
        try:
            tokens_list.append(dictionary[ word ])
        except Exception as e:
            tokens_list.append(dictionary[ '<unk>' ])
        
    return preprocessing.sequence.pad_sequences([tokens_list] , maxlen = MAX_LEN , padding='post')


# ### New Input

# In[ ]:


def new_input(file_path, sentence):
    X = []
    features = extract_features(file_path)
    X.append(features)
    X = np.expand_dims(np.array(X), axis = 3)
    encoding = encoding_input(sentence)

    y_pred = main_model.predict([X, encoding])
    emotion = np.argmax(y_pred[0])
    print(emotion)


# In[ ]:




