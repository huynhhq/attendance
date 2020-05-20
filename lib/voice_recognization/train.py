import IPython.display as ipd
import os
import pandas as pd
import librosa
import glob
import librosa.display
import random


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import utils
from sklearn import metrics

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import regularizers

from sklearn.preprocessing import LabelEncoder

from datetime import datetime

import os
from os.path import isdir, join
from os import listdir

import config

# list the files
filelist = os.listdir(config.DATASET_PATH)
all_users = [name for name in listdir(
    config.DATASET_PATH) if isdir(join(config.DATASET_PATH, name))]
print(all_users)


idx = 0
df = []
for user_name in all_users:
    file_names = listdir(join(config.DATASET_PATH, user_name))  # , 'wav'))
    # for file in file_names:
    #     file_path = join(config.DATASET_PATH, user_name, file) #'wav', file)

    # read them into pandas
    df_files = pd.DataFrame(file_names)
    # print('-----------------------------')
    # print(df_files)
    df_files['label'] = "" + str(idx)
    df_files = df_files.rename(columns={0: 'file'})
    df_files.head()
    # print(df_files)
    df_files[df_files['file'] == '.DS_Store']
    # df_files.drop(981, inplace=True)
    # Resetting the index since we dropped a row
    # for f in df_files.file:
    #     # print(f)
    #     f = join(user_name, f)
    df_files.file = config.DATASET_NAME + "/" + user_name + "/" + df_files.file
    if idx == 0:
        df = df_files
    else:
        # df_files = df_files.reset_index(drop=True)
        df = pd.concat([df_files, df], ignore_index=True)
    idx = idx + 1

# TODO: skip line 35 - 41

df.head()
df = df.sample(frac=1).reset_index(drop=True)
df_train_size = int(0.8 * len(df))
df_train = df[:df_train_size]  # 9188

df_train['label'].value_counts(normalize=True)
print(df_train['label'].value_counts(normalize=True))
# df_validation = df[9188:11813]
df_validation_size = int(0.9 * len(df))
df_validation = df[df_train_size:df_validation_size]
df_validation['label'].value_counts(normalize=True)
# df_test = df[11813:13125]
df_test = df[df_validation_size:]
df_test['label'].value_counts(normalize=True)
print('-------------------')
print(df)
print('-------------------')
# print(df_train)


def extract_features(files):
    # Sets the name to be the path to where the file is in my computer
    file_name = os.path.join(config.ROOT_PATH, str(files.file))
    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
    mfccs = np.mean(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))
    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(
        S=stft, sr=sample_rate).T, axis=0)
    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(
        S=stft, sr=sample_rate).T, axis=0)
    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)
    # We add also the classes of each file as a label at the end
    label = files.label

    return mfccs, chroma, mel, contrast, tonnetz, label


startTime = datetime.now()
features_label = df.apply(extract_features, axis=1)
print(datetime.now() - startTime)
print(features_label)
# features_label
features = []
for i in range(0, len(features_label)):
    features.append(np.concatenate((features_label[i][0], features_label[i][1],
                                    features_label[i][2], features_label[i][3],
                                    features_label[i][4]), axis=0))
print('len(features) = ', len(features))

speaker = []
for i in range(0, len(df)):
    speaker.append(df['file'][i].split('/')[1])
df['speaker'] = speaker

df.head()
print(df.head())
print('-----------------------------------')
print(df['speaker'].nunique())
print('speaker = ', speaker)
labels = speaker
print(len(labels))
np.unique(labels, return_counts=True)

X = np.array(features)
y = np.array(labels)
lb = LabelEncoder()
print('-----------------------------------------------')
print(lb.fit_transform(y))
y = to_categorical(lb.fit_transform(y))
print('----------------------y-------------------------')
print(y)

################################################
y2 = np.array(labels)
lb2 = LabelEncoder()
encode = lb2.fit(np.array(all_users))
encode.classes_
print('-----------------------------------------------')
print(encode.transform(y2))
y2 = to_categorical(encode.transform(y2))
print('----------------------y2-------------------------')
print(y2)
if np.array(y).all() == np.array(y2).all():
    print('----------------------EQUAL-------------------------')

print(X.shape)
print(y.shape)

X_train = X[:df_train_size]
y_train = y[:df_train_size]

X_val = X[df_train_size:df_validation_size]
y_val = y[df_train_size:df_validation_size]

X_test = X[df_validation_size:]
y_test = y[df_validation_size:]

########################################################################
np.savez(config.FEATURE_SET_FILE,
         all_users=all_users,
         X=X,
         y=y,
         labels=labels,
         X_train=X_train,
         y_train=y_train,
         X_val=X_val,
         y_val=y_val,
         X_test=X_test,
         y_test=y_test)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_val = ss.transform(X_val)
X_test = ss.transform(X_test)

sample_shape = X_train.shape[1:]
print('sample_shape: ', sample_shape)
output_size = len(all_users)

model = Sequential()

# ----------default ----------------
# model.add(Dense(64, input_shape=(193,), activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(all_users), activation='softmax'))

# ------------- thu mo hinh 1
# model.add(Dense(64, input_shape=(193,), activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(len(all_users), activation='softmax'))

# ------------- thu mo hinh 2
model.add(Dense(128, input_shape=sample_shape, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

# Classifier
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_size, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')

early_stop = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

history = model.fit(X_train, y_train, batch_size=len(X_train), epochs=100,
                    validation_data=(X_val, y_val)
                    # ,
                    # callbacks=[early_stop]
                    )

models.save_model(model, config.MODEL_FILE_PATH)

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Set figure size.
plt.figure(figsize=(12, 8))

# Generate line plot of training, testing loss over epochs.
plt.plot(train_accuracy, label='Training Accuracy', color='#185fad')
plt.plot(val_accuracy, label='Validation Accuracy', color='orange')

# Set title
plt.title('Training and Validation Accuracy by Epoch', fontsize=25)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Categorical Crossentropy', fontsize=18)
plt.xticks(range(0, 100, 5), range(0, 100, 5))

plt.legend(fontsize=18)
plt.show()


preds = model.predict_classes(X_test)
# We transform back our predictions to the speakers ids
preds = lb.inverse_transform(preds)
# We slice our dataframe to our test dataframe
df_test = df[df_validation_size:]
# We create a new column called preds and set it equal to our predictions
df_test['preds'] = preds
print(df_test)
