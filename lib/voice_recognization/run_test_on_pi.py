import IPython.display as ipd
import os
import pandas as pd
import librosa
import glob
import librosa.display
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

unknown_label = 'UNKNOWN'

model = models.load_model(config.MODEL_FILE_PATH)
# summarize model.
# model.summary()
# probability_model = Sequential([model, tf.keras.layers.Softmax()])

# load saved preprocessed data
feature_sets = np.load(config.FEATURE_SET_FILE, allow_pickle=True)
X_train_original = feature_sets['X_train']
labels = feature_sets['labels']
y = np.array(labels)
lb = LabelEncoder()
y = to_categorical(lb.fit_transform(y))
ss = StandardScaler()
X_train_original = ss.fit_transform(X_train_original)


def extract_features(files):
    # Sets the name to be the path to where the file is in my computer
    file_name = str(files.file)
    print('file_name = ', file_name)
    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast', sr=16000)
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


# files = []
# files.append(config.DATA_TEST_PATH + '/' +'Minh_1.wav')
def predict_speaker(audio_file):
    df_files = pd.DataFrame([audio_file])
    # print('-----------------------------')
    # print(df_files)
    df_files['label'] = unknown_label
    df_files = df_files.rename(columns={0: 'file'})
    df_files[df_files['file'] == '.DS_Store']
    df_files = df_files.sample(frac=1).reset_index(drop=True)

    features_label = df_files.apply(extract_features, axis=1)
    features = []
    for i in range(0, len(features_label)):
        features.append(np.concatenate((features_label[i][0], features_label[i][1],
                                        features_label[i][2], features_label[i][3],
                                        features_label[i][4]), axis=0))
    X = np.array(features)
    X_train = ss.transform(X)

    output_data = model.predict(X_train)
    # print('--------------------------------------')
    # for i in range(0, output_data.shape[0]):
    index = np.argmax(output_data[0])
    val = max(output_data[0])
    # print('val = ', val, ', index = ', index,
    #       ', name = ', lb.inverse_transform([index])[0])
    if (val >= config.THRESHOLD):
        return lb.inverse_transform([index])[0]
    return unknown_label


print(predict_speaker('assets/video/attendance_today.wav'))
