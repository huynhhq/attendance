from tkinter import *
from tkinter import filedialog
from tkinter import Tk, RIGHT, BOTH, RAISED
from tkinter.ttk import Frame, Button, Style
import tkinter as tk
from tkinter.filedialog import askopenfilename

import mtcnn
from mtcnn.mtcnn import MTCNN

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

from numpy import savez_compressed
from numpy import load
from numpy import expand_dims
from numpy import asarray

from keras.models import load_model
from os import path
from os import listdir
from os.path import isdir

from PIL import ImageTk, Image
from matplotlib import pyplot
from skimage import io
from random import choice
from pathlib import Path
from os.path import basename, expanduser, isfile, join as joined

import cv2
import numpy
import time
# import dlib
import cv2
import sys

imagePath = sys.argv[1]

FACES_EMBEDDING_ZIP = "/home/huynhhq/work_space/attendance_manager/lib/face_recognization/faces_embedding.npz"

detector = MTCNN()
# load FaceNet
modelFaceNet = load_model('/home/huynhhq/work_space/attendance_manager/lib/face_recognization/facenet_keras.h5')
# load model
modelTrained = joblib.load('/home/huynhhq/work_space/attendance_manager/lib/face_recognization/Model.pkl')

# load face embeddings
data = load(FACES_EMBEDDING_ZIP)
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)

# normalize input vectors
in_encoder = Normalizer(norm='l2')

# get the face embedding for one face


def get_embedding(model, face_pixels):

    face_pixels = face_pixels.astype('float32')

    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()

    face_pixels = (face_pixels - mean) / std

    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)

    # make prediction to get embedding
    yhat = model.predict(samples)

    return yhat[0]


def face_recognition(imagePath):    
    face_labels = list()

    # convert image to nsarray data
    imageNsArray = cv2.imread(imagePath)

    faces = detector.detect_faces(imageNsArray)

    for i in range(0, len(faces)):

        face_label = "Unknown"

        face = faces[i]
        faceOriginal = faces[i]

        x1, y1, width, height = face['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face = imageNsArray[y1:y2, x1:x2]

        face = Image.fromarray(face)
        face = face.resize((160, 160))
        face = asarray(face)

        facesEmbeddingArray = list()
        faceEmbedding = get_embedding(modelFaceNet, face)
        facesEmbeddingArray.append(faceEmbedding)
        facesEmbeddingArray = asarray(facesEmbeddingArray)
        facesEmbeddingArray = in_encoder.transform(facesEmbeddingArray)

        samples = expand_dims(facesEmbeddingArray[0], axis=0)
        yhat_class = modelTrained.predict(samples)
        yhat_prob = modelTrained.predict_proba(samples)

        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)

        if not predict_names:
            face_label = "Unknown"
        # if not predict_names or class_probability < 50:
        #     face_label = "Unknown"
        else:
            face_label = predict_names[0]

        face_labels.append(face_label)

    return face_labels


# Example
print(face_recognition(imagePath))
