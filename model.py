import numpy as np
import scipy
import pandas
import jams
import librosa as lb
import tensorflow as tf
import keras
from keras import datasets, layers, models
from keras import backend as K

NUM_FRETS = 21
NUM_STRINGS = 6
MAX_LEN = 915


def lossFun(out, truth):
    #todo, this is naive
    #return sum(np.abs(out - truth))
    loss = 0
    for i in range(0, NUM_STRINGS):
        loss += K.categorical_crossentropy(truth[i, :], out[i, :])
    return(loss)

def avg_acc(truth, out):
    return K.mean(K.equal(K.argmax(truth, axis=1), K.argmax(out, axis=1)))


pathTab = "annotation/"
pathSound = "audio_mono-mic/"



#data = pr.readDt(pathSound)
data = np.load("ffts.npy")
classes = np.load("tabs.npy")
classesTest = np.load("tabsTest.npy")
print(data.shape)
print(classes.shape)

#Onsets and frames ---- > poglej loadanje, kako malo dela na kitari

model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (data.shape[1], data.shape[0], 1)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(NUM_STRINGS*NUM_FRETS)) #NUM_FRETS incorporated as max number on each string
model.add(layers.Reshape((NUM_FRETS, NUM_STRINGS)))

model.compile(loss=lossFun, optimizer='adam', metrics=['accuracy'])
model.fit(np.transpose(data), np.transpose(classes), epochs=10)
model.save("frstModel")
