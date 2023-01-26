import keras
from keras import datasets, layers, models
from keras.layers import Activation, Dropout, Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from keras.optimizers.optimizer_v2 import adadelta
from keras import backend as K

input_shape = (192, 9, 1)

output_shape = (6, 21)

NUM_STRINGS = 6
NUM_FRETS = 21

def catCrossLoss(out, truth):
    loss = 0
    for i in range(0, NUM_STRINGS):
        loss += K.categorical_crossentropy(out[:, i, :], truth[:, i, :])  # maybe swap indices
    return(loss)


def stringSoftmax(rez):
    softmaxByString = []
    for i in range(NUM_STRINGS):
        softmax = K.softmax(rez[:, i, :])
        softmaxByString.append(softmax)
    softmaxByString = K.stack(softmaxByString, axis=1)
    return softmaxByString


def avg_acc(truth, out):  # tweak here
    return K.mean(K.equal(K.argmax(truth, axis=-1), K.argmax(out, axis=-1)))


def variantLSTM():
    # Define the model
    model= keras.Sequential()

    # Define convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(192, 9, 1)))
    # model2.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model2.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Define recurrent layer
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Reshape((1, 128)))
    model.add(layers.LSTM(128, return_sequences=True))

    # reshape the output
    model.add(layers.Flatten())
    model.add(layers.Dense(NUM_STRINGS * NUM_FRETS))
    model.add(layers.Reshape((NUM_STRINGS, NUM_FRETS)))

    # Define the output layer
    model.add(Activation(stringSoftmax))

    model.compile(loss=catCrossLoss, optimizer=adadelta.Adadelta(learning_rate=1.0), metrics=[avg_acc])
    return model

def variantCONV(): #in progress
    modelConv = keras.Sequential()
    # layer0
    modelConv.add(layers.Conv2D(64, (3, 3), input_shape=(1, 192, 9)))  # 32 exchangable -> this is adapted from oaf transcriber.py
    modelConv.add(layers.BatchNormalization(momentum=0.15, axis=-1))
    modelConv.add(layers.ReLU)

    # layer1
    modelConv.add(layers.Conv2D(64, (3, 3)))  # maybe add padding="same"
    modelConv.add(layers.BatchNormalization(momentum=0.15, axis=-1))
    modelConv.add(layers.ReLU)

    # layer2
    modelConv.add(layers.MaxPooling2D(pool_size=(1, 2)))
    modelConv.add(layers.Dropout(0.25))
    modelConv.add(layers.Conv2D(128, (3, 3)))
    modelConv.add(layers.BatchNormalization(momentum=0.15, axis=-1))
    modelConv.add(layers.ReLU)

    modelConv.add(layers.MaxPooling2D(pool_size=(1, 2)))
    modelConv.add(layers.Dropout(0.25))


def variantOAF_like():
    model = keras.Sequential()

    # layer 1
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # layer 2
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # layer 3
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # layer 4
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # layer 5
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # layer 6
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # layer 7
    model.add(layers.Flatten())
    model.add(layers.Dense(2048))
    model.add(layers.ReLU())
    model.add(Dropout(0.5))

    # layer 8
    model.add(layers.Dense(128))
    model.add(layers.Dense(126))
    model.add(layers.Reshape(output_shape))
    model.add(Activation(stringSoftmax))

    model.compile(loss=catCrossLoss, optimizer=adadelta.Adadelta(learning_rate=1.0), metrics=[avg_acc])
    return model

#OAF-like architecture
#recurent architecture
#different kernel shapes -> vertical, horizontal, combination
