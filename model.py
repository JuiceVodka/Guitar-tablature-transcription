import numpy as np
from comet_ml import Experiment
import keras
from keras import datasets, layers, models
from keras.layers import Activation, Dropout
from keras.optimizers.optimizer_v2 import adadelta
from keras import backend as K
from TabDataGenerator import TabDataGenerator
import metrics


"""experiment = Experiment(api_key="mtbzRxnK1pFMS91Zk4gDMg4Xa",
                        project_name="guitar-tablature-transcription",
                        workspace="juicevodka",
                        auto_metric_logging=True,
                        auto_param_logging=True,
                        auto_histogram_weight_logging=True,
                        auto_histogram_gradient_logging=True,
                        auto_histogram_activation_logging=True,
                        )"""

NUM_FRETS = 21
NUM_STRINGS = 6
MAX_LEN = 915


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


def avg_accuracy(truth, out):  # tweak here
    return K.mean(K.equal(K.argmax(truth, axis=-1), K.argmax(out, axis=-1)))

#directly from article
def softmax_by_string(t):
    sh = K.shape(t)
    string_sm = []
    for i in range(NUM_STRINGS):
        string_sm.append(K.expand_dims(K.softmax(t[:,i,:]), axis=1))
    return K.concatenate(string_sm, axis=1)

def catcross_by_string(target, output):
    loss = 0
    for i in range(NUM_STRINGS):
        loss += K.categorical_crossentropy(target[:,i,:], output[:,i,:])
    return loss

def avg_acc(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
#--------------



pathTab = "annotation/"
pathSound = "audio_mono-mic/"
pathList = "listSlices/"

listSlices = np.load(pathList + "ids.npy")

partition = {"training": [], "validation": []}

for slice in listSlices:
    artist = slice.split("_")[0]
    if int(artist) > 0:
        partition["training"].append(slice)
    else:
        partition["validation"].append(slice)

trainingDataGenerator = TabDataGenerator(partition["training"])
validationDataGenerator = TabDataGenerator(partition["validation"])

print("data gen completed")
# vX,y = trainingDataGenerator.__getitem__(0)
# print(X.shape)
# print(y.shape)
# stringSoftmax(y[1:2,:,:])

# data = pr.readDt(pathSound)
# data = np.load("ffts.npy")
# classes = np.load("tabs.npy")
# classesTest = np.load("tabsTest.npy")
# print(data.shape)
# print(classes.shape)

# Onsets and frames ---- > poglej loadanje, kako malo dela na kitari

model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(192, 9, 1)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(NUM_STRINGS * NUM_FRETS))
model.add(layers.Reshape((NUM_STRINGS, NUM_FRETS)))
model.add(Activation(softmax_by_string))

model.compile(loss=catCrossLoss, optimizer=adadelta.Adadelta(), metrics=[avg_acc])
model.fit_generator(generator=trainingDataGenerator, validation_data=None, epochs=10)
model.save("./models/fifthModel-512")



#evaluation
X_test, y_truth = validationDataGenerator[0]
y_pred = model.predict(X_test)

print("pitch precision: " + metrics.pitchPrecision(y_pred, y_truth))
print("pitch recall: " + metrics.pitchRecall(y_pred, y_truth))
print("tab precision: " + metrics.tabPrecision(y_pred, y_truth))
print("tab recall: " + metrics.tabRecall(y_pred, y_truth))
print("pitch f: " + metrics.fMeasure(y_pred, y_truth, False))
print("tab f: " + metrics.fMeasure(y_pred, y_truth, True))


# accuracy not very high... try more/different layers or different centering of data


# plan:
# -bugfix to get better results
# -najdi vec podatkou (ali naredi) -> found something, look into it
# -sprobaj exsisting model na teh podatkih
# -probaj se kak drugacen model (npr onsets and frames)
# -


# onsets and frames
# pretrained modeli, unsupervised extractajo feature, ki se potem specializirajo za taske
