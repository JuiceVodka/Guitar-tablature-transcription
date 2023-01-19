import math

import numpy as np
import readJam as rj
import keras
import keras.utils
#h5py import -> hdf5 format -> za sharnjevanje binarnih podatkov in za loadanje dolocenih chunkov

class TabDataGenerator(keras.utils.Sequence):

    def __init__(self, listSlices, batch_size=128, labelDim=(6,21), dataPath="./spec_tab/", padLen=9, shuffle=True):
        print(len(listSlices))
        self.listSlices = listSlices
        self.batchSize = batch_size
        self.labelDim = labelDim
        self.dataPath = dataPath
        self.padLen = padLen
        self.shuffle = shuffle
        #self.X_dim = (self.batchSize, padLen, 168, 1) #maybe switch pad len and 168?
        #self.y_dim = (self.batchSize, self.labelDim[0], self.labelDim[1])
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.listSlices))
        if(self.shuffle):
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batchSize : (index+1) * self.batchSize]

        list_IDs_temp = [self.listSlices[i] for i in indexes]

        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def __len__(self):
        return int(np.floor(len(self.listSlices) / self.batchSize))

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batchSize, 192, self.padLen, 1)) #maybe switch pad len and 168?
        y = np.empty((self.batchSize, self.labelDim[0], self.labelDim[1]))

        for i, ID in enumerate(list_IDs_temp):
            components = ID.split("_")
            songID = components[1]
            sliceID = components[2]

            spec_tab = np.load(self.dataPath + songID + ".npz")

            spec = np.pad(spec_tab["spec"], ((0, 0), (math.floor(self.padLen/2), math.floor(self.padLen/2))), 'constant')
            tab = spec_tab["tab"]
            specSlice = spec[:, int(sliceID):int(sliceID) + self.padLen] #tweak this window

            X[i,] = np.expand_dims(specSlice, -1)
            y[i,] = tab[int(sliceID)]#rj.annotateHandPosition(tab[:, int(sliceID)])

        return X, y


#dg = TabDataGenerator()
#print(len(np.load("./listSlices/ids.npy")))


#1/5 celih komadou v validation, ostalo v training -> te idje lahko podas datageneratorju, lahko naredis se en dict,
# ki za vsak komad hrani koliko ima framou song
#v clanku ze pri idjih naredi za vsak slice svoj id