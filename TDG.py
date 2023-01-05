import numpy as np
import readJam as rj

import tensorflow as tf
#h5py import -> hdf5 format -> za sharnjevanje binarnih podatkov in za loadanje dolocenih chunkov

"""class TDG(keras.utils.Sequence):
    def __init__(self):
        self.batchSize = 128
        self.dataIndices = np.arange(300)
        print(len(self.dataIndices))
        print(self.dataIndices)

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
"""

#dg = TDG()

c = np.array([[3.,4], [5.,6], [6.,7]])
step = tf.reduce_mean(c, 1)
with tf.Session() as sess:
    print(sess.run(step))