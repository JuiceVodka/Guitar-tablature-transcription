import math

import librosa
import numpy as np
import librosa as lb
from scipy.fftpack import fft, fftfreq
from scipy.io import wavfile
from matplotlib import pyplot as plt
import os
import readJam as rj


def toFFTvec(path):
    maxLen = 915
    snd, sr = lb.load(path, sr=16000)
    sndNorm = snd / (max(snd))
    fftComb = []
    # pf = fftfreq(sr//20, 1/sr)[0:sr//40]

    for i in range(0, len(sndNorm), sr // 20):
        patch = sndNorm[i:i + sr // 20]
        if (patch.shape[0] < sr // 20):
            patch = np.pad(patch, (0, np.abs(patch.shape[0] - sr // 20)), 'constant')
        patchFFT = np.abs(fft(patch)[0:len(patch) // 2])
        patchFFT = np.floor(patchFFT)  # square and divide by length of patch -> from tis, sound removal
        #print(max(patchFFT))
        # patchFFT = patchFFT.reshape((len(patchFFT), 1))
        # pf = fftfreq(len(patch), 1/sr)[0:len(patch)//2]
        #print(patchFFT.shape)
        fftComb.append(patchFFT)
        #todo window function + overlapping patchi + >= 16k sampling rate

    fftComb = np.transpose(np.array(fftComb).astype(np.float32))
    if(fftComb.shape[1] < maxLen):
        fftComb = np.hstack((fftComb, np.zeros((fftComb.shape[0], np.abs(fftComb.shape[1] - maxLen)))))
    #print(fftComb)
    return fftComb

def readDt(dirPath):
    sndStack = []
    check = False
    for file in os.listdir(dirPath):
        snd = toFFTvec(os.path.join(dirPath, file))
        if not check:
            sndStack = snd
            check = True
        else:
            sndStack = np.dstack((sndStack, snd))
    return sndStack

#1 constantq for 1 patch of data, each patch is 1 entry into neural network
def cnstntQ(filePath, padLen=9):
    #snd, sr = lb.load(filePath, sr=44100, mono=True)#, offset=start, duration=dur)
    sr, snd = wavfile.read(filePath)
    snd = snd.astype(float)
    sndNorm = librosa.util.normalize(snd)
    sndNorm = librosa.resample(sndNorm, orig_sr=sr, target_sr=22050)

    #better results without min freq at 75
    constantq = lb.cqt(sndNorm, sr=22050, hop_length=512, n_bins=192, bins_per_octave=24) #in article hop len 512, num bins 192 -> for this samplinmg freq has to be higher

    cqtOut = constantq
    constantqMagnifie = librosa.magphase(constantq)[0]
    constantqDB = librosa.core.amplitude_to_db(constantqMagnifie, ref=constantqMagnifie.max())
    cqtOut = np.copy(constantqDB)
    #cqtOut[cqtOut < -60] = -60
    cqtOut[:, :] += np.abs(min(cqtOut.flatten()))
    #cqtOut = np.pad(constantq, ((0, 0), (math.floor(padLen/2), math.floor(padLen/2))), 'constant')

    print("-------")
    print(filePath.split("/")[1])
    print(cqtOut.shape)
    return np.abs(cqtOut)


def readDTcnstQ(sndPath, tabPath, save=False):
    songSliceArtistList = []
    for i, file in enumerate(os.listdir(sndPath)):
        spec_tab_dict = {}

        tabFile = f"{file[:-8]}.jams"

        filePath = os.path.join(sndPath, file)
        constQ = cnstntQ(filePath)
        jam = rj.readSingleJam(tabFile, tabPath, constQ.shape[1], hop_size=512)

        spec_tab_dict["spec"] = constQ
        spec_tab_dict["tab"] = jam

        artist = file[1]
        songID = i
        #print(jam.shape[1])
        songSliceArtistList.extend([f"{artist}_{songID}_{j}" for j in range(jam.shape[0])])

        if(save):
            np.savez(f"./spec_tab2/{i}", **spec_tab_dict) #saves each individual song-annotation pair
        else:
            plt.imshow(constQ.astype(float), cmap="jet", interpolation='nearest', aspect='auto')
            plt.show()
    #if(save):
    #    songSliceArtistList = np.array(songSliceArtistList)
    #    np.save("./listSlices/ids.npy", songSliceArtistList)


if __name__ == "__main__":
    pathTab = "annotation/"
    pathSound = "audio_mono-mic/"

    """game plan:
    embed audio recordings -> sound to vec (slice up int segments, fft majbe?)
    pass into neuralnetwork (simple cnn to start, 3 convolutional layers, 1 pooling, etc etc, papers for reference)
    loss function with comparison of probability of right class (first have to learn how to work .jam files)"""

    # audio preprocesing

    # normalization
    #exampleSnd, sr = lb.load(pathSound + "00_BN1-129-Eb_comp_mic.wav", sr=4800)

    #exampleSndNorm = exampleSnd / (max(exampleSnd))
    #print(max(exampleSnd))
    #print(max(exampleSndNorm))
    #print(len(exampleSndNorm) / sr)
    #print((len(exampleSndNorm) / sr) * 20)

    # cutting up into slices (20 slices per second should suffice)
    # 4800/20 = 240
    # constant-q transform
    # (ali pa mel skala) -> veliko v lobrosi poglej dokuimentacijo (mel spectrogram)

    #im = toFFTvec(pathSound + "00_BN1-129-Eb_comp_mic.wav")
    #plt.imshow(im, cmap="gray")
    #plt.show()

    #data = readDt(pathSound)
    #np.save("ffts.npy", data)

    #readDTcnstQ(pathSound, pathTab, save=True)
    readDTcnstQ(pathSound, pathTab)
