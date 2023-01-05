import librosa
import numpy as np
import librosa as lb
from scipy.fftpack import fft, fftfreq
from matplotlib import pyplot as plt
import os


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
def cnstntQ(filePath): # todo add padding for sliding window and save to disk to use in data generator
    snd, sr = lb.load(filePath, sr=44100, mono=True)#, offset=start, duration=dur)
    snd = librosa.resample(snd, orig_sr=44100, target_sr=22050)
    sndNorm = librosa.util.normalize(snd)

    constantq = lb.cqt(sndNorm, sr=22050, hop_length=1024, fmin=75, n_bins=190, bins_per_octave=24) #in article hop len 512, num bins 192 -> for this samplinmg freq has to be higher
    constantqMagnifie = librosa.magphase(constantq)[0]
    constantqDB = librosa.core.amplitude_to_db(constantqMagnifie, ref=constantqMagnifie.max())
    cqtOut = np.copy(constantqDB)
    cqtOut[cqtOut < -60] = -120
    print("-------")
    print(filePath.split("/")[1])
    print(cqtOut.shape)
    return cqtOut


def readDTcnstQ(dirPath):
    for file in os.listdir(dirPath):
        filePath = os.path.join(dirPath, file)
        constQ = cnstntQ(filePath)

        plt.imshow(constQ, cmap="jet", interpolation='nearest', aspect='auto')
        plt.show()


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

    readDTcnstQ(pathSound)
