import librosa
import numpy as np
import jams
import os
import math

NUM_STRINGS = 6
NUM_FRETS = 21 #19 + open + nothing playing(optional)
MAX_LEN = 985 #max duration * sampling rate / num of bins per sample
MAX_DUR = 985/(22050/1024)

#strings: 0->E (6th string, lowest), 5-> e (1st string, highest)

def annotateHandPosition(tabSlice):
    tabFrame = np.zeros((NUM_STRINGS, NUM_FRETS))
    for string, fret in enumerate(tabSlice):
        tabFrame[string, int(fret)+1] = 1
    return tabFrame


def parseJamRealLength(jm, sr, hop_size, dur):
    stringsMidiDict = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
    sliceIndices = range(dur)
    times = librosa.frames_to_time(sliceIndices, sr=sr, hop_length=hop_size)

    stringAnnotations = []
    for i, string in enumerate(jm):
        samples = string.to_samples(times)

        for j in sliceIndices:
            if(samples[j] == []):
                samples[j] = -1
            else:
                samples[j] = int(round(samples[j][0] - stringsMidiDict[i]))
        stringAnnotations.append(samples)
    stringAnnotations = np.array(stringAnnotations)
    handPositionTab = []
    for i in range(stringAnnotations.shape[1]):
        handPositionTab.append(annotateHandPosition(stringAnnotations[:, i]))
    handPositionTab = np.array(handPositionTab)
    print(handPositionTab.shape)
    return handPositionTab
"""
    tab = np.zeros((math.ceil(jm[0].duration * (sr / hop_size)), NUM_STRINGS))
    tab.fill(-1)  # test
    tabFull = []
    for i, gString in enumerate(jm):
        # each string has its own annotations; i is the string number 0-5
        for note in gString:
            startTimeSlice = round(note[0] * (sr / hop_size))
            fret = round(note[2] - stringsMidiDict[i])
            stride = round(note[1] * (sr / hop_size))
            tab[startTimeSlice:startTimeSlice + stride, i] = fret  # +1 #test
    for i in range(tab.shape[0]):
        tabFull.append(annotateHandPosition(tab[i, :]))
    tabFull = np.array(tabFull)
    # print(tab)
    return (tabFull)"""


def parseJam(jm, sr=22050, hop_size=1024):
    stringsMidiDict = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
    parseJamRealLength(jm)
    tab = np.zeros((NUM_STRINGS, MAX_LEN))
    tabFrame = np.zeros((NUM_STRINGS, NUM_FRETS))
    tab.fill(-1) #test
    #1 hot encoding 6*21 array
    print(math.ceil(jm[0].duration * (sr / hop_size)))
    for i, gString in enumerate(jm):
        #each string has its own annotations; i is the string number 0-5
        for note in gString:
            startTimeSlice = round(note[0] * (sr/hop_size))
            fret = round(note[2] - stringsMidiDict[i])
            stride = round(note[1] * (sr/hop_size))
            tab[i, startTimeSlice:startTimeSlice+stride] = fret  # +1 #test

    #print(tab)
    return(tab)

def readJams(path):
    tabCorpus = []
    empty = True
    for file in os.listdir(path):
        jm = jams.load(path + file)
        jmNotes = jm.search(namespace="note_midi")
        tab = parseJam(jmNotes)
        if(empty):
            tabCorpus = tab
            empty = False
        else:
            tabCorpus = np.dstack((tabCorpus, tab))
    return(tabCorpus)

def readSingleJam(fileName, pathTab, dur, sr=22050, hop_size=1024):
    path = os.path.join(pathTab, fileName)
    print(path)
    jm = jams.load(path)
    jmNotes = jm.search(namespace="note_midi")
    tab = parseJamRealLength(jmNotes, sr, hop_size, dur)
    return tab


def jam2MidiArray(jamFilePath, n_bins, save=False, nameToSave="", save_path="./gt_midi_arrays/"):
    jm = jams.load(jamFilePath)
    jmNotes = jm.search(namespace="note_midi")
    midiArray = np.zeros((88, n_bins))
    duration = jmNotes[0].duration
    for string in jmNotes:
        for note in string:
            startTime = note[0] * (n_bins/duration)
            tone = note[2] - 21
            stride = note[1] * (n_bins/duration)
            midiArray[round(tone), round(startTime):round(stride)] = 1
    if(save):
        np.save("{}{}".format(save_path, nameToSave), midiArray)


def parseSecondDataset():
    pathData = "./second_dataset/tablature_dataset/tablature_dataset/tablature_labels/"
    pathCSV = "./second_dataset/tablature_dataset/tablature_dataset/timestamps.csv"
    #dataset does not contain any info about stride


if __name__ == "__main__":
    pathTab = "annotation/"
    tabs = readJams(pathTab)
    #tabs = np.load("tabs.npy")
    #annotateHandPosition(tabSlice)
    np.save("tabs.npy", tabs)
    print(tabs[:, :20, 0])
    #np.save("tabsTest.npy", tabs)


