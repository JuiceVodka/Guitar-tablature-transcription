import numpy as np
import jams
import os
import math

NUM_STRINGS = 6
NUM_FRETS = 21 #19 + open + nothing playing(optional)
MAX_LEN = 985 #max duration * sampling rate / num of bins per sample
MAX_DUR = 985/(22050/1024)

#strings: 0->E (6th string, lowest), 5-> e (1st string, highest)

def annotateHandPosition(tabSlice): #todo incorporate this, maybe import in datagenerator and use it there
    tabFrame = np.zeros((NUM_STRINGS, NUM_FRETS))
    for string, fret in enumerate(tabSlice):
        print(string)
        tabFrame[string, int(fret)+1] = 1
    return tabFrame


def parseJam(jm):
    stringsMidiDict = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}

    tab = np.zeros((NUM_STRINGS, MAX_LEN))
    tabFrame = np.zeros((NUM_STRINGS, NUM_FRETS))
    tab.fill(-1) #test
    #1 hot encoding 6*21 array
    for i, gString in enumerate(jm):
        #each string has its own annotations; i is the string number 0-5
        for note in gString:
            startTimeSlice = math.ceil(note[0] * (22050/1024))
            fret = round(note[2] - stringsMidiDict[i])
            stride = round(note[1] * (22050/1024))
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


if __name__ == "__main__":
    pathTab = "annotation/"
    tabs = readJams(pathTab)
    #tabs = np.load("tabs.npy")
    #annotateHandPosition(tabSlice)
    np.save("tabs.npy", tabs)
    print(tabs[:, :20, 0])
    #np.save("tabsTest.npy", tabs)


