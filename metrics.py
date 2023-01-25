import numpy as np


# pitch metrics
def tabToPitch(tab):
    stringsMidiDict = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
    pitchVector = np.zeros(44)
    for string in range(tab.shape[0]):
        fret = tab[string, :]
        played = np.argmax(fret, -1)
        if(played > 0):
            pitch = played + stringsMidiDict[string] - 41  # to move down by pitch of lower e string + 1 for string not played class
            pitchVector[pitch] = 1
    return pitchVector

def tabToVec(tab):
    fretBoard = np.zeros((6, 20))
    for string in range(tab.shape[0]):
        stringVec = tab[string, :]
        played = np.argmax(stringVec, -1)
        if(played > 0):
            fret = played - 1
            fretBoard[string][fret] = 1
    return fretBoard



def pitchPrecision(prediction, truth):
    pitchPred = np.array(list(map(tabToPitch, prediction)))
    pitchTruth = np.array(list(map(tabToPitch, truth)))
    precision = np.sum(np.multiply(pitchPred, pitchTruth).flatten()) / np.sum(pitchPred.flatten())
    return precision


def pitchRecall(prediction, truth):
    pitchPred = np.array(list(map(tabToPitch, prediction)))
    pitchTruth = np.array(list(map(tabToPitch, truth)))
    recall = np.sum(np.multiply(pitchPred, pitchTruth).flatten()) / np.sum(pitchTruth.flatten())
    return recall


# tab metrics

def tabPrecision(prediction, truth):
    tabPred = np.array(list(map(tabToVec, prediction)))
    tabTruth = np.array(list(map(tabToVec, truth)))
    precision = np.sum(np.multiply(tabPred, tabTruth).flatten()) / np.sum(tabPred.flatten())
    return precision


def tabRecall(prediction, truth):
    tabPred = np.array(list(map(tabToVec, prediction)))
    tabTruth = np.array(list(map(tabToVec, truth)))
    recall = np.sum(np.multiply(tabPred, tabTruth).flatten()) / np.sum(tabTruth.flatten())
    return recall



def fMeasure(prediction, truth, isTab):
    precision = tabPrecision(prediction, truth) if isTab else pitchPrecision(prediction, truth)
    recall = tabRecall(prediction, truth) if isTab else pitchRecall(prediction, truth)
    f = (2 * precision * recall) / (precision + recall)
    return f


#midiArray metrics


def midiArrPrecision(prediction, truth):
    numenator = np.sum(np.multiply(prediction, truth).flatten())
    denominator = np.sum(prediction.flatten())
    return (numenator / denominator)

def midiArrRecall(prediction, truth):
    numenator = np.sum(np.multiply(prediction, truth).flatten())
    denominator = np.sum(truth.flatten())
    return (numenator / denominator)

def midiArrfMeasure(prediction, truth):
    precision = midiArrPrecision(prediction, truth)
    recall = midiArrRecall(prediction, truth)
    f = (2 * precision * recall) / (precision + recall)
    return f

