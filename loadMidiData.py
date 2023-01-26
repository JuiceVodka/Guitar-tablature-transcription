import numpy as np
import mido
import os
import readJam as rj
import metrics as m
import pandas as pd

def mid2array(mid):
    totalTicks = 0
    for e in mid.tracks[1]:
        totalTicks += e.time

    midiArray = np.zeros((88, totalTicks))
    onsetArray = np.zeros(88)

    midiDict = {}

    currentTick = 0
    for msg in mid.tracks[1]:
        # print(msg.type)
        currentTick += msg.time
        if (msg.type == "note_on" or msg.type == "note_off"):
            note = msg.note - 21
            if (msg.velocity > 0):
                midiArray[note, currentTick] = 1
                onsetArray[note] = currentTick
            else:
                midiArray[note, int(onsetArray[note]):int(currentTick)] = 1

    midiDict["array"] = midiArray
    midiDict["metadata"] = np.array([mid.length, mid.ticks_per_beat, mid.tracks[0][0].tempo])

    return midiDict


if(__name__ == "__main__"):
    musicPath = "C:/Users/Niko/Documents/Faks/onsets-and-frames/"
    jamPath = "./annotation/"
    savePath = "./onsets_and_frames_predictions/"

    mid = mido.MidiFile(musicPath + "prediction0.mid", clip=True)
    print(mid.tracks[0][0].tempo)

    """for i, file in enumerate(os.listdir(jamPath)):
        if(i >= 60): break
        filenameMid = os.path.join(musicPath, f"prediction{i}.mid")
        filenameJam = os.path.join(jamPath, file)

        mid = mido.MidiFile(filenameMid, clip=True)
        parsedMid = mid2array(mid)
        np.savez(f"{savePath}predArr{i}", **parsedMid)

        rj.jam2MidiArray(filenameJam, parsedMid["array"].shape[1], True, nameToSave=f"jamMidiArray_{i}")"""

    metrics = {"o_a_f_precision" : [],
               "o_a_f_recall" : [],
               "o_a_f_fMeasure" : []}

    for i, file in enumerate(os.listdir(savePath)):
        predMid = np.load("{}predArr{}.npz".format(savePath, i))["array"]
        gtMid = np.load("./gt_midi_arrays/jamMidiArray_{}.npy".format(i))
        print(m.midiArrPrecision(predMid, gtMid))
        print(m.midiArrRecall(predMid, gtMid))
        print("--------------")
        metrics["o_a_f_precision"].append(m.midiArrPrecision(predMid, gtMid))
        metrics["o_a_f_recall"].append(m.midiArrRecall(predMid, gtMid))

    metrics["o_a_f_precision"] = [np.sum(metrics["o_a_f_precision"]) / len(metrics["o_a_f_precision"])]
    metrics["o_a_f_recall"] = [np.sum(metrics["o_a_f_recall"]) / len(metrics["o_a_f_recall"])]
    metrics["o_a_f_fMeasure"] = [(2 * metrics["o_a_f_precision"][0] * metrics["o_a_f_recall"][0]) / (metrics["o_a_f_precision"][0] + metrics["o_a_f_recall"][0])]

    out = pd.DataFrame.from_dict(metrics)
    out.to_csv("./onsets_and_frames_metrics/result.csv")
