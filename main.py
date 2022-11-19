## Run this program to save MFCC featurs

import librosa
import csv	
import os
import numpy as np

from utils import compute_vad  # GMM based Voice activity detection program

###############################################################################

def makeFiles(path):
    source_path = path  # path to source folder containing the wav files
    s2 = source_path.removeprefix(r'D:\ISL\PR_A2')[1:]
    destination_path = r"D:\ISL\Files\\" + s2
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    n_fft = 320  # window length of stft
    hop_length = 160  # hop length of stft (50% overlap)
    n_mfcc = 13  # number of mfcc to be extracted
    target_sr = 8000  # resampling the wav file to trageted sample rate
    ##################################################################################
    os.chdir(source_path)
    for file in os.listdir(source_path):
        if file.endswith("wav"):
            audio_file = file

            signal, sr = librosa.load(audio_file)  # loading the wav file

            signal_8k = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
            print(signal_8k)
            mfccs = librosa.feature.mfcc(y=signal_8k, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc, sr=target_sr,
                                         center=False)  # mfcc feacture extractation

            delta_mfccs = librosa.feature.delta(mfccs)  # computation of first order delta over the mfcc feature

            delta2_mfccs = librosa.feature.delta(mfccs,
                                                 order=2)  # computation of second order delta over the mfcc feature

            comprehensive_mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
            data = comprehensive_mfccs.T  # transposing the concatenated feature

        vad = compute_vad(signal_8k, win_length=n_fft, win_overlap=hop_length)  # voice activity detection computation

        fn = os.path.splitext(file)[0]  # wav file name extracting from path
        print('creating {}.csv'.format(fn))

        # writing concatenated features of those frames where voice activity is detected into csv file
        with open(destination_path + "/{}.csv".format(fn), 'w', encoding='utf16') as f1:
            writer = csv.writer(f1)
            for i in range(len(vad)):
                if vad[i] == 1:
                    writer.writerow(data[i])

    ########################################################################################


for folder in os.listdir(r"D:\ISL\PR_A2"):
    directory=r"D:\ISL\PR_A2"
    folder=os.path.join(directory , folder)
    for subfolder in os.listdir(folder):
         directory=folder
         subfolder=os.path.join(directory , subfolder)
         makeFiles(subfolder)

