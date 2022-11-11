#!/usr/bin/env python3
"""
Iterates over a supplied directory, loads WAV
files and creates Mel spectrograms for each file.
The spectrogram is named as filename_melspec.png.
Configuration options are set in melspec.cfg
"""
import glob
import os
import sys
import librosa
import json
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import specgram
from librosa import display


def get_config() -> str:
    """
    gets config file for app
    """
    data_file = ("configs/train_image.cfg")
    config = json.loads(open(data_file,'r').read())
    return config


def resize_image(img_file) -> None:
    """
    Takes pyplot.plt image and saves as
    dedicated 128x128 pixel image
    """
    print("resize_image(): Resizing %s to 128x128" % img_file)
    size = 128, 128
    im = Image.open(img_file)
    im.thumbnail(size)
    im.save(img_file)

def training_images() -> None:
    """
    Uses librosa to create bw 128x128 png for files.
    """
    config = get_config()
    hop_length = config["hop_length"]
    fmin = config["fmin"]
    fmax = config["fmax"]
    n_fft = config["n_fft"]
    n_mels = config["n_mels"]
    cmap = config["cmap"]
    wav_dir = config["wav_dir"]
    wav_files = []
    train_dir = config["train_dir"]

    if len(sys.argv) == 1:
        print("NO ARGS. RUNNING IN BATCH MODE.")
        batch = True
    else:
        print("SINGLE FILE MODE.")
        batch = False
        args = get_args()

    if batch:
        wav_files = get_file_names(wav_dir)
        if len(wav_files) == 0:
            print("mel_spec(): No files found.")
            sys.exit()
    else:
        wav_files.append(args["file"])

    for wav_file in wav_files:
        print("training_image(): Generating 128x128 for %s" % wav_file)
        base = os.path.basename(wav_file)
        no_ext = os.path.splitext(wav_file)[0]
        bits, rate = librosa.core.load(wav_file)
        plt.figure(figsize=(1,1), dpi=300)


        mel_spec = (librosa.feature.melspectrogram(bits, n_fft=n_fft,
                    hop_length=hop_length, n_mels = n_mels, sr=rate,
                    power=1.0, fmax=fmax))

        mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_spec_db,sr=rate, cmap='gray_r', \
            hop_length=hop_length, x_axis='time',y_axis='mel', fmax=fmax)

        save_file = "%s/%s_training.png" % (train_dir, base)
        print("training_image(): Writing %s" % save_file)
        try:
            plt.axis('off')
            plt.savefig(save_file,bbox_inches='tight', pad_inches=0, dpi=300, framecolor='false')
        except IOError as e:
            print("Failed to write %s. Error: %s" (save_file, e))
        try:
            resize_image(save_file)
        except IOError as e:
            print("Failed to write %s. Error: %s" (save_file, e))
        plt.close()


def get_args() -> str:
    """
    Gets command line args
    """
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("-f", "--file", required="true",
                       help="filename of audio file")
    arg_p.add_argument("-s", "--start", required="true",
                       help="beginning pos in audio file")
    arg_p.add_argument("-e", "--end", required="true",
                       help="ending  pos in audio file")

    args = vars(arg_p.parse_args())
    return args


def get_file_names(wav_dir) -> list:
    """
    gets wav file names directory tree  and returns as list
    """
    wav_files = []
    print("get_file_names(): %s" % wav_dir)
    for root, dirs, files in os.walk(wav_dir):
        if len(files) != 0:
            for wav_file in files:
                # WAV files only
                if '.wav' in wav_file:
                    file_name = "%s/%s" % (root, wav_file)
                    wav_files.append(file_name)
    return wav_files


if __name__ == '__main__':
    training_images()
