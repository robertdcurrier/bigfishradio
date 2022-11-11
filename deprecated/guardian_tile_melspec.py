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
from matplotlib.pyplot import specgram
from librosa import display
from PIL import Image

def resize_image(img_file) -> None:
    """
    Takes pyplot.plt image and saves as
    dedicated 128x128 pixel image
    """
    print("resize_image(): Resizing %s to 70x350" % img_file)
    size = (70, 350)
    try:
        im = Image.open(img_file)
    except IOError as e:
        print("Could not open %s for resizing" % img_file)
        sys.exit()
    try:
        im.resize(size)
        im.save(img_file)
    except:
        print("Could not resize %s" % img_file)

def get_args() -> str:
    """
    Gets command line args
    """
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("-f", "--file", required="true",
                       help="filename of audio file")
    arg_p.add_argument("-o", "--org", required="true",
                       help="Organization")
    arg_p.add_argument("-p", "--project", required="true",
                       help="Project")
    args = vars(arg_p.parse_args())
    return args


def get_config() -> str:
    """
    gets config file for app
    """
    data_file = ("configs/tiles.cfg")
    config = json.loads(open(data_file,'r').read())
    return config


def melspec_tile() -> None:
    """
    DOCSTRING
    """
    config = get_config()
    hop_length = config["hop_length"]
    fmin = config["fmin"]
    fmax = config["fmax"]
    n_fft = config["n_fft"]
    n_mels = config["n_mels"]
    cmap = config["cmap"]
    wav_files = []
    wav_dir = config["chunk_dir"]

    if len(sys.argv) == 1:
        print("NO ARGS. RUNNING IN BATCH MODE.")
        batch = True
    else:
        print("SINGLE FILE MODE.")
        batch = False
        args = get_args()

    if batch:
        wav_files = get_chunk_names(wav_dir)
    else:
        wav_files.append(args["file"])

    for wav_file in wav_files:
        print("melspec_tile(): Generating tile for %s" % wav_file)
        base = os.path.basename(wav_file)
        no_ext = os.path.splitext(base)[0]
        bits, rate = librosa.core.load(wav_file)
        plt.figure(figsize=(.101, .506), dpi=900)
        mel_spec = (librosa.feature.melspectrogram(bits, n_fft=n_fft,
                    hop_length=hop_length, n_mels = n_mels, sr=rate,
                    power=1.0, fmax=fmax))

        mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_spec_db,sr=rate, cmap=cmap, \
                hop_length=hop_length, x_axis='time',y_axis='mel', fmax=fmax)

        if batch:
            save_file = "%s/%s_melspec_tile.png" % (config["tile_dir"], no_ext)
        else:
            save_file = "%s_melspec_tile.png" % (no_ext)

        print("melspec_tile(): Writing %s" % save_file)
        try:
            plt.axis('off')
            plt.savefig(save_file, bbox_inches='tight', dpi=900, pad_inches=0, framecolor='false')
        except IOError as e:
            print("Failed to write %s. Error: %s" (save_file, e))
        plt.close()


def get_chunk_names(chunk_dir) -> list:
    """
    gets wav file names directory tree  and returns as list
    """
    chunk_files = []
    print("get_chunk_names(): %s" % chunk_dir)
    for root, dirs, files in os.walk(chunk_dir):
        if len(files) != 0:
            for chunk_file in files:
                # WAV files only
                if '.wav' in chunk_file:
                    file_name = "%s/%s" % (root, chunk_file)
                    chunk_files.append(file_name)
    return chunk_files


def init_app() -> None:
    """
    Kick it!
    """
    melspec_tile()

if __name__ == '__main__':
    init_app()
