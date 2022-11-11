#!/usr/bin/env python3
"""
"""
import os
import sys
import librosa
import json
import argparse
import numpy as np
import pydub


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
    arg_p.add_argument("-b", "--boost", required="true",
                               help="boost in DBs")
    args = vars(arg_p.parse_args())
    return args


def get_config() -> str:
    """
    gets config file for app
    """
    data_file = ("configs/boost.cfg")
    config = json.loads(open(data_file,'r').read())
    return config


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



def boost_audio(wav_file, boost):
    """
    DOCSTRING
    """
    print("boost_audio(): %s" % wav_file)
    audio = pydub.AudioSegment.from_wav(wav_file)
    audio = audio + boost
    audio.export(wav_file,format='wav')



def init_app() -> None:
    """
    Kick it!
    """
    # Single file
    if len(sys.argv) > 1:
        args = get_args()
        wav_file = args["file"]
        boost = args["boost"]
        boost_audio(wav_file, boost)
    else:
        # Batch mode
        print("BATCH MODE")
        config = get_config()
        boost = config['boost']
        wav_files = get_file_names(config['wav_dir'])
        for wav_file in wav_files:
            boost_audio(wav_file, boost)

if __name__ == '__main__':
    init_app()


