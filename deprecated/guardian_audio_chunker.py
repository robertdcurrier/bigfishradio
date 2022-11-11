#!/usr/bin/env python3
"""
Audio chunker for GUARDIAN testing. This app takes an
input file and carves up into n-second chunks w/n being
defined in chunker.cfg.  The chunks are named as follows:
org_project_beginningEpoch_chunk.wav Example:
cobi_SNAP_1567014982_1.wav and cobi_SNAP_1567014982_2.wav
"""
import os
import sys
import librosa
import json
import argparse
import numpy as np


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
    arg_p.add_argument("-c", "--chunk_size", required="true",
                               help="Chunksize")
    arg_p.add_argument("-d", "--data_dir", required="true",
                                   help="Data_dir")
    args = vars(arg_p.parse_args())
    return args


def get_config() -> str:
    """
    gets config file for app
    """
    data_file = ("configs/chunker.cfg")
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


def chunk_file(wav_file, config) -> None:
    """
    Uses librosa to chunk file.
    Parameters are obtained from chunker.cfg
    """
    org = config["org"]
    project = config["project"]
    chunk_size = int(config["chunk_size"])

    print("chunk_file(): Chunking %s into %d second chunks" %
          ( wav_file, chunk_size))
    base = os.path.basename(wav_file)
    no_ext = os.path.splitext(base)[0]
    bits, rate = librosa.core.load(wav_file)
    chunks = round(len(bits)/(chunk_size*rate))
    print("chunk_file(): %s has %d bits and a rate of %d bps" %
          (wav_file, len(bits), rate))
    bits_per_chunk = int(len(bits)/chunks)
    start = 0
    end = bits_per_chunk


    for chunk_num in range(chunks):
        # don't want a chunk_num of 0 so we add 1
        chunk = bits[start:end]
        start = start + bits_per_chunk
        end = end + bits_per_chunk
        data_dir = config['data_dir']
        file_name = "%s/%s_%s_%s_chunk_%d.wav" % (data_dir, org, project,
                                               no_ext, chunk_num+1)
        try:
            librosa.output.write_wav(file_name, chunk, rate)
        except IOError as e:
            print("chunk_file(): Failed to write %s. Error: %s" %
                  (file_name, e))


def init_app() -> None:
    """
    Kick it!
    """
    # Single file
    if len(sys.argv) > 1:
        args = get_args()
        wav_file = args["file"]
        chunk_file(wav_file, args)
    else:
        # Batch mode
        print("BATCH MODE")
        config = get_config()
        wav_files = get_file_names(config['wav_dir'])
        for wav_file in wav_files:
            chunk_file(wav_file, config)

if __name__ == '__main__':
    init_app()


