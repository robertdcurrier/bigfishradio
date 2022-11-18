#!/usr/bin/env python3
"""
bfr_create_movie.py
This app takes PA audio, writes out WAV
and MP3 files,  generates MEL spectrograms for each file and concatenates
the spectrograms into one image. This image is then sliced into
n frames, where n  is the number of needed frames to match the
audio at 24FPS. Using ffmpeg a movie is created from the frames
and synched with the audio into an MP4 container.

TO DO: Integrate the ffmpeg code in create_movie.sh into the python3
source. Right now we call ffmpeg from the shell.
"""
import logging
import time
import sys
import multiprocessing as mp
# Utility imports
from bfr_utils_BETA import (mel_spec, get_config, clean_tmp_files, get_cli_args,
get_wav_file_names, do_singles, create_report_db)


def bfr() -> None:
    """
    Created:    2021-11-08
    Author:     robertdcurrier@gmail.com
    Modified:   2022-11-07
    Notes:      Moving to doing single file movies...
    """
    config = get_config()
    args = get_cli_args()
    target = args['target']
    try:
        wav_dir = config['targets'][target]["wav_dir"]
    except KeyError:
        logging.warning('Invalid target %s', target)
        sys.exit()    
    # set up logging
    create_report_db(target)

    wav_files = []
    wav_files = get_wav_file_names(wav_dir)
    num_files = len(wav_files)
    # Use Multiprocessing to expedite across all cores
    pool = mp.Pool()
    pool.map(do_singles, wav_files)
    # For testing
    #for file in wav_files:
    #    log_entry = do_singles(file)
    #    detections.append(log_entry)
    return num_files


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    start_time = time.time()
    clean_tmp_files()
    num_files = bfr()
    clean_tmp_files()
    end_time = time.time()
    minutes = ((end_time - start_time) / 60)
    logging.info('Processed %d files in %0.2f minutes', num_files, minutes)
