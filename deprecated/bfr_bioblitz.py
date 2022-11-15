#!/usr/bin/env python3
"""
Name:		bfr_bioblitz.py
Author:		robertdcurrier@gmail.com
Created:	2022-11-07
Modified:	2022-11-07
Notes:		This tool loads up a non-annotated melspec.png file
and attempts to find ROIs signifying biologicals. If found bboxes 
are drawn around the ROIs. Not sure if CORAL will be useful for this
(optically detecting) or if we need to perform signal analysis on
the raw WAV data. (power spectrum detecting.)  Let the hacking commence!
"""
import logging
import time
import sys
import multiprocessing as mp
# Utility imports
from bfr_utils import (mel_spec, get_config, get_cli_args,
get_melspec_file_names, get_wav_file_names, seek_biologics_wav,
seek_biologics_png)


def do_wavs(wav_files):
	"""
	"""
	# Takin' a dip in the MP pool...
	pool = mp.Pool()
	pool.map(seek_biologics_wav, wav_files)
	return len(wav_files)


def do_pngs(png_files):
	"""
	"""
	# Takin' a dip in the MP pool...
	pool = mp.Pool()
	pool.map(seek_biologics_png, png_files)
	return len(png_files)


def bfr_bioblitz():
	"""
	Name:		bfr_bioblitz.py
	Author:		robertdcurrier@gmail.com
	Created:	2022-11-07
	Modified:	2022-11-07
	Notes:		Main entry point
	"""
	# Get config info
	config = get_config()
	args = get_cli_args()
	target = args['target']


	"""
	NOTE: Not batching WAVs while we work on CORAL/ROI for PNG
	# Batch process WAVs generating MP4 and PNG files
	wav_dir = config['targets'][target]['wav_dir']
	wav_files = get_wav_file_names(wav_dir)
	start_time = time.time()
	num_files = do_wavs(wav_files)
	end_time = time.time()
	minutes = ((end_time - start_time) / 60)
	if num_files > 0:
		logging.info('Processed %d WAV files in %0.2f minutes', num_files, minutes)
	else:
		logging.warning('No WAV files processed')
	"""
	
	# Batch process PNG files generating contours and using CORAL
	png_dir = config['targets'][target]['processed_dir']
	png_files = get_melspec_file_names(png_dir)
	
	start_time = time.time()
	num_files = do_pngs(png_files)
	end_time = time.time()
	minutes = ((end_time - start_time) / 60)
	if num_files > 0:
		logging.info('Processed %d PNG files in %0.2f minutes', num_files, minutes)
	else:
 		logging.warning('No PNG files processed')



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    bfr_bioblitz()
    