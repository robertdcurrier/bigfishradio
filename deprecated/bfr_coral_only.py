#!/usr/bin/env python3
"""
Name:		bfr_bioblitz.py
Author:		robertdcurrier@gmail.com
Created:	2022-11-07
Modified:	2022-11-16
Notes:		This tool loads up a non-annotated melspec.png file
and attempts to find ROIs signifying biologicals. If found bboxes
are drawn around the ROIs. THiS IS ONLY ANNOTATING the raw mels and 
not creating the full plot w/colorbar etc. This is a speedy way of testing
changes to the CORAL settings without having to re-process all the WAV files.

"""
import logging
import time
import sys
import multiprocessing as mp
# Utility imports
from bfr_utils_BETA import (mel_spec, get_config, get_cli_args,
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
	logging.info('do_pngs()')
	pool = mp.Pool()
	pool.map(seek_biologics_png, png_files)
	return len(png_files)


def annotate_melspec(png_file, bboxes):
	"""
	Name:		annotate_melspec
	Author:		robertdcurrier@gmail.com
	Created:	2022-11-16
	Modified:	2022-11-16
	Notes:		Takes bboxes and draws bboxes .
	"""

def bfr_coral_only():
	"""
	Name:		bfr_coral_only.py
	Author:		robertdcurrier@gmail.com
	Created:	2022-11-07
	Modified:	2022-11-16
	Notes:		Main entry point
	"""
	# Get config info
	config = get_config()
	args = get_cli_args()
	target = args['target']

	# Batch process PNG files generating contours and using CORAL
	png_dir = config['targets'][target]['processed_dir']
	png_files = get_melspec_file_names(png_dir)

	start_time = time.time()

	for pfile in png_files:
		bboxes = seek_biologics_png(pfile)
		print(bboxes)

	end_time = time.time()
	minutes = ((end_time - start_time) / 60)
	if len(png_files) > 0:
		logging.info('Processed %d PNG files in %0.2f minutes', len(png_files), minutes)
	else:
 		logging.warning('No PNG files processed')



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    bfr_coral_only()

