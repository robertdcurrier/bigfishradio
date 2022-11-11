#!/usr/bin/env python3
"""
Stitches and slides, yo
"""
import sys
import os
import glob
import json
import argparse
import cv2
from PIL import Image
from natsort import natsorted
import numpy as np


DISPLAY = False


def get_file_names(config) -> list:
    """
    gets png file names
    """
    img_dir = config["image_dir"]
    img_files = []
    print("get_file_names(): %s" % img_dir)
    for root, dirs, files in os.walk(img_dir):
        if len(files) != 0:
            for img_file in files:
                # PNG files only
                if 'mel.png' in img_file:
                    file_name = "%s/%s" % (root, img_file)
                    img_files.append(file_name)
    print("get_file_names() found %d images" % len(img_files))
    img_files = natsorted(img_files)
    return img_files


def load_stitch(config, img_files):
    """
    Opens image files and fails if not
    returns stitched PIL image
    """
    np_files = []
    header = cv2.imread(config["header"])
    np_files.append(header)
    for imfile in img_files:
        print(imfile)
        np_files.append(cv2.imread(imfile))
    # Do header footer here
    #footer = cv2.imread(config["footer"])
    #np_files.append(footer)
    stitched = np.array(np_files)
    #stitched = cv2.hconcat(stitched)
    #return stitched
    cv2.imwrite('BAPNG.png', stitched) 

def clean_frames() -> None:
    """
    Notes: Empties frame_dir prior to run.
    Modified: 2020-03-19
    """
    config = get_config()
    spec_dir = config['frame_dir']
    path = '%s/*.png' % spec_dir
    print("clean_frames(): Removing %s" % path)
    files = glob.glob(path)
    for png_file in files:
        os.remove(png_file)
    print("clean_frames(): Removed all files in %s" % path)


def slide_win(config, stitched):
    """
    Slides window across stitched
    writing out frames as it goes
    """
    frame_dir = config["frame_dir"]
    frame_num = 0
    hpix = stitched.shape[1]
    print(stitched.shape[1])
    xstart = 240
    xend =   720
    shift = 1
    # First frame
    fname = ("%s/frame_%04d.png" % (frame_dir, frame_num))
    frame = stitched[0:240, xstart:xend]
    cv2.imwrite(fname, frame)
    while xend < hpix:
        frame_num += 1
        xstart += shift
        xend += shift
        frame = stitched[0:240, xstart:xend]
        fname = ("%s/frame_%04d.png" % (frame_dir, frame_num))
        print("slide_win(): Writing %s" % fname)
        cv2.imwrite(fname, frame)

def get_config() -> str:
    """
    gets config file for app
    """
    data_file = ("configs/stitchslide.cfg")
    config = json.loads(open(data_file,'r').read())
    return config


def init_app() -> None:
    """
    Kick it!
    """
    config = get_config()
    print(config)
    clean_frames()
    img_files = get_file_names(config)
    if len(img_files) == 0:
        print("init_app(): No files found.")
        sys.exit()
    stitched = load_stitch(config, img_files)
    slide_win(config, stitched)

if __name__ == '__main__':
    init_app()
