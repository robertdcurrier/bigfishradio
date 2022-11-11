#!/usr/bin/env python3
"""
Takes as input a directory path and a year. If the directory/year
combo doesn't exist it is created. If it does, the create fails
and the program exits with an error.

Once the path is created, full-name months are populated
in the directory/year path.

Days are created in each /directory/year/month path based
on the number of days in the month.

Finally, hours 0-24 are created in each day of /directory/year/month.

Notes:
    2019-08-22 Added -c arg so we don't have to rm -rf data structure
    each time we want to copy files.
"""
import os
import sys
import json
import glob
import datetime
import argparse
import numpy as np
from calendar import monthrange
from time import strptime
from shutil import copyfile

DEBUG = True

def new_struct(data_dir) -> None:
    """
    Takes path, looks to see if it
    exists, if so, return fail. If not, build
    Year -> [Jan - Dec] -> DOM -> [0-23] structure
    """
    full_path='%s' % (data_dir)
    months = ['January', 'February', 'March', 'April', 'May',
              'June', 'July', 'August', 'September', 'October',
              'November', 'December']

    print("new_struct(): Creating %s" % full_path)
    try:
        os.makedirs(full_path)
    except OSError as e:
        print("Failed to create %s. Error: %s" % (full_path, e))
        sys.exit()
    os.chdir(full_path)
    # Months
    for month in months:
        try:
            os.mkdir(month)
        except OSError as e:
            print("Failed to create %s. Error: %s" % (month, e))
            sys.exit()
    # Days
    for month in months:
        year = data_dir.split('/')[-1]
        maxdays = monthrange(int(year), strptime(month, '%B').tm_mon)[1]
        os.chdir('%s/%s' % (full_path, month))
        print("new_struct(): Populating %s with days" % month)
        day = 1
        hours = range(0,24)
        while day <= maxdays:
            try:
                os.mkdir(str("%02d") % day)
            except OSError as e:
                print("Failed to create %s. Error: %s" % (month, e))
                sys.exit
            # Hours
            for hour in hours:
                hour = "%02d" % (hour)
                try:
                    os.mkdir('%s/%s/%s/%s' % (full_path, month, (str("%02d") % day), hour))
                except OSError as e:
                    print("Failed to create %s. Error: %s" % (month, e))
                    sys.exit()
            day+=1


def get_args():
    """
    Gets command line args
    """
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("-c", "--create", action="store_true",
                       help="Create data structure")

    args = vars(arg_p.parse_args())
    return args


def get_config() -> str:
    """
    gets config file for app
    """
    data_file = ("configs/data_struct.cfg")
    config = json.loads(open(data_file,'r').read())
    return config

def parse_timestruct(data_dir, wav_files) -> None:
    """
    Reads basename of file, gets year/month/day/hour,
    moves to proper folder
    """
    year = data_dir.split('/')[-1]
    print(("parse_timestruct(): Copying %d files to %s" %
          (len(wav_files), data_dir)))
    for wav_file in wav_files:
        base = os.path.basename(wav_file)
        epoch = int(os.path.splitext(base)[0])
        year = (datetime.datetime.fromtimestamp(epoch).strftime("%Y"))
        month = (datetime.datetime.fromtimestamp(epoch).strftime("%B"))
        day = (datetime.datetime.fromtimestamp(epoch).strftime("%d"))
        hour = (datetime.datetime.fromtimestamp(epoch).strftime("%H"))
        destination = "%s/%s/%s/%s/%s" % (data_dir,month,day,hour,base)
        if DEBUG:
            print("parse_timestruct(%s)" % wav_file)
            print(("parse_timestruct(): Moving %s to %s" %
                  (wav_file, destination)))
        try:
            copyfile(wav_file, destination)
        except IOError as e:
            print("Failed to copy %s" % wav_file)
            sys.exit()


def get_file_names(raw_dir) -> list:
    """
    gets wav file names in data_dir and returns as list
    """
    wav_files = []
    print("get_file_names(): %s" % raw_dir)
    file_names = glob.glob(raw_dir + "/*.wav")
    file_names.sort(key=os.path.getmtime)
    for wav_file in file_names:
        wav_files.append(wav_file)
    return wav_files

def init_app() -> None:
    """
    kick it!
    """
    args = get_args()
    config = get_config()
    print('%s %s' % (config['title'], config['version']))
    data_dir = config['data_dir']
    if args["create"]:
        print("CREATING NEW DATA STRUCTURE")
        new_struct(data_dir)

    print("COPYING FILES TO EXISTING DATA STRUCTURE...")
    raw_dir = config['raw_dir']
    wav_files = get_file_names(raw_dir)
    parse_timestruct(data_dir, wav_files)

if __name__ == '__main__':
    init_app()
