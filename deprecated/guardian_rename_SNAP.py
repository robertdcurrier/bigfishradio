#!/usr/bin/env python3
"""
One-off tool for renaming files from Stu Fulton.
The files lost their date/time info when being copied
to Google Drive but Stu knew the exact timestamp for the
first and last files as well as the increment. This code
renames as epochtime.wav using the calculated epoch time.
"""
import os
import shutil

file_count = 0
epoch_start = 1547787720
time_bump = 320

target_dir = '/data/guardian/bigfish/raw'
for folderName, subfolders, filenames in os.walk(target_dir):
    print('The current folder is ' + folderName)

for filename in sorted(filenames):
    # get prefix
    base = os.path.splitext(filename)[0]
    newfile = filename.replace(base, str(epoch_start))
    shutil.move(folderName +'/'+filename, folderName + '/' + newfile)
    print('Converted ' + folderName + ': '+ filename + ' to ' + newfile)
    file_count += 1
    epoch_start = epoch_start + time_bump

print('Renamed %d files...' % file_count)

