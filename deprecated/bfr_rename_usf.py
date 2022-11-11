#!/usr/bin/env python3
"""
One-off tool for renaming files from Chad and Catherine
"""
import os
import shutil

file_count = 0

target_dir = '/data/bigfishradio/raw/projects/USF_Right_Whales'
for folderName, subfolders, filenames in os.walk(target_dir):
    print('The current folder is ' + folderName)

for filename in sorted(filenames):
    # get prefix
    base = os.path.splitext(filename)[0]
    print(base)
    fixed = base.replace('.', '_')
    fixed = fixed.replace(' ', '_')
    fixed = fixed.replace('__','_')
    final = '/data/bigfishradio/raw/projects/USF_Right_Whales/%s.wav' % fixed
    shutil.move(folderName +'/'+ filename, final)
    print('Converted %s to %s' % (filename, final))
    file_count += 1

print('Renamed %d files...' % file_count)

