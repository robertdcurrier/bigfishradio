#!/usr/bin/env python
import os
recording_seconds = 380
frame_x = 12160
frame_y = 320
loop_command = """ffmpeg  -loop 1 -i 'tmp/combinedPNG.png' -vf "crop=w=%d:h=ih:x='(iw-%d)*t/%d':y=0" -r 25 -pix_fmt yuv420p out.mp4""" % (frame_x, frame_x, recording_seconds)
os.system(loop_command)

