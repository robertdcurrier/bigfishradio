#!/bin/bash
#
#This works great for Will's 20 second files. Note: We'll need to have a 160px
#black footer appended so we move the spectrogram until the last of it rolls 
#out of stage left.
#
#Loop it
ffmpeg -y -loop 1 -t 60 -i tmp/0002_DSG_DECD_HMS_17_30_0_DMY_3_2_20_boosted_final.png -vf "crop=w=640:h=ih:x='(iw-640)*t/60':y=0" -r 24 -pix_fmt yuv420p tmp/test.mp4

#Mix it
#ffmpeg -y -i tmp/test.mp4  -i tmp/20200212T170000_boosted.mp3 -vf scale=640:320 -framerate 24 -c:v libx264 -pix_fmt yuv420p -profile:v baseline -level 3.0 -crf 20 -preset veryslow -c:a aac -strict experimental -movflags +faststart -threads 0 tmp/test_mixed.mp4
