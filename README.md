# bigfishradio


## What it does

* Built as Docker Stack with a single container
* Requires ffmpeg, ffmpy, librosa, matplotlib, natsort, pydub and opencv
* Reads WAV files from passive acoustic installation 
* Concatenates list of WAV files into single file
* Writes concatenated audio as WAV and MP3 files
* Generates mel spectrogram of concatenated audio file
* Resizes mel spectrogram
* Calculates number of frames needed: n = audio_length_secondsx24 fps
* Chops mel spectrogram into n frames
* Uses ffmpeg to concatenate frames into MP4
* Uses ffmpeg to combine MP4 and MP3 into QT-compliant MP4 container
* Rules the universe


## Feedback

This is a work in progress and will change on a daily basis

## License

Copyright 2019 GCOOS-RA, Inc.

Licensed to the Apache Software Foundation (ASF) under one or more contributor
license agreements. See the NOTICE file distributed with this work for
additional information regarding copyright ownership. The ASF licenses this
file to you under the Apache License, Version 2.0 (the “License”); you may not
use this file except in compliance with the License. You may obtain a copy of
the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
# gandalf-docker
# bigfishradio
# bigfishradio
