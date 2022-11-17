"""

Author: robertdcurrier@gmail.com
Created:    2022-11-07
Modified:   2022-11-07
Notes:      Utility routines for bfr to avoid duping in many tools
"""
import glob
import logging
import os
import sys
import time
import librosa
import json
import argparse
import itertools
import numpy as np
import cv2 as cv2
import sox as sox
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from librosa import display
from PIL import Image
from natsort import natsorted
from pydub import AudioSegment
from scipy.signal import butter, filtfilt


def get_cli_args():
    """What it say.

    Author: robertdcurrier@gmail.com
    Created:    2018-11-06
    Modified:   2021-12-06

    Notes: Starting out with just -t for target
    """
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("-t", "--target", help="target as defined in config file",
                       required='true')
    args = vars(arg_p.parse_args())
    return args


def get_config() -> str:
    """
    Name:       get_config
    Author:     robertdcurrier@gmail.com
    Created:    2021-11-10
    Modified:   2021-11-10
    Notes:      Gets config file for all settings
    """
    data_file = ("configs/bfr_configs.cfg")
    config = json.loads(open(data_file,'r').read())
    return config


def mel_spec(wav_file, target) -> None:
    """
    Name:       mel_spec
    Author:     robertdcurrier@gmail.com
    Created:    2021-11-10
    Modified:   2022-11-12
    Notes:      Generates mel spec. Way too long; we need to break out
    into some shorter routines -> spec gen, rescale, fig gen.
    """
    config = get_config()
    target = target
    taxa = 'beta'
    # Config settings
    wav_dir = config['targets'][target]["wav_dir"]
    processed_dir = config["targets"][target]["processed_dir"]
    pi = config["targets"][target]["pi"]
    project = config["targets"][target]["project"]
    tmp_dir = config["targets"][target]["tmp_dir"]
    frame_x = config['targets'][target]['frame_x']
    frame_y = config['targets'][target]['frame_y']
    annotated_x = config['targets'][target]['annotated_x']
    annotated_y = config['targets'][target]['annotated_y']
    hop_length = config['targets'][target]["hop_length"]
    spec_fmin = config['targets'][target]["spec_fmin"]
    spec_fmax = config['targets'][target]["spec_fmax"]
    spec_power = config['targets'][target]["spec_power"]
    n_fft = config['targets'][target]["n_fft"]
    n_mels = config['targets'][target]["n_mels"]
    cmap = config['targets'][target]["cmap"]
    base = os.path.basename(wav_file)
    no_ext = os.path.splitext(base)[0]
    fig_x = config['targets'][target]['fig_x']
    fig_y = config['targets'][target]['fig_y']
    spec_fsteps = config['targets'][target]['spec_fsteps']
    dpi = config['targets'][target]['dpi']
    edge_color = config['targets'][target]['taxa'][taxa]["edge_color"]

    # End config settings

    base = os.path.basename(wav_file)
    no_ext = os.path.splitext(base)[0]
    tmp_mel = "%s/%s_mel.png" % (tmp_dir, no_ext)
    # Save in procoessed for AI
    raw_mel = "%s/%s_mel.png" % (processed_dir, no_ext)

    logging.info("mel_spec(): Generating mel spec for %s", wav_file)
  
    bits, rate = librosa.core.load(wav_file)
    plt.figure(figsize=(fig_x, fig_y), dpi=dpi)

    mel_spec = (librosa.feature.melspectrogram(bits, n_fft=n_fft,
                hop_length=hop_length, n_mels = n_mels, sr=rate,
                power=spec_power, fmax=spec_fmax, fmin=spec_fmin))

    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)


    img = librosa.display.specshow(mel_spec_db,sr=rate, \
            hop_length=hop_length, x_axis='time',y_axis='mel',
            fmax=spec_fmax, fmin=spec_fmin, cmap=cmap)

    try:
        plt.axis('off')
        plt.savefig(tmp_mel, bbox_inches='tight', dpi=dpi, pad_inches=0)
        plt.savefig(raw_mel, bbox_inches='tight', dpi=dpi, pad_inches=0)
    except IOError as e:
        logging.warning("Failed to write fig. Error: %s", e)
        plt.close()
    
    # Resize from tmp_mel and raw_mel size so we're cool for ffmpeg
    image = (cv2.imread(tmp_mel))
    image = cv2.resize(image,(frame_x,frame_y))
    cv2.imwrite(tmp_mel, image)
    image = (cv2.imread(raw_mel))
    image = cv2.resize(image,(frame_x,frame_y))
    cv2.imwrite(raw_mel, image)

    fig, ax = plt.subplots()

    img = librosa.display.specshow(mel_spec_db,sr=rate, \
            hop_length=hop_length, x_axis='time',y_axis='mel',
            fmax=spec_fmax, fmin=spec_fmin, cmap=cmap)

    
    bboxes = seek_biologics_png(raw_mel)
    parameters = get_transform_parameters(config, target) 
    for bbox in bboxes:
        (ax1,ay1,aw1,ah1) = transform_axes(bbox, parameters)
        ax.add_patch(Rectangle((ax1, ay1), aw1, ah1,
                        edgecolor = edge_color,
                        fill=False,
                        lw=1))

    fig.colorbar(img, ax=ax, format="%+2.f dB")
    fig.gca().set_ylabel("Hz", fontsize=8)
    fig.gca().set_xlabel("Seconds", fontsize=8)
    fig.gca().set_yticks(range(spec_fmin, spec_fmax, spec_fsteps))

    dts = wav_file.split('/')[1]
    dts = dts.split('_')[0]
    title = "\n%s: %s %s\n" % (pi, project, dts)
    plt.title(title, fontsize=6, horizontalalignment='center')
    
    
    try:
        annotated_out = '%s/%s_annotated.png' % (processed_dir, no_ext)
        plt.axis('on')
        plt.savefig(annotated_out, bbox_inches='tight', dpi=dpi, pad_inches=0)
    except IOError as e:
        logging.warning("Failed to write %s. Error: %s", annotated_out, e)
        plt.close()
    plt.close()
    # Resize so we're cool for viewing
    image = (cv2.imread(annotated_out))
    image = cv2.resize(image,(annotated_x,annotated_y))
    cv2.imwrite(annotated_out, image)
    logging.debug("mel_spec(): Closing fig")
    plt.close()


def get_transform_parameters(config, target):
    """
    Name:       get_transform_parameters
    Author:     robertdcurrier@gmail.com
    Created:    2022-11-14
    Modified:   2022-11-14
    Notes:      Get parameters from config file for performing
                axes transformation: rt, ph, pw, xfactor, yfactor and fmin,
                fmax, etc.

                Return fmin, fmax, xfac, yfac and ph 
    """
    spec_fmin = config['targets'][target]['spec_fmin']
    spec_fmax = config['targets'][target]['spec_fmax']
    pw = config['targets'][target]['frame_x']
    ph = config['targets'][target]['frame_y']
    rt = config['targets'][target]['recording_seconds']
    xfac = pw/rt 
    yfac = ph/spec_fmax # <---- 0 top 320 bottom
    zfac = spec_fmax/ph # <---- 0 bottom 320 top
    parameters = { 
                    "fmin" : spec_fmin,
                    "fmax" : spec_fmax,
                    "ph" : ph,
                    "xfac" : xfac,
                    "yfac" : yfac,
                    "zfac" : zfac
                }

    return parameters


def transform_axes(bbox, parameters):
    """
    Name:       transform_axes
    Author:     robertdcurrier@gmail.com
    Created:    2022-11-11
    Modified:   2022-11-15
    Notes:      Maps x pixels to x seconds and y pixels to y freq 
    Okay, this will make your brain bleed. On the raw mel png we have an x axis of 
    0 to 640 pixels. This has to be mapped to 0 to recording_time seconds for the annotated
    file.  The mel y axis is 0 to 320 pixels, with 0 at the TOP. This has to be mapped
    to fmin to fmax hz, with 0 at the BOTTOM.  We pre-calculate the scaling factors with
    xfac = pixels_wide (640) / recording_time in seconds. Y is MUCH more complicated. First, 
    we add returned bbox height to bbox y1, as we are INVERTED, giving us ay1. 
    This is the starting y point. We then subtract ay1 from pixels_high (320) and multiply
    by yfac. This gives us the bottom end of the y range in hz. We finally use zfac to do
    the inverse to get height in hz by multiplying h1 by zfac. If ay1 is < 0 which can happen
    when we set fmin to a value > 0, we add ay1 to fmin, otherwise ay1 is set to fmin. 
    """
    xfac = parameters["xfac"]
    yfac = parameters["yfac"]
    zfac = parameters["zfac"]
    fmin = parameters["fmin"]
    fmax = parameters["fmax"]
    ph = parameters["ph"]

    x1 = bbox[0]
    y1 = bbox[1]
    w1 = bbox[2]
    h1 = bbox[3]

    ax1 = x1/xfac
    ay1 = y1+h1 #<--- because of inverted scale we have to add h1 to get starting y1
    ay1 = (ph-ay1)*yfac

    if ay1 <=0:
        if fmin > 0:
            ay1 = ay1+fmin
        else:
            ay1 = fmin
    ay1 = ay1
    aw1 = w1/xfac
    ah1 = zfac*h1
    
    logging.debug('parameters are: %0.2f, %0.2f, %d, %d', xfac, yfac, ph, fmin)
    logging.debug("bbox is: %d,%d,%d,%d", x1,y1,w1,h1)
    logging.debug("transformed: %0.4f,%0.4f,%0.4f,%0.4f",ax1, ay1, aw1, ah1)
    return(ax1, ay1, aw1, ah1)


def combine_wav(target) -> None:
    """
    Name:       combine_wav
    Author:     robertdcurrier@gmail.com
    Created:    2021-11-10
    Modified:   2022-01-26
    Notes:      Not currently in use. We now build one movie per wav file
                Update: Going to use to build aggregated movie as this
                is something Will wants. Note: We will need to boost and
                SOX the combined wave file as we're working with the raw
                WAVs to start...
    """
    config = get_config()
    wav_dir = config['targets'][target]['wav_dir']
    wav_out = config['targets'][target]['wav_out']
    fps = config['targets'][target]['fps']

    wav_files = []
    logging.info("combine_wav(): Getting list of wav files...")

    wav_files = get_file_names(wav_dir)
    if len(wav_files) == 0:
        logging.info("combine_wav(): No files found.")
        sys.exit()

    # Create and empty to get us going
    combinedWavFile = AudioSegment.empty()

    # Now iterate
    for wav_file in wav_files:
        logging.info("combine_wav(): Adding %s to combinedWavFile", wav_file)
        combinedWavFile = combinedWavFile + AudioSegment.from_file(wav_file)

    # Get length so we can calculate frame count
    seconds_long = combinedWavFile.duration_seconds
    logging.debug("combine_wav(): combinedWavFile is %d seconds long", seconds_long)
    logging.debug("combine_wav(): Writing combinedWavFile as wav")
    combinedWavFile.export(wav_out, format='wav')


def soxfilter(wav_file, target) -> None:
    """
    Name:       wav_to_mp3
    Author:     robertdcurrier@gmail.com
    Created:    2021-11-08
    Modified:   2022-02-24
    Notes:      Changed name and doing high/low in one def
    """
    config = get_config()
    wav_dir = config['targets'][target]["wav_dir"]
    fps = config['targets'][target]['fps']
    lowpass = int(config['targets'][target]['lowpass'])
    highpass = int(config['targets'][target]['highpass'])
    base = os.path.basename(wav_file)
    no_ext = os.path.splitext(base)[0]
    sox_out = "tmp/%s_sox.wav" % no_ext
    sox_tmp = "tmp/%s_sox_tmp.wav" % no_ext


    # Run sox lowpass
    logging.info('soxfilter(): Running lowpass on %s with cutoff %d', wav_file, lowpass)
    tfm = sox.Transformer()
    tfm.lowpass(lowpass)
    tfm.build_file(wav_file, sox_out)
    return sox_out


def get_wav_file_names(wav_dir) -> list:
    """
    gets wav file names directory tree  and returns as list
    """
    wav_files = []
    logging.debug("get_wav_file_names(): %s", wav_dir)
    for root, dirs, files in os.walk(wav_dir):
        if len(files) != 0:
            for wav_file in files:
                # WAV files only
                if '.wav' in wav_file:
                    file_name = "%s/%s" % (root, wav_file)
                    wav_files.append(file_name)
    wav_files = natsorted(wav_files)
    return wav_files


def get_melspec_file_names(png_dir) -> list:
    """
    gets melspec.png file names directory tree and returns list 
    of non-annotated PNG files. 
    """
    png_files = []
    logging.debug("get_melspec_file_names(): %s", png_dir)
    for root, dirs, files in os.walk(png_dir):
        if len(files) != 0:
            for png_file in files:
                # Non-annotated PNG files only
                if 'sox_mel' in png_file:
                    file_name = "%s/%s" % (root, png_file)
                    png_files.append(file_name)
    png_files = natsorted(png_files)
    return png_files


def clean_tmp_files():
    """
    Name: clean_tmp_files
    Author: robertdcurrier@gmail.com
    Created: 2021-11-09
    Modified: 2021-11-10
    Notes: Empties tmp folder prior to beginning run
    """
    files = glob.glob('tmp/*')
    for file in files:
        os.remove(file)
    logging.debug("clean_tmp_files(): Removed all files in tmp")


def ffmpeg_it(wav_file, target):
    """
    Name: ffmpeg_it
    Author: robertdcurrier@gmail.com
    Created: 2021-11-09
    Modified: 2021-11-10
    Notes: Incorporated into this code to eliminate need for
    shell script.  For now we use the os command but plan on
    integrating into ffmpeg library for Python3
    """
    logging.info('ffmpeg_it(%s)', wav_file)
    config = get_config()
    processed_dir = config['targets'][target]['processed_dir']
    base = os.path.basename(wav_file)
    no_ext = os.path.splitext(base)[0]
    target = target
    wav_dir = config['targets'][target]["wav_dir"]
    sox_in = "%s/%s_boosted_sox.wav" % (processed_dir, no_ext)
    processed_dir = config["targets"][target]["processed_dir"]
    recording_seconds = int(config['targets'][target]['recording_seconds'])
    frame_x = int(config['targets'][target]['frame_x'])


    loop_command = """ffmpeg -loglevel quiet -y -loop 1 -t %d -i tmp/%s_final.png -vf "crop=w=%d:h=ih:x='(iw-%d)*t/%d':y=0" -r 24 -pix_fmt yuv420p tmp/%s_frames.mp4""" % (recording_seconds, no_ext, frame_x, frame_x, recording_seconds, no_ext)
    logging.debug(loop_command)
    os.system(loop_command)
    logging.debug('ffmpeg_it(): Synchronizing audio and video for %s', wav_file)
    mix_command = """ffmpeg -loglevel quiet -y -i tmp/%s_frames.mp4  -i tmp/%s.wav -vf scale=%d:320 -framerate 24 -c:v libx264 -pix_fmt yuv420p -profile:v baseline -level 3.0 -crf 20 -preset veryslow -c:a aac -strict experimental -movflags +faststart -threads 0 %s/%s_processed.mp4""" % (no_ext, no_ext, frame_x, processed_dir, no_ext)
    os.system(mix_command)
    logging.debug(mix_command)
    logging.info('ffmpeg_it(): Finished building %s_final.mp4', no_ext)


def boost_audio(wav_file, target):
    """
    Name:       boost_audio
    Author:     robertdcurrier@gmail.com
    Created:    2021-11-11
    Modified:   2021-11-11
    Notes:      Boosts audio files by 'boost' dB.
    """
    config = get_config()
    base = os.path.basename(wav_file)
    no_ext = os.path.splitext(base)[0]
    boost = config['targets'][target]['boost']
    logging.info("boost_audio(): Boosting %s by %d dB", wav_file, boost)
    try:
        audio = AudioSegment.from_wav(wav_file)
        audio = audio + boost
    except:
        logging.warning('boost_audio(): Failed to boost %s', wav_file)
    boost_file = 'tmp/%s_boosted.wav' % no_ext
    audio.export(boost_file,format='wav')
    return(boost_file)


def header_footer(wav_file, target):
    """
    Notes: Loads tmp_mel and adds header and footer.
    Modified: 2021-11-10
    """
    logging.debug("header_footer(): Loading %s", wav_file)
    config = get_config()
    wav_dir = config['targets'][target]["wav_dir"]
    tmp_dir = config['targets'][target]["tmp_dir"]
    processed_dir = config["targets"][target]["processed_dir"]
    fps = config['targets'][target]['fps']

    base = os.path.basename(wav_file)
    no_ext = os.path.splitext(base)[0]
    tmp_mel = "%s/%s_mel.png" % (tmp_dir, no_ext)
    final_out = "%s/%s_final.png" % (tmp_dir, no_ext)
    frame_x = config['targets'][target]['frame_x']
    frame_y = config['targets'][target]['frame_y']
    header_x = config['targets'][target]['header_x']
    footer_x = config['targets'][target]['footer_x']

    header = config['targets'][target]["header"]
    footer = config['targets'][target]["footer"]

    try:

        image = (cv2.imread(tmp_mel))
        header = (cv2.imread(header))
        image = cv2.resize(image,(frame_x,frame_y))
        header = cv2.resize(header,(header_x,frame_y))
        image = np.append(header, image, axis=1)
        cv2.imwrite(final_out, image)
    except Exception as e:
        logging.warning("header_footer(): Encountered error %s", e)
        sys.exit()


def seek_biologics_wav(wav_file):
    """
    Name:       seek_biologics_wav
    Author:     robertdcurrier@gmail.com
    Created:    2022-11-07
    Modified:   2022-11-07
    Notes:      Hunts for biological signatures in wav files.  This will be moved
    to brf_utils.py when fully debugged. 
    """
    logging.info('seek_biologics(%s)', wav_file)
    audio = AudioSegment.from_wav(wav_file)
    # Here is where the magic lives...


def seek_biologics_png(png_file):
    """
    Name:       seek_biologics_png
    Author:     robertdcurrier@gmail.com
    Created:    2022-11-07
    Modified:   2022-11-16
    Notes:      Hunts for biological signatures using CORAL. We removed the annotation
    code and now this function returns only bounding boxes. Annotation is done elsewhere. 
    """

    config = get_config()
    args = get_cli_args()
    target = args['target']
    taxa = "beta"

    debug_dir = config['targets'][target]['debug_dir']
    logging.info('seek_biologics_png(%s)', png_file)
        
    try:
        img = cv2.imread(png_file)
    except:
        logging.warning('load_image(): Failed to open %s' % png_file)
        return
    
    contours = gen_cons(png_file)
    circ_cons = gen_coral(img.copy(), contours, png_file)
    bboxes = gen_bboxes(circ_cons)
    
    rect_color = eval('%s' % config['targets'][target]['taxa'][taxa]["rect_color"])
   
    line_thick = config['targets'][target]['taxa'][taxa]["line_thick"]
    y1_max = config['targets'][target]['taxa'][taxa]["y1_max"]
    for bbox in bboxes:
        x1 = bbox[0]
        y1 = bbox[1]
        width = bbox[2]
        height = bbox[3]
        x2 = x1+width
        y2 = y1+height
    
        if y1 > y1_max:
            cv2.rectangle(img,(x1,y1),(x2,y2), (rect_color), line_thick)    
    (root, fname) = os.path.split(png_file)
    no_ext = os.path.splitext(fname)[0]
    cons_f = "%s/%s_CONS.png" % (debug_dir, no_ext)
    logging.debug('seek_biologics_png(): Writing contours %s', cons_f)
    cv2.imwrite(cons_f, img)
    return bboxes


def gen_coral(img, cons, png_file):
    """
    Name:       gen_coral
    Author:     robertdcurrier@gmail.com
    Created:    2022-07-11
    Modified:   2022-11-08
    Notes:      Iterates over PNG. Returns circle cons
    for generating bounding boxes.
    """
    logging.debug('gen_coral(%s)', png_file)
    config = get_config()
    args = get_cli_args()
    target = args['target']
    taxa = "beta"
    debug = config['targets'][target]["debug"]

    debug_dir = config['targets'][target]['debug_dir']
    line_thick = config['targets'][target]['taxa'][taxa]['line_thick']
    radius_boost = config['targets'][target]['taxa'][taxa]['radius_boost']

    # circles
    circle_img = img.copy()
    for con in cons:
        (x,y), radius = cv2.minEnclosingCircle(con)
        center = (int(x), int(y))
        radius = int(radius+radius_boost)
        cv2.circle(circle_img, center, radius, (0,0,0), -1)
    
    edges_min = config['targets'][target]['taxa'][taxa]['coral_edges_min']
    edges_max = config['targets'][target]['taxa'][taxa]['coral_edges_max']
    thresh_min = config['targets'][target]['taxa'][taxa]['thresh_min']
    thresh_max = config['targets'][target]['taxa'][taxa]['thresh_max']

    gray  = cv2.cvtColor(circle_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), cv2.BORDER_WRAP)
    threshold = cv2.threshold(blurred, thresh_min, thresh_max,
                              cv2.THRESH_BINARY)[1]
    edges = cv2.Canny(threshold, edges_min, edges_max)
    circ_cons, _ = (cv2.findContours(edges, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE))
    coral_img = cv2.drawContours(img, circ_cons, -1, (0,255,0), 2)
    logging.debug('gen_coral(%s): found %d circ_cons', png_file, len(circ_cons))
    (root, fname) = os.path.split(png_file)
    no_ext = os.path.splitext(fname)[0]

    if debug:
        cons_f = "%s/%s_CONS.png" % (debug_dir, no_ext)
        edges_f = "%s/%s_EDGES.png" % (debug_dir, no_ext)
        coral_f = "%s/%s_CORAL.png" % (debug_dir, no_ext)
        cv2.imwrite(cons_f, img)
        cv2.imwrite(edges_f, edges)
        cv2.imwrite(coral_f, coral_img)
    return(circ_cons)


def gen_cons(png_file):
    """
    Name:       gen_cons
    Author:     robertdcurrier@gmail.com
    Created:    2022-07-04
    Modified:   2022-11-07
    Notes:      Back to contours and edges. Mask works great in the
                lab with clear water but barfs in the wild. Another negative
                for mask is inability to deal with lighting variations.
    """
    # Taxa settings -- not in use during beta
    args = get_cli_args()
    target = args['target']
    taxa = "beta"

    config = get_config()
    edges_min = config['targets'][target]['taxa'][taxa]['con_edges_min']
    edges_max = config['targets'][target]['taxa'][taxa]['con_edges_max']
    

    logging.debug('gen_cons(%s)' % png_file)

    img = frame = cv2.imread(png_file)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), cv2.BORDER_WRAP)
    edges = cv2.Canny(blurred, edges_min, edges_max)
    contours, _ = (cv2.findContours(edges, cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_NONE))
    (root, fname) = os.path.split(png_file)
    no_ext = os.path.splitext(fname)[0]
    return contours


def gen_bboxes(cons):
    """
    Name:       gen_bboxes
    Author:     robertdcurrier@gmail.com
    Created:    2022-07-11
    Modified:   2022-11-09
    Notes:      
    """
    args = get_cli_args()
    target = args['target']
    config = get_config()
    taxa = "beta"
    bboxes = []
    good_cons = []
    ncons = len(cons)
    logging.debug('gen_bboxes(): %d circ_cons' % ncons)

    min_roi_area = config['targets'][target]['taxa'][taxa]['min_roi']
    max_roi_area = config['targets'][target]['taxa'][taxa]['max_roi']
    for con in cons:
        rect = cv2.boundingRect(con)
        x1 = rect[0]
        y1 = rect[1]
        width = rect[2]
        height = rect[3]
        x2 = x1+width
        y2 = y1+height
        
        area = width*height
        logging.debug('gen_bboxes() Area: %d', area)
        if area > min_roi_area and area < max_roi_area:
            bboxes.append([x1,y1,width,height])
    bboxes = list(bboxes for bboxes,_ in itertools.groupby(bboxes))
    logging.debug('gen_bboxes(): Found %d ROIs' % (len(bboxes)))
    return (bboxes)


def do_singles(file) -> None:
    """
    Created:    2021-12-06
    Author:     robertdcurrier@gmail.com
    Modified:   2022-01-26
    Notes:      This replaces the for file in loop we used previously
    """
    args = get_cli_args()
    # Single files
    target = args['target']
    logging.debug('do_singles(): processing file %s', file)
    boost_file = boost_audio(file, target)
    sox_file = soxfilter(boost_file, target)
    # need to run spec on SOX file, not just BOOST file yo!
    mel_spec(sox_file, target)
    header_footer(sox_file, target)
    ffmpeg_it(sox_file, target)
    logging.debug('bfr(): Finished single file processing %s', file)
