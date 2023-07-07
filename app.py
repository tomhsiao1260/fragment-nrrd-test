import os
import json
import glob
import nrrd
import math
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

if not os.path.exists('config.json'):
    print('config.json not found')
    exit()

# config & path
with open('config.json') as f:
    config = json.load(f)

CLIP        = config['CLIP']
TIF_INPUT   = config['TIF_INPUT']
LABEL_INPUT = config['LABEL_INPUT']
MASK_INPUT  = config['MASK_INPUT']

RAW_SAMPLING  = config['RAW_SAMPLING']
TIF_SAMPLING  = config['TIF_SAMPLING']
SUBCLIP_LIMIT = config['SUBCLIP_LIMIT']

NRRD_OUTPUT = './output/volume'
NRRD_INFO   = './output/volume/meta.json'

def nrrd_list(CLIP, SUBCLIP_LIMIT):
    NRRD_LIST = []
    NUM_W = math.ceil(CLIP['w'] / SUBCLIP_LIMIT['w'])
    NUM_H = math.ceil(CLIP['h'] / SUBCLIP_LIMIT['h'])
    NUM_D = math.ceil(CLIP['d'] / SUBCLIP_LIMIT['d'])
    DIGIT_NUM = len(str(NUM_W * NUM_H * NUM_D))
    SUBCLIP_ID = 0

    mask = Image.open(MASK_INPUT)
    label = Image.open(LABEL_INPUT)

    for i in range(NUM_W):
        for j in range(NUM_H):
            for k in range(NUM_D):
                SUB_CLIP = CLIP.copy()
                SUB_CLIP['x'] = CLIP['x'] + SUBCLIP_LIMIT['w'] * i
                SUB_CLIP['y'] = CLIP['y'] + SUBCLIP_LIMIT['h'] * j
                SUB_CLIP['z'] = CLIP['z'] + SUBCLIP_LIMIT['d'] * k
                SUB_CLIP['w'] = SUBCLIP_LIMIT['w']
                SUB_CLIP['h'] = SUBCLIP_LIMIT['h']
                SUB_CLIP['d'] = SUBCLIP_LIMIT['d']

                if (i == NUM_W - 1):
                    SUB_CLIP['w'] = CLIP['x'] + CLIP['w'] - SUB_CLIP['x']
                if (j == NUM_H - 1):
                    SUB_CLIP['h'] = CLIP['y'] + CLIP['h'] - SUB_CLIP['y']
                if (k == NUM_D - 1):
                    SUB_CLIP['d'] = CLIP['z'] + CLIP['d'] - SUB_CLIP['z']

                mask_crop = mask.crop((SUB_CLIP['x'], SUB_CLIP['y'], SUB_CLIP['x'] + SUB_CLIP['w'], SUB_CLIP['y'] + SUB_CLIP['h']))
                mask_crop = np.array(mask_crop, dtype=np.float32)
                if (np.mean(mask_crop) < 0.1): continue

                label_crop = label.crop((SUB_CLIP['x'], SUB_CLIP['y'], SUB_CLIP['x'] + SUB_CLIP['w'], SUB_CLIP['y'] + SUB_CLIP['h']))
                label_crop = np.array(label_crop, dtype=np.float32)
                ink_density = float(np.mean(label_crop))

                info = {}
                info['id'] = str(SUBCLIP_ID).zfill(DIGIT_NUM)
                info['label'] = round(ink_density, 2)
                info['clip'] = SUB_CLIP

                NRRD_LIST.append(info)
                SUBCLIP_ID += 1

    return NRRD_LIST

def read_tif(TIF_INPUT, CLIP, RAW_SAMPLING, TIF_SAMPLING):
    # change to tif data clip
    rs = RAW_SAMPLING
    c  = { key: round(value / rs) for key, value in CLIP.items() }

    names = sorted(glob.glob(TIF_INPUT + '/*tif'))
    names = names[c['z'] : c['z'] + c['d']]
    image_stack = np.zeros((c['w'], c['h'], c['d']), dtype=np.float32)

    for i, filename in enumerate(tqdm(names)):
        image = Image.open(filename)
        image = image.crop((c['x'], c['y'], c['x']+c['w'], c['y']+c['h']))
        image = np.array(image, dtype=np.float32)
        image /= 65535.0
        image_stack[:, :, i] = np.transpose(image, (1, 0))

    # sampling tif stack
    ts = TIF_SAMPLING

    pad_x = (ts - image_stack.shape[0] % ts) % ts
    pad_y = (ts - image_stack.shape[1] % ts) % ts
    pad_z = (ts - image_stack.shape[2] % ts) % ts

    image_stack = np.pad(image_stack, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='edge')

    image_stack = image_stack.reshape((math.ceil(c['w'] / ts), ts, math.ceil(c['h'] / ts), ts, math.ceil(c['d'] / ts), ts))
    image_stack = image_stack.mean(axis=(1, 3, 5))

    return image_stack

def read_nrrd(NRRD_DIR):
    data, header = nrrd.read(NRRD_DIR)

def write_nrrd(NRRD_DIR, data):
    # header = {'spacings': [1.0, 1.0, 1.0]}
    # nrrd.write(NRRD_DIR, data, header)
    nrrd.write(NRRD_DIR, data)

# clear .nrrd output folder
shutil.rmtree(NRRD_OUTPUT, ignore_errors=True)
os.makedirs(NRRD_OUTPUT)

# generate nrrd list
NRRD_LIST = nrrd_list(CLIP, SUBCLIP_LIMIT)

# generate .nrrd files from .tif files
for NRRD_CHUNK in NRRD_LIST:
    NRRD_ID = NRRD_CHUNK['id']
    NRRD_SUBCLIP = NRRD_CHUNK['clip']

    # extract image stack from .tif files
    image_stack = read_tif(TIF_INPUT, NRRD_SUBCLIP, RAW_SAMPLING, TIF_SAMPLING)

    NRRD_CHUNK['shape'] = {}
    NRRD_CHUNK['shape']['w'] = image_stack.shape[0]
    NRRD_CHUNK['shape']['h'] = image_stack.shape[1]
    NRRD_CHUNK['shape']['d'] = image_stack.shape[2]

    # generate .nrrd file from image stack
    write_nrrd(f'{NRRD_OUTPUT}/{NRRD_ID}.nrrd', image_stack)

# save relevant info and copy to client
meta = {}
meta['nrrd'] = NRRD_LIST

with open(NRRD_INFO, "w") as outfile:
    json.dump(meta, outfile, indent=4)
