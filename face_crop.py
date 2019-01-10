#!/usr/bin/env python
import os
import numpy as np
import face_recognition
import argparse
from PIL import Image, ExifTags

parser = argparse.ArgumentParser(description='Crop faces from photo')
parser.add_argument('--in-dir', help='path to input dir')
parser.add_argument('--out-dir', default='out', help='path to output dir')
parser.add_argument('--max-size', type=int, default=512, help='max size of output image')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
print(args)
for (dirpath, dirnames, filenames) in os.walk(args.in_dir):
    for file in filenames:
        # f = os.path.join(os.path.dirname(os.path.realpath(__file__)), dirpath, file)
        f, ext = os.path.splitext(file)
        if ext.lower() not in ['.jpg', '.jpeg', '.png']: continue
        infile = os.path.join(args.in_dir, file)
        print(infile)
        image = Image.open(infile)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)

        face_locations = face_recognition.face_locations(np.array(image))  # top, right, bottom, left
        for i, (t, r, b, l) in enumerate(face_locations):
            w = r - l
            h = b - t
            l = max(0, l - w / 4)
            r = min(image.width, r + w / 4)
            t = max(0, t - h / 2)
            b = min(image.height, b + h / 4)

            im = image.crop([l, t, r, b])
            im.thumbnail((args.max_size, args.max_size), Image.ANTIALIAS)
            extra = "" if i == 0 else f'.{i}'
            outfile = os.path.join(args.out_dir, f'{f}{extra}{ext}')
            im.save(outfile)
