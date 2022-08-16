#!/usr/bin/env python
import os
import sys
import argparse
from PIL import Image

"""
Reduce images size

Example: python image_resize.py -d /home/user/images -o /home/user/output_dir -s 1024 768
"""
def cmp(a, b):
    return (a > b) - (a < b) 

def resizeImage(infile, output_dir, size):
    outfile = os.path.splitext(os.path.basename(infile))[0]
    extension = os.path.splitext(infile)[1]

    if (cmp(extension, ".jpg")):
        return

    if infile != outfile:
        try:
            im = Image.open(infile)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(os.path.join(output_dir, outfile+extension),"JPEG")
        except IOError:
            print("cannot reduce image for ", infile)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="Directory to look up for images")
    parser.add_argument("-o", help="Output directory")
    parser.add_argument("-s", nargs=2, type=int, help="Output size")
    args = parser.parse_args()

    input_dir = os.path.normpath(args.d) if args.d else os.getcwd()
    output_dir = os.path.normpath(args.o) if args.o else os.path.join(os.getcwd(), 'resized')
    output_size = tuple(args.s) if args.s else (1024,768)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for file in os.listdir(input_dir):
        resizeImage(os.path.join(input_dir, file), output_dir, output_size)
