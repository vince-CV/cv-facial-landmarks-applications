# Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED
#
# This code is made available to the students of
# the online course titled "Computer Vision for Faces"
# by Satya Mallick for personal non-commercial use.
#
# Sharing this code is strictly prohibited without written
# permission from Big Vision LLC.
#
# For licensing and other inquiries, please email
# spmallick@bigvisionllc.com
#

import os
import sys

# Default values
fldDir = "../data/facial_landmark_data"

if len(sys.argv) == 2:
  # facial landmark data directory
  fldDir = sys.argv[1]

# Path to image_names file
imageNamesFilepath = os.path.join(fldDir, 'image_names.txt')

# Check whether path to image_names file exists within facial_landmark_data directory
if os.path.exists(imageNamesFilepath):
  # If image_names.txt exists, read it
  with open(imageNamesFilepath) as d:
    imageNames = [x.strip() for x in d.readlines()]
else:
  print('Pass path to facial_landmark_data as argument to this script')

# Out of 70 points, we have to pick 33 points.
# Here we are writing indices of those 33 points.
# IMPORTANT: Numbers shown on image are natural numbers. They
# start from 1 whereas indices in Python start from 0.
# So to match these indices with facial landmark numbers on
# sample image, you should add 1.
points33Indices = [
                   1, 3, 5, 8, 11, 13, 15,     # Jaw line
                   17, 19, 21,                 # Left eyebrow
                   22, 24, 26,                 # Right eyebrow
                   30, 31,                     # Nose bridge
                   33, 35,                     # Lower nose
                   36, 37, 38, 39, 40, 41,     # Left eye
                   42, 43, 44, 45, 46, 47,     # Right Eye
                   48, 51, 54, 57              # Outer lip
                  ]
numImages = len(imageNames)

# Iterate over all image names
for n, imageName in enumerate(imageNames):
  # Just pretty printing the progress
  print('{}/{} - {}'.format(n+1, numImages, imageName))

  # path to image
  imagePath = os.path.join(fldDir, imageName)
  # We points annotation file has prefix _bv70.txt in end
  # whereas image has .jpg. So we are creating the path to 70points
  # annotation file by replacing .jpg of image path with _bv70.txt.
  points70Path = os.path.splitext(imagePath)[0] + '_bv70.txt'
  # Similarly for 30points annotation files.
  points33Path = os.path.splitext(imagePath)[0] + '_bv33.txt'

  # Check if path to annotation file exists
  if os.path.exists(points70Path):
    # open file
    with open(points70Path, 'r') as f:
      # read all lines
      points70 = f.readlines()
      # select lines whose indices are in our points33Indices list
      points33 = [points70[i] for i in points33Indices]
      # open points33 file
      with open(points33Path, 'w') as g:
        # write 33 points to file
        g.writelines(points33)

  else:
    print('Unable to find path:{}'.format(points70Path))
