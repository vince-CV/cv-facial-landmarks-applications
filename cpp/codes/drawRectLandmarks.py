import os
import sys
import cv2
import random

# create a directory if it doesn't exist
def create_dir(folder):
  try:
    os.makedirs(folder)
  except:
    print('{} already exists.'.format(folder))

# draw rectangle on image
def drawRectangle(im, bbox):
  x1, y1, x2, y2 = bbox
  cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255), thickness=5, lineType=cv2.LINE_8)

# draw landmarks on image
def drawLandmarks(im, parts):
  for i, part in enumerate(parts):
    # print shape.num_parts()
    px, py = part
    # draw circle at each landmark
    cv2.circle(im, (px, py), 1, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    # write landmark number at each landmark
    cv2.putText(im, str(i+1), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 100), 4)


# define scale so that points can be printed well
scale = 4
# we will draw facial rectangles and landmarks on
# a randomly sampled small subset of all images
numSamples = 50

# Default values
fldDir = "../data/facial_landmark_data"
numPoints = 70

# facial landmark data directory
if len(sys.argv) == 2:
  fldDir = sys.argv[1]
# number of facial landmarks; pass 70 or 33
if len(sys.argv) == 3:
  numPoints = sys.argv[2]


# Prepare output dirs
# as we know we have a mirrored image corresponding
# to each image. We will results for mirrored and original
# images in separate directory. Although this is not
# important for you because we already have annotation
# files for mirrored images.
#
# This step was crucial, when we got eye-centers annotated
# by data team. Data team only annotated eye-centers for
# original images. Then we generated annotation files
# for mirrored images from annotation files of original images
outputDir = os.path.join(fldDir, 'output')
outputMirrorDir = os.path.join(outputDir, 'mirror')
outputOriginalDir = os.path.join(outputDir, 'original')
create_dir(outputMirrorDir)
create_dir(outputOriginalDir)

# Path to image_names file
imageNamesFilepath = os.path.join(fldDir, 'image_names.txt')

# Check whether path to image_names file exists
# within facial_landmark_data directory
if os.path.exists(imageNamesFilepath):
  # If image_names.txt exists, read it
  with open(imageNamesFilepath) as d:
    imageNames = [x.strip() for x in d.readlines()]
else:
  print('Pass path to facial_landmark_data as argument to this script')



# Randomly shuffle list cntaining image names
random.shuffle(imageNames)
# select numSamples image names from list
imageNamesSampled = imageNames[:numSamples]

# Iterate over image names
for k, imageName in enumerate(imageNamesSampled):
  print("Processing file: {}".format(imageName))

  # create image path
  imagePath = os.path.join(fldDir, imageName)
  # read image
  im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
  # scale up image
  im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
  # create path to face rectangle file
  rectPath = os.path.splitext(imagePath)[0] + '_rect.txt'

  # open rectangle file and read
  with open(rectPath) as f:
    line = f.readline()
    # read annotations
    left, top, width, height = [float(n) for n in line.strip().split()]
    # calculate coordinates of bottom right corner of rectangle
    right = left + width
    bottom = top + height
    # scale up face reactangle coordinates
    x1, y1, x2, y2 = int(scale*left), int(scale*top), int(scale*right), int(scale*bottom)
    # save coordinates to a list. this is also called bounding box
    # it is a term to denote coordinates of an object
    bbox = [x1, y1, x2, y2]

  # open facial landmarks file and read coordinates
  pointsPath = os.path.splitext(imagePath)[0] + '_bv' + str(numPoints) + '.txt'
  parts = []
  # open points file and read
  with open(pointsPath) as g:
    # read lines. each line has coordinates of a landmark point
    lines = [x.strip() for x in g.readlines()]
    # iterate over all lines
    for line in lines:
      # each line has two numbers (x, y of each landmark)
      left, right = [float(n) for n in line.split()]
      # scale up landmark coordinates
      px, py = int(scale*left), int(scale*right)
      parts.append([px, py])

  # draw face rectangle and landmarks
  drawRectangle(im, bbox)
  drawLandmarks(im, parts)

  # basename is filename in a filepath
  imageBasename = os.path.basename(imagePath)
  # if basename has mirror, output image will be stored
  # in mirror output directory
  # else in original outout directory
  if 'mirror' in imageBasename:
    outputImagePath = os.path.join(outputMirrorDir, imageBasename)
  else:
    outputImagePath = os.path.join(outputOriginalDir, imageBasename)

  # save output image
  cv2.imwrite(outputImagePath, im)
