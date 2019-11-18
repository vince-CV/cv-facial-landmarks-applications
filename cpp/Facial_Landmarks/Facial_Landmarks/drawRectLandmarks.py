import os
import sys
import cv2
import random


def create_dir(folder):
  try:
    os.makedirs(folder)
  except:
    print('{} already exists.'.format(folder))

def drawRectangle(im, bbox):
  x1, y1, x2, y2 = bbox
  cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255), thickness=5, lineType=cv2.LINE_8)

def drawLandmarks(im, parts):
  for i, part in enumerate(parts):
    px, py = part
    cv2.circle(im, (px, py), 1, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(im, str(i+1), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 100), 4)


# define scale so that points can be printed well
scale = 4
numSamples = 50

fldDir = "../data/facial_landmark_data"
numPoints = 70

if len(sys.argv) == 2:
  fldDir = sys.argv[1]
if len(sys.argv) == 3:
  numPoints = sys.argv[2]


outputDir = os.path.join(fldDir, 'output')
outputMirrorDir = os.path.join(outputDir, 'mirror')
outputOriginalDir = os.path.join(outputDir, 'original')
create_dir(outputMirrorDir)
create_dir(outputOriginalDir)

imageNamesFilepath = os.path.join(fldDir, 'image_names.txt')


if os.path.exists(imageNamesFilepath):
  with open(imageNamesFilepath) as d:
    imageNames = [x.strip() for x in d.readlines()]
else:
  print('Pass path to facial_landmark_data as argument to this script')


random.shuffle(imageNames)
imageNamesSampled = imageNames[:numSamples]


for k, imageName in enumerate(imageNamesSampled):
  print("Processing file: {}".format(imageName))

  imagePath = os.path.join(fldDir, imageName)
  im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
  im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
  rectPath = os.path.splitext(imagePath)[0] + '_rect.txt'


  with open(rectPath) as f:
    line = f.readline()

    left, top, width, height = [float(n) for n in line.strip().split()]
    right = left + width
    bottom = top + height

    x1, y1, x2, y2 = int(scale*left), int(scale*top), int(scale*right), int(scale*bottom)
    bbox = [x1, y1, x2, y2]


  pointsPath = os.path.splitext(imagePath)[0] + '_bv' + str(numPoints) + '.txt'
  parts = []

  with open(pointsPath) as g:
    lines = [x.strip() for x in g.readlines()]
    for line in lines:
      left, right = [float(n) for n in line.split()]
      px, py = int(scale*left), int(scale*right)
      parts.append([px, py])

  drawRectangle(im, bbox)
  drawLandmarks(im, parts)

  imageBasename = os.path.basename(imagePath)

  if 'mirror' in imageBasename:
    outputImagePath = os.path.join(outputMirrorDir, imageBasename)
  else:
    outputImagePath = os.path.join(outputOriginalDir, imageBasename)

  cv2.imwrite(outputImagePath, im)
