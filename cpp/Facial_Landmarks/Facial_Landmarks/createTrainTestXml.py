import sys
import os
import random
try:
  from lxml import etree as ET
except ImportError:
  print('install lxml using pip')
  print('pip install lxml')


def createXml(imageNames, xmlName, numPoints):

  dataset = ET.Element('dataset')
  ET.SubElement(dataset, "name").text = "Training Faces"
  images = ET.SubElement(dataset, "images")

  numFiles = len(imageNames)
  print('{0} : {1} files'.format(xmlName, numFiles))

  for k, imageName in enumerate(imageNames):

    print('{}:{} - {}'.format(k+1, numFiles, imageName))
    rect_name = os.path.splitext(imageName)[0] + '_rect.txt'
    with open(os.path.join(fldDatadir, rect_name), 'r') as file:
      rect = file.readline()
    rect = rect.split()
    left, top, width, height = rect[0:4]

    image = ET.SubElement(images, "image", file=imageName)

    box = ET.SubElement(image, 'box', top=top, left=left, width=width, height=height)

    points_name = os.path.splitext(imageName)[0] + '_bv' + str(numPoints) + '.txt'
    with open(os.path.join(fldDatadir, points_name), 'r') as file:
      for i, point in enumerate(file):
        x, y = point.split()
        x = str(int(float(x)))
        y = str(int(float(y)))

        name = str(i).zfill(2)
        ET.SubElement(box, 'part', name=name, x=x, y=y)

  tree = ET.ElementTree(dataset)

  print('writing on disk: {}'.format(xmlName))

  tree.write(xmlName, pretty_print=True, xml_declaration=True, encoding="UTF-8")


if __name__ == '__main__':

  fldDatadir = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/facial_landmark_data"
  numPoints = 70

  if len(sys.argv) == 2:

    fldDatadir = sys.argv[1]
  if len(sys.argv) == 3:

    numPoints = sys.argv[2]

  with open(os.path.join(fldDatadir, 'image_names.txt')) as d:
    imageNames = [x.strip() for x in d.readlines()]


  # decrease training data if enough RAM
  n = 1000
  imageNames = random.sample(imageNames, n)

  totalNumFiles = len(imageNames)
  numTestFiles = int(0.05 * totalNumFiles)

  testFiles = random.sample(imageNames, numTestFiles)
  trainFiles = list(set(imageNames) - set(testFiles))

  createXml(trainFiles, os.path.join(fldDatadir, 'training_with_face_landmarks.xml'), numPoints)
  createXml(testFiles, os.path.join(fldDatadir, 'testing_with_face_landmarks.xml'), numPoints)
