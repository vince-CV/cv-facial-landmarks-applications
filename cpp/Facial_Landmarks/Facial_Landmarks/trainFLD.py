

import os
import sys
import dlib

#print("USAGE : python trainFLD.py <path to facial_landmark_data folder> <number of points>")

# Default values
fldDatadir = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/facial_landmark_data/"
numPoints = 70
'''
if len(sys.argv) == 2:
  fldDatadir = sys.argv[1]
if len(sys.argv) == 3:
  numPoints = sys.argv[2]
'''
modelName = 'shape_predictor_{}_face_landmarks.dat'.format(numPoints)

options = dlib.shape_predictor_training_options()
options.cascade_depth = 10
options.num_trees_per_cascade_level = 500
options.tree_depth = 4
options.nu = 0.1
options.oversampling_amount = 20
options.feature_pool_size = 400
options.feature_pool_region_padding = 0
options.lambda_param = 0.1
options.num_test_splits = 20

# Tell the trainer to print status messages to the console so we can
# see training options and how long the training will take.
options.be_verbose = True


trainingXmlPath = os.path.join(fldDatadir, "training_with_face_landmarks.xml")
testingXmlPath = os.path.join(fldDatadir, "testing_with_face_landmarks.xml")
outputModelPath = os.path.join(fldDatadir, modelName)

# check whether path to XML files is correct
if os.path.exists(trainingXmlPath) and os.path.exists(testingXmlPath):

  # dlib.train_shape_predictor() does the actual training.  It will save the
  # final predictor to predictor.dat.  The input is an XML file that lists the
  # images in the training dataset and also contains the positions of the face
  # parts.
  dlib.train_shape_predictor(trainingXmlPath, outputModelPath, options)

  # Now that we have a model we can test it.  dlib.test_shape_predictor()
  # measures the average distance between a face landmark output by the
  # shape_predictor and ground truth data.

  print("\nTraining accuracy: {}".format(
    dlib.test_shape_predictor(trainingXmlPath, outputModelPath)))

  # The real test is to see how well it does on data it wasn't trained on.
  print("Testing accuracy: {}".format(
    dlib.test_shape_predictor(testingXmlPath, outputModelPath)))
else:
  print('training and test XML files not found.')
  print('Please check paths:')
  print('train: {}'.format(trainingXmlPath))
  print('test: {}'.format(testingXmlPath))
