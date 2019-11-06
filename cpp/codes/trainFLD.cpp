// The contents of this file are in the public domain.
/*
The intent of the example programs supplied with the dlib C++ library is
to both instruct users and to also provide a simple body of code they
may copy and paste from.  To make this as painless as possible all the
example programs have been placed into the public domain.


This work is hereby released into the Public Domain.
To view a copy of the public domain dedication, visit
http://creativecommons.org/licenses/publicdomain/ or send a
letter to
    Creative Commons
    171 Second Street
    Suite 300,
    San Francisco, California, 94105, USA.



Public domain dedications are not recognized by some countries.  So
if you live in an area where the above dedication isn't valid then
you can consider the example programs to be licensed under the Boost
Software License.
*/

#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <iostream>
#include <string>

using namespace dlib;
using namespace std;


double interocular_distance (
  const full_object_detection& det,
  int numPoints
)
{
  dlib::vector<double,2> l, r;
  double cnt = 0;
  // Find the center of the left eye by averaging the points around
  // the eye.
  if (numPoints == 70)
  {
    for (unsigned long i = 36; i <= 41; ++i)
    {
      l += det.part(i);
      ++cnt;
    }
  }
  else // 33 point model
  {
    for (unsigned long i = 17; i <= 22; ++i)
    {
      l += det.part(i);
      ++cnt;
    }
  }
  l /= cnt;

  // Find the center of the right eye by averaging the points around
  // the eye.
  cnt = 0;
  if (numPoints==70)
  {
    for (unsigned long i = 42; i <= 47; ++i)
    {
      r += det.part(i);
      ++cnt;
    }
  }
  else // 33 point model
  {
    for (unsigned long i = 23; i <= 28; ++i)
    {
      r += det.part(i);
      ++cnt;
    }
  }
  r /= cnt;

  // Now return the distance between the centers of the eyes
  return length(l-r);
}

std::vector<std::vector<double> > get_interocular_distances (
  const std::vector<std::vector<full_object_detection> >& objects,
  int numPoints
)
{
  std::vector<std::vector<double> > temp(objects.size());
  for (unsigned long i = 0; i < objects.size(); ++i)
  {
    for (unsigned long j = 0; j < objects[i].size(); ++j)
    {
      temp[i].push_back(interocular_distance(objects[i][j], numPoints));
    }
  }
  return temp;
}

int main(int argc, char** argv)
{
  try
  {
    // In this example we are going to train a shape_predictor using
    // facial_landmark_data. So the first thing we do is load dataset.
    // This means you need to supply the path to this faces folder
    // as a command line argument so we will know where it is.
    if (argc != 3)
    {
      cout << "Give the path to the facial_landmark_data " << endl;
      cout << "directory as the argument to this program.  " << endl;
      cout << " ./build/trainFLD ../data/facial_landmark_data 70" << endl;
      cout << endl;
      return 0;
    }
    const std::string fldDatadir = argv[1];
    int numPoints = atoi(argv[2]);
    const std::string modelName = "shape_predictor_" + to_string(numPoints) + "_face_landmarks.dat";
    const std::string modelPath = fldDatadir + "/" + modelName;

    // Create shape_predictor_trainer object for training the model.
    shape_predictor_trainer trainer;

    // some parts of training process can be parallelized.
    // Trainer will use this count of threads when possible
    trainer.set_num_threads(2);

    trainer.set_cascade_depth(10);
    trainer.set_num_trees_per_cascade_level(500);
    trainer.set_tree_depth(4);
    trainer.set_nu(0.1);
    trainer.set_oversampling_amount(20);
    trainer.set_feature_pool_size(400);
    trainer.set_feature_pool_region_padding(0);
    trainer.set_lambda(0.1);
    trainer.set_num_test_splits(20);

    // Tell the trainer to print status messages to the console so we can
    // see training options and how long the training will take.
    trainer.be_verbose();

    // Now we will create the variables that will hold our dataset.
    // images_train will hold training images and faces_train holds
    // the locations and poses of each face in the training images.
    dlib::array<array2d<unsigned char> > images_train, images_test;
    std::vector<std::vector<full_object_detection> > faces_train, faces_test;

    // Now we load the data.  These XML files list the images in each
    // dataset and also contain the positions of the face boxes and
    // landmarks (called parts in the XML file).
    load_image_dataset(images_train, faces_train, fldDatadir + "/training_with_face_landmarks.xml");
    load_image_dataset(images_test, faces_test, fldDatadir + "/testing_with_face_landmarks.xml");

    // Now finally generate the shape model
    shape_predictor sp = trainer.train(images_train, faces_train);


    // Now that we have a model we can test it. This function measures the
    // average distance between a face landmark output by the
    // shape_predictor and ground truth data.
    // Note that there is an optional 4th argument that lets us normalize the
    // distances.  Here we are normalizing the error using the interocular
    // distance, as is customary when evaluating face landmarking systems.
    cout << "mean training error: " <<
      test_shape_predictor(sp, images_train, faces_train, get_interocular_distances(faces_train, numPoints )) << endl;

    // The real test is to see how well it does on data it wasn't trained
    // on.
    cout << "mean testing error:  " <<
      test_shape_predictor(sp, images_test, faces_test, get_interocular_distances(faces_test, numPoints)) << endl;

    // Finally, we save the model to disk so we can use it later.
    serialize(modelPath) << sp;
  }
  catch (exception& e)
  {
    cout << "\nexception thrown!" << endl;
    cout << e.what() << endl;
  }
}
