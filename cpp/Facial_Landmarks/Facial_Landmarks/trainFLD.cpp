#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <iostream>
#include <string>

using namespace dlib;
using namespace std;


double interocular_distance (const full_object_detection& det, int numPoints)
{
  dlib::vector<double,2> l, r;
  double cnt = 0;
  // Find the center of the left eye by averaging the points around the eye.
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

  // Find the center of the right eye by averaging the points around the eye.
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

std::vector<std::vector<double> > get_interocular_distances (const std::vector<std::vector<full_object_detection> >& objects,int numPoints)
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
    // train a shape_predictor using facial_landmark_data. 

    if (argc != 3)
    {
      cout << "path to the facial_landmark_data " << endl;
      cout << "directory " << endl;
      cout << " ./build/trainFLD ../data/facial_landmark_data 70" << endl;
      cout << endl;
      return 0;
    }
    const std::string fldDatadir = argv[1];
    int numPoints = atoi(argv[2]);
    const std::string modelName = "shape_predictor_" + to_string(numPoints) + "_face_landmarks.dat";
    const std::string modelPath = fldDatadir + "/" + modelName;

    // Create shape_predictor_trainer object
    shape_predictor_trainer trainer;

    trainer.set_num_threads(2); // Parallelize training process

    trainer.set_cascade_depth(10);
    trainer.set_num_trees_per_cascade_level(500);
    trainer.set_tree_depth(4);
    trainer.set_nu(0.1);
    trainer.set_oversampling_amount(20);
    trainer.set_feature_pool_size(400);
    trainer.set_feature_pool_region_padding(0);
    trainer.set_lambda(0.1);
    trainer.set_num_test_splits(20);

    trainer.be_verbose();


    // images_train: training images
	// faces_train: the locations and poses of each face in the training images.
    dlib::array<array2d<unsigned char> > images_train, images_test;
    std::vector<std::vector<full_object_detection> > faces_train, faces_test;

    load_image_dataset(images_train, faces_train, fldDatadir + "/training_with_face_landmarks.xml");
    load_image_dataset(images_test, faces_test, fldDatadir + "/testing_with_face_landmarks.xml");

    shape_predictor sp = trainer.train(images_train, faces_train);


    cout << "mean training error: " <<
      test_shape_predictor(sp, images_train, faces_train, get_interocular_distances(faces_train, numPoints )) << endl;

    cout << "mean testing error:  " <<
      test_shape_predictor(sp, images_test, faces_test, get_interocular_distances(faces_test, numPoints)) << endl;

    serialize(modelPath) << sp;  // save model

  }

  catch (exception& e)
  {
    cout << "\nexception thrown!" << endl;
    cout << e.what() << endl;
  }
}
