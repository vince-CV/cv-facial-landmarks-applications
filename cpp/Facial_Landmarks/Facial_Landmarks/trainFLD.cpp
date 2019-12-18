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

  if (numPoints == 70)
  {
    for (unsigned long i = 36; i <= 41; ++i)
    {
      l += det.part(i);
      ++cnt;
    }
  }
  else
  {
    for (unsigned long i = 17; i <= 22; ++i)
    {
      l += det.part(i);
      ++cnt;
    }
  }
  l /= cnt;

  cnt = 0;
  if (numPoints==70)
  {
    for (unsigned long i = 42; i <= 47; ++i)
    {
      r += det.part(i);
      ++cnt;
    }
  }
  else
  {
    for (unsigned long i = 23; i <= 28; ++i)
    {
      r += det.part(i);
      ++cnt;
    }
  }
  r /= cnt;

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

int trainFLD()
{
  try
  {


	const std::string fldDatadir = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/facial_landmark_data"; //argv[1];
	int numPoints = 70; //atoi(argv[2]);
    const std::string modelName = "shape_predictor_" + to_string(numPoints) + "_face_landmarks.dat";
    const std::string modelPath = fldDatadir + "/" + modelName;

    shape_predictor_trainer trainer;

    trainer.set_num_threads(1);

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



    dlib::array<array2d<unsigned char> > images_train, images_test;
    std::vector<std::vector<full_object_detection> > faces_train, faces_test;

	cout << fldDatadir + "/training_with_face_landmarks.xml" << endl;

    load_image_dataset(images_train, faces_train, fldDatadir + "/training_with_face_landmarks.xml");
	cout << "Loaded!" << endl;
    load_image_dataset(images_test, faces_test, fldDatadir + "/testing_with_face_landmarks.xml");
	
	

    shape_predictor sp = trainer.train(images_train, faces_train);


    cout << "mean training error: " <<
      test_shape_predictor(sp, images_train, faces_train, get_interocular_distances(faces_train, numPoints )) << endl;

    cout << "mean testing error:  " <<
      test_shape_predictor(sp, images_test, faces_test, get_interocular_distances(faces_test, numPoints)) << endl;

    serialize(modelPath) << sp;  // save model

	return 1;

  }

  catch (exception& e)
  {
    cout << "\n exception thrown!" << endl;
    cout << e.what() << endl;
  }
}
