#include <iostream>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace dlib;

#define FACE_DOWNSAMPLE_RATIO_DLIB 1   

#ifndef M_PI
  #define M_PI 3.14159
#endif

static Mat barrel(Mat &src, float k)
{
  int w = src.cols;
  int h = src.rows;


  Mat Xd = cv::Mat::zeros(src.size(), CV_32F);
  Mat Yd = cv::Mat::zeros(src.size(), CV_32F);

  float Xu, Yu;
  for (int y = 0; y < h; y++)
  {
    for (int x = 0; x < w; x++)
    {

      Xu = ( (float) x / w )- 0.5;
      Yu = ( (float) y / h )- 0.5;


      float r = sqrt(Xu * Xu + Yu * Yu);


      float dr = k * r * cos(M_PI * r);

      if (r > 0.5) dr = 0;


      float rn = r - dr;


      Xd.at<float>(y,x) =  w * (rn * Xu / r + 0.5);
      Yd.at<float>(y,x) = h * (rn * Yu / r + 0.5);

    }
  }


  Mat dst;
  cv::remap( src, dst, Xd, Yd, INTER_CUBIC, BORDER_CONSTANT, Scalar(0,0, 0) );
  return dst;
}

int bug_eyes_image(int argc, char** argv)
{
  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor pose_model;

  string modelPath = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/models/shape_predictor_68_face_landmarks.dat";

  string filename = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/2.jpg";

  float bulge_amount = .85;

  int radius = 30;


  cout << "USAGE" << endl << "./bugeyeImage <bulge_amount default : .5 > < radius around eye default : 30 > <filename> " << endl;

  if (argc == 2)
  {
    bulge_amount = atof(argv[1]);
  }
  else if(argc == 3)
  {
    bulge_amount = atof(argv[1]);
    radius = atoi(argv[2]);
  }
  else if(argc == 4)
  {
    bulge_amount = atof(argv[1]);
    radius = atoi(argv[2]);
    filename = argv[3];
  }

  deserialize(modelPath) >> pose_model;

  // load a nice picture
  double t = (double)cv::getTickCount();

  Mat src = imread(filename);

  cv_image<bgr_pixel> cimg(src);
  std::vector<dlib::rectangle> faces;

  faces = detector(cimg);

  // Find the pose of each face.
  full_object_detection landmarks;

  // Find the landmark points using DLIB Facial landmarks detector
  landmarks = pose_model(cimg, faces[0]);

  // Find the roi for left and right Eye
  Rect roiEyeRight ( (landmarks.part(43).x()-radius)
                    , (landmarks.part(43).y()-radius)
                    , ( landmarks.part(46).x() - landmarks.part(43).x() + 2*radius )
                    , ( landmarks.part(47).y() - landmarks.part(43).y() + 2*radius ) );
  Rect roiEyeLeft ( (landmarks.part(37).x()-radius)
                   , (landmarks.part(37).y()-radius)
                   , ( landmarks.part(40).x() - landmarks.part(37).x() + 2*radius )
                   , ( landmarks.part(41).y() - landmarks.part(37).y() + 2*radius ) );

  // Find the patch and apply the transformation
  Mat eyeRegion, output;
  output = src.clone();
  src(roiEyeRight).copyTo(eyeRegion);
  eyeRegion = barrel(eyeRegion, bulge_amount);
  eyeRegion.copyTo(output(roiEyeRight));
  src(roiEyeLeft).copyTo(eyeRegion);
  eyeRegion = barrel(eyeRegion, bulge_amount);
  eyeRegion.copyTo(output(roiEyeLeft));

  cout << "time taken " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << endl;
  // imshow("distorted",dst);
  imshow("final",output);

  imwrite("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/results/fayer.jpg",output);
  waitKey(0);

  return 0;
}
