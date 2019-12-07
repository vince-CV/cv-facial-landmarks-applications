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

#define RESIZE_HEIGHT 360
#define FACE_DOWNSAMPLE_RATIO_DLIB 1.5    //affects dlib's face detector

#ifndef M_PI
  #define M_PI 3.14159
#endif

Mat barrel(Mat &src, float k)
{
  int w = src.cols;
  int h = src.rows;

  // Meshgrid of destiation image
  Mat Xd = cv::Mat::zeros(src.size(), CV_32F);
  Mat Yd = cv::Mat::zeros(src.size(), CV_32F);

  float Xu, Yu;
  for (int y = 0; y < h; y++)
  {
    for (int x = 0; x < w; x++)
    {
      // Normalize x and y
      Xu = ( (float) x / w )- 0.5;
      Yu = ( (float) y / h )- 0.5;

      // Radial distance from center
      float r = sqrt(Xu * Xu + Yu * Yu);

      // Implementing the following equation
      // dr = k * r * cos(pi*r)
      float dr = k * r * cos(M_PI * r);

      // Outside the maximum radius dr is set to 0
      if (r > 0.5) dr = 0;

      // Remember we need to provide inverse mapping to remap
      // Hence the negative sign before dr
      float rn = r - dr;

      // Applying the distortion on the grid
      // Back to un-normalized coordinates
      Xd.at<float>(y,x) =  w * (rn * Xu / r + 0.5);
      Yd.at<float>(y,x) = h * (rn * Yu / r + 0.5);

    }
  }

  // Interpolation of points
  Mat dst;
  cv::remap( src, dst, Xd, Yd, INTER_CUBIC, BORDER_CONSTANT, Scalar(0,0, 0) );
  return dst;
}

void constrainRect(Rect &r, Size sz)
{
  if(r.x < 0)
    r.x = 0;

  if(r.y < 0)
    r.y = 0;

  if(r.width + r.x > sz.width)
    r.width = sz.width - r.x;

  if(r.height + r.y > sz.height)
    r.height = sz.height - r.y;

}
int main(int argc, char** argv)
{
  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor pose_model;

  string modelPath = "../data/models/shape_predictor_68_face_landmarks.dat";

  float bulgeAmount = .5;

  int radius = 30;

  // accept command line arguments for model path for shape detector and image file
  cout << "USAGE" << endl << "./bugeyeVideo <bulge_amount default : .5 > < radius around eye default : 30 > " << endl;

  if (argc == 2)
  {
    bulgeAmount = atof(argv[1]);
  }
  else if(argc == 3)
  {
    bulgeAmount = atof(argv[1]);
    radius = atoi(argv[2]);
  }

  deserialize(modelPath) >> pose_model;

  cv::VideoCapture cap(0);
  if (!cap.isOpened())
  {
    cerr << "Unable to connect to camera" << endl;
    return 1;
  }

  Mat src, eyeRegion, output;
  std::vector<dlib::rectangle> faces ;
  while(1)
  {
    // Grab a frame
    double time_total = (double)cv::getTickCount();
    cap >> src;
    int height = src.rows;
    float IMAGE_RESIZE = (float)height/RESIZE_HEIGHT;
    cv::resize(src, src, cv::Size(), 1.0/IMAGE_RESIZE, 1.0/IMAGE_RESIZE);

    cv::Size size = src.size();

    cv::Mat src_small;
    cv::resize(src, src_small, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO_DLIB, 1.0/FACE_DOWNSAMPLE_RATIO_DLIB);

    cv_image<bgr_pixel> cimg(src);
    cv_image<bgr_pixel> cimg_small(src_small);

    // Detect face
    faces = detector(cimg_small);
    cout << "Time taken by detector at scale " << FACE_DOWNSAMPLE_RATIO_DLIB << " = "<< ((double)cv::getTickCount() - time_total)/cv::getTickFrequency() << endl;

    if (!faces.size())
    {
      putText(src, "Unable to detect face, Please check proper lighting", Point(10, 50), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255), 1, LINE_AA);
      putText(src, "Or Decrease FACE_DOWNSAMPLE_RATIO", Point(10, 150), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255), 1, LINE_AA);
      imshow("Bug Eye Demo",src);
      if ((waitKey(1) & 0xFF) == 27)
        return 0;
      continue;
    }
    dlib::rectangle r(
                (long)(faces[0].left() * FACE_DOWNSAMPLE_RATIO_DLIB),
                (long)(faces[0].top() * FACE_DOWNSAMPLE_RATIO_DLIB),
                (long)(faces[0].right() * FACE_DOWNSAMPLE_RATIO_DLIB),
                (long)(faces[0].bottom() * FACE_DOWNSAMPLE_RATIO_DLIB)
                );

    // Find the pose of each face.
    full_object_detection landmarks;

    // Find the landmark points using DLIB Facial landmarks detector
    landmarks = pose_model(cimg, r);


    // Find the roi for left and right Eye
    Rect roiEyeRight ( (landmarks.part(43).x()-radius)
                      , (landmarks.part(43).y()-radius)
                      , ( landmarks.part(46).x() - landmarks.part(43).x() + 2*radius )
                      , ( landmarks.part(47).y() - landmarks.part(43).y() + 2*radius ) );
    Rect roiEyeLeft ( (landmarks.part(37).x()-radius)
                     , (landmarks.part(37).y()-radius)
                     , ( landmarks.part(40).x() - landmarks.part(37).x() + 2*radius )
                     , ( landmarks.part(41).y() - landmarks.part(37).y() + 2*radius ) );

    constrainRect(roiEyeRight, src.size());
    constrainRect(roiEyeLeft, src.size());

    // Find the atch and apply the transform
    output = src.clone();
    src(roiEyeRight).copyTo(eyeRegion);
    eyeRegion = barrel(eyeRegion, bulgeAmount);
    eyeRegion.copyTo(output(roiEyeRight));

    src(roiEyeLeft).copyTo(eyeRegion);
    eyeRegion = barrel(eyeRegion, bulgeAmount);
    eyeRegion.copyTo(output(roiEyeLeft));

    cout << "Total Time taken at scale " << FACE_DOWNSAMPLE_RATIO_DLIB << " = "<< ((double)cv::getTickCount() - time_total)/cv::getTickFrequency() << endl;
    imshow("Bug Eye Demo",output);

    int k = cv::waitKey(1);
    // Quit if 'q' or ESC is pressed
    if ( k == 'q' || k == 27)
    {
      break;
    }
  }

cap.release();
destroyAllWindows();
return 0;
}
