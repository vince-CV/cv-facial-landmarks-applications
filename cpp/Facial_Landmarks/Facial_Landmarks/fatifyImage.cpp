#include <iostream>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <opencv2/core/core.hpp>
#include "faceBlendCommon.h"
#include "mls.h"

using namespace dlib;

#define RESIZE_HEIGHT 360
#define FACE_DOWNSAMPLE_RATIO 1.5


int fatify_image(int argc, char** argv)
{

  frontal_face_detector faceDetector = dlib::get_frontal_face_detector();
  shape_predictor landmarkDetector;

  deserialize("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/models/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

  float offset = 1.5;

  int anchorPoints[] = {1, 15, 30};
  std::vector<int> anchorPointsArray (anchorPoints, anchorPoints + sizeof(anchorPoints) / sizeof(int) );

  int deformedPoints[] = {5, 6, 8, 10, 11};
  std::vector<int> deformedPointsArray (deformedPoints, deformedPoints + sizeof(deformedPoints) / sizeof(int) );

  if (argc == 2)
  {
    offset = atof(argv[1]);
  }

  double t = (double)cv::getTickCount();

  string filename = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/hillary_clinton.jpg";
  Mat src = imread(filename);
  int height = src.rows;
  float IMAGE_RESIZE = (float)height/RESIZE_HEIGHT;
  cv::resize(src, src, cv::Size(), 1.0/IMAGE_RESIZE, 1.0/IMAGE_RESIZE);

  std::vector<Point2f> landmarks;
  landmarks = getLandmarks(faceDetector, landmarkDetector, src, (float)FACE_DOWNSAMPLE_RATIO);


  Point2f center (landmarks[30]);

  
  std::vector<Point2f> srcPoints, dstPoints;


  for( int i = 0; i < anchorPointsArray.size(); i++)
  {
    srcPoints.push_back(landmarks[anchorPointsArray[i]]);
    dstPoints.push_back(landmarks[anchorPointsArray[i]]);
  }
  for( int i = 0; i < deformedPointsArray.size(); i++)
  {
    srcPoints.push_back(landmarks[deformedPointsArray[i]]);
    Point2f pt( offset*(landmarks[deformedPointsArray[i]].x - center.x) + center.x, offset*(landmarks[deformedPointsArray[i]].y - center.y) + center.y);
    dstPoints.push_back(pt);
  }

  getEightBoundaryPoints(src.size(), srcPoints);
  getEightBoundaryPoints(src.size(), dstPoints);


  Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
  MLSWarpImage( src, srcPoints, dst, dstPoints, 0 );

  cout << "time taken " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << endl;
  Mat combined;
  hconcat(src, dst, combined);
  imshow("Distorted",combined);


  waitKey(0);
  destroyAllWindows();

  return 0;
}
