#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <opencv2/imgproc.hpp> 
#include <iostream>
#include <fstream>
#include <string>
#include <dlib/opencv.h>
#include <stdlib.h>
#include "faceBlendCommon.h"

using namespace cv;
using namespace std;
using namespace dlib;

#define RESIZE_HEIGHT 480
#define FACE_DOWNSAMPLE_RATIO 1

static int selectedpoints[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31, 32, 33, 34, 35, 55, 56, 57, 58, 59};
static std::vector<int> selectedIndex (selectedpoints, selectedpoints + sizeof(selectedpoints) / sizeof(int) );


static std::vector<Point2f> getSavedPoints(string pointsFileName)
{
  std::vector<Point2f> points;
  ifstream ifs(pointsFileName.c_str());
  float x, y;
  if (!ifs)
    cout << "Unable to open file" << endl;
  while(ifs >> x >> y)
  {
    points.push_back(Point2f(x,y));
  }
  return points;
}


int beard_image(int argc, char** argv)
{
  string overlayFile = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/beard1.png";
  string imageFile = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/feier.jpg";

  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor predictor;
  string modelPath = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/models/shape_predictor_68_face_landmarks.dat";
  deserialize(modelPath) >> predictor;

  if (argc == 2)
  {
    imageFile = argv[1];
  }
  else if (argc == 3)
  {
    imageFile = argv[1];
    overlayFile = argv[2];
  }


  Mat beard, targetImage, beardAlphaMask;
  Mat imgWithMask = imread(overlayFile,IMREAD_UNCHANGED);
  std::vector<Mat> rgbaChannels(4);


  split(imgWithMask, rgbaChannels);


  std::vector<Mat> bgrchannels;
  bgrchannels.push_back(rgbaChannels[0]);
  bgrchannels.push_back(rgbaChannels[1]);
  bgrchannels.push_back(rgbaChannels[2]);

  merge(bgrchannels, beard);
  beard.convertTo(beard, CV_32F, 1.0/255.0);


  std::vector<Mat> maskchannels;
  maskchannels.push_back(rgbaChannels[3]);
  maskchannels.push_back(rgbaChannels[3]);
  maskchannels.push_back(rgbaChannels[3]);

  merge(maskchannels, beardAlphaMask);
  beardAlphaMask.convertTo(beardAlphaMask, CV_32FC3);


  std::vector<Point2f> featurePoints1 = getSavedPoints( overlayFile + ".txt");


  Rect rect = boundingRect(featurePoints1);

  std::vector< std::vector<int> > dt;
  calculateDelaunayTriangles(rect, featurePoints1, dt);

  float time_detector = (double)cv::getTickCount();

  targetImage = imread(imageFile);
  int height = targetImage.rows;
  float IMAGE_RESIZE = (float)height/RESIZE_HEIGHT;
  cv::resize(targetImage, targetImage, cv::Size(), 1.0/IMAGE_RESIZE, 1.0/IMAGE_RESIZE);

  std::vector<Point2f> points2 = getLandmarks(detector, predictor, targetImage, (float)FACE_DOWNSAMPLE_RATIO);

  std::vector<Point2f> featurePoints2;
  for( int i = 0; i < selectedIndex.size(); i++)
  {
    featurePoints2.push_back(points2[selectedIndex[i]]);
    constrainPoint(featurePoints2[i], targetImage.size());
  }

  targetImage.convertTo(targetImage, CV_32F, 1.0/255.0);


  Mat beardWarped = Mat::zeros(targetImage.size(), beard.type());
  Mat beardAlphaMaskWarped = Mat::zeros(targetImage.size(), beardAlphaMask.type());


  for(size_t i = 0; i < dt.size(); i++)
  {
    std::vector<Point2f> t1, t2;
 
    for(size_t j = 0; j < 3; j++)
    {
      t1.push_back(featurePoints1[dt[i][j]]);
      t2.push_back(featurePoints2[dt[i][j]]);
    }
    warpTriangle(beard, beardWarped, t1, t2);
    warpTriangle(beardAlphaMask, beardAlphaMaskWarped, t1, t2);


  }
  imshow("beardWarped",beardWarped);
  imshow("beardAlphaMaskWarped",beardAlphaMaskWarped);

  Mat mask1;
  beardAlphaMaskWarped.convertTo(mask1, CV_32FC3, 1.0/255.0);


  Mat mask2 = Scalar(1.0,1.0,1.0) - mask1;
  Mat temp1 = targetImage.mul(mask2);
  Mat temp2 = beardWarped.mul(mask1);

  imshow("mask1",mask1);
  imshow("mask2",mask2);
  imshow("temp1",temp1);
  imshow("temp2",temp2);

  Mat result = temp1 + temp2;
  imshow("result",result);

  cout << "Time taken  " << ((double)cv::getTickCount() - time_detector)/cv::getTickFrequency() << endl;
  char c = (char)waitKey(0);
  if( c == 's' )
    imwrite("warped_" + imageFile + "_" + overlayFile, result*255);

  return 0;
}
