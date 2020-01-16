#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include "faceBlendCommon.h"

using namespace cv;
using namespace std;
using namespace dlib;

int faceswap( int argc, char** argv)
{

  
  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor predictor;
  string model_path = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/models/shape_predictor_68_face_landmarks.dat";
  deserialize(model_path) >> predictor;

  double t = (double)cv::getTickCount();

  string filename1 = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks//data/images/dw.png";
  string filename2 = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks//data/images/xw.png";

  Mat img1 = imread(filename1);
  Mat img2 = imread(filename2);
  Mat img1Warped = img2.clone();

  std::vector<Point2f> points1, points2;
  points1 = getLandmarks(detector, predictor, img1);
  points2 = getLandmarks(detector, predictor, img2);


  img1.convertTo(img1, CV_32F);
  img1Warped.convertTo(img1Warped, CV_32F);



  std::vector<Point2f> hull1;
  std::vector<Point2f> hull2;
  std::vector<int> hullIndex;

  convexHull(points2, hullIndex, false, false);

  for(int i = 0; i < hullIndex.size(); i++)
  {
    hull1.push_back(points1[hullIndex[i]]);
    hull2.push_back(points2[hullIndex[i]]);
  }


  std::vector< std::vector<int> > dt;
  Rect rect(0, 0, img1Warped.cols, img1Warped.rows);
  calculateDelaunayTriangles(rect, hull2, dt);


  for(size_t i = 0; i < dt.size(); i++)
  {
    std::vector<Point2f> t1, t2;

    for(size_t j = 0; j < 3; j++)
    {
      t1.push_back(hull1[dt[i][j]]);
      t2.push_back(hull2[dt[i][j]]);
    }
    warpTriangle(img1, img1Warped, t1, t2);
  }
  cout << "Time taken for faceswap " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << endl;

  double tClone = (double)cv::getTickCount();


  std::vector<Point> hull8U;
  for(int i = 0; i < hull2.size(); i++)
  {
    Point pt(hull2[i].x, hull2[i].y);
    hull8U.push_back(pt);
  }

  Mat mask = Mat::zeros(img2.rows, img2.cols, img2.depth());
  fillConvexPoly(mask,&hull8U[0], hull8U.size(), Scalar(255,255,255));


  Rect r = boundingRect(hull2);
  Point center = (r.tl() + r.br()) / 2;

  Mat output;
  img1Warped.convertTo(img1Warped, CV_8UC3);
  seamlessClone(img1Warped, img2, mask, center, output, NORMAL_CLONE);

  cout << "Time taken for seamless cloning " << ((double)cv::getTickCount() - tClone)/cv::getTickFrequency() << endl;
  cout << "Total Time taken " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << endl;

  resize(img1Warped, img1Warped, Size(), 0.5, 0.5);
  resize(output, output, Size(), 0.5, 0.5);
  imshow("Face Swapped before seamless cloning", img1Warped);
  imshow("Face Swapped after seamless cloning", output);
  imwrite("results/faceswap.jpg", output);
  waitKey(0);
  destroyAllWindows();

  return 1;
}
