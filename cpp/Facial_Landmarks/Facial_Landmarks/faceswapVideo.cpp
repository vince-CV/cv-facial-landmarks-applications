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
#include "colorCorrection.h"


using namespace cv;
using namespace std;
using namespace dlib;

#define RESIZE_HEIGHT 480
#define FACE_DOWNSAMPLE_RATIO 1.5
#define SKIP_FRAMES 2

int main()
{

  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor predictor;
  string modelPath = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/models/shape_predictor_68_face_landmarks.dat";

  string filename1 = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/f.jpg";


  deserialize(modelPath) >> predictor;


  std::vector<Point2f> points1;
  Mat img1 = imread(filename1);

  int height = img1.rows;
  float IMAGE_RESIZE = (float)height/RESIZE_HEIGHT;
  cv::resize(img1, img1, cv::Size(), 1.0/IMAGE_RESIZE, 1.0/IMAGE_RESIZE);


  points1 = getLandmarks(detector, predictor, img1, (float)FACE_DOWNSAMPLE_RATIO);
  img1.convertTo(img1, CV_32F);


  std::vector<int> hullIndex;
  convexHull(points1, hullIndex, false, false);


  for(int i=48; i<59;i++)
  {
    hullIndex.push_back(i);
  }


  std::vector< std::vector<int> > dt;
  Rect rect(0, 0, img1.cols, img1.rows);
  std::vector<Point2f> hull1 ;

  for(int i = 0; i < hullIndex.size(); i++)
  {
    hull1.push_back(points1[hullIndex[i]]);
  }
  calculateDelaunayTriangles(rect, hull1, dt);

  cout << "processed input image";

  cv::VideoCapture cap(0);
  cv::Mat img2;


  cap >> img2;
  height = img2.rows;
  IMAGE_RESIZE = (float)height/RESIZE_HEIGHT;


  std::vector<Point2f> points2;


  int count = 0;
  double t = (double)cv::getTickCount();
  double fps = 30.0;

  // optical flow 
  TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
  Size subPixWinSize(10,10), winSize(101,101);
  double eyeDistance, sigma;
  bool eyeDistanceNotCalculated = true;

  std::vector<Point2f> hull2Prev ;
  std::vector<Point2f> hull2Next ;

  Mat img2Gray, img2GrayPrev;

  Mat result, output, img1Warped;

  namedWindow("After Blending");


  while(cap.read(img2))
  {
    if ( count == 0 )
      t = (double)cv::getTickCount();

    double time_detector = (double)cv::getTickCount();

    cv::resize(img2, img2, cv::Size(), 1.0/IMAGE_RESIZE, 1.0/IMAGE_RESIZE);


    if (count % SKIP_FRAMES == 0)
    {
      points2 = getLandmarks(detector, predictor, img2, (float)FACE_DOWNSAMPLE_RATIO);
      cout << "Face Detector" << endl;
    }


    if(points2.size() != 68)
    {
      cout << "Points not detected" << endl;
      continue;
    }


    img1Warped = img2.clone();
    img1Warped.convertTo(img1Warped, CV_32F);


    std::vector<Point2f> hull2 ;

    for(int i = 0; i < hullIndex.size(); i++)
    {
      hull2.push_back(points2[hullIndex[i]]);
    }

   // optical flow and stabilization of landmarks 
    if(!hull2Prev.size())
    {
      hull2Prev = hull2;
    }

    double t1 = (double)cv::getTickCount();

    if ( eyeDistanceNotCalculated )
    {
      eyeDistance = norm(points2[36] - points2[45]);
      winSize = cv::Size(2 * int(eyeDistance/4) + 1,  2 * int(eyeDistance/4) + 1);
      eyeDistanceNotCalculated = false;
      sigma = eyeDistance * eyeDistance / 400;
    }

    cvtColor(img2, img2Gray, COLOR_BGR2GRAY);

    if(img2GrayPrev.empty())
      img2GrayPrev = img2Gray.clone();

    std::vector<uchar> status;
    std::vector<float> err;

    //optical flow based estimate of the point in this frame
    calcOpticalFlowPyrLK(img2GrayPrev, img2Gray, hull2Prev, hull2Next, status, err, winSize, 5, termcrit, 0, 0.001);

    //final landmark points are a weighted average of detected landmarks and tracked landmarks
    for (unsigned long k = 0; k < hull2.size(); ++k)
    {
      double n = norm(hull2Next[k] - hull2[k]);
      double alpha = exp(-n*n/sigma);
      hull2[k] = (1 - alpha) * hull2[k] + alpha * hull2Next[k];
      constrainPoint(hull2[k], img2.size());
    }


    hull2Prev = hull2;
    img2GrayPrev = img2Gray.clone();

    // stabilization done

   
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

    cout << "Stabilize and Warp time" << ((double)cv::getTickCount() - t1)/cv::getTickFrequency() << endl;



    img1Warped.convertTo(img1Warped, CV_8UC3);

   
    output = correctColours(img2, img1Warped, points2);

    Rect re = boundingRect(hull2);
    Point center = (re.tl() + re.br()) / 2;
    std::vector<Point> hull3;

    for(int i = 0; i < hull2.size()-12; i++)
    {
      Point pt1( 0.95*(hull2[i].x - center.x) + center.x, 0.95*(hull2[i].y - center.y) + center.y);
      hull3.push_back(pt1);
    }
    Mat mask1 = Mat::zeros(img2.rows, img2.cols, img2.type());

    fillConvexPoly(mask1,&hull3[0], hull3.size(), Scalar(255,255,255));

    cv::GaussianBlur(mask1,mask1, Size (21, 21),10);

    Mat mask2 = Scalar(255,255,255) - mask1;

    Mat temp1 = output.mul(mask1, 1.0/255);
    Mat temp2 = img2.mul(mask2,1.0/255);
    result =  temp1 + temp2;


    cout << "Total time" << ((double)cv::getTickCount() - time_detector)/cv::getTickFrequency() << endl;
    imshow("After Blending", result);

    int k = cv::waitKey(1);

    if (k == 27)
    {
      break;
    }

    count++;

    if ( count == 10)
    {
      fps = 10.0 * cv::getTickFrequency() / ((double)cv::getTickCount() - t);
      count = 0;
    }
    cout << "FPS " << fps << endl;
  }

  cap.release();
  return 1;
}
