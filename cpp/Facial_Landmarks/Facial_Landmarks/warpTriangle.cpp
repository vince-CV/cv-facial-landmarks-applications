#include <opencv2/opencv.hpp>
#include <stdlib.h>

using namespace cv;
using namespace std;


void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> tri1, vector<Point2f> tri2)
{

  Rect r1 = boundingRect(tri1);
  Rect r2 = boundingRect(tri2);


  Mat img1Cropped;
  img1(r1).copyTo(img1Cropped);


  vector<Point2f> tri1Cropped, tri2Cropped;
  vector<Point> tri2CroppedInt;
  for(int i = 0; i < 3; i++)
  {
    tri1Cropped.push_back( Point2f( tri1[i].x - r1.x, tri1[i].y -  r1.y) );
    tri2Cropped.push_back( Point2f( tri2[i].x - r2.x, tri2[i].y - r2.y) );

    tri2CroppedInt.push_back( Point((int)(tri2[i].x - r2.x), (int)(tri2[i].y - r2.y)) );

  }


  Mat warpMat = getAffineTransform( tri1Cropped, tri2Cropped );

  Mat img2Cropped = Mat::zeros(r2.height, r2.width, img1Cropped.type());
  warpAffine( img1Cropped, img2Cropped, warpMat, img2Cropped.size(), INTER_LINEAR, BORDER_REFLECT_101);

  Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
  fillConvexPoly(mask, tri2CroppedInt, Scalar(1.0, 1.0, 1.0), 16, 0);


  multiply(img2Cropped,mask, img2Cropped);
  multiply(img2(r2), Scalar(1.0,1.0,1.0) - mask, img2(r2));
  img2(r2) = img2(r2) + img2Cropped;

}

int triangle_warping( int argc, char** argv)
{

  Mat imgIn = imread("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/kingfisher.jpg");

 
  imgIn.convertTo(imgIn, CV_32FC3, 1/255.0);


  Mat imgOut = Mat::ones(imgIn.size(), imgIn.type());
  imgOut = Scalar(1.0,1.0,1.0);


  vector <Point2f> triIn;
  triIn.push_back(Point2f(360,50));
  triIn.push_back(Point2d(60,100));
  triIn.push_back(Point2f(300,400));


  vector <Point2f> triOut;
  triOut.push_back(Point2f(400,200));
  triOut.push_back(Point2f(160,270));
  triOut.push_back(Point2f(400,400));


  warpTriangle(imgIn, imgOut, triIn, triOut);





  imgIn.convertTo(imgIn, CV_8UC3, 255.0);
  imgOut.convertTo(imgOut, CV_8UC3, 255.0);


  Scalar color = Scalar(0, 0, 0);


  vector <Point> triInInt, triOutInt;
  for(int i=0; i < 3; i++)
  {
    triInInt.push_back(Point(triIn[i].x,triIn[i].y));
    triOutInt.push_back(Point(triOut[i].x,triOut[i].y));
  }


  int lineWidth = 2;
  polylines(imgIn, triInInt, true, color, lineWidth, LINE_AA);
  polylines(imgOut, triOutInt, true, color, lineWidth, LINE_AA);


  namedWindow("Input");
  imshow("Input", imgIn);
  imwrite("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/results/kingfisherInputTriangle.jpg", imgIn);

  namedWindow("Output");
  imshow("Output", imgOut);
  imwrite("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/results/kingfisherOutputTriangle.jpg", imgOut);

  waitKey(0);

  return 0;
}
