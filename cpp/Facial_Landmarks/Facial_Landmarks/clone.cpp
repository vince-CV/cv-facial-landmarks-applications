#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

void seamlessCloningExample()
{
  Mat src = imread("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/airplane.jpg");
  Mat dst = imread("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/sky.jpg");

  Mat srcMask = Mat::zeros(src.rows, src.cols, src.depth());

  Point poly[1][7];
  poly[0][0] = Point(4, 80);
  poly[0][1] = Point(30, 54);
  poly[0][2] = Point(151,63);
  poly[0][3] = Point(254,37);
  poly[0][4] = Point(298,90);
  poly[0][5] = Point(272,134);
  poly[0][6] = Point(43,122);

  const Point* polygons[1] = { poly[0] };
  int numPoints[] = { 7 };

  fillPoly(srcMask, polygons, numPoints, 1, Scalar(255,255,255));

  Point center(800,100);

  Mat output;
  seamlessClone(src, dst, srcMask, center, output, NORMAL_CLONE);

  namedWindow("Seamless Cloning Example");
  imshow("Seamless Cloning Example", output);


}

void normalVersusMixedCloningExample()
{

  Mat src = imread("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/iloveyouticket.jpg");
  Mat dst = imread("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/wood-texture.jpg");

  Mat srcMask = 255 * Mat::ones(src.rows, src.cols, src.depth());

  Point center(dst.cols/2,dst.rows/2);

  Mat normalClone;
  Mat mixedClone;

  seamlessClone(src, dst, srcMask, center, normalClone, NORMAL_CLONE);
  seamlessClone(src, dst, srcMask, center, mixedClone, MIXED_CLONE);

  namedWindow("NORMAL_CLONE Example");
  namedWindow("MIXED_CLONE Example");
  imshow("NORMAL_CLONE Example", normalClone);
  imshow("MIXED_CLONE Example", mixedClone);
  waitKey(0);


}


int clone( int argc, char** argv )
{

  seamlessCloningExample();
  normalVersusMixedCloningExample();

}
