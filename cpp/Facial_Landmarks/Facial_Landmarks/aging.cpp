#include "faceBlendCommon.h"

using namespace cv;
using namespace std;
using namespace dlib;

#define FACE_DOWNSAMPLE_RATIO 1


Mat& alphaBlend(Mat& alpha, Mat& foreground, Mat& background, Mat& outImage)
{
  Mat fore, back;
  multiply(alpha, foreground, fore, 1/255.0);
  multiply(Scalar::all(255)-alpha, background, back,1/255.0);
  add(fore, back, outImage);

  return outImage;
}


void desaturateImage(Mat &im, double scaleBy)
{
 
  Mat imgHSV;
  cv::cvtColor(im,imgHSV,COLOR_BGR2HSV);


  std::vector<Mat> channels(3);
  split(imgHSV,channels);

  
  channels[1] = scaleBy * channels[1];


  merge(channels,imgHSV);


  cv::cvtColor(imgHSV,im,COLOR_HSV2BGR);

}

void removePolygonFromMask(Mat &mask, std::vector<Point2f> points, std::vector<int> pointsIndex)
{
  std::vector<Point> hullPoints;
  for(int i = 0; i < pointsIndex.size(); i++)
  {
    Point pt( points[pointsIndex[i]].x , points[pointsIndex[i]].y );
    hullPoints.push_back(pt);
  }
  fillConvexPoly(mask,&hullPoints[0], hullPoints.size(), Scalar(0,0,0));
}

void appendForeheadPoints(std::vector<Point2f> &points)
{

  double offsetScalp = 3.0;

  static int brows[] = {25, 23, 20, 18 };
  std::vector<int> browsIndex (brows, brows + sizeof(brows) / sizeof(brows[0]) );
  static int browsReference[] = {45, 47, 40, 36};
  std::vector<int> browsReferenceIndex (browsReference, browsReference + sizeof(browsReference) / sizeof(browsReference[0]) );

  for (unsigned long k = 0; k < browsIndex.size(); ++k)
  {
    Point2f foreheadPoint = offsetScalp * ( points[ browsIndex[k] ] - points[ browsReferenceIndex[k]]) + points[browsReferenceIndex[k]];
    points.push_back(foreheadPoint);
  }

}

Mat getFaceMask(Size size, std::vector<Point2f> points)
{


  static int leftEye[] = {36, 37, 38, 39, 40, 41};
  std::vector<int> leftEyeIndex (leftEye, leftEye + sizeof(leftEye) / sizeof(leftEye[0]) );


  static int rightEye[] = {42, 43, 44, 45, 46, 47};
  std::vector<int> rightEyeIndex (rightEye, rightEye + sizeof(rightEye) / sizeof(rightEye[0]) );


  static int mouth[] = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
  std::vector<int> mouthIndex (mouth, mouth + sizeof(mouth) / sizeof(mouth[0]) );


  static int nose[] = {28, 31, 33, 35};
  std::vector<int> noseIndex (nose, nose + sizeof(nose) / sizeof(nose[0]) );

 
  std::vector<Point2f> hull;
  convexHull(points, hull, false, true);


  std::vector<Point> hullInt;
  for(int i = 0; i < hull.size(); i++)
  {
    Point pt( hull[i].x , hull[i].y );
    hullInt.push_back(pt);
  }


  Mat mask = Mat::zeros(size.height, size.width, CV_8UC3);
  fillConvexPoly(mask,&hullInt[0], hullInt.size(), Scalar(255,255,255));


  removePolygonFromMask(mask, points, leftEyeIndex);
  removePolygonFromMask(mask, points, rightEyeIndex);
  removePolygonFromMask(mask, points, noseIndex);
  removePolygonFromMask(mask, points, mouthIndex);

  return mask;

}


int aging( int argc, char** argv)
{
  frontal_face_detector faceDetector = get_frontal_face_detector();

  shape_predictor landmarkDetector;
  deserialize("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/models/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;


  string filename1 = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/wrinkle2.jpg";

  string filename2 = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/ted_cruz.jpg";


  cout << "USAGE" << endl << "./aging <wrinkle file> <original file>" << endl;
  if (argc == 2)
  {
    filename1 = argv[1];
  }
  else if (argc == 3)
  {
    filename1 = argv[1];
    filename2 = argv[2];
  }



  Mat img1 = imread(filename1);
  Mat img2 = imread(filename2);


  std::vector<Point2f> points1, points2;

  points1 = getLandmarks(faceDetector, landmarkDetector, img1, (float)FACE_DOWNSAMPLE_RATIO);
  points2 = getLandmarks(faceDetector, landmarkDetector, img2, (float)FACE_DOWNSAMPLE_RATIO);

  appendForeheadPoints(points1);
  appendForeheadPoints(points2);

  std::vector< std::vector<int> > dt;
  Rect rect(0, 0, img1.cols, img1.rows);
  calculateDelaunayTriangles(rect, points1, dt);


  img1.convertTo(img1, CV_32F);
  img2.convertTo(img2, CV_32F);


  Mat img1Warped = img2.clone();
  warpImage(img1,img1Warped, points1, points2, dt);
  img1Warped.convertTo(img1Warped, CV_8UC3);
  img2.convertTo(img2, CV_8UC3);


  Mat mask = getFaceMask(img2.size(), points2);


  Rect r1 = boundingRect(points2);
  Point center1 = (r1.tl() + r1.br()) / 2;
  Mat clonedOutput;
  seamlessClone(img1Warped,img2, mask, center1, clonedOutput, MIXED_CLONE);


  Size size = mask.size();
  Mat maskSmall;
  resize(mask, maskSmall, Size(256, int((size.height) * 256.0/double(size.width))));
  erode(maskSmall, maskSmall, Mat(), Point(-1,-1), 5);
  GaussianBlur(maskSmall, maskSmall, Size(15,15), 0, 0);
  resize(maskSmall, mask, size);

  Mat agedImage = clonedOutput.clone();
  alphaBlend(mask, clonedOutput, img2, agedImage);


  desaturateImage(agedImage, 0.8);


  Mat displayImage;
  hconcat(img2,agedImage,displayImage);
  namedWindow("Output", WINDOW_NORMAL);
  imshow("Output",displayImage);
  int k = cv::waitKey(0);
  return EXIT_SUCCESS;
}
