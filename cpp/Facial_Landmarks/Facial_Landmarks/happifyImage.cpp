#include "faceBlendCommon.h"
#include "mls30.h"

using namespace dlib;

#define RESIZE_HEIGHT 360
#define FACE_DOWNSAMPLE_RATIO 1.5

int happify_image(int argc, char** argv)
{

  frontal_face_detector faceDetector = dlib::get_frontal_face_detector();

  shape_predictor landmarkDetector;

  deserialize("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/models/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

  float offset1 = 1.5;
  float offset2 = 1.5;

  int anchorPoints[] = {8, 30};
  std::vector<int> anchorPointsArray (anchorPoints, anchorPoints + sizeof(anchorPoints) / sizeof(int) );

  int deformedPoints1[] = {48, 57, 54};
  std::vector<int> deformedPoints1Array (deformedPoints1, deformedPoints1 + sizeof(deformedPoints1) / sizeof(int) );

  int deformedPoints2[] = {21, 22, 36, 45};
  std::vector<int> deformedPoints2Array (deformedPoints2, deformedPoints2 + sizeof(deformedPoints2) / sizeof(int) );

  double t = (double)cv::getTickCount();

  string filename = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/f.jpg";
  Mat src = imread(filename);
  int height = src.rows;
  float IMAGE_RESIZE = (float)height/RESIZE_HEIGHT;
  cv::resize(src, src, cv::Size(), 1.0/IMAGE_RESIZE, 1.0/IMAGE_RESIZE);

  std::vector<Point2f> landmarks;
  landmarks = getLandmarks(faceDetector, landmarkDetector, src, (float)FACE_DOWNSAMPLE_RATIO);

  Point2f center1 (landmarks[8]);
  Point2f center2 (landmarks[28]);

  std::vector<Point2f> srcPoints, dstPoints;

  for( int i = 0; i < anchorPointsArray.size(); i++)
  {
    srcPoints.push_back(landmarks[anchorPointsArray[i]]);
    dstPoints.push_back(landmarks[anchorPointsArray[i]]);
  }
  for( int i = 0; i < deformedPoints1Array.size(); i++)
  {
    srcPoints.push_back(landmarks[deformedPoints1Array[i]]);
    Point2f pt = offset1 * (landmarks[deformedPoints1Array[i]] - center1) + center1;
    dstPoints.push_back(pt);
  }
  for( int i = 0; i < deformedPoints2Array.size(); i++)
  {
    srcPoints.push_back(landmarks[deformedPoints2Array[i]]);
    Point2f pt =  offset2 * (landmarks[deformedPoints2Array[i]] - center2) + center2;
    dstPoints.push_back(pt);
  }

  getEightBoundaryPoints(src.size(), srcPoints);
  getEightBoundaryPoints(src.size(), dstPoints);

  Mat dst = src.clone();
  MLSWarpImage( src, srcPoints, dst, dstPoints, 0 );

  cout << "time taken " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << endl;
  Mat combined;
  hconcat(src, dst, combined);
  imshow("Distorted",combined);

  //imwrite("results/happy.jpg",dst);
  waitKey(0);
  destroyAllWindows();

  return 0;
}
