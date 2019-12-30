#include "faceBlendCommon.h"

int main( int argc, char** argv)
{

  dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();


  dlib::shape_predictor landmarkDetector;


  dlib::deserialize("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/models/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;


  Mat img1 = imread("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/y.jpg");
  Mat img2 = imread("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/f.jpg");


  vector<Point2f> points1 = getLandmarks(faceDetector, landmarkDetector, img1);
  vector<Point2f> points2 = getLandmarks(faceDetector, landmarkDetector, img2);


  img1.convertTo(img1, CV_32FC3, 1/255.0);
  img2.convertTo(img2, CV_32FC3, 1/255.0);


  Size size(600,600);


  Mat imgNorm1, imgNorm2;


  normalizeImagesAndLandmarks(size,img1,imgNorm1, points1, points1);
  normalizeImagesAndLandmarks(size,img2,imgNorm2, points2, points2);


  vector<Point2f> pointsAvg;
  for(int i = 0; i < points1.size(); i++)
  {
    pointsAvg.push_back((points1[i] + points2[i])/2);
  }


  vector <Point2f> boundaryPts;
  getEightBoundaryPoints(size, boundaryPts);

  for(int i = 0; i < boundaryPts.size(); i++)
  {
    pointsAvg.push_back(boundaryPts[i]);
    points1.push_back(boundaryPts[i]);
    points2.push_back(boundaryPts[i]);
  }


  vector< vector<int> > delaunayTri;
  calculateDelaunayTriangles(Rect(0,0,size.width,size.height), pointsAvg, delaunayTri);


  double alpha = 0;
  bool increaseAlpha = true;

  while (1)
  {

    vector<Point2f> points;
    for(int i = 0; i < points1.size(); i++)
    {
      Point2f pointMorph = (1-alpha) * points1[i] + alpha * points2[i];
      points.push_back(pointMorph);
    }


    Mat imgOut1, imgOut2;
    warpImage(imgNorm1, imgOut1, points1, points, delaunayTri);
    warpImage(imgNorm2, imgOut2, points2, points, delaunayTri);

 
    Mat imgMorph = ( 1 - alpha ) * imgOut1 + alpha * imgOut2;

    if (alpha <= 0 && !increaseAlpha) increaseAlpha = true;
    if (alpha >= 1 && increaseAlpha) increaseAlpha = false;
    if(increaseAlpha) alpha += 0.010;
    else alpha -= 0.025;



    imshow("Morphed Face", imgMorph);
    int key = waitKey(8);


    if ( key == 27) break;

  }
  return EXIT_SUCCESS;
}
