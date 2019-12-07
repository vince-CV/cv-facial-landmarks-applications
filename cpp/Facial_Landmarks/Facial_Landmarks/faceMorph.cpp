#include "faceBlendCommon.hpp"

int main( int argc, char** argv)
{
  // Get the face detector
  dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();

  // The landmark detector is implemented in the shape_predictor class
  dlib::shape_predictor landmarkDetector;

  // Load the landmark model
  dlib::deserialize("../data/models/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

  //Read two images
  Mat img1 = imread("../data/images/hillary-clinton.jpg");
  Mat img2 = imread("../data/images/presidents/bill-clinton.jpg");

  // Detect landmarks in both images.
  vector<Point2f> points1 = getLandmarks(faceDetector, landmarkDetector, img1);
  vector<Point2f> points2 = getLandmarks(faceDetector, landmarkDetector, img2);

  // Convert image to floating point in the range 0 to 1
  img1.convertTo(img1, CV_32FC3, 1/255.0);
  img2.convertTo(img2, CV_32FC3, 1/255.0);

  // Dimensions of output image
  Size size(600,600);

  // Variables for storing normalized images.
  Mat imgNorm1, imgNorm2;

  // Normalize image to output coordinates.
  normalizeImagesAndLandmarks(size,img1,imgNorm1, points1, points1);
  normalizeImagesAndLandmarks(size,img2,imgNorm2, points2, points2);

  // Calculate average points. Will be used for Delaunay triangulation.
  vector<Point2f> pointsAvg;
  for(int i = 0; i < points1.size(); i++)
  {
    pointsAvg.push_back((points1[i] + points2[i])/2);
  }

  // 8 Boundary points for Delaunay Triangulation
  vector <Point2f> boundaryPts;
  getEightBoundaryPoints(size, boundaryPts);

  for(int i = 0; i < boundaryPts.size(); i++)
  {
    pointsAvg.push_back(boundaryPts[i]);
    points1.push_back(boundaryPts[i]);
    points2.push_back(boundaryPts[i]);
  }

  // Calculate Delaunay triangulation.
  vector< vector<int> > delaunayTri;
  calculateDelaunayTriangles(Rect(0,0,size.width,size.height), pointsAvg, delaunayTri);

  // Start animation.
  double alpha = 0;
  bool increaseAlpha = true;

  while (1)
  {
    // Compute landmark points based on morphing parameter alpha
    vector<Point2f> points;
    for(int i = 0; i < points1.size(); i++)
    {
      Point2f pointMorph = (1-alpha) * points1[i] + alpha * points2[i];
      points.push_back(pointMorph);
    }

    // Warp images such that normalized points line up with morphed points.
    Mat imgOut1, imgOut2;
    warpImage(imgNorm1, imgOut1, points1, points, delaunayTri);
    warpImage(imgNorm2, imgOut2, points2, points, delaunayTri);

    // Blend warped images based on morphing parameter alpha
    Mat imgMorph = ( 1 - alpha ) * imgOut1 + alpha * imgOut2;

    // Keep animating by ensuring alpha stays between 0 and 1.
    if (alpha <= 0 && !increaseAlpha) increaseAlpha = true;
    if (alpha >= 1 && increaseAlpha) increaseAlpha = false;
    if(increaseAlpha) alpha += 0.025;
    else alpha -= 0.025;


    // Display morphed image.
    imshow("Morphed Face", imgMorph);
    int key = waitKey(15);

    // Exit when ESC is pressed.
    if ( key == 27) break;

  }
  return EXIT_SUCCESS;
}
