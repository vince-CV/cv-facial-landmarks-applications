#include "faceBlendCommon.hpp"

int main( int argc, char** argv)
{
  // Get the face detector
  dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();

  // The landmark detector is implemented in the shape_predictor class
  dlib::shape_predictor landmarkDetector;

  // Load the landmark model
  dlib::deserialize("../data/models/shape_predictor_5_face_landmarks.dat") >> landmarkDetector;

  //Read image
  Mat im = imread("../data/images/face2.png");

  // Detect landmarks
  vector<Point2f> points = getLandmarks(faceDetector, landmarkDetector, im);

  // Convert image to floating point in the range 0 to 1
  im.convertTo(im, CV_32FC3, 1/255.0);

  // Dimensions of output image
  Size size(600,600);

  // Variables for storing normalized image
  Mat imNorm;

  // Normalize image to output coordinates.
  normalizeImagesAndLandmarks(size, im, imNorm, points, points);

  imNorm.convertTo(imNorm, CV_8UC3, 255);

  imshow("Original Face", im);
  imshow("Aligned Face", imNorm);
  waitKey(0);
  destroyAllWindows();
  return EXIT_SUCCESS;
}
