#include "faceBlendCommon.h"


int faceAlignment(int argc, char** argv)
{
	// Get the face detector
	dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();

	// The landmark detector is implemented in the shape_predictor class
	dlib::shape_predictor landmarkDetector;

	// Load the landmark model
	dlib::deserialize("C:/Users/xwen2/Desktop/Computer Vision Projects/1. Face landmarks/data/models/shape_predictor_5_face_landmarks.dat") >> landmarkDetector;

	//Read image
	Mat im = imread("C:/Users/xwen2/Desktop/Computer Vision Projects/1. Face landmarks/data/images/face1.png");

	// Detect landmarks
	vector<Point2f> points = getLandmarks(faceDetector, landmarkDetector, im);

	// Convert image to floating point in the range 0 to 1
	im.convertTo(im, CV_32FC3, 1 / 255.0);

	// Dimensions of output image
	Size size(300, 300);

	Mat imNorm;
	// Normalize image to output coordinates.
	normalizeImagesAndLandmarks(size, im, imNorm, points, points);

	imNorm.convertTo(imNorm, CV_8UC3, 255);

	imshow("Original Face", im);
	imshow("Aligned Face", imNorm);
	waitKey(0);
	destroyAllWindows();
	return 0;
}
