#include "faceBlendCommon.h"


int main(int argc, char** argv)
{

	dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();
	dlib::shape_predictor landmarkDetector;


	dlib::deserialize("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/models/shape_predictor_5_face_landmarks.dat") >> landmarkDetector;


	Mat im = imread("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/39.png");
	vector<Point2f> points = getLandmarks(faceDetector, landmarkDetector, im);

	im.convertTo(im, CV_32FC3, 1 / 255.0);

	Size size(300, 300);

	Mat imNorm;

	normalizeImagesAndLandmarks(size, im, imNorm, points, points);

	imNorm.convertTo(imNorm, CV_8UC3, 255);

	resize(im, im, Size(), 0.2, 0.2);
	imshow("Original Face", im);
	imshow("Aligned Face", imNorm);
	waitKey(0);
	destroyAllWindows();
	return 0;
}
