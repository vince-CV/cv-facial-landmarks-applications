#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include "renderFace.h"
#include <iostream>



using namespace dlib;
using namespace std;

static dlib::rectangle openCVRectToDlib(cv::Rect r)
{
	return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

int arylla_landmark() 
{
	

	shape_predictor landmarkDetector; 

	deserialize("C:/Users/xwen2/Desktop/shape_predictor_18_face_landmarks.dat") >> landmarkDetector;

	string imageFilename("C:/Users/xwen2/Desktop/lm/1.jpg");
	cv::Mat im = cv::imread(imageFilename);

	string landmarksBasename("C:/Users/xwen2/Desktop/");

	cv_image<bgr_pixel> dlibIm(im); 

	cv::Rect rect(0, 0, 786, 935);

	dlib::rectangle imageRect= openCVRectToDlib(rect);


	std::vector<full_object_detection> landmarksAll;
	
	full_object_detection landmarks = landmarkDetector(dlibIm, imageRect);

	std::vector<cv::Point2f> points;

	for (int i = 0; i < landmarks.num_parts(); i++)
	{
		cv::Point2f pt(landmarks.part(i).x(), landmarks.part(i).y());
		points.push_back(pt);
	}

	for (int i = 0; i < points.size(); i++)
	{
		cv::circle(im, points[i], 12, cv::Scalar(0, 200, 100), -1);
	}
	cv::imshow("Facial Landmark Detected", im);
	cv::waitKey(0);

}