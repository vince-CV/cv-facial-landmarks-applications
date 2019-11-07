#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>


static void drawPolyline(cv::Mat &img, const dlib::full_object_detection& landmarks, const int start, const int end, bool isClosed = false)
{
	std::vector <cv::Point> points;
	for (int i = start; i <= end; ++i)
	{
		points.push_back(cv::Point(landmarks.part(i).x(), landmarks.part(i).y()));
	}
	cv::polylines(img, points, isClosed, cv::Scalar(255, 200, 0), 2, 16); // link points into a line

}

static void renderFace(cv::Mat &img, const dlib::full_object_detection& landmarks)
{
	drawPolyline(img, landmarks, 0, 16);      
	drawPolyline(img, landmarks, 17, 21);       
	drawPolyline(img, landmarks, 22, 26);       
	drawPolyline(img, landmarks, 27, 30);          
	drawPolyline(img, landmarks, 30, 35, true);    
	drawPolyline(img, landmarks, 36, 41, true);    
	drawPolyline(img, landmarks, 42, 47, true);    
	drawPolyline(img, landmarks, 48, 59, true);    
	drawPolyline(img, landmarks, 60, 67, true);    

}

static void renderFace(cv::Mat &img, const std::vector<cv::Point2f> &points, cv::Scalar color, int radius = 3) 
{
	for (int i = 0; i < points.size(); i++)
	{
		cv::circle(img, points[i], radius, color, -1);
	}
}


