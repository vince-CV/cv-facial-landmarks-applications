#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include "renderFace.h"
#include <math.h>

using namespace dlib;
using namespace std;



void constrainPoint(cv::Point2f &p, cv::Size sz)
{
	p.x = cv::min(cv::max((double)p.x, 0.0), (double)(sz.width - 1));
	p.y = cv::min(cv::max((double)p.y, 0.0), (double)(sz.height - 1));

}

double interEyeDistance(full_object_detection &shape)
{
	cv::Point2f leftEyeLeftCorner(shape.part(36).x(), shape.part(36).y());
	cv::Point2f rightEyeRightCorner(shape.part(45).x(), shape.part(45).y());
	double distance = norm(rightEyeRightCorner - leftEyeLeftCorner);
	return distance;
}

int stableFLD()
{
	int RESIZE_HEIGHT = 360;
	int NUM_FRAMES_FOR_FPS = 100;
	int SKIP_FRAMES = 1;
	
	try
	{
		string winName("Stabilized Facial Landmark Detector");
		cv::namedWindow(winName, cv::WINDOW_NORMAL);

		cv::VideoCapture cap(0);
		
		cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
		cv::Size winSize(101, 101);
		double eyeDistance, dotRadius, sigma;
		bool eyeDistanceNotCalculated = true;
		int maxLevel = 5;
		std::vector<uchar> status;
		std::vector<float> err;

		double fps = 30.0;

		cv::Mat im, imPrev, imGray, imGrayPrev;
		std::vector<cv::Mat> imGrayPyr, imGrayPrevPyr;

		cap >> imPrev;

		cv::cvtColor(imPrev, imGrayPrev, cv::COLOR_BGR2GRAY);
		cv::buildOpticalFlowPyramid(imGrayPrev, imGrayPrevPyr, winSize, maxLevel);
		cv::Size size = imPrev.size();
		cv::Mat imSmall;
		

		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor landmarkDetector;
		deserialize("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/models/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

		std::vector<rectangle> faces;

		std::vector<cv::Point2f> points, pointsPrev, pointsDetectedCur, pointsDetectedPrev;

		for (unsigned long k = 0; k < landmarkDetector.num_parts(); ++k)
		{
			pointsPrev.push_back(cv::Point2f(0, 0));
			points.push_back(cv::Point2f(0, 0));
			pointsDetectedCur.push_back(cv::Point2f(0, 0));
			pointsDetectedPrev.push_back(cv::Point2f(0, 0));
		}


		bool isFirstFrame = true;

		bool showStabilized = true;


		int count = 0;
		double t;

		while (1)
		{
			if (count == 0) t = (double)cv::getTickCount();

			cap >> im;

			cv::cvtColor(im, imGray, cv::COLOR_BGR2GRAY);
			float height = im.rows;
			float IMAGE_RESIZE = height / RESIZE_HEIGHT;
			cv::resize(im, imSmall, cv::Size(), 1.0 / IMAGE_RESIZE, 1.0 / IMAGE_RESIZE);

			cv_image<bgr_pixel> cimg_small(imSmall);
			cv_image<bgr_pixel> cimg(im);

			if (count % SKIP_FRAMES == 0)
			{
				faces = detector(cimg_small);
			}

			if (faces.size() < 1) continue;


			std::vector<full_object_detection> shapes;

			for (unsigned long i = 0; i < faces.size(); ++i)
			{
				rectangle r(
					(long)(faces[i].left() * IMAGE_RESIZE),
					(long)(faces[i].top() * IMAGE_RESIZE),
					(long)(faces[i].right() * IMAGE_RESIZE),
					(long)(faces[i].bottom() * IMAGE_RESIZE)
				);

				full_object_detection shape = landmarkDetector(cimg, r);

				shapes.push_back(shape);

				for (unsigned long k = 0; k < shape.num_parts(); ++k)
				{

					if (isFirstFrame)
					{
						pointsPrev[k].x = pointsDetectedPrev[k].x = shape.part(k).x();
						pointsPrev[k].y = pointsDetectedPrev[k].y = shape.part(k).y();
					}
					else
					{
						pointsPrev[k] = points[k];
						pointsDetectedPrev[k] = pointsDetectedCur[k];
					}

					// pointsDetectedCur stores results returned by the facial landmark detector
					// points stores the stabilized landmark points
					points[k].x = pointsDetectedCur[k].x = shape.part(k).x();
					points[k].y = pointsDetectedCur[k].y = shape.part(k).y();
				}

				if (eyeDistanceNotCalculated)
				{
					eyeDistance = interEyeDistance(shape);
					winSize = cv::Size(2 * int(eyeDistance / 4) + 1, 2 * int(eyeDistance / 4) + 1);
					eyeDistanceNotCalculated = false;
					dotRadius = eyeDistance > 100 ? 3 : 2;
					sigma = eyeDistance * eyeDistance / 400;
				}

				cv::buildOpticalFlowPyramid(imGray, imGrayPyr, winSize, maxLevel);
				cv::calcOpticalFlowPyrLK(imGrayPrevPyr, imGrayPyr, pointsPrev, points, status, err, winSize, maxLevel, termcrit, 0, 0.0001);

				// Final landmark points are a weighted average of detected landmarks and tracked landmarks
				for (unsigned long k = 0; k < shape.num_parts(); ++k)
				{
					double n = norm(pointsDetectedPrev[k] - pointsDetectedCur[k]);
					double alpha = exp(-n * n / sigma);
					points[k] = (1 - alpha) * pointsDetectedCur[k] + alpha * points[k];
					//constrainPoint(points[k], imGray.size());
				}

				if (showStabilized)
				{
					renderFace(im, points, cv::Scalar(255, 0, 0), dotRadius);
				}
				else
				{
					renderFace(im, pointsDetectedCur, cv::Scalar(0, 0, 255), dotRadius);
				}
			}

			cv::imshow(winName, im);
			char key = cv::waitKey(1);

			if (key == 32)
			{
				showStabilized = !showStabilized;
			}
			else if (key == 27)
			{
				return EXIT_SUCCESS;
			}

			imPrev = im.clone();
			imGrayPrev = imGray.clone();
			imGrayPrevPyr = imGrayPyr;
			imGrayPyr = std::vector<cv::Mat>();

			isFirstFrame = false;

			count++;
			if (count == NUM_FRAMES_FOR_FPS)
			{
				t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
				fps = NUM_FRAMES_FOR_FPS / t;
				count = 0;
			}
			//cv::putText(im, cv::format("fps %.2f", fps), cv::Point(50, size.height - 50), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
		}
		cap.release();
		cv::destroyAllWindows();
	}
	catch (serialization_error& e)
	{
		
		cout << endl;
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}
}
