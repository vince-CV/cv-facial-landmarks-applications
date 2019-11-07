
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include "renderFace.h"

using namespace dlib;
using namespace std;

#define RESIZE_HEIGHT 480
#define SKIP_FRAMES 2
#define OPENCV_FACE_RENDER


int fastWebcamFLD()
{
	try
	{
		string winName("Fast Facial Landmark Detector");
		cv::namedWindow(winName, cv::WINDOW_NORMAL);
		
		cv::VideoCapture cap(0);
		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			return 1;
		}

		double fps = 30.0;

		cv::Mat im;
		cap >> im;

		cv::Mat imSmall, imDisplay;
		float height = im.rows;


		float RESIZE_SCALE = height / RESIZE_HEIGHT;
		cv::Size size = im.size();


		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor predictor;
		deserialize("C:/Users/xwen2/Desktop/Computer Vision Projects/Face landmarks/data/models/shape_predictor_68_face_landmarks.dat") >> predictor;

		double t = (double)cv::getTickCount();
		int count = 0;

		std::vector<rectangle> faces;
		
		while (1)
		{
			if (count == 0) t = cv::getTickCount();
			cap >> im;
			cv::resize(im, imSmall, cv::Size(), 1.0 / RESIZE_SCALE, 1.0 / RESIZE_SCALE);
			
			cv_image<bgr_pixel> cimgSmall(imSmall);
			cv_image<bgr_pixel> cimg(im);

			if (count % SKIP_FRAMES == 0) // skip frame
			{
				faces = detector(cimgSmall);
			}


			std::vector<full_object_detection> shapes;

			for (unsigned long i = 0; i < faces.size(); ++i)
			{
				rectangle r((long)(faces[i].left() * RESIZE_SCALE),(long)(faces[i].top() * RESIZE_SCALE),
					(long)(faces[i].right() * RESIZE_SCALE),(long)(faces[i].bottom() * RESIZE_SCALE)); // scale up
				
				full_object_detection shape = predictor(cimg, r);
				shapes.push_back(shape);

				renderFace(im, shape);
			}

			//cv::putText(im, cv::format("fps %.2f", fps), cv::Point(50, size.height - 50), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);


			cv::imshow(winName, im);
	
			char key = cv::waitKey(1);
			if (key == 27)
			{
				return EXIT_SUCCESS;
			}

			count++;

			if (count == 100)
			{
				t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
				fps = 100.0 / t;
				count = 0;
			}
		}
		cap.release();
		cv::destroyAllWindows();
	}
	catch (serialization_error& e)
	{

		cout << "No model found!" << endl;
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}
}
