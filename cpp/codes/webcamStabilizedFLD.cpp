/*
 Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED

 This code is made available to the students of
 the online course titled "Computer Vision for Faces"
 by Satya Mallick for personal non-commercial use.

 Sharing this code is strictly prohibited without written
 permission from Big Vision LLC.

 For licensing and other inquiries, please email
 spmallick@bigvisionllc.com
 */

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include "renderFace.hpp"
#include <math.h>

using namespace dlib;
using namespace std;

#define RESIZE_HEIGHT 360
#define NUM_FRAMES_FOR_FPS 100
#define SKIP_FRAMES 1


// Constrains points to be inside boundary
void constrainPoint(cv::Point2f &p, cv::Size sz)
{
  p.x = cv::min(cv::max( (double)p.x, 0.0), (double)(sz.width - 1));
  p.y = cv::min(cv::max( (double)p.y, 0.0), (double)(sz.height - 1));

}

double interEyeDistance( full_object_detection &shape)
{
	cv::Point2f leftEyeLeftCorner(shape.part(36).x(), shape.part(36).y());
	cv::Point2f rightEyeRightCorner(shape.part(45).x(), shape.part(45).y());
	double distance = norm(rightEyeRightCorner - leftEyeLeftCorner);
	return distance;
}

int main()
{
  try
  {
    string winName("Stabilized Facial Landmark Detector");
    cv::namedWindow(winName, cv::WINDOW_NORMAL);

    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
      cerr << "Unable to connect to camera" << endl;
      return EXIT_FAILURE;
    }

    // Set up optical flow params
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
    cv::Size winSize(101,101);
    double eyeDistance, dotRadius, sigma;
    bool eyeDistanceNotCalculated = true;
    int maxLevel = 5;
    std::vector<uchar> status;
    std::vector<float> err;

    // Just a place holder for frame rate.
    // Actual value calculated after 100 frames.
    double fps = 30.0;

    // Space for current frame, previous frame, and the grayscale versions.
    cv::Mat im, imPrev, imGray, imGrayPrev;

    // Vector of images for storing image pyramids for optical flow
    std::vector<cv::Mat> imGrayPyr, imGrayPrevPyr;

    // Get first frame and allocate memory.
    cap >> imPrev;

    // Convert to grayscale for optical flow calculation
    cv::cvtColor(imPrev, imGrayPrev, cv::COLOR_BGR2GRAY);

    // Build image pyramid for fast optical flow calculation
    cv::buildOpticalFlowPyramid(imGrayPrev, imGrayPrevPyr, winSize, maxLevel);

    // Get image size
    cv::Size size = imPrev.size();

    // imSmall will be used for storing a resized image.
    cv::Mat imSmall;


    // Load Dlib's face detection
    frontal_face_detector detector = get_frontal_face_detector();

    // Load Facial Landmark Detector
    shape_predictor landmarkDetector;
    deserialize("../data/models/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

    // Vector to store face rectangles
    std::vector<rectangle> faces;

    // Space for landmark points
    std::vector<cv::Point2f> points, pointsPrev, pointsDetectedCur, pointsDetectedPrev;

    // Initialize point arrays with (0,0)
    for (unsigned long k = 0; k < landmarkDetector.num_parts(); ++k)
    {
      pointsPrev.push_back(cv::Point2f(0,0));
      points.push_back(cv::Point2f(0,0));
      pointsDetectedCur.push_back(cv::Point2f(0,0));
      pointsDetectedPrev.push_back(cv::Point2f(0,0));
    }

    // First frame is handled differently.
    bool isFirstFrame = true;

    // Show stabilized video flag
    bool showStabilized = false;

    // Variables used for Frame rate calculation
    int count = 0;
    double t;

    while(1)
    {
      if ( count == 0 ) t = (double)cv::getTickCount();

      // Grab a frame
      cap >> im;

      cv::cvtColor(im, imGray, cv::COLOR_BGR2GRAY);
    	float height = im.rows;
    	float IMAGE_RESIZE = height/RESIZE_HEIGHT;
      // Resize image for faster face detection
      cv::resize(im, imSmall, cv::Size(), 1.0/IMAGE_RESIZE, 1.0/IMAGE_RESIZE);

      // Change to dlib's image format. No memory is copied.
      cv_image<bgr_pixel> cimg_small(imSmall);
      cv_image<bgr_pixel> cimg(im);


      // Detect faces. Some frames are skipped for speed.
      if ( count % SKIP_FRAMES == 0 )
      {
        faces = detector(cimg_small);
      }

      if(faces.size() < 1) continue;

      // Space for landmarks on multiple faces.
      std::vector<full_object_detection> shapes;

      // Loop over all faces
      for (unsigned long i = 0; i < faces.size(); ++i)
      {

        // Face detector was found over a smaller image.
        // So, we scale face rectangle to correct size.
        rectangle r(
                    (long)(faces[i].left() * IMAGE_RESIZE),
                    (long)(faces[i].top() * IMAGE_RESIZE),
                    (long)(faces[i].right() * IMAGE_RESIZE),
                    (long)(faces[i].bottom() * IMAGE_RESIZE)
                    );

        // Run landmark detector on current frame
        full_object_detection shape = landmarkDetector(cimg, r);

        // Save current face in a vector
        shapes.push_back(shape);

        // Loop over every point
        for (unsigned long k = 0; k < shape.num_parts(); ++k)
        {

          if (isFirstFrame)
          {
            // If it is the first frame copy the current frame points
            pointsPrev[k].x = pointsDetectedPrev[k].x = shape.part(k).x();
            pointsPrev[k].y = pointsDetectedPrev[k].y = shape.part(k).y();
          }
          else
          {
            // If not the first frame, copy points from previous frame.
            pointsPrev[k] = points[k];
            pointsDetectedPrev[k] = pointsDetectedCur[k];
          }

          // pointsDetectedCur stores results returned by the facial landmark detector
          // points stores the stabilized landmark points
          points[k].x = pointsDetectedCur[k].x = shape.part(k).x();
          points[k].y = pointsDetectedCur[k].y = shape.part(k).y();
        }

        if ( eyeDistanceNotCalculated )
        {
          eyeDistance = interEyeDistance(shape);
          winSize = cv::Size(2 * int(eyeDistance/4) + 1,  2 * int(eyeDistance/4) + 1);
          eyeDistanceNotCalculated = false;
          dotRadius = eyeDistance > 100 ? 3 : 2;
          sigma = eyeDistance * eyeDistance / 400;
        }

        // Build an image pyramid to speed up optical flow
        cv::buildOpticalFlowPyramid(imGray, imGrayPyr, winSize, maxLevel);

        // Predict landmarks based on optical flow. points stores the new location of points.
        cv::calcOpticalFlowPyrLK(imGrayPrevPyr, imGrayPyr, pointsPrev, points, status, err, winSize, maxLevel, termcrit, 0, 0.0001);

        // Final landmark points are a weighted average of
        // detected landmarks and tracked landmarks

        for (unsigned long k = 0; k < shape.num_parts(); ++k)
        {
          double n = norm(pointsDetectedPrev[k] - pointsDetectedCur[k]);
          double alpha = exp(-n*n/sigma);
          points[k] = (1 - alpha) * pointsDetectedCur[k] + alpha * points[k];
          // constrainPoint(points[k], imGray.size());
        }

        if(showStabilized)
        {
          // Show optical flow stabilized points
         renderFace(im, points, cv::Scalar(255,0,0), dotRadius);
        }
        else
        {
          // Show landmark points (unstabilized)
          renderFace(im, pointsDetectedCur, cv::Scalar(0,0,255), dotRadius);
        }
      }

      // Display on screen
      cv::imshow(winName, im);

      // Wait for keypress
      char key = cv::waitKey(1);

      if ( key == 32)
      {
        // If space is pressed toggle showStabilized
        showStabilized = !showStabilized;
      }
      else if (key == 27) // ESC
      {
        // If ESC is pressed, exit.
        return EXIT_SUCCESS;
      }

      // Get ready for next frame.
      imPrev = im.clone();
      imGrayPrev = imGray.clone();
      imGrayPrevPyr = imGrayPyr;
      imGrayPyr = std::vector<cv::Mat>();

      isFirstFrame = false;

      // Calculate framerate
      count++;
      if ( count == NUM_FRAMES_FOR_FPS)
      {
        t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        fps = NUM_FRAMES_FOR_FPS/t;
        count = 0;
      }
      cv::putText(im, cv::format("fps %.2f",fps), cv::Point(50, size.height - 50), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
    }
    cap.release();
    cv::destroyAllWindows();
  }
  catch(serialization_error& e)
  {
    cout << "You need dlib's default face landmarking model file to run this example." << endl;
    cout << "You can get it from the following URL: " << endl;
    cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
    cout << endl << e.what() << endl;
  }
  catch(exception& e)
  {
    cout << e.what() << endl;
  }
}
