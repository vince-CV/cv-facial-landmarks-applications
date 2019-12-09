#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include "renderFace.h"

using namespace dlib;
using namespace std;

#define FACE_DOWNSAMPLE_RATIO 2
#define SKIP_FRAMES 10
#define OPENCV_FACE_RENDER


// 3D Model Points of selected landmarks in an arbitrary frame of reference
static std::vector<cv::Point3d> get3dModelPoints()
{
  std::vector<cv::Point3d> modelPoints;

  modelPoints.push_back(cv::Point3d(0.0f, 0.0f, 0.0f)); //The first must be (0,0,0) while using POSIT
  modelPoints.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));
  modelPoints.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));
  modelPoints.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));
  modelPoints.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));
  modelPoints.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));

  return modelPoints;

}

// 2D landmark points from all landmarks
static std::vector<cv::Point2d> get2dImagePoints(full_object_detection &d)
{
  std::vector<cv::Point2d> imagePoints;
  imagePoints.push_back( cv::Point2d( d.part(30).x(), d.part(30).y() ) );    // Nose tip
  imagePoints.push_back( cv::Point2d( d.part(8).x(), d.part(8).y() ) );      // Chin
  imagePoints.push_back( cv::Point2d( d.part(36).x(), d.part(36).y() ) );    // Left eye left corner
  imagePoints.push_back( cv::Point2d( d.part(45).x(), d.part(45).y() ) );    // Right eye right corner
  imagePoints.push_back( cv::Point2d( d.part(48).x(), d.part(48).y() ) );    // Left Mouth corner
  imagePoints.push_back( cv::Point2d( d.part(54).x(), d.part(54).y() ) );    // Right mouth corner
  return imagePoints;

}

// Camera Matrix from focal length and focal center
static cv::Mat getCameraMatrix(float focal_length, cv::Point2d center)
{
  cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
  return cameraMatrix;
}

int webcam_head()
{
  try
  {
    // Create a VideoCapture object
    cv::VideoCapture cap(0);
    // Check if OpenCV is able to read feed from camera
    if (!cap.isOpened())
    {
      cerr << "Unable to connect to camera" << endl;
      return 1;
    }

    // Just a place holder. Actual value calculated after 100 frames.
    double fps = 30.0;
    cv::Mat im;

    // Get first frame and allocate memory.
    cap >> im;
    cv::Mat imSmall, imDisplay;

    // Resize image to reduce computations
    cv::resize(im, imSmall, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);
    cv::resize(im, imDisplay, cv::Size(), 0.5, 0.5);

    cv::Size size = im.size();

    // Load face detection and pose estimation models.
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor predictor;
    deserialize("../data/models/shape_predictor_68_face_landmarks.dat") >> predictor;

    // initiate the tickCounter
    int count = 0;
    double t = (double)cv::getTickCount();

    // variable to store face rectangles
    std::vector<rectangle> faces;

    // Grab and process frames until the main window is closed by the user.
    while(1)
    {

      // start tick counter if count is zero
      if ( count == 0 )
        t = cv::getTickCount();

      // Grab a frame
      cap >> im;

      // Create imSmall by resizing image for face detection
      cv::resize(im, imSmall, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);

      // Change to dlib's image format. No memory is copied.
      cv_image<bgr_pixel> cimgSmall(imSmall);
      cv_image<bgr_pixel> cimg(im);

      // Process frames at an interval of SKIP_FRAMES.
      // This value should be set depending on your system hardware
      // and camera fps.
      // To reduce computations, this value should be increased
      if ( count % SKIP_FRAMES == 0 )
      {
        // Detect faces
        faces = detector(cimgSmall);
      }

      // Pose estimation
      std::vector<cv::Point3d> modelPoints = get3dModelPoints();


      // Iterate over faces
      std::vector<full_object_detection> shapes;
      for (unsigned long i = 0; i < faces.size(); ++i)
      {
        // Since we ran face detection on a resized image,
        // we will scale up coordinates of face rectangle
        rectangle r(
              (long)(faces[i].left() * FACE_DOWNSAMPLE_RATIO),
              (long)(faces[i].top() * FACE_DOWNSAMPLE_RATIO),
              (long)(faces[i].right() * FACE_DOWNSAMPLE_RATIO),
              (long)(faces[i].bottom() * FACE_DOWNSAMPLE_RATIO)
              );

        // Find face landmarks by providing reactangle for each face
        full_object_detection shape = predictor(cimg, r);
        shapes.push_back(shape);

        // Draw landmarks over face
        renderFace(im, shape);

        // get 2D landmarks from Dlib's shape object
        std::vector<cv::Point2d> imagePoints = get2dImagePoints(shape);

        // Camera parameters
        double focal_length = im.cols;
        cv::Mat cameraMatrix = getCameraMatrix(focal_length, cv::Point2d(im.cols/2,im.rows/2));

        // Assume no lens distortion
        cv::Mat distCoeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type);

        // calculate rotation and translation vector using solvePnP
        cv::Mat rotationVector;
        cv::Mat translationVector;
        cv::solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs, rotationVector, translationVector);

        // Project a 3D point (0, 0, 1000.0) onto the image plane.
        // We use this to draw a line sticking out of the nose
        std::vector<cv::Point3d> noseEndPoint3D;
        std::vector<cv::Point2d> noseEndPoint2D;
        noseEndPoint3D.push_back(cv::Point3d(0,0,1000.0));
        cv::projectPoints(noseEndPoint3D, rotationVector, translationVector, cameraMatrix, distCoeffs, noseEndPoint2D);

        // draw line between nose points in image and 3D nose points
        // projected to image plane
        cv::line(im,imagePoints[0], noseEndPoint2D[0], cv::Scalar(255,0,0), 2);

      }

      // Print actual FPS
      cv::putText(im, cv::format("fps %.2f",fps), cv::Point(50, size.height - 50), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);

      // Display it all on the screen

      // Resize image for display
      imDisplay = im;
      cv::resize(im, imDisplay, cv::Size(), 0.5, 0.5);
      cv::imshow("webcam Head Pose", imDisplay);

      // WaitKey slows down the runtime quite a lot
      // So check every 15 frames
      if ( count % 15 == 0)
      {
        int k = cv::waitKey(1);
        // Quit if 'q' or ESC is pressed
        if ( k == 'q' || k == 27)
        {
          break;
        }
      }

      // Calculate actual fps
      // increment frame counter
      count++;
      // calculate fps at an interval of 100 frames
      if ( count == 100)
      {
        t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        fps = 100.0/t;
        count = 0;
      }
    }
  }
  catch(serialization_error& e)
  {
    cout << "Shape predictor model file not found" << endl;
    cout << "Put shape_predictor_68_face_landmarks in models directory" << endl;
    cout << endl << e.what() << endl;
  }
  catch(exception& e)
  {
    cout << e.what() << endl;
  }
}
