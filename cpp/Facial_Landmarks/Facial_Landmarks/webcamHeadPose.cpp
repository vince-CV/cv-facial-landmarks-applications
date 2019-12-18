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


static std::vector<cv::Point3d> get3dModelPoints()
{
  std::vector<cv::Point3d> modelPoints;

  modelPoints.push_back(cv::Point3d(0.0f, 0.0f, 0.0f)); //when using POSIT
  modelPoints.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));
  modelPoints.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));
  modelPoints.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));
  modelPoints.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));
  modelPoints.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));

  return modelPoints;

}


static std::vector<cv::Point2d> get2dImagePoints(full_object_detection &d)
{
  std::vector<cv::Point2d> imagePoints;
  imagePoints.push_back( cv::Point2d( d.part(30).x(), d.part(30).y() ) );    
  imagePoints.push_back( cv::Point2d( d.part(8).x(), d.part(8).y() ) );     
  imagePoints.push_back( cv::Point2d( d.part(36).x(), d.part(36).y() ) );    
  imagePoints.push_back( cv::Point2d( d.part(45).x(), d.part(45).y() ) );   
  imagePoints.push_back( cv::Point2d( d.part(48).x(), d.part(48).y() ) );  
  imagePoints.push_back( cv::Point2d( d.part(54).x(), d.part(54).y() ) );   
  return imagePoints;

}


static cv::Mat getCameraMatrix(float focal_length, cv::Point2d center)
{
  cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
  return cameraMatrix;
}

int head_PRY()
{
  try
  {
    
    cv::VideoCapture cap(0);
   
    double fps = 30.0;
    cv::Mat im;


    cap >> im;
    cv::Mat imSmall, imDisplay;


    cv::resize(im, imSmall, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);
    cv::resize(im, imDisplay, cv::Size(), 0.5, 0.5);

    cv::Size size = im.size();


    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor predictor;
    deserialize("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/models/shape_predictor_68_face_landmarks.dat") >> predictor;


    int count = 0;
    double t = (double)cv::getTickCount();


    std::vector<rectangle> faces;


    while(1)
    {

      if ( count == 0 )
        t = cv::getTickCount();

      cap >> im;

      cv::resize(im, imSmall, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);

      cv_image<bgr_pixel> cimgSmall(imSmall);
      cv_image<bgr_pixel> cimg(im);

      if ( count % SKIP_FRAMES == 0 )
      {
        faces = detector(cimgSmall);
      }


      std::vector<cv::Point3d> modelPoints = get3dModelPoints();

      std::vector<full_object_detection> shapes;
      for (unsigned long i = 0; i < faces.size(); ++i)
      {
        rectangle r(
              (long)(faces[i].left() * FACE_DOWNSAMPLE_RATIO),
              (long)(faces[i].top() * FACE_DOWNSAMPLE_RATIO),
              (long)(faces[i].right() * FACE_DOWNSAMPLE_RATIO),
              (long)(faces[i].bottom() * FACE_DOWNSAMPLE_RATIO)
              );


        full_object_detection shape = predictor(cimg, r);
        shapes.push_back(shape);

        renderFace(im, shape);

        std::vector<cv::Point2d> imagePoints = get2dImagePoints(shape);

        double focal_length = im.cols;
        cv::Mat cameraMatrix = getCameraMatrix(focal_length, cv::Point2d(im.cols/2,im.rows/2));

        cv::Mat distCoeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type);

        cv::Mat rotationVector;
        cv::Mat translationVector;
        cv::solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs, rotationVector, translationVector);

        std::vector<cv::Point3d> noseEndPoint3D;
        std::vector<cv::Point2d> noseEndPoint2D;
        noseEndPoint3D.push_back(cv::Point3d(0,0,1000.0));
        cv::projectPoints(noseEndPoint3D, rotationVector, translationVector, cameraMatrix, distCoeffs, noseEndPoint2D);

        cv::line(im,imagePoints[0], noseEndPoint2D[0], cv::Scalar(255,0,0), 2);

      }

      cv::putText(im, cv::format("fps %.2f",fps), cv::Point(50, size.height - 50), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);


      imDisplay = im;
      cv::resize(im, imDisplay, cv::Size(), 0.5, 0.5);
      cv::imshow("webcam Head Pose", imDisplay);


      if ( count % 15 == 0)
      {
        int k = cv::waitKey(1);

        if ( k == 'q' || k == 27)
        {
          break;
        }
      }


      count++;

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
