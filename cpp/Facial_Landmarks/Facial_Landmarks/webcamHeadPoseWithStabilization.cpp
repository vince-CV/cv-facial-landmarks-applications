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
  double Model3D[58][3]={{-7.308957,0.913869,0.000000}, {-6.775290,-0.730814,-0.012799}, {-5.665918,-3.286078,1.022951}, {-5.011779,-4.876396,1.047961}, {-4.056931,-5.947019,1.636229},   {-1.833492,-7.056977,4.061275}, {0.000000,-7.415691,4.070434}, {1.833492,-7.056977,4.061275}, {4.056931,-5.947019,1.636229}, {5.011779,-4.876396,1.047961}, {5.665918,-3.286078,1.022951}, {6.775290,-0.730814,-0.012799}, {7.308957,0.913869,0.000000}, {5.311432,5.485328,3.987654}, {4.461908,6.189018,5.594410}, {3.550622,6.185143,5.712299}, {2.542231,5.862829,4.687939}, {1.789930,5.393625,4.413414}, {2.693583,5.018237,5.072837}, {3.530191,4.981603,4.937805}, {4.490323,5.186498,4.694397}, {-5.311432,5.485328,3.987654}, {-4.461908,6.189018,5.594410}, {-3.550622,6.185143,5.712299}, {-2.542231,5.862829,4.687939}, {-1.789930,5.393625,4.413414}, {-2.693583,5.018237,5.072837}, {-3.530191,4.981603,4.937805}, {-4.490323,5.186498,4.694397}, {1.330353,7.122144,6.903745}, {2.533424,7.878085,7.451034}, {4.861131,7.878672,6.601275}, {6.137002,7.271266,5.200823}, {6.825897,6.760612,4.402142}, {-1.330353,7.122144,6.903745}, {-2.533424,7.878085,7.451034}, {-4.861131,7.878672,6.601275}, {-6.137002,7.271266,5.200823}, {-6.825897,6.760612,4.402142}, {-2.774015,-2.080775,5.048531}, {-0.509714,-1.571179,6.566167}, {0.000000,-1.646444,6.704956}, {0.509714,-1.571179,6.566167}, {2.774015,-2.080775,5.048531}, {0.589441,-2.958597,6.109526}, {0.000000,-3.116408,6.097667}, {-0.589441,-2.958597,6.109526}, {-0.981972,4.554081,6.301271}, {-0.973987,1.916389,7.654050}, {-2.005628,1.409845,6.165652}, {-1.930245,0.424351,5.914376}, {-0.746313,0.348381,6.263227}, {0.000000,1.400000,8.063430}, {0.746313,0.348381,6.263227}, {1.930245,0.424351,5.914376}, {2.005628,1.409845,6.165652}, {0.973987,1.916389,7.654050}, {0.981972,4.554081,6.301271}};
   int alpha=-1;


  modelPoints.push_back(cv::Point3d(Model3D[13][0], Model3D[13][1]  , -alpha*(Model3D[13][2]- Model3D[52][2]))); 
  modelPoints.push_back(cv::Point3d(Model3D[17][0], Model3D[17][1]  , -alpha*(Model3D[17][2]-Model3D[52][2])));
  modelPoints.push_back(cv::Point3d(Model3D[25][0], Model3D[25][1]  , -alpha*(Model3D[25][2]-Model3D[52][2])));
  modelPoints.push_back(cv::Point3d(Model3D[21][0], Model3D[21][1]  , -alpha*(Model3D[21][2]-Model3D[52][2])));
  modelPoints.push_back(cv::Point3d(Model3D[43][0], Model3D[43][1]  , -alpha*(Model3D[43][2]-Model3D[52][2])));
  modelPoints.push_back(cv::Point3d(Model3D[39][0], Model3D[39][1]  , -alpha*(Model3D[39][2]-Model3D[52][2])));
  modelPoints.push_back(cv::Point3d(Model3D[52][0], Model3D[52][1]  , -alpha*(Model3D[52][2]-Model3D[52][2])));


  return modelPoints;

}

static std::vector<cv::Point2d> get2dImagePoints(full_object_detection &d)
{
  std::vector<cv::Point2d> imagePoints;

  imagePoints.push_back( cv::Point2d( d.part(36).x(), d.part(36).y() ) );    
  imagePoints.push_back( cv::Point2d( d.part(39).x(), d.part(39).y() ) );   
  imagePoints.push_back( cv::Point2d( d.part(42).x(), d.part(42).y() ) );  
  imagePoints.push_back( cv::Point2d( d.part(45).x(), d.part(45).y() ) );   
  imagePoints.push_back( cv::Point2d( d.part(48).x(), d.part(48).y() ) );  
  imagePoints.push_back( cv::Point2d( d.part(54).x(), d.part(54).y() ) );  
  imagePoints.push_back( cv::Point2d( d.part(30).x(), d.part(30).y() ) );  
  return imagePoints;

}

static cv::Mat getCameraMatrix(float focal_length, cv::Point2d center)
{
  focal_length=2*focal_length;
  cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
  return cameraMatrix;
}

int Stablized_head_pose()
{
  try
  {

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
        noseEndPoint3D.push_back(cv::Point3d(0,0,20.0));
        cv::projectPoints(noseEndPoint3D, rotationVector, translationVector, cameraMatrix, distCoeffs, noseEndPoint2D);

        cv::line(im,imagePoints[6], noseEndPoint2D[0], cv::Scalar(255,0,0), 2);

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
