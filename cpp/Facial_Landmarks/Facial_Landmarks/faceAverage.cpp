#include "faceBlendCommon.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <algorithm>
#include <vector>


#ifdef _WIN32
  #include "dirent.h"
#elif __APPLE__
  #include "TargetConditionals.h"
  #if TARGET_OS_MAC
    #include <dirent.h>
  #else
    #error "Not Mac. Find al alternative to dirent"
  #endif
#elif __linux__
  #include <dirent.h>
#elif __unix__ // all unices not caught above
  #include <dirent.h>
#else
  #error "Unknown compiler"
#endif


void readFileNames(string dirName, vector<string> &imageFnames)
{
  DIR *dir;
  struct dirent *ent;
  int count = 0;

  //image extensions
  string imgExt = "jpg";
  vector<string> files;

  if ((dir = opendir (dirName.c_str())) != NULL)
  {

    while ((ent = readdir (dir)) != NULL)
    {
      if(strcmp(ent->d_name,".") == 0 || strcmp(ent->d_name,"..") == 0 )
      {
        continue;
      }
      string temp_name = ent->d_name;
      files.push_back(temp_name);

    }
    std::sort(files.begin(),files.end());
    for(int it=0;it<files.size();it++)
    {
      string path = dirName;
      string fname=files[it];


      if (fname.find(imgExt, (fname.length() - imgExt.length())) != std::string::npos)
      {
        path.append(fname);
        imageFnames.push_back(path);
      }
    }
    closedir (dir);
  }

}

int face_ave( int argc, char** argv)
{

  dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();

  dlib::shape_predictor landmarkDetector;

  dlib::deserialize("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/models/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

  string dirName = "C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/presidents";

  if (!dirName.empty() && dirName.back() != '/')
    dirName += '/';

  vector<string> imageNames, ptsNames;
  readFileNames(dirName, imageNames);

  if(imageNames.empty())exit(EXIT_FAILURE);

  vector<vector<Point2f> > allPoints;


  vector<Mat> images;
  for(size_t i = 0; i < imageNames.size(); i++)
  {
    Mat img = imread(imageNames[i]);
    if(!img.data)
    {
      cout << "image " << imageNames[i] << " not read properly" << endl;
    }
    else
    {

      vector<Point2f> points = getLandmarks(faceDetector, landmarkDetector, img);
      if (points.size() > 0)
      {
        allPoints.push_back(points);
        img.convertTo(img, CV_32FC3, 1/255.0);
        images.push_back(img);
      }
    }
  }

  if(images.empty())
  {
    cout << "No images found " << endl;
    exit(EXIT_FAILURE);
  }

  int numImages = images.size();

  vector <Mat> imagesNorm;
  vector < vector <Point2f> > pointsNorm;

  vector <Point2f> pointsAvg(allPoints[0].size());

  Size size(600,600);

  vector <Point2f> boundaryPts;
  getEightBoundaryPoints(size, boundaryPts);


  for(size_t i = 0; i < images.size(); i++)
  {

    vector <Point2f> points = allPoints[i];

    Mat img;
    normalizeImagesAndLandmarks(size,images[i],img, points, points);

    for ( size_t j = 0; j < points.size(); j++)
    {
      pointsAvg[j] += points[j] * ( 1.0 / numImages);
    }

    for ( size_t j = 0; j < boundaryPts.size(); j++)
    {
      points.push_back(boundaryPts[j]);
    }

    pointsNorm.push_back(points);
    imagesNorm.push_back(img);

  }

  for ( size_t j = 0; j < boundaryPts.size(); j++)
  {
    pointsAvg.push_back(boundaryPts[j]);
  }


  Rect rect(0, 0, size.width, size.height);
  vector< vector<int> > dt;
  calculateDelaunayTriangles(rect, pointsAvg, dt);


  Mat output = Mat::zeros(size, CV_32FC3);


  for(size_t i = 0; i < numImages; i++)
  {
    Mat img;
    warpImage(imagesNorm[i],img, pointsNorm[i], pointsAvg, dt);
    output = output + img;

  }


  output = output / (double)numImages;


  cv::imshow("image", output);
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
