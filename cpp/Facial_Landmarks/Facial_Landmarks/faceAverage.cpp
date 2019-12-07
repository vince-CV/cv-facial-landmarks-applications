#include "faceBlendCommon.hpp"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <algorithm>
#include <vector>

// dirent.h is pre-included with *nix like systems
// but not for Windows. So we are trying to include
// this header files based on Operating System
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

// Read jpg files from the directory
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
    /* print all the files and directories within directory */
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

int main( int argc, char** argv)
{
  // Get the face detector
  dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();

  // The landmark detector is implemented in the shape_predictor class
  dlib::shape_predictor landmarkDetector;

  // Load the landmark model
  dlib::deserialize("../data/models/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

  // Directory containing images.
  string dirName = "../data/images/presidents";

  // Add slash to directory name if missing
  if (!dirName.empty() && dirName.back() != '/')
    dirName += '/';

  // Read images in the directory
  vector<string> imageNames, ptsNames;
  readFileNames(dirName, imageNames);

  // Exit program if no images are found or if the number of image files does not match with the number of point files
  if(imageNames.empty())exit(EXIT_FAILURE);

  // Vector of vector of points for all image landmarks.
  vector<vector<Point2f> > allPoints;

  // Read images and perform landmark detection.
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

  // Space for normalized images and points.
  vector <Mat> imagesNorm;
  vector < vector <Point2f> > pointsNorm;

  // Space for average landmark points
  vector <Point2f> pointsAvg(allPoints[0].size());

  // Dimensions of output image
  Size size(600,600);

  // 8 Boundary points for Delaunay Triangulation
  vector <Point2f> boundaryPts;
  getEightBoundaryPoints(size, boundaryPts);

  // Warp images and transform landmarks to output coordinate system,
  // and find average of transformed landmarks.

  for(size_t i = 0; i < images.size(); i++)
  {

    vector <Point2f> points = allPoints[i];

    Mat img;
    normalizeImagesAndLandmarks(size,images[i],img, points, points);

    // Calculate average landmark locations
    for ( size_t j = 0; j < points.size(); j++)
    {
      pointsAvg[j] += points[j] * ( 1.0 / numImages);
    }

    // Append boundary points. Will be used in Delaunay Triangulation
    for ( size_t j = 0; j < boundaryPts.size(); j++)
    {
      points.push_back(boundaryPts[j]);
    }

    pointsNorm.push_back(points);
    imagesNorm.push_back(img);

  }

  // Append boundary points to average points.
  for ( size_t j = 0; j < boundaryPts.size(); j++)
  {
    pointsAvg.push_back(boundaryPts[j]);
  }

  // Calculate Delaunay triangles
  Rect rect(0, 0, size.width, size.height);
  vector< vector<int> > dt;
  calculateDelaunayTriangles(rect, pointsAvg, dt);

  // Space for output image
  Mat output = Mat::zeros(size, CV_32FC3);

  // Warp input images to average image landmarks
  for(size_t i = 0; i < numImages; i++)
  {
    Mat img;
    warpImage(imagesNorm[i],img, pointsNorm[i], pointsAvg, dt);
    // Add image intensities for averaging
    output = output + img;

  }

  // Divide by numImages to get average
  output = output / (double)numImages;

  // Display result
  imshow("image", output);
  waitKey(0);

  return EXIT_SUCCESS;
}
