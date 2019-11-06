/*
 Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED

 This program is distributed WITHOUT ANY WARRANTY to the
 Plus and Premium membership students of the online course
 titled "Computer Visionfor Faces" by Satya Mallick for
 personal non-commercial use.

 Sharing this code is strictly prohibited without written
 permission from Big Vision LLC.

 For licensing and other inquiries, please email
 spmallick@bigvisionllc.com

 */

#ifndef BIGVISION_faceBlendCommon_HPP_
#define BIGVISION_faceBlendCommon_HPP_


#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

using namespace cv;
using namespace std;

#ifndef M_PI
  #define M_PI 3.14159
#endif


// Constrains points to be inside boundary
void constrainPoint(Point2f &p, Size sz)
{
  p.x = min(max( (double)p.x, 0.0), (double)(sz.width - 1));
  p.y = min(max( (double)p.y, 0.0), (double)(sz.height - 1));

}

// Returns 8 points on the boundary of a rectangle
void getEightBoundaryPoints(Size size, vector<Point2f>& boundaryPts)
{
  int h = size.height, w = size.width;
  boundaryPts.push_back(Point2f(0,0));
  boundaryPts.push_back(Point2f(w/2, 0));
  boundaryPts.push_back(Point2f(w-1,0));
  boundaryPts.push_back(Point2f(w-1, h/2));
  boundaryPts.push_back(Point2f(w-1, h-1));
  boundaryPts.push_back(Point2f(w/2, h-1));
  boundaryPts.push_back(Point2f(0, h-1));
  boundaryPts.push_back(Point2f(0, h/2));
}

// Converts Dlib landmarks into a vector for Point2f
void dlibLandmarksToPoints(dlib::full_object_detection &landmarks, vector<Point2f>& points)
{
  // Loop over all landmark points
  for (int i = 0; i < landmarks.num_parts(); i++)
  {
    Point2f pt(landmarks.part(i).x(), landmarks.part(i).y());
    points.push_back(pt);
  }
}

// Compute similarity transform given two pairs of corresponding points.
// OpenCV requires 3 points for calculating similarity matrix.
// We are hallucinating the third point.
void similarityTransform(std::vector<cv::Point2f>& inPoints, std::vector<cv::Point2f>& outPoints, cv::Mat &tform)
{

  double s60 = sin(60 * M_PI / 180.0);
  double c60 = cos(60 * M_PI / 180.0);

  vector <Point2f> inPts = inPoints;
  vector <Point2f> outPts = outPoints;

  // Placeholder for the third point.
  inPts.push_back(cv::Point2f(0,0));
  outPts.push_back(cv::Point2f(0,0));

  // The third point is calculated so that the three points make an equilateral triangle
  inPts[2].x =  c60 * (inPts[0].x - inPts[1].x) - s60 * (inPts[0].y - inPts[1].y) + inPts[1].x;
  inPts[2].y =  s60 * (inPts[0].x - inPts[1].x) + c60 * (inPts[0].y - inPts[1].y) + inPts[1].y;

  outPts[2].x =  c60 * (outPts[0].x - outPts[1].x) - s60 * (outPts[0].y - outPts[1].y) + outPts[1].x;
  outPts[2].y =  s60 * (outPts[0].x - outPts[1].x) + c60 * (outPts[0].y - outPts[1].y) + outPts[1].y;

  // Now we can use estimateRigidTransform for calculating the similarity transform.
  tform = cv::estimateAffinePartial2D(inPts, outPts);
}

// Normalizes a facial image to a standard size given by outSize.
// The normalization is done based on Dlib's landmark points passed as pointsIn
// After the normalization the left corner of the left eye is at (0.3 * w, h/3 )
// and the right corner of the right eye is at ( 0.7 * w, h / 3) where w and h
// are the width and height of outSize.
void normalizeImagesAndLandmarks(Size outSize, Mat &imgIn, Mat &imgOut, vector<Point2f>& pointsIn, vector<Point2f>& pointsOut)
{
  int h = outSize.height;
  int w = outSize.width;


  vector<Point2f> eyecornerSrc;
  if (pointsIn.size() == 68)
  {
    // Get the locations of the left corner of left eye
    eyecornerSrc.push_back(pointsIn[36]);
    // Get the locations of the right corner of right eye
    eyecornerSrc.push_back(pointsIn[45]);
  }
  else if(pointsIn.size() == 5)
  {
    // Get the locations of the left corner of left eye
    eyecornerSrc.push_back(pointsIn[2]);
    // Get the locations of the right corner of right eye
    eyecornerSrc.push_back(pointsIn[0]);
  }

  vector<Point2f> eyecornerDst;
  // Location of the left corner of left eye in normalized image.
  eyecornerDst.push_back(Point2f( 0.3*w, h/3));
  // Location of the right corner of right eye in normalized image.
  eyecornerDst.push_back(Point2f( 0.7*w, h/3));

  // Calculate similarity transform
  Mat tform;
  similarityTransform(eyecornerSrc, eyecornerDst, tform);

  // Apply similarity transform to input image
  imgOut = Mat::zeros(h, w, CV_32FC3);
  warpAffine(imgIn, imgOut, tform, imgOut.size());

  // Apply similarity transform to landmarks
  transform( pointsIn, pointsOut, tform);

}

// In a vector of points, find the index of point closest to input point.
static int findIndex(vector<Point2f>& points, Point2f &point)
{
  int minIndex = 0;
  double minDistance = norm(points[0] - point);
  for(int i = 1; i < points.size(); i++)
  {
    double distance = norm(points[i] - point);
    if( distance < minDistance )
    {
      minIndex = i;
      minDistance = distance;
    }

  }
  return minIndex;
}


// Calculate Delaunay triangles for set of points
// Returns the vector of indices of 3 points for each triangle
static void calculateDelaunayTriangles(Rect rect, vector<Point2f> &points, vector< vector<int> > &delaunayTri){

  // Create an instance of Subdiv2D
  Subdiv2D subdiv(rect);

  // Insert points into subdiv
  for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
    subdiv.insert(*it);

  // Get Delaunay triangulation
  vector<Vec6f> triangleList;
  subdiv.getTriangleList(triangleList);

  // Variable to store a triangle ( 3 points )
  vector<Point2f> pt(3);

  // Variable to store a triangle as indices from list of points
  vector<int> ind(3);

  for( size_t i = 0; i < triangleList.size(); i++ )
  {
    // The triangle returned by getTriangleList is
    // a list of 6 coordinates of the 3 points in
    // x1, y1, x2, y2, x3, y3 format.
    Vec6f t = triangleList[i];

    // Store triangle as a vector of three points
    pt[0] = Point2f(t[0], t[1]);
    pt[1] = Point2f(t[2], t[3]);
    pt[2] = Point2f(t[4], t[5]);


    if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
    {
      // Find the index of each vertex in the points list
      for(int j = 0; j < 3; j++)
      {
        ind[j] = findIndex(points, pt[j]);
      }
      // Store triangulation as a list of indices
      delaunayTri.push_back(ind);
    }
  }

}

// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(Mat &warpImage, Mat &src, vector<Point2f> &srcTri, vector<Point2f> &dstTri)
{
  // Given a pair of triangles, find the affine transform.
  Mat warpMat = getAffineTransform( srcTri, dstTri );

  // Apply the Affine Transform just found to the src image
  warpAffine( src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

// Warps and alpha blends triangular regions from img1 and img2 to img
void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> t1, vector<Point2f> t2)
{
  // Find bounding rectangle for each triangle
  Rect r1 = boundingRect(t1);
  Rect r2 = boundingRect(t2);
  // Offset points by left top corner of the respective rectangles
  vector<Point2f> t1Rect, t2Rect;
  vector<Point> t2RectInt;
  for(int i = 0; i < 3; i++)
  {
    //tRect.push_back( Point2f( t[i].x - r.x, t[i].y -  r.y) );
    t2RectInt.push_back( Point((int)(t2[i].x - r2.x), (int)(t2[i].y - r2.y)) ); // for fillConvexPoly

    t1Rect.push_back( Point2f( t1[i].x - r1.x, t1[i].y -  r1.y) );
    t2Rect.push_back( Point2f( t2[i].x - r2.x, t2[i].y - r2.y) );
  }

  // Get mask by filling triangle
  Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
  fillConvexPoly(mask, t2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);

  // Apply warpImage to small rectangular patches
  Mat img1Rect, img2Rect;
  img1(r1).copyTo(img1Rect);

  Mat warpImage = Mat::zeros(r2.height, r2.width, img1Rect.type());

  applyAffineTransform(warpImage, img1Rect, t1Rect, t2Rect);

  // Copy triangular region of the rectangular patch to the output image
  multiply(warpImage,mask, warpImage);
  multiply(img2(r2), Scalar(1.0,1.0,1.0) - mask, img2(r2));
  img2(r2) = img2(r2) + warpImage;

}


// Compare dlib rectangle
bool rectAreaComparator(dlib::rectangle &r1, dlib::rectangle &r2)
{ return r1.area() < r2.area(); }


vector<Point2f> getLandmarks(dlib::frontal_face_detector &faceDetector, dlib::shape_predictor &landmarkDetector, Mat &img, float FACE_DOWNSAMPLE_RATIO = 1 )
{

  vector<Point2f> points;

  Mat imgSmall;
  cv::resize(img, imgSmall, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);

  // Convert OpenCV image format to Dlib's image format
  dlib::cv_image<dlib::bgr_pixel> dlibIm(img);
  dlib::cv_image<dlib::bgr_pixel> dlibImSmall(imgSmall);


  // Detect faces in the image
  std::vector<dlib::rectangle> faceRects = faceDetector(dlibImSmall);

  if(faceRects.size() > 0)
  {
    // Pick the biggest face
    dlib::rectangle rect = *std::max_element(faceRects.begin(), faceRects.end(), rectAreaComparator);

    dlib::rectangle scaledRect(
                    (long)(rect.left() * FACE_DOWNSAMPLE_RATIO),
                    (long)(rect.top() * FACE_DOWNSAMPLE_RATIO),
                    (long)(rect.right() * FACE_DOWNSAMPLE_RATIO),
                    (long)(rect.bottom() * FACE_DOWNSAMPLE_RATIO)
                    );

    dlib::full_object_detection landmarks = landmarkDetector(dlibIm, scaledRect);
    dlibLandmarksToPoints(landmarks, points);
  }

  return points;

}


// Warps an image in a piecewise affine manner.
// The warp is defined by the movement of landmark points specified by pointsIn
// to a new location specified by pointsOut. The triangulation beween points is specified
// by their indices in delaunayTri.
void warpImage(Mat &imgIn, Mat &imgOut, vector<Point2f> &pointsIn, vector<Point2f> &pointsOut, vector< vector<int> > &delaunayTri)
{
  // Specify the output image the same size and type as the input image.
  Size size = imgIn.size();
  imgOut = Mat::zeros(size, imgIn.type());

  // Warp each input triangle to output triangle.
  // The triangulation is specified by delaunayTri
  for(size_t j = 0; j < delaunayTri.size(); j++)
  {
    // Input and output points corresponding to jth triangle
    vector<Point2f> tin, tout;

    for(int k = 0; k < 3; k++)
    {
      // Extract a vertex of input triangle
      Point2f pIn = pointsIn[delaunayTri[j][k]];
      // Make sure the vertex is inside the image.
      constrainPoint(pIn, size);

      // Extract a vertex of the output triangle
      Point2f pOut = pointsOut[delaunayTri[j][k]];
      // Make sure the vertex is inside the image.
      constrainPoint(pOut,size);

      // Push the input vertex into input triangle
      tin.push_back(pIn);
      // Push the output vertex into output triangle
      tout.push_back(pOut);
    }
    // Warp pixels inside input triangle to output triangle.
    warpTriangle(imgIn, imgOut, tin, tout);
  }
}



#endif // BIGVISION_faceBlendCommon_HPP_
