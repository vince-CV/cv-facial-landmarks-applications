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

#ifndef BIGVISION_renderFace_H_
#define BIGVISION_renderFace_H_
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>


// Draw an open or closed polygon between
// start and end indices of full_object_detection
void drawPolyline
(
  cv::Mat &img,
  const dlib::full_object_detection& landmarks,
  const int start,
  const int end,
  bool isClosed = false
)
{
    std::vector <cv::Point> points;
    for (int i = start; i <= end; ++i)
    {
        points.push_back(cv::Point(landmarks.part(i).x(), landmarks.part(i).y()));
    }
    cv::polylines(img, points, isClosed, cv::Scalar(255, 200,0), 2, 16);

}

// Draw face for the 68-point model.
void renderFace(cv::Mat &img, const dlib::full_object_detection& landmarks)
{
    drawPolyline(img, landmarks, 0, 16);           // Jaw line
    drawPolyline(img, landmarks, 17, 21);          // Left eyebrow
    drawPolyline(img, landmarks, 22, 26);          // Right eyebrow
    drawPolyline(img, landmarks, 27, 30);          // Nose bridge
    drawPolyline(img, landmarks, 30, 35, true);    // Lower nose
    drawPolyline(img, landmarks, 36, 41, true);    // Left eye
    drawPolyline(img, landmarks, 42, 47, true);    // Right Eye
    drawPolyline(img, landmarks, 48, 59, true);    // Outer lip
    drawPolyline(img, landmarks, 60, 67, true);    // Inner lip

}

// Draw points on an image.
// Works for any number of points.
void renderFace
(
  cv::Mat &img, // Image to draw the points on
  const std::vector<cv::Point2f> &points, // Vector of points
  cv::Scalar color, // color points
  int radius = 3) // Radius of points.
{

  for (int i = 0; i < points.size(); i++)
  {
    cv::circle(img, points[i], radius, color, -1);
  }

}


#endif // BIGVISION_renderFace_H_
