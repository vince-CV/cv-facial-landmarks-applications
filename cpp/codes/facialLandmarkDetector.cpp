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
#include <dlib/image_processing.h>
#include "renderFace.hpp"

using namespace dlib;
using namespace std;

// Write landmarks to file
void writeLandmarksToFile(full_object_detection &landmarks, const string &filename)
{
  // Open file
	std::ofstream ofs;
	ofs.open(filename);

  // Loop over all landmark points
  for (int i = 0; i < landmarks.num_parts(); i++)
	{
    // Print x and y coordinates to file
		ofs << landmarks.part(i).x() << " " << landmarks.part(i).y() << endl;

	}
  // Close file
	ofs.close();
}

int main(int argc, char** argv)
{

  // Get the face detector
  frontal_face_detector faceDetector = get_frontal_face_detector();

  // The landmark detector is implemented in the shape_predictor class
  shape_predictor landmarkDetector;

  // Load the landmark model
  deserialize("../data/models/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

  // Read Image
  string imageFilename("../data/images/family.jpg");
  cv::Mat im = cv::imread(imageFilename);

  // landmarks will be stored in results/famil_0.txt
  string landmarksBasename("results/family");

  // Convert OpenCV image format to Dlib's image format
  cv_image<bgr_pixel> dlibIm(im);

  // Detect faces in the image
  std::vector<rectangle> faceRects = faceDetector(dlibIm);
  cout << "Number of faces detected: " << faceRects.size() << endl;

  // Vector to store landmarks of all detected faces
  std::vector<full_object_detection> landmarksAll;

  // Loop over all detected face rectangles
  for (int i = 0; i < faceRects.size(); i++)
  {
    // For every face rectangle, run landmarkDetector
    full_object_detection landmarks = landmarkDetector(dlibIm, faceRects[i]);

    // Print number of landmarks
    if (i == 0) cout << "Number of landmarks : " << landmarks.num_parts() << endl;

    // Store landmarks for current face
    landmarksAll.push_back(landmarks);

    // Draw landmarks on face
    renderFace(im, landmarks);

    // Write landmarks to disk
    std::stringstream landmarksFilename;
    landmarksFilename << landmarksBasename <<  "_"  << i << ".txt";
    cout << "Saving landmarks to " << landmarksFilename.str() << endl;
    writeLandmarksToFile(landmarks, landmarksFilename.str());

  }

  // Save image
  string outputFilename("results/familyLandmarks.jpg");
  cout << "Saving output image to " << outputFilename << endl;
  cv::imwrite(outputFilename, im);

  // Display image
  cv::imshow("Facial Landmark Detector", im);
  cv::waitKey(0);

  return EXIT_SUCCESS;
}

